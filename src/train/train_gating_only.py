# src/train/train_gating_only.py
"""
Pre-train the gating network only (freeze α,μ) for GSE-Balanced plugin approach.
This learns a good mixture of experts before applying the plugin algorithm.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group_by_threshold
from src.data.datasets import get_cifar100_lt_counts

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'grouping': {
        'threshold': 20,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/',
    },
    'gating_params': {
        'epochs': 25,         # More epochs for better convergence
        'batch_size': 256,
        'lr': 8e-4,          # Slightly reduced for stability
        'weight_decay': 2e-4, # Increased weight decay 
        'balanced_training': True,  # Enable tail-aware training
        'tail_weight': 1.0,  # Even softer weighting for optimal balance
        'use_freq_weighting': True,  # Use frequency-based soft weighting
        'entropy_penalty': 0.0000,  # Giảm entropy_penalty về 0 để tránh ép uniform
        'diversity_penalty': 0.002,  # usage_balance nhỏ để tránh collapse
        'gradient_clip': 0.5,  # NEW: Gradient clipping for stability
    },
    'selective': {
        # Core selective parameters
        'tau': 0.70,              # global target coverage backup
        'tau_by_group': [0.56, 0.44],  # per-group τ_k (head, tail) - updated for Pinball
        'tau_head': 0.56,         # head coverage target (for Pinball loss)
        'tau_tail': 0.44,         # tail coverage target (for Pinball loss)
        'beta_tail': 2.0,         # tail weighting in L_sel (head=1, tail=β_tail)
        'kappa': 25.0,            # sharpness κ for sigmoid smoothing (increased for better calibration)
        'lambda_cov': 15.0,       # λ_cov global (legacy, kept for compatibility)
        'lambda_cov_g': 20.0,     # λ_cov-g group coverage penalty (legacy)
        'lambda_q': 1.0,          # λ_q for Pinball quantile loss
        'lambda_cov_pinball': 20.0, # λ_cov for per-group coverage penalty in Pinball mode
        'lambda_H': 0.01,         # λ_H entropy regularizer
        'lambda_GA': 0.05,        # λ_GA group-aware prior KL
        # Scheduling
        'stageA_epochs': 5,       # Stage A (warm-up) epochs (mixture CE)
        'cycles': 6,              # M cycles (Stage B alternating)
        'epochs_per_cycle': 3,    # B1 epochs per cycle
        'alpha_steps': 2,         # B2 fixed-point steps for α per cycle
        'update_alpha': True,     # Whether to run α updates (disable if relying on cov-g)
        'use_quantile_t': True,   # Update t by quantile each epoch (else learnable not yet supported)
        'alpha_min': 0.80,        # Expanded α range for better tail control
        'alpha_max': 1.60,        # Wider range allows more aggressive tail boosting
        'gamma_alpha': 0.20,      # EMA factor for α
        # μ / λ grid search (B3)
    'lambda_grid': [round(x,2) for x in np.linspace(-2.0,2.0,41)],
        'opt_objective': 'worst',  # 'worst' or 'balanced'
        # Priors & temperatures
        'prior_tail_boost': 1.5,
        'prior_head_boost': 1.5,
        'temperature': {},        # dict name->T (fallback 1.0)
        # Logging
        'log_interval': 50,
        'eps': 1e-8,
    },
    'output': {
        'checkpoints_dir': './checkpoints/gating_pretrained/',
    },
    'seed': 42
}

def gating_diversity_regularizer(gating_weights, mode="usage_balance"):
    """
    Gating diversity regularizer with gradient for gating weights.
    usage_balance: khuyến khích tần suất dùng expert gần đều để tránh collapse
    """
    p_bar = gating_weights.mean(dim=0) + 1e-8   # [E]
    # KL(p_bar || Uniform) = sum p_bar * log(p_bar * E) >= 0 (min=0 tại đều)
    return torch.sum(p_bar * torch.log(p_bar * gating_weights.size(1)))

def mixture_cross_entropy_loss(expert_logits, labels, gating_weights, sample_weights=None, 
                            entropy_penalty=0.0, diversity_penalty=0.0):
    """
    Enhanced cross-entropy loss with diversity promotion.
    L = -log(Σ_e w_e * softmax(logits_e)[y]) + entropy_penalty * H(gating_weights) + diversity_penalty * D(gating)
    """
    # expert_logits: [B, E, C]
    # gating_weights: [B, E]  
    # labels: [B]
    # sample_weights: [B] optional
    
    expert_probs = torch.softmax(expert_logits, dim=-1)  # [B, E, C]
    mixture_probs = torch.einsum('be,bec->bc', gating_weights, expert_probs)  # [B, C]
    
    # Clamp for numerical stability
    mixture_probs = torch.clamp(mixture_probs, min=1e-7, max=1.0-1e-7)
    
    # Cross-entropy: -log(p_y)
    log_probs = torch.log(mixture_probs)  # [B, C]
    nll = torch.nn.functional.nll_loss(log_probs, labels, reduction='none')  # [B]
    
    # Apply sample weights if provided
    if sample_weights is not None:
        nll = nll * sample_weights
    
    ce_loss = nll.mean()
    
    # Add entropy penalty to encourage diversity in gating weights
    entropy_loss = 0.0
    if entropy_penalty > 0:
        # H(p) = -Σ p*log(p), we want to maximize entropy (minimize negative entropy)
        gating_log_probs = torch.log(gating_weights + 1e-8)  # [B, E]
        entropy = -(gating_weights * gating_log_probs).sum(dim=1).mean()  # [B] -> scalar
        entropy_loss = -entropy_penalty * entropy  # Negative because we want to maximize entropy
    
    # Add diversity penalty to promote usage balance of gating (có gradient)
    diversity_loss = 0.0
    if diversity_penalty > 0:
        div_reg = gating_diversity_regularizer(gating_weights, mode="usage_balance")
        diversity_loss = diversity_penalty * div_reg

    return ce_loss + entropy_loss + diversity_loss

def compute_frequency_weights(labels, class_counts, smoothing=0.5):
    """
    Compute frequency-based soft weights: w_i = (freq(y_i))^(-smoothing)
    """
    # Get frequencies for each class
    unique_labels = labels.unique()
    freq_weights = torch.ones_like(labels, dtype=torch.float)
    
    for label in unique_labels:
        class_freq = class_counts[label.item()]
        weight = (class_freq + 1) ** (-smoothing)  # +1 for smoothing
        freq_weights[labels == label] = weight
    
    # Normalize so mean weight = 1
    freq_weights = freq_weights / freq_weights.mean()
    return freq_weights

def load_data_from_logits(config):
    """Load pre-computed logits for training gating."""
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    expert_names = config['experts']['names']
    num_experts = len(expert_names)
    num_classes = config['dataset']['num_classes']
    
    # Use tuneV for gating training 
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    indices_path = splits_dir / 'tuneV_indices.json'
    indices = json.loads(indices_path.read_text())

    # Stack expert logits
    stacked_logits = torch.zeros(len(indices), num_experts, num_classes)
    for i, expert_name in enumerate(expert_names):
        logits_path = logits_root / expert_name / "tuneV_logits.pt"
        stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu')

    labels = torch.tensor(np.array(cifar_train_full.targets)[indices])
    dataset = TensorDataset(stacked_logits, labels)
    
    return DataLoader(dataset, batch_size=config['gating_params']['batch_size'], 
                     shuffle=True, num_workers=4)

# ---------------------------- Selective Mode Utilities ---------------------------- #

def load_two_splits_from_logits(config):
    """Load tuneV (S1) and val_lt (S2) splits with stacked expert logits."""
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    expert_names = config['experts']['names']
    num_experts = len(expert_names)
    num_classes = config['dataset']['num_classes']

    # Base datasets
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    cifar_test_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)

    split_specs = [
        ('tuneV', cifar_train_full, 'tuneV_indices.json'),
        ('val_lt', cifar_test_full, 'val_lt_indices.json')
    ]
    out = {}
    for split_name, base_ds, fname in split_specs:
        idx_path = splits_dir / fname
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing indices file: {idx_path}")
        indices = json.loads(idx_path.read_text())
        stacked = torch.zeros(len(indices), num_experts, num_classes)
        for i, ename in enumerate(expert_names):
            logit_path = logits_root / ename / f"{split_name}_logits.pt"
            if not logit_path.exists():
                raise FileNotFoundError(f"Missing logits file: {logit_path}")
            stacked[:, i, :] = torch.load(logit_path, map_location='cpu')
        labels = torch.tensor(np.array(base_ds.targets)[indices])
        dataset = TensorDataset(stacked, labels)
        out[split_name] = DataLoader(dataset, batch_size=CONFIG['gating_params']['batch_size'], shuffle=True if split_name=='tuneV' else False, num_workers=4)
    return out['tuneV'], out['val_lt']

def build_group_priors(expert_names, K, head_boost=1.5, tail_boost=1.5):
    """Construct simple group-aware priors π_g over experts.
    K groups (assume 2 if not otherwise). For head group boost head-friendly methods (ce / irm),
    for tail group boost tail-friendly methods (balsoftmax/logitadjust/ride/ldam/disalign).
    Returns tensor [K, E]."""
    E = len(expert_names)
    pi = torch.ones(K, E, dtype=torch.float32)
    head_keywords = ['ce', 'irm']
    tail_keywords = ['balsoft', 'logitadjust', 'ride', 'ldam', 'disalign']
    for e, name in enumerate(expert_names):
        lname = name.lower()
        if any(k in lname for k in head_keywords):
            pi[0, e] *= head_boost
        if K > 1 and any(k in lname for k in tail_keywords):
            pi[1, e] *= tail_boost
    # Normalize each group
    pi = pi / pi.sum(dim=1, keepdim=True)
    return pi

def temperature_scale_logits(expert_logits, expert_names, temp_cfg):
    """Apply per-expert temperature scaling if provided (dict name->T)."""
    if not temp_cfg:
        return expert_logits
    scaled = expert_logits.clone()
    for i, name in enumerate(expert_names):
        T = float(temp_cfg.get(name, 1.0))
        if abs(T - 1.0) > 1e-6:
            scaled[:, i, :] = scaled[:, i, :] / T
    return scaled

def fit_temperature_scaling(expert_logits, labels, expert_names, device='cuda'):
    """
    Fit per-expert temperature scaling using Platt scaling on validation set.
    Returns dictionary of optimal temperatures for each expert.
    
    Args:
        expert_logits: [B, E, C] logits from experts
        labels: [B] ground truth labels
        expert_names: list of expert names
        device: computation device
    
    Returns:
        dict: expert_name -> optimal_temperature
    """
    print("Fitting per-expert temperature scaling...")
    temperatures = {}
    
    for i, name in enumerate(expert_names):
        # Extract logits for this expert
        logits_i = expert_logits[:, i, :].to(device)  # [B, C]
        labels_i = labels.to(device)
        
        # Optimize temperature via grid search (simple but effective)
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            scaled_logits = logits_i / temp
            nll = F.cross_entropy(scaled_logits, labels_i).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        temperatures[name] = best_temp
        print(f"  {name}: T={best_temp:.2f} (NLL={best_nll:.4f})")
    
    return temperatures

def compute_mixture_and_w(model, expert_logits):
    """Return gating weights w [B,E] and mixture posterior η [B,C]."""
    with torch.no_grad():
        pass
    gating_features = model.feature_builder(expert_logits)
    w = torch.softmax(model.gating_net(gating_features), dim=1)  # [B,E]
    expert_probs = torch.softmax(expert_logits, dim=-1)          # [B,E,C]
    eta = torch.einsum('be,bec->bc', w, expert_probs)            # [B,C]
    return w, eta, expert_probs

def compute_raw_margin(eta, alpha, mu, class_to_group):
    """m_raw(x) = max_y α_{g(y)} η_y - Σ_y (1/α_{g(y)} - μ_{g(y)}) η_y"""
    device = eta.device
    alpha = alpha.to(device)
    mu = mu.to(device)
    class_to_group = class_to_group.to(device)
    alpha_per_class = alpha[class_to_group]         # [C]
    score = (alpha_per_class * eta).max(dim=1).values
    coeff = 1.0 / alpha_per_class - mu[class_to_group]
    threshold_term = (coeff.unsqueeze(0) * eta).sum(dim=1)
    return score - threshold_term  # [B]

def update_alpha_fixed_point_conditional(eta, y, alpha, mu, t, class_to_group, K, gamma=0.2, alpha_min=0.85, alpha_max=1.15, rho=0.03):
    """Conditional acceptance fixed-point for α (reuse simplified logic)."""
    device = eta.device
    with torch.no_grad():
        raw = compute_raw_margin(eta, alpha, mu, class_to_group)
        accepted = raw > t
        y_groups = class_to_group[y].to(device)
        alpha_hat = torch.ones_like(alpha)
        for k in range(K):
            mask = (y_groups == k)
            if mask.any():
                grp_acc = (accepted & mask).float().sum() / mask.float().sum()
                alpha_hat[k] = grp_acc + 1e-3
        # EMA
        new_alpha = (1 - gamma) * alpha + gamma * alpha_hat
        # Project: enforce geomean=1 then clamp
        new_alpha = new_alpha.clamp_min(alpha_min)
        loga = new_alpha.log()
        new_alpha = torch.exp(loga - loga.mean())
        new_alpha = new_alpha.clamp(min=alpha_min, max=alpha_max)
    # Add directional prior u=[-1,+1] for K=2 to gently push tail up
    if K == 2 and rho > 0:
        u = torch.tensor([-1.0, 1.0], device=new_alpha.device)
        new_alpha = new_alpha + rho * u
        # Re-project
        new_alpha = new_alpha.clamp_min(alpha_min)
        loga2 = new_alpha.log()
        new_alpha = torch.exp(loga2 - loga2.mean()).clamp(min=alpha_min, max=alpha_max)
    return new_alpha

def project_alpha(a, alpha_min=0.85, alpha_max=1.15):
    a = a.clamp_min(alpha_min)
    loga = a.log()
    a = torch.exp(loga - loga.mean())
    return a.clamp(min=alpha_min, max=alpha_max)

def mu_from_lambda_grid(lambdas, K):
    mus = []
    for lam in lambdas:
        if K == 2:
            mus.append(torch.tensor([lam/2.0, -lam/2.0], dtype=torch.float32))
        else:
            raise NotImplementedError("Provide μ grid for K>2")
    return mus

def evaluate_split_with_learned_thresholds(eta, y, alpha, mu, t_param, class_to_group, K, objective='worst_err'):
    """Evaluate split using learned per-group thresholds t_param instead of fitting."""
    with torch.no_grad():
        m_raw = compute_raw_margin(eta, alpha, mu, class_to_group)
        y_groups = class_to_group[y]
        t_groups = t_param[y_groups]  # Select threshold for each sample's group
        s = torch.sigmoid(10.0 * (m_raw - t_groups))  # Use fixed kappa=10.0
        
        pred = eta.argmax(dim=1)
        correct = (pred == y).float()
        
        # Group-wise errors
        group_errs = []
        for g in range(K):
            mask = (y_groups == g)
            if mask.sum() > 0:
                s_g = s[mask]
                correct_g = correct[mask]
                # Rejection is (1 - s), so accepted error is: wrong predictions with high s
                accepted_wrong = (1 - correct_g) * s_g
                total_accepted = s_g.sum().clamp(min=1e-6)
                err_g = accepted_wrong.sum() / total_accepted
                group_errs.append(err_g.item())
            else:
                group_errs.append(0.0)
        
        # Objective computation
        if objective == 'worst_err':
            return max(group_errs), group_errs
        elif objective == 'balanced_err':
            return sum(group_errs) / K, group_errs
        else:
            return group_errs[0], group_errs  # Default to first group

def evaluate_split(eta, y, alpha, mu, t, class_to_group, K, objective='worst'):
    """Compute error on accepted samples under objective."""
    with torch.no_grad():
        raw = compute_raw_margin(eta, alpha, mu, class_to_group)
        accepted = raw > t
        if not accepted.any():
            return 1.0, [1.0]*K
        alpha_per_class = alpha[class_to_group]
        preds = (alpha_per_class.unsqueeze(0) * eta).argmax(dim=1)
        groups = class_to_group[y]
        errs = []
        for k in range(K):
            m = (groups == k) & accepted
            if m.any():
                acc = (preds[m] == y[m]).float().mean().item()
                errs.append(1-acc)
            else:
                errs.append(1.0)
        if objective == 'balanced':
            return float(np.mean(errs)), errs
        else:
            return float(max(errs)), errs

def selective_losses_with_pinball(expert_logits, labels, model, alpha, mu, t_param, cfg_sel, class_to_group, pi_by_group):
    """Compute all selective-training losses including learnable per-group thresholds with Pinball Loss."""
    eps = cfg_sel['eps']
    device = expert_logits.device
    
    # Forward gating & mixture
    gating_features = model.feature_builder(expert_logits)
    raw_w = model.gating_net(gating_features)
    w = torch.softmax(raw_w, dim=1)  # [B,E]
    expert_probs = torch.softmax(expert_logits, dim=-1)  # [B,E,C]
    eta = torch.einsum('be,bec->bc', w, expert_probs)    # [B,C]

    # Raw margin computation
    m_raw = compute_raw_margin(eta, alpha, mu, class_to_group)  # [B]
    
    # Get ground truth groups and corresponding thresholds
    y_groups = class_to_group[labels]  # [B] group indices
    t_g = t_param[y_groups]            # [B] per-sample thresholds
    
    # Soft acceptance probability for selective loss
    s = torch.sigmoid(cfg_sel['kappa'] * (m_raw - t_g))  # [B]

    # p^α: q = α_{g(y)} * η_y then normalize  
    alpha_per_class = alpha[class_to_group].to(eta.device)  # [C]
    q = eta * alpha_per_class.unsqueeze(0)  # [B,C]
    q = q / (q.sum(dim=1, keepdim=True) + eps)
    ce = F.nll_loss(torch.log(q + eps), labels, reduction='none')  # [B]

    # Tail-aware weighting in L_sel
    if len(cfg_sel.get('tau_by_group', [])) == 2 and cfg_sel.get('beta_tail', 1.0) != 1.0:
        beta_tail = cfg_sel['beta_tail']
        sample_w = torch.where(y_groups == 1, torch.tensor(beta_tail, device=device), torch.tensor(1.0, device=device))
        L_sel = (s * ce * sample_w).sum() / (s * sample_w).sum().clamp_min(eps)
    else:
        L_sel = (s * ce).sum() / (s.sum() + eps)

    # Pinball loss for learning quantile thresholds
    tau_head, tau_tail = cfg_sel.get('tau_head', 0.56), cfg_sel.get('tau_tail', 0.44)
    tau_quantile = torch.where(y_groups == 0, 
                               torch.tensor(1 - tau_head, device=device),
                               torch.tensor(1 - tau_tail, device=device))  # quantile level = 1 - coverage
    z = m_raw - t_g  # residual
    pinball = tau_quantile * torch.relu(z) + (1 - tau_quantile) * torch.relu(-z)
    L_q = pinball.mean()

    # Per-group coverage penalty  
    cov_head = s[y_groups == 0].mean() if (y_groups == 0).any() else torch.tensor(0., device=device)
    cov_tail = s[y_groups == 1].mean() if (y_groups == 1).any() else torch.tensor(0., device=device)
    L_cov_pinball = (cov_head - tau_head)**2 + (cov_tail - tau_tail)**2

    # Entropy regularizer λ_H * E[-Σ w log w]
    H_w = -(w * torch.log(w + eps)).sum(dim=1).mean()
    L_H = cfg_sel['lambda_H'] * H_w

    # Group-aware prior KL λ_GA * E[ KL(w || π_{g(y)}) ]
    pi = pi_by_group[y_groups]  # [B,E]
    KL = (w * (torch.log(w + eps) - torch.log(pi + eps))).sum(dim=1).mean()
    L_GA = cfg_sel['lambda_GA'] * KL

    # Total loss with Pinball terms
    lambda_q = cfg_sel.get('lambda_q', 1.0)
    lambda_cov_pinball = cfg_sel.get('lambda_cov_pinball', 20.0)
    
    total = L_sel + lambda_q * L_q + lambda_cov_pinball * L_cov_pinball + L_H + L_GA
    
    diagnostics = {
        'L_sel': L_sel.item(), 'L_q': L_q.item(), 'L_cov_pinball': L_cov_pinball.item(),
        'L_H': L_H.item(), 'L_GA': L_GA.item(), 
        'cov_head': cov_head.item() if torch.is_tensor(cov_head) else cov_head,
        'cov_tail': cov_tail.item() if torch.is_tensor(cov_tail) else cov_tail,
        'entropy_w': H_w.item(), 'kl_w': KL.item(),
        't_head': t_param[0].item(), 't_tail': t_param[1].item(),
    }
    return total, diagnostics, m_raw.detach(), s.detach()

def selective_losses(expert_logits, labels, model, alpha, mu, t, cfg_sel, class_to_group, pi_by_group):
    """Compute all selective-training losses and diagnostics for one batch."""
    eps = cfg_sel['eps']
    # Forward gating & mixture
    gating_features = model.feature_builder(expert_logits)
    raw_w = model.gating_net(gating_features)
    w = torch.softmax(raw_w, dim=1)  # [B,E]
    expert_probs = torch.softmax(expert_logits, dim=-1)  # [B,E,C]
    eta = torch.einsum('be,bec->bc', w, expert_probs)    # [B,C]

    # p^α: q = α_{g(y)} * η_y then normalize
    alpha_per_class = alpha[class_to_group].to(eta.device)  # [C]
    q = eta * alpha_per_class.unsqueeze(0)  # [B,C]
    q = q / (q.sum(dim=1, keepdim=True) + eps)

    ce = F.nll_loss(torch.log(q + eps), labels, reduction='none')  # [B]

    # Raw margin
    m_raw = compute_raw_margin(eta, alpha, mu, class_to_group)
    s = torch.sigmoid(cfg_sel['kappa'] * (m_raw - t))  # [B]

    # Tail-aware weighting in L_sel
    y_groups = class_to_group[labels]
    if len(cfg_sel.get('tau_by_group', [])) == 2 and cfg_sel.get('beta_tail', 1.0) != 1.0:
        beta_tail = cfg_sel['beta_tail']
        sample_w = torch.where(y_groups == 1, torch.tensor(beta_tail, device=eta.device), torch.tensor(1.0, device=eta.device))
        L_sel = (s * ce * sample_w).sum() / (s * sample_w).sum().clamp_min(eps)
    else:
        L_sel = (s * ce).sum() / (s.sum() + eps)
    cov_mean = s.mean()
    L_cov = cfg_sel['lambda_cov'] * (cov_mean - cfg_sel['tau'])**2

    # Group coverage (E[s 1{y∈Gk}] - tau/K)^2
    L_cov_g = torch.zeros(1, device=expert_logits.device)
    if cfg_sel['lambda_cov_g'] > 0:
        K = pi_by_group.size(0)
        tau_by_group = cfg_sel.get('tau_by_group', [])
        if len(tau_by_group) != K:
            tau_by_group = [cfg_sel['tau']/K] * K
        cov_terms = []
        for k in range(K):
            mask = (y_groups == k)
            s_mask_mean = (s[mask].mean() if mask.any() else torch.zeros((), device=s.device))
            cov_k = (s_mask_mean - tau_by_group[k])**2
            cov_terms.append(cov_k)
        L_cov_g = cfg_sel['lambda_cov_g'] * torch.stack(cov_terms).sum()

    # Entropy regularizer λ_H * E[-Σ w log w]
    H_w = -(w * torch.log(w + eps)).sum(dim=1).mean()
    L_H = cfg_sel['lambda_H'] * H_w

    # Group-aware prior KL λ_GA * E[ KL(w || π_{g(y)}) ]
    y_groups = class_to_group[labels]
    pi = pi_by_group[y_groups]  # [B,E]
    KL = (w * (torch.log(w + eps) - torch.log(pi + eps))).sum(dim=1).mean()
    L_GA = cfg_sel['lambda_GA'] * KL

    total = L_sel + L_cov + L_cov_g + L_H + L_GA
    diagnostics = {
        'L_sel': L_sel.item(), 'L_cov': L_cov.item(), 'L_cov_g': L_cov_g.item(),
        'L_H': L_H.item(), 'L_GA': L_GA.item(), 'coverage': cov_mean.item(),
        'mean_s_head': (s[y_groups==0].mean().item() if (y_groups==0).any() else 0.0),
        'mean_s_tail': (s[y_groups==1].mean().item() if (y_groups==1).any() else 0.0),
        'entropy_w': H_w.item(), 'kl_w': KL.item(),
    }
    return total, diagnostics, m_raw.detach(), s.detach()

def run_selective_mode():
    """End-to-end selective gating training (Stage A + alternating Stage B)."""
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    print("=== Selective Gating Training (AR-GSE) ===")
    sel_cfg = CONFIG['selective']

    # Load splits
    S1_loader, S2_loader = load_two_splits_from_logits(CONFIG)
    print(f"Loaded S1 (tuneV) batches: {len(S1_loader)} | S2 (val_lt) batches: {len(S2_loader)}")

    # Calibrate expert temperatures on S1 before any training
    print("\n-- Temperature Calibration --")
    temp_logits = []
    temp_labels = []
    for expert_logits, labels in S1_loader:
        temp_logits.append(expert_logits)
        temp_labels.append(labels)
        if len(temp_logits) * expert_logits.size(0) > 2000:  # Use ~2k samples for calibration
            break
    
    temp_logits = torch.cat(temp_logits)[:2000]
    temp_labels = torch.cat(temp_labels)[:2000]
    
    temperatures = fit_temperature_scaling(temp_logits, temp_labels, CONFIG['experts']['names'], DEVICE)
    sel_cfg['temperature'] = temperatures
    print(f"✅ Fitted temperatures: {temperatures}")

    # Grouping
    class_counts_full = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group_by_threshold(class_counts_full, threshold=CONFIG['grouping']['threshold'])
    K = (class_to_group.max().item() + 1)
    print(f"Groups: K={K} (head={(class_to_group==0).sum().item()} tail={(class_to_group==1).sum().item() if K>1 else 0})")

    # Build model (dynamic feature dim)
    num_experts = len(CONFIG['experts']['names'])
    with torch.no_grad():
        dummy = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        tmp = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], K, 1).to(DEVICE)
        feat_dim = tmp.feature_builder(dummy).size(-1)
        del tmp
    # Recreate model with correct enriched feature dimension (builder now outputs 7E+3)
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], K, feat_dim).to(DEVICE)

    # Init α, μ
    alpha = torch.ones(K, device=DEVICE)
    mu = torch.zeros(K, device=DEVICE)
    
    # Initialize learnable per-group thresholds
    t_param = torch.nn.Parameter(torch.full((K,), -0.70, device=DEVICE))  # init near reasonable level
    print(f"✅ Initialized learnable thresholds t_param: {t_param.tolist()}")

    # Optimizer (gating params + learnable thresholds)
    optimizer = optim.Adam(
        list(model.gating_net.parameters()) + [t_param], 
        lr=CONFIG['gating_params']['lr'], 
        weight_decay=CONFIG['gating_params']['weight_decay']
    )

    # Priors π_g
    pi_by_group = build_group_priors(CONFIG['experts']['names'], K, sel_cfg['prior_head_boost'], sel_cfg['prior_tail_boost']).to(DEVICE)
    print(f"Group priors π_g shape: {pi_by_group.shape}")

    # Stage A: warm-up mixture CE
    print(f"\n-- Stage A: Warm-up ({sel_cfg['stageA_epochs']} epochs) --")
    t = torch.tensor(0.0, device=DEVICE)
    for epoch in range(sel_cfg['stageA_epochs']):
        model.train()
        running = 0.0
        n_batches = 0
        raw_margins_epoch = []
        for b,(expert_logits, labels) in enumerate(S1_loader):
            expert_logits = expert_logits.to(DEVICE)
            labels = labels.to(DEVICE)
            # Temperature scaling (in-place transform)
            expert_logits = temperature_scale_logits(expert_logits, CONFIG['experts']['names'], sel_cfg['temperature'])
            optimizer.zero_grad()
            w, eta, expert_probs = compute_mixture_and_w(model, expert_logits)
            # Standard mixture CE
            mixture_probs = eta.clamp(min=1e-8)
            ce = F.nll_loss(mixture_probs.log(), labels)
            ce.backward()
            optimizer.step()
            running += ce.item()
            n_batches += 1
            # Collect raw margins for threshold init
            with torch.no_grad():
                rm = compute_raw_margin(eta.detach(), alpha, mu, class_to_group.to(DEVICE))
                raw_margins_epoch.append(rm.cpu())
        avg = running / max(1, n_batches)
        raw_cat = torch.cat(raw_margins_epoch)
        if sel_cfg['use_quantile_t']:
            t = torch.quantile(raw_cat.to(DEVICE), 1 - sel_cfg['tau'])
        print(f"StageA Epoch {epoch+1}: loss={avg:.4f} | t={t.item():.4f}")

    # Stage B: alternating cycles
    print(f"\n-- Stage B: Alternating (cycles={sel_cfg['cycles']}, epochs_per_cycle={sel_cfg['epochs_per_cycle']}) --")
    lambda_grid = sel_cfg['lambda_grid']
    mu_candidates = [m.to(DEVICE) for m in mu_from_lambda_grid(lambda_grid, K)]
    best_mu = mu.clone()
    best_score = float('inf')
    
    # Initialize cycle logging
    cycle_logs = []

    for cycle in range(sel_cfg['cycles']):
        print(f"\nCycle {cycle+1}/{sel_cfg['cycles']}")
        
        # Initialize cycle diagnostics
        cycle_diag = {
            'cycle': cycle+1,
            'alpha': alpha.tolist(),
            'mu': mu.tolist(),
            'threshold': t.item()
        }
        # B1: optimize gating with selective-aware loss (using pinball for thresholds)
        for ep in range(sel_cfg['epochs_per_cycle']):
            model.train()
            epoch_diag = []
            raw_collect = []
            for batch_idx,(expert_logits, labels) in enumerate(S1_loader):
                expert_logits = temperature_scale_logits(expert_logits.to(DEVICE), CONFIG['experts']['names'], sel_cfg['temperature'])
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                # Forward pass through gating network only
                w, eta, _ = compute_mixture_and_w(model, expert_logits)
                m_raw = compute_raw_margin(eta, alpha, mu, class_to_group.to(DEVICE))
                
                # Use pinball loss for learning thresholds
                total_loss, diagnostics, m_raw_detached, s_detached = selective_losses_with_pinball(
                    expert_logits, labels, model, alpha, mu, t_param, CONFIG['selective'], 
                    class_to_group.to(DEVICE), pi_by_group
                )
                
                # Extract values from diagnostics for compatibility
                selective_loss = diagnostics['L_sel']
                coverage_current = diagnostics.get('coverage', 0.0)
                coverage_head = diagnostics.get('mean_s_head', 0.0) 
                coverage_tail = diagnostics.get('mean_s_tail', 0.0)
                
                total_loss.backward()
                if CONFIG['gating_params'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(list(model.gating_net.parameters()) + [t_param], CONFIG['gating_params']['gradient_clip'])
                optimizer.step()
                
                # Diagnostics (compatible format)
                with torch.no_grad():
                    w, eta, _ = compute_mixture_and_w(model, expert_logits)
                    m_raw = compute_raw_margin(eta, alpha, mu, class_to_group.to(DEVICE))
                    raw_collect.append(m_raw.cpu())
                    
                    diag = {
                        'L_sel': selective_loss,
                        'L_cov': 0.0,  # Placeholder for coverage loss component
                        'coverage': coverage_current,
                        'mean_s_head': coverage_head,
                        'mean_s_tail': coverage_tail,
                        'entropy_w': w.entropy(dim=1).mean().item() if hasattr(w, 'entropy') else 0.0
                    }
                    epoch_diag.append(diag)
                
                if (batch_idx+1) % sel_cfg['log_interval'] == 0:
                    print(f"  [B1 e{ep+1} b{batch_idx+1}] L_sel={diag['L_sel']:.3f} cov={diag['coverage']:.3f} t_h={t_param[0].item():.3f} t_t={t_param[1].item():.3f}")
            
            # No need to update threshold t via quantile - t_param is learned
            # Summarize epoch diagnostics
            keys = ['L_sel','coverage','mean_s_head','mean_s_tail']
            avg_diag = {k: float(np.mean([d[k] for d in epoch_diag])) for k in keys}
            print(f"  B1 Epoch {ep+1}: t_head={t_param[0].item():.4f} t_tail={t_param[1].item():.4f} | " + 
                  ", ".join([f"{k}={avg_diag[k]:.3f}" for k in keys]))

        # Cache η for S1/S2 for α & μ updates
        def cache_eta(loader):
            model.eval()
            etas = []
            ys = []
            with torch.no_grad():
                for expert_logits, labels in loader:
                    expert_logits = temperature_scale_logits(expert_logits.to(DEVICE), CONFIG['experts']['names'], sel_cfg['temperature'])
                    w, eta_batch, _ = compute_mixture_and_w(model, expert_logits)
                    etas.append(eta_batch.cpu())
                    ys.append(labels.clone())
            return torch.cat(etas).to(DEVICE), torch.cat(ys).to(DEVICE)
        eta_S1, y_S1 = cache_eta(S1_loader)
        eta_S2, y_S2 = cache_eta(S2_loader)

        # B2: update α (optional)
        if sel_cfg['update_alpha']:
            for _ in range(sel_cfg['alpha_steps']):
                alpha = update_alpha_fixed_point_conditional(eta_S1, y_S1, alpha, mu, t, class_to_group.to(DEVICE), K,
                                                             gamma=sel_cfg['gamma_alpha'], alpha_min=sel_cfg['alpha_min'], alpha_max=sel_cfg['alpha_max'], rho=0.03)
            print(f"  Updated α: {alpha.tolist()}")

        # B3: sweep μ via λ grid on S2 (use learned thresholds t_param)
        mu_best_local = mu.clone()
        score_best_local = float('inf')
        for j, mu_cand in enumerate(mu_candidates):
            # Use current learned thresholds directly - no fitting needed
            score, group_errs = evaluate_split_with_learned_thresholds(eta_S2, y_S2, alpha, mu_cand, t_param, class_to_group.to(DEVICE), K, objective=sel_cfg['opt_objective'])
            print(f"  λ[{lambda_grid[j]:.2f}] -> {sel_cfg['opt_objective']} err={score:.4f} t_h={t_param[0].item():.4f} t_t={t_param[1].item():.4f} | groups={['{:.3f}'.format(e) for e in group_errs]}")
            if score < score_best_local - 1e-6:
                score_best_local = score
                mu_best_local = mu_cand.clone()
        # EMA stabilize μ after sweeping candidates
        mu = 0.5 * mu + 0.5 * mu_best_local
        print(f"  Selected μ={mu.tolist()} (score={score_best_local:.4f})")
        if score_best_local < best_score - 1e-6:
            best_score = score_best_local
            best_mu = mu.clone()

        # EMA stabilize α
        alpha = 0.5 * alpha + 0.5 * project_alpha(alpha, sel_cfg['alpha_min'], sel_cfg['alpha_max'])
        
        # Log cycle diagnostics
        cycle_diag.update({
            'final_alpha': alpha.tolist(),
            'final_mu': mu.tolist(), 
            'best_score': score_best_local,
            'global_best': best_score
        })
        cycle_logs.append(cycle_diag)
        
        print(f"  End Cycle {cycle+1}: best_score_so_far={best_score:.4f}")

    # Save checkpoint with learned thresholds
    output_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'gating_net_state_dict': model.gating_net.state_dict(),
        'alpha': alpha.cpu(),
        'mu': best_mu.cpu(),
        't_param': t_param.cpu().detach(),  # Save learned thresholds
        'pi_by_group': pi_by_group.cpu(),
        'config': CONFIG,
        'mode': 'selective_pinball',  # Mark as pinball mode
        'cycle_logs': cycle_logs,
        'temperatures': temperatures
    }
    torch.save(ckpt, output_dir / 'gating_selective.ckpt')
    
    # Save cycle logs to JSON for analysis
    import json
    with open(output_dir / 'selective_training_logs.json', 'w') as f:
        json.dump(cycle_logs, f, indent=2)
    
    print(f"\n✅ Selective Pinball training complete. Saved checkpoint to {output_dir / 'gating_selective.ckpt'}")
    print(f"✅ Cycle logs saved to {output_dir / 'selective_training_logs.json'}")
    print(f"Final α={alpha.tolist()} | μ={best_mu.tolist()} | learned t={t_param.tolist()} | best_score={best_score:.4f}")
    print(f"📊 Learned thresholds - t_head: {t_param[0].item():.4f}, t_tail: {t_param[1].item():.4f}")


def train_gating_only():
    """Train only the gating network with fixed α=1, μ=0."""
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("=== Training Gating Network Only ===")
    
    # Load data
    train_loader = load_data_from_logits(CONFIG)
    print(f"✅ Loaded training data: {len(train_loader)} batches")
    
    # Get split counts from tuneV for sample weighting (không phải full counts)
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    indices_path = Path(CONFIG['dataset']['splits_dir']) / 'tuneV_indices.json'
    indices = json.loads(indices_path.read_text())
    split_labels = torch.tensor(np.array(cifar_train_full.targets)[indices])
    split_counts = torch.bincount(split_labels, minlength=CONFIG['dataset']['num_classes']).float()
    print("✅ Using tuneV split counts (not full counts) for sample weighting")
    
    # Set up grouping (for model creation, but α/μ won't be used)
    class_counts = get_cifar100_lt_counts(imb_factor=100)  # class_to_group still uses threshold from full LT
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=CONFIG['grouping']['threshold'])
    num_groups = class_to_group.max().item() + 1
    
    # Create model with dynamic feature dimension
    num_experts = len(CONFIG['experts']['names'])
    
    # Compute gating feature dimension dynamically
    with torch.no_grad():
        dummy = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        temp_model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, 1).to(DEVICE)
        gating_feature_dim = temp_model.feature_builder(dummy).size(-1)
        del temp_model
    
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    print(f"✅ Dynamic gating feature dimension: {gating_feature_dim}")
    
    # Freeze α, μ at reasonable values
    with torch.no_grad():
        model.alpha.fill_(1.0)  # α = 1 for all groups
        model.mu.fill_(0.0)     # μ = 0 for all groups
        
    # Only optimize gating network parameters (feature_builder has no parameters)
    optimizer = optim.Adam(
        model.gating_net.parameters(),
        lr=CONFIG['gating_params']['lr'],
        weight_decay=CONFIG['gating_params']['weight_decay']
    )
    
    # Training loop with optional balanced training
    best_loss = float('inf')
    
    # Set up class grouping for balanced training
    if CONFIG['gating_params']['balanced_training']:
        # class_counts dùng cho weighting => đổi sang split_counts từ tuneV
        class_counts = split_counts  # Sử dụng tần suất từ tuneV thay vì full counts
        # class_to_group giữ nguyên theo threshold toàn tập, KHÔNG đổi.
        tail_weight = CONFIG['gating_params']['tail_weight']
        use_freq_weighting = CONFIG['gating_params']['use_freq_weighting']
        entropy_penalty = CONFIG['gating_params']['entropy_penalty']
        
        print("✅ Balanced training enabled:")
        print(f"   - Tail weight: {tail_weight}")
        print(f"   - Frequency weighting: {use_freq_weighting}")
        print(f"   - Entropy penalty: {entropy_penalty}")
        print("   - Using tuneV split frequencies for weighting")
    
    for epoch in range(CONFIG['gating_params']['epochs']):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (expert_logits, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            expert_logits = expert_logits.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass: get gating weights
            gating_features = model.feature_builder(expert_logits)
            raw_weights = model.gating_net(gating_features)
            gating_weights = torch.softmax(raw_weights, dim=1)  # [B, E]
            
            # Compute sample weights for balanced training
            sample_weights = None
            if CONFIG['gating_params']['balanced_training']:
                if use_freq_weighting:
                    # Use soft frequency-based weighting
                    sample_weights = compute_frequency_weights(labels.cpu(), class_counts, smoothing=0.5).to(DEVICE)
                else:
                    # Use hard group-based weighting
                    with torch.no_grad():
                        g = class_to_group[labels.cpu()].to(DEVICE)  # 0=head, 1=tail, move to device
                        sample_weights = torch.where(g == 0, 
                                                    torch.tensor(1.0, device=DEVICE),
                                                    torch.tensor(tail_weight, device=DEVICE))
            
            # Mixture cross-entropy loss with sample weights, entropy penalty, and diversity penalty
            loss = mixture_cross_entropy_loss(expert_logits, labels, gating_weights, 
                                            sample_weights, 
                                            entropy_penalty=CONFIG['gating_params']['entropy_penalty'],
                                            diversity_penalty=CONFIG['gating_params']['diversity_penalty'])
            
            loss.backward()
            # Apply gradient clipping if specified
            if CONFIG['gating_params'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.gating_net.parameters(), 
                                             max_norm=CONFIG['gating_params']['gradient_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{CONFIG['gating_params']['epochs']} | Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            output_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'gating_net_state_dict': model.gating_net.state_dict(),
                'num_experts': num_experts,
                'num_classes': CONFIG['dataset']['num_classes'],
                'num_groups': num_groups,
                'gating_feature_dim': gating_feature_dim,
                'config': CONFIG,
            }
            
            ckpt_path = output_dir / 'gating_pretrained.ckpt'
            torch.save(checkpoint, ckpt_path)
            print(f"💾 New best! Saved to {ckpt_path}")
    
    print(f"✅ Gating training complete. Best loss: {best_loss:.4f}")

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Train gating network (pretrain or selective)')
    p.add_argument('--mode', choices=['pretrain','selective'], default='pretrain', help='Training mode')
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == 'pretrain':
        train_gating_only()
    else:
        run_selective_mode()

if __name__ == '__main__':
    main()