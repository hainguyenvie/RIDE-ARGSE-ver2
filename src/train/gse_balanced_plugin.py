# src/train/gse_balanced_plugin.py
"""
GSE-Balanced implementation using plug-in approach with S1/S2 splits.
This avoids the complex primal-dual training and uses fixed-point matching instead.
"""
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torchvision

# Import our custom modules
from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group_by_threshold
from src.data.datasets import get_cifar100_lt_counts

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- CONFIGURATION ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'grouping': {
        'threshold': 20,  # classes with >threshold samples are head
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/',
    },
    'plugin_params': {
        'c': 0.2,  # rejection cost
        'M': 12,   # More iterations for better convergence  
        'gamma': 0.20,  # Reduced EMA for stability
        'alpha_min': 0.85,   # Focused bounds around optimal region
        'alpha_max': 1.15,   # Tighter optimal range
        'lambda_grid': [round(x, 2) for x in np.linspace(-2.0, 2.0, 41)],  # Expanded grid for comprehensive search
        'cov_target': 0.58,  # Lower target for better precision
        'objective': 'worst',  # Switch to 'worst' for tail-focused optimization
        'hybrid_beta': 0.2,  # weight for balanced term in hybrid objective
        'alpha_steps': 5,  # More steps for precision
        'use_conditional_alpha': True,  # use conditional acceptance for alpha updates
        'tie_break_balanced': True,  # use balanced AURC for tie-breaking when optimizing worst
        'use_eg_outer': False,  # use EG-outer for worst-group optimization
        'eg_outer_T': 20,  # EG outer iterations
        'eg_outer_xi': 1.0,  # EG step size
        'use_ema_mu': True,  # Apply EMA smoothing to μ updates
        # --- New selective init options ---
        'use_selective_init': True,       # load gating_selective.ckpt if available
        'freeze_alpha': False,            # if True, keep α from selective init
        'freeze_mu': False,               # if True, skip μ grid optimization (keep μ)
        'expand_grid_if_frozen_mu': False # ignore lambda grid when mu frozen
    },
    'output': {
        'checkpoints_dir': './checkpoints/argse_balanced_plugin/',
    },
    'seed': 42
}

@torch.no_grad()
def cache_eta_mix(gse_model, loader, class_to_group):
    """
    Cache mixture posteriors η̃(x) for S1, S2 by freezing all GSE components.
    
    Args:
        gse_model: Trained AR_GSE model (frozen)
        loader: DataLoader for split 
        class_to_group: [C] class to group mapping
        
    Returns:
        eta_mix: [N, C] mixture posteriors
        labels: [N] ground truth labels
    """
    gse_model.eval()
    etas, labels = [], []
    
    for logits, y in tqdm(loader, desc="Caching η̃"):
        logits = logits.to(DEVICE)
        
        # Get mixture posterior (no margin computation needed)
        expert_posteriors = torch.softmax(logits, dim=-1)  # [B, E, C]
        
        # Get gating weights
        gating_features = gse_model.feature_builder(logits)
        w = torch.softmax(gse_model.gating_net(gating_features), dim=1)  # [B, E]
        
        # Mixture: η̃_y(x) = Σ_e w^(e)(x) * p^(e)(y|x)
        eta = torch.einsum('be,bec->bc', w, expert_posteriors)  # [B, C]
        
        etas.append(eta.cpu())
        labels.append(y.cpu())
    
    return torch.cat(etas), torch.cat(labels)

def compute_raw_margin(eta, alpha, mu, class_to_group):
    # Ensure all tensors are on same device
    device = eta.device
    alpha = alpha.to(device)
    mu = mu.to(device)
    class_to_group = class_to_group.to(device)
    
    # score = max_y α_{g(y)} * η̃_y
    score = (alpha[class_to_group] * eta).max(dim=1).values  # [N]
    # threshold = Σ_y (1/α_{g(y)} - μ_{g(y)}) * η̃_y
    coeff = 1.0 / alpha[class_to_group] - mu[class_to_group]
    threshold = (coeff.unsqueeze(0) * eta).sum(dim=1)        # [N]
    return score - threshold         

def compute_margin(eta, alpha, mu, c, class_to_group):
    raw = compute_raw_margin(eta, alpha, mu, class_to_group)
    return raw - c

def c_for_target_coverage_from_raw(raw_margins, target_cov=0.6):
    # c = quantile(raw_margin, 1 - target_cov)
    return torch.quantile(raw_margins, 1.0 - target_cov).item()


def accepted_and_pred(eta, alpha, mu, c, class_to_group):
    """
    Compute acceptance decisions and predictions.
    
    Returns:
        accepted: [N] boolean mask for accepted samples
        preds: [N] predicted class labels
        margin: [N] margin scores
    """
    margin = compute_margin(eta, alpha, mu, c, class_to_group)
    accepted = (margin >= 0)  # >= giúp bền vững khi có ties
    preds = (alpha[class_to_group] * eta).argmax(dim=1)
    return accepted, preds, margin

def balanced_error_on_S(eta, y, alpha, mu, c, class_to_group, K):
    """
    Compute balanced error rate on a split S.
    
    Returns:
        bal_error: balanced error (average of per-group errors)
        group_errors: list of per-group error rates
    """
    accepted, preds, _ = accepted_and_pred(eta, alpha, mu, c, class_to_group)
    
    if accepted.sum() == 0:
        # No samples accepted -> return worst possible error
        return 1.0, [1.0] * K
    
    y_groups = class_to_group[y]  # [N]
    group_errors = []
    
    for k in range(K):
        mask_k = (y_groups == k) & accepted
        if mask_k.sum() == 0:
            # No accepted samples in group k
            group_errors.append(1.0)
        else:
            group_acc = (preds[mask_k] == y[mask_k]).float().mean().item()
            group_errors.append(1.0 - group_acc)
    
    return float(np.mean(group_errors)), group_errors

def worst_error_on_S(eta, y, alpha, mu, c, class_to_group, K):
    """
    Compute worst-group error rate on a split S.
    
    Returns:
        worst_error: worst-group error (max of per-group errors)
        group_errors: list of per-group error rates
    """
    accepted, preds, _ = accepted_and_pred(eta, alpha, mu, c, class_to_group)
    
    if accepted.sum() == 0:
        # No samples accepted -> return worst possible error
        return 1.0, [1.0] * K
    
    y_groups = class_to_group[y]  # [N]
    group_errors = []
    
    for k in range(K):
        mask_k = (y_groups == k) & accepted
        if mask_k.sum() == 0:
            # No accepted samples in group k
            group_errors.append(1.0)
        else:
            group_acc = (preds[mask_k] == y[mask_k]).float().mean().item()
            group_errors.append(1.0 - group_acc)
    
    return float(max(group_errors)), group_errors

def worst_error_on_S_with_per_group_thresholds(eta, y, alpha, mu, t_group, class_to_group, K):
    """
    Compute worst-group error rate using per-group thresholds t_k.
    
    Args:
        eta: mixture posteriors [N, C]
        y: labels [N]  
        alpha, mu: per-group parameters [K]
        t_group: per-group thresholds [K]
        class_to_group: class -> group mapping [C]
        K: number of groups
        
    Returns:
        worst_error: worst-group error (max of per-group errors)
        group_errors: list of per-group error rates
    """
    # Compute raw margins
    raw_margins = compute_raw_margin(eta, alpha, mu, class_to_group)  # [N]
    
    # Get group for each sample based on ground truth labels
    y_groups = class_to_group[y]  # [N]
    
    # Per-sample threshold based on group
    thresholds_per_sample = t_group[y_groups]  # [N]
    
    # Acceptance based on per-group thresholds
    accepted = (raw_margins > thresholds_per_sample)  # [N]
    
    if accepted.sum() == 0:
        return 1.0, [1.0] * K
    
    # Predictions  
    alpha_per_class = alpha[class_to_group]  # [C]
    preds = (alpha_per_class * eta).argmax(dim=1)  # [N]
    
    group_errors = []
    for k in range(K):
        mask_k = (y_groups == k) & accepted
        if mask_k.sum() == 0:
            group_errors.append(1.0)
        else:
            group_acc = (preds[mask_k] == y[mask_k]).float().mean().item()
            group_errors.append(1.0 - group_acc)
    
    return float(max(group_errors)), group_errors

def hybrid_error_on_S(eta, y, alpha, mu, c, class_to_group, K, beta=0.2):
    """
    Compute hybrid error: worst_error + beta * balanced_error
    
    Returns:
        hybrid_error: worst + beta * balanced
        (worst_error, balanced_error): individual components
    """
    worst_err, group_errs = worst_error_on_S(eta, y, alpha, mu, c, class_to_group, K)
    bal_err = float(np.mean(group_errs))
    
    hybrid_err = worst_err + beta * bal_err
    return hybrid_err, (worst_err, bal_err)

def update_alpha_fixed_point_conditional(eta_S1, y_S1, alpha, mu, c, class_to_group, K, 
                                        gamma=0.3, alpha_min=0.75, alpha_max=1.35):
    """
    Fixed-point alpha update using conditional acceptance rates per group.
    α_k ← (1-γ)α_k + γ·r̂_k where r̂_k = #{acc ∧ y∈G_k} / #{y∈G_k}
    """
    accepted, _, _ = accepted_and_pred(eta_S1, alpha, mu, c, class_to_group)
    y_groups = class_to_group[y_S1]  # [N]
    
    # Conditional acceptance per group: r̂_k = acceptance rate within group k
    alpha_hat = torch.zeros(K, dtype=torch.float32, device=eta_S1.device)
    for k in range(K):
        group_mask = (y_groups == k)
        if group_mask.sum() > 0:  # Group has samples
            group_acceptance_rate = (accepted & group_mask).sum().float() / group_mask.sum().float()
            # Smooth with small constant to avoid zeros
            alpha_hat[k] = group_acceptance_rate + 1e-3
        else:
            alpha_hat[k] = 1.0  # Default value for empty groups
    
    # EMA update
    new_alpha = (1 - gamma) * alpha + gamma * alpha_hat
    
    # Project: geomean=1, then clamp
    new_alpha = new_alpha.clamp_min(alpha_min)
    log_alpha = new_alpha.log()
    new_alpha = torch.exp(log_alpha - log_alpha.mean())
    new_alpha = new_alpha.clamp(min=alpha_min, max=alpha_max)
    
    return new_alpha

def project_alpha(a, alpha_min=0.75, alpha_max=1.35):
    """Project alpha to valid range with geometric mean normalization."""
    a = a.clamp_min(alpha_min)
    loga = a.log()
    a = torch.exp(loga - loga.mean())   # chuẩn hoá geomean=1
    return a.clamp(min=alpha_min, max=alpha_max)

def update_alpha_fixed_point_blend(eta_S1, y_S1, alpha, mu, c, class_to_group, K,
                                   gamma=0.25, blend_lambda=0.25, alpha_min=0.75, alpha_max=1.35):
    """
    Blended alpha update combining joint and conditional methods for stability.
    """
    a_joint = update_alpha_fixed_point(eta_S1, y_S1, alpha, mu, c, class_to_group, K,
                                       gamma=gamma, alpha_min=alpha_min, alpha_max=alpha_max,
                                       use_conditional=False)
    a_cond  = update_alpha_fixed_point_conditional(eta_S1, y_S1, alpha, mu, c, class_to_group, K,
                                                   gamma=gamma, alpha_min=alpha_min, alpha_max=alpha_max)
    a_new = (1 - blend_lambda) * a_joint + blend_lambda * a_cond
    return project_alpha(a_new, alpha_min, alpha_max)

def update_alpha_fixed_point(eta_S1, y_S1, alpha, mu, c, class_to_group, K, 
                           gamma=0.3, alpha_min=0.75, alpha_max=1.35, use_conditional=True):
    """
    Fixed-point alpha update with option for conditional or joint acceptance.
    """
    if use_conditional:
        return update_alpha_fixed_point_conditional(eta_S1, y_S1, alpha, mu, c, class_to_group, K, 
                                                  gamma, alpha_min, alpha_max)
    else:
        # Original joint acceptance method
        accepted, _, _ = accepted_and_pred(eta_S1, alpha, mu, c, class_to_group)
        y_groups = class_to_group[y_S1]  # [N]
        N = y_S1.numel()
        
        # Joint acceptance per group: (1/N) * sum 1{accept}1{y∈G_k}
        joint = torch.zeros(K, dtype=torch.float32, device=eta_S1.device)
        for k in range(K):
            joint[k] = (accepted & (y_groups == k)).sum().float() / float(N)
        
        alpha_hat = K * joint  # target α_k
        
        # EMA update
        new_alpha = (1 - gamma) * alpha + gamma * alpha_hat
        
        # Project: geomean=1, then clamp
        new_alpha = new_alpha.clamp_min(alpha_min)
        log_alpha = new_alpha.log()
        new_alpha = torch.exp(log_alpha - log_alpha.mean())
        new_alpha = new_alpha.clamp(min=alpha_min, max=alpha_max)
        
        return new_alpha

def mu_from_lambda_grid(lambdas, K):
    """
    Convert lambda grid to mu vectors with constraint Σμ_k = 0.
    For K=2: μ = [λ/2, -λ/2]
    """
    mus = []
    for lam in lambdas:
        if K == 2:
            mus.append(torch.tensor([+lam/2.0, -lam/2.0], dtype=torch.float32))
        else:
            # For K>2: could use orthogonal vectors, but not implemented here
            raise NotImplementedError("Provide a mu grid for K>2")
    return mus

def gse_balanced_plugin(eta_S1, y_S1, eta_S2, y_S2, class_to_group, K,
                    c, M=10, lambda_grid=None,
                    alpha_init=None, gamma=0.3, cov_target=0.6, 
                    objective='balanced', hybrid_beta=0.2, alpha_steps=4,
                    use_conditional_alpha=True, tie_break_balanced=True, use_ema_mu=True):
    """
    Main GSE-Balanced plugin algorithm with improved optimization strategies.
    
    Args:
        eta_S1, y_S1: cached posteriors and labels for S1 (tuning split)
        eta_S2, y_S2: cached posteriors and labels for S2 (validation split)
        class_to_group: [C] class to group mapping
        K: number of groups
        c: rejection cost (unused, kept for compatibility)
        M: number of plugin iterations
        lambda_grid: values of λ to sweep over (defaults to expanded [-2.0, 2.0])
        alpha_init: initial alpha values
        gamma: EMA factor for alpha updates
        cov_target: target coverage for threshold fitting
        objective: 'balanced' or 'worst' group optimization
        use_ema_mu: whether to apply EMA smoothing to μ updates
        objective: 'worst', 'balanced', or 'hybrid'
        hybrid_beta: weight for balanced term if objective='hybrid'
        alpha_steps: number of fixed-point steps for alpha updates
        use_conditional_alpha: use conditional acceptance for alpha updates
        tie_break_balanced: use balanced AURC for tie-breaking in worst optimization
        
    Returns:
        best_alpha: optimal α* 
        best_mu: optimal μ*
        best_score: best error on S2 (according to objective)
    """
    device = eta_S1.device
    y_S1 = y_S1.to(device)
    y_S2 = y_S2.to(device)
    class_to_group = class_to_group.to(device)
    
    # Default to expanded λ grid if not provided
    if lambda_grid is None:
        lambda_grid = [round(x, 2) for x in np.linspace(-2.0, 2.0, 41)]
        
    mu_candidates = [mu.to(device) for mu in mu_from_lambda_grid(lambda_grid, K)]
    print(f"Sweeping {len(mu_candidates)} μ candidates (λ ∈ [{min(lambda_grid):.1f}, {max(lambda_grid):.1f}])")
    
    # Initialize α
    if alpha_init is None:
        alpha = torch.ones(K, dtype=torch.float32, device=device)
    else:
        alpha = alpha_init.clone().float().to(device)
    
    best_alpha = alpha.clone()
    best_mu = torch.zeros(K, dtype=torch.float32, device=device)
    best_score = float('inf')
    best_balanced_score = float('inf')  # For tie-breaking
    
    # EMA buffer for μ if enabled  
    mu_ema = torch.zeros(K, dtype=torch.float32, device=device) if use_ema_mu else None
    
    print(f"Starting GSE-Balanced plugin with {M} iterations, {len(mu_candidates)} μ candidates")
    print(f"Objective: {objective}, Coverage target: {cov_target:.2f}")
    print(f"Alpha method: {'conditional' if use_conditional_alpha else 'joint'}, EMA μ: {use_ema_mu}")
    if objective == 'hybrid':
        print(f"Hybrid beta: {hybrid_beta}")
    
    best_t = None
    best_lambda_idx = None  # Track best lambda index for adaptive expansion
    
    for m in range(M):
        print(f"\n--- Plugin Iteration {m+1}/{M} ---")
        for i, mu in enumerate(mu_candidates):
            alpha_cur = alpha.clone()
            t_cur = None

            # Multiple fixed-point steps for α; refit threshold each step
            for step in range(alpha_steps):
                raw_S1 = compute_raw_margin(eta_S1, alpha_cur, mu, class_to_group)
                t_cur = c_for_target_coverage_from_raw(raw_S1, cov_target)  # fit threshold on S1
                # update α using blended fixed-point approach for stability
                if use_conditional_alpha:
                    alpha_cur = update_alpha_fixed_point_blend(
                        eta_S1, y_S1, alpha_cur, mu, t_cur, class_to_group, K,
                        gamma=gamma, blend_lambda=0.25  # có thể tune 0.15–0.35
                    )
                else:
                    alpha_cur = update_alpha_fixed_point(
                        eta_S1, y_S1, alpha_cur, mu, t_cur, class_to_group, K,
                        gamma=gamma, use_conditional=use_conditional_alpha
                    )

            # Evaluate on S2 with same t_cur according to objective
            if objective == 'worst':
                error_score, group_errs = worst_error_on_S(
                    eta_S2, y_S2, alpha_cur, mu, t_cur, class_to_group, K
                )
                error_type = "worst"
                # Also compute balanced for tie-breaking
                balanced_score = float(np.mean(group_errs))
            elif objective == 'balanced':
                error_score, group_errs = balanced_error_on_S(
                    eta_S2, y_S2, alpha_cur, mu, t_cur, class_to_group, K
                )
                error_type = "bal"
                balanced_score = error_score
            elif objective == 'hybrid':
                error_score, (worst_err, bal_err) = hybrid_error_on_S(
                    eta_S2, y_S2, alpha_cur, mu, t_cur, class_to_group, K, hybrid_beta
                )
                error_type = f"hybrid(w={worst_err:.3f},b={bal_err:.3f})"
                balanced_score = bal_err
            else:
                raise ValueError(f"Unknown objective: {objective}")

            # Selection logic with tie-breaking
            is_better = False
            if error_score < best_score - 1e-6:  # Significant improvement
                is_better = True
            elif abs(error_score - best_score) < 1e-6 and tie_break_balanced:  # Tie-break
                if balanced_score < best_balanced_score - 1e-6:
                    is_better = True
            
            if is_better:
                best_score = error_score
                best_balanced_score = balanced_score
                best_alpha = alpha_cur.clone()
                
                # Apply EMA to μ updates if enabled
                if use_ema_mu and mu_ema is not None:
                    mu_ema = 0.7 * mu_ema + 0.3 * mu
                    best_mu = mu_ema.clone()
                else:
                    best_mu = mu.clone()
                    
                best_t = t_cur
                best_lambda_idx = i  # Track best lambda index
                tie_info = f" (tie-break: bal={balanced_score:.4f})" if abs(error_score - best_score) < 1e-6 else ""
                print(f"  λ={lambda_grid[i]:.2f}: NEW BEST! {error_type}={error_score:.4f}, "
                    f"α=[{alpha_cur[0]:.3f},{alpha_cur[1]:.3f}], "
                    f"μ=[{best_mu[0]:.3f},{best_mu[1]:.3f}], t={t_cur:.3f}{tie_info}")
            else:
                print(f"  λ={lambda_grid[i]:.2f}: {error_type}={error_score:.4f} (t={t_cur:.3f})")

        alpha = (0.5 * alpha + 0.5 * best_alpha).clone()
        print(f"[Iter {m+1}] Current best: {objective}={best_score:.4f}, t*={best_t:.3f}")
        
        # Adaptive lambda grid expansion when best hits boundary
        if best_lambda_idx is not None and best_lambda_idx in [0, len(lambda_grid)-1]:
            step = lambda_grid[1] - lambda_grid[0] if len(lambda_grid) > 1 else 0.25
            if best_lambda_idx == 0:
                new_min = lambda_grid[0] - 4*step
                lambda_grid = np.linspace(new_min, lambda_grid[-1], len(lambda_grid)+4).tolist()
            else:
                new_max = lambda_grid[-1] + 4*step
                lambda_grid = np.linspace(lambda_grid[0], new_max, len(lambda_grid)+4).tolist()
            mu_candidates = [mu.to(device) for mu in mu_from_lambda_grid(lambda_grid, K)]
            print(f"↔️ Expanded lambda_grid to [{lambda_grid[0]:.2f}, {lambda_grid[-1]:.2f}] ({len(lambda_grid)} pts)")
            # Reset best_lambda_idx for next iteration
            best_lambda_idx = None

    return best_alpha.cpu(), best_mu.cpu(), best_score, float(best_t)

def calculate_optimal_c_from_eta(eta, target_coverage=0.6):
    """
    Calculate optimal c based on quantile of max_y η̃_y to achieve target coverage.
    c = 1 - quantile_q(max η̃)
    """
    max_probs = eta.max(dim=1).values  # [N]
    optimal_c = 1.0 - torch.quantile(max_probs, target_coverage)
    return optimal_c.item()

def load_data_from_logits(config):
    """Load pre-computed logits for tuneV (S1) and val_lt (S2) splits."""
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    expert_names = config['experts']['names']
    num_experts = len(expert_names)
    num_classes = config['dataset']['num_classes']
    
    dataloaders = {}
    
    # Base datasets
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    cifar_test_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    
    # Use tuneV (S1) and val_lt (S2) splits
    splits_config = [
        {'split_name': 'tuneV', 'base_dataset': cifar_train_full, 'indices_file': 'tuneV_indices.json'},
        {'split_name': 'val_lt', 'base_dataset': cifar_test_full, 'indices_file': 'val_lt_indices.json'}
    ]
    
    for split in splits_config:
        split_name = split['split_name']
        base_dataset = split['base_dataset']
        indices_path = splits_dir / split['indices_file']
        print(f"Loading data for split: {split_name}")
        
        if not indices_path.exists():
            raise FileNotFoundError(f"Missing indices file: {indices_path}")
        indices = json.loads(indices_path.read_text())

        # Stack expert logits
        stacked_logits = torch.zeros(len(indices), num_experts, num_classes)
        for i, expert_name in enumerate(expert_names):
            logits_path = logits_root / expert_name / f"{split_name}_logits.pt"
            if not logits_path.exists():
                raise FileNotFoundError(f"Missing logits file: {logits_path}")
            stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu')

        labels = torch.tensor(np.array(base_dataset.targets)[indices])
        dataset = TensorDataset(stacked_logits, labels)
        dataloaders[split_name] = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    return dataloaders['tuneV'], dataloaders['val_lt']

def main():
    """Main GSE-Balanced plugin training."""
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("=== GSE-Balanced Plugin Training ===")
    
    # 1) Load data
    S1_loader, S2_loader = load_data_from_logits(CONFIG)
    print(f"✅ Loaded S1 (tuneV): {len(S1_loader)} batches")
    print(f"✅ Loaded S2 (val_lt): {len(S2_loader)} batches")
    
    # 2) Set up grouping
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=CONFIG['grouping']['threshold'])
    num_groups = class_to_group.max().item() + 1
    head = (class_to_group == 0).sum().item()
    tail = (class_to_group == 1).sum().item()
    print(f"✅ Groups: {head} head classes, {tail} tail classes")
    
    # 3) Load frozen GSE model (with pre-trained gating if available)
    num_experts = len(CONFIG['experts']['names'])
    
    # Dynamic gating feature dimension computation
    with torch.no_grad():
        dummy_logits = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        temp_model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, 1).to(DEVICE)
        gating_feature_dim = temp_model.feature_builder(dummy_logits).size(-1)
        del temp_model
    print(f"✅ Dynamic gating feature dim: {gating_feature_dim}")
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    
    # Try to load pre-trained gating weights
    gating_ckpt_path = Path('./checkpoints/gating_pretrained/') / CONFIG['dataset']['name'] / 'gating_pretrained.ckpt'
    
    selective_ckpt_path = gating_ckpt_path.parent / 'gating_selective.ckpt'
    selective_loaded = False
    if CONFIG['plugin_params'].get('use_selective_init', False) and selective_ckpt_path.exists():
        try:
            sel_ckpt = torch.load(selective_ckpt_path, map_location=DEVICE, weights_only=False)
            
            # Check dimension compatibility before loading
            saved_state = sel_ckpt['gating_net_state_dict']
            current_state = model.gating_net.state_dict()
            
            compatible = True
            for key in saved_state.keys():
                if key in current_state and saved_state[key].shape != current_state[key].shape:
                    print(f"⚠️  Dimension mismatch for {key}: saved {saved_state[key].shape} vs current {current_state[key].shape}")
                    compatible = False
            
            if compatible:
                model.gating_net.load_state_dict(saved_state)
                print(f"📂 Loaded selective gating checkpoint: {selective_ckpt_path}")
                # Initialize α, μ, threshold from selective if not frozen logic will apply later
                init_alpha = sel_ckpt.get('alpha', None)
                init_mu = sel_ckpt.get('mu', None) 
                init_t = sel_ckpt.get('t', None)
                selective_loaded = True
            else:
                print("❌ Selective checkpoint incompatible with enriched features. Using random init.")
        except Exception as e:
            print(f"⚠️ Failed to load selective checkpoint ({e}). Fallback to gating_pretrained or random.")
    if not selective_loaded:
        if gating_ckpt_path.exists():
            try:
                print(f"📂 Loading pre-trained gating from {gating_ckpt_path}")
                gating_ckpt = torch.load(gating_ckpt_path, map_location=DEVICE, weights_only=False)
                
                # Check dimension compatibility 
                saved_state = gating_ckpt['gating_net_state_dict']
                current_state = model.gating_net.state_dict()
                
                compatible = True
                for key in saved_state.keys():
                    if key in current_state and saved_state[key].shape != current_state[key].shape:
                        print(f"⚠️  Dimension mismatch for {key}: saved {saved_state[key].shape} vs current {current_state[key].shape}")
                        compatible = False
                
                if compatible:
                    model.gating_net.load_state_dict(saved_state)
                    print("✅ Pre-trained gating loaded successfully!")
                else:
                    print("❌ Pre-trained checkpoint incompatible with enriched features. Using random init.")
                    
                init_alpha = None
                init_mu = None
                init_t = None
            except Exception as e:
                print(f"⚠️ Failed to load pre-trained checkpoint ({e}). Using random initialization.")
                init_alpha = None
                init_mu = None
                init_t = None
        else:
            print("⚠️  No gating checkpoint found. Using random initialization.")
            print("   Consider running train_gating_only.py --mode pretrain or --mode selective first.")
            init_alpha = None
            init_mu = None
            init_t = None
    
    # Initialize α, μ with reasonable values (will be optimized by plugin)
    with torch.no_grad():
        model.alpha.fill_(1.0)
        model.mu.fill_(0.0)
    
    # 4) Cache η̃ for both splits
    print("\n=== Caching mixture posteriors ===")
    eta_S1, y_S1 = cache_eta_mix(model, S1_loader, class_to_group)
    eta_S2, y_S2 = cache_eta_mix(model, S2_loader, class_to_group) 
    
    print(f"✅ Cached η̃_S1: {eta_S1.shape}, y_S1: {y_S1.shape}")
    print(f"✅ Cached η̃_S2: {eta_S2.shape}, y_S2: {y_S2.shape}")
    
    # Optional: save cached posteriors for multiple experiments
    cache_dir = Path('./cache/eta_mix')
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'eta': eta_S1, 'y': y_S1}, cache_dir / 'S1_tuneV.pt')
    torch.save({'eta': eta_S2, 'y': y_S2}, cache_dir / 'S2_val_lt.pt')
    print(f"💾 Saved cached posteriors to {cache_dir}")
    
    # 5) Auto-calibrate c if needed
    current_c = CONFIG['plugin_params']['c']
    optimal_c = calculate_optimal_c_from_eta(eta_S1, target_coverage=0.6)
    print(f"📊 Current c={current_c:.3f}, Optimal c for 60% coverage={optimal_c:.3f}")
    
    # Use current c or switch to optimal
    
    # 6) Run GSE-Balanced plugin with improved parameters
    cov_target = CONFIG['plugin_params']['cov_target']
    objective = CONFIG['plugin_params']['objective']
    
    # Check if we should use EG-outer for worst-group optimization
    if objective == 'worst' and CONFIG['plugin_params']['use_eg_outer']:
        print(f"\n=== Running GSE Worst-Group with EG-Outer (T={CONFIG['plugin_params']['eg_outer_T']}, xi={CONFIG['plugin_params']['eg_outer_xi']}) ===")
        
        from src.train.gse_worst_eg import worst_group_eg_outer
        
        alpha_star, mu_star, t_group_star, beta_star, eg_hist = worst_group_eg_outer(
            eta_S1.to(DEVICE), y_S1.to(DEVICE),
            eta_S2.to(DEVICE), y_S2.to(DEVICE),
            class_to_group.to(DEVICE), K=num_groups,
            T=CONFIG['plugin_params']['eg_outer_T'], 
            xi=CONFIG['plugin_params']['eg_outer_xi'],
            lambda_grid=np.linspace(-1.2, 1.2, 41).tolist(),
            M=8, alpha_steps=4, 
            target_cov_by_group=[0.55, 0.45] if num_groups==2 else [cov_target]*num_groups,
            gamma=CONFIG['plugin_params']['gamma'],
            use_conditional_alpha=CONFIG['plugin_params']['use_conditional_alpha']  # 🔧 Fix: Use config flag
        )
        
        
        # ✅ Use the per-group thresholds directly from EG-outer (consistent!)
        t_group = t_group_star.cpu().numpy().tolist() if hasattr(t_group_star, 'cpu') else t_group_star
        print(f"✅ Using per-group thresholds from EG-outer: {t_group}")
        
        # No need to re-fit - maintains consistency between optimization and evaluation!
        
        # Compute final best score using per-group thresholds
        from src.train.gse_balanced_plugin import worst_error_on_S_with_per_group_thresholds  
        best_score, _ = worst_error_on_S_with_per_group_thresholds(eta_S2.to(DEVICE), y_S2.to(DEVICE), 
                                       alpha_star.to(DEVICE), mu_star.to(DEVICE), 
                                       t_group_star.to(DEVICE), class_to_group.to(DEVICE), num_groups)
        
        
    else:
        print(f"\n=== Running GSE-Balanced Plugin (target_coverage={cov_target:.2f}, objective={objective}) ===")
        # If selective init loaded and freeze flags set, adapt grid / parameters
        lambda_grid_cfg = CONFIG['plugin_params']['lambda_grid']
        if selective_loaded and CONFIG['plugin_params'].get('freeze_mu', False):
            print("🔒 freeze_mu=True -> skipping μ grid search, using μ from selective checkpoint")
            # Set grid to single dummy to keep code path consistent
            lambda_grid_cfg = [0.0]
            if isinstance(init_mu, torch.Tensor):
                mu_init_tensor = init_mu.to(DEVICE)
            else:
                mu_init_tensor = None
        alpha_init = init_alpha if (selective_loaded and init_alpha is not None and CONFIG['plugin_params'].get('freeze_alpha', False)) else None
        if alpha_init is not None:
            print(f"🔒 freeze_alpha=True -> using α from selective init: {alpha_init.tolist()}")
        if selective_loaded and init_t is not None:
            print(f"ℹ️ Using selective raw-margin threshold init t={init_t:.4f} (will refit per plugin iteration)")

        alpha_star, mu_star, best_score, t_star = gse_balanced_plugin(
            eta_S1=eta_S1.to(DEVICE),
            y_S1=y_S1.to(DEVICE),
            eta_S2=eta_S2.to(DEVICE), 
            y_S2=y_S2.to(DEVICE),
            class_to_group=class_to_group.to(DEVICE),
            K=num_groups,
            c=None,
            M=CONFIG['plugin_params']['M'],
            lambda_grid=lambda_grid_cfg,
            alpha_init=alpha_init.to(DEVICE) if isinstance(alpha_init, torch.Tensor) else None,
            gamma=CONFIG['plugin_params']['gamma'],
            cov_target=CONFIG['plugin_params']['cov_target'],
            objective=CONFIG['plugin_params']['objective'],
            hybrid_beta=CONFIG['plugin_params']['hybrid_beta'],
            alpha_steps=CONFIG['plugin_params']['alpha_steps'],
            use_conditional_alpha=CONFIG['plugin_params']['use_conditional_alpha'],
            tie_break_balanced=CONFIG['plugin_params']['tie_break_balanced'],
            use_ema_mu=CONFIG['plugin_params'].get('use_ema_mu', True),
        )
        # If μ was frozen, overwrite with init μ
        if selective_loaded and CONFIG['plugin_params'].get('freeze_mu', False) and 'mu_init_tensor' in locals() and mu_init_tensor is not None:
            mu_star = mu_init_tensor.cpu()
    
    # print(f"Best raw-margin threshold t* (fitted on S1): {t_star:.3f}")  # Skip since using per-group

    # Use global threshold or per-group thresholds based on method
    if objective == 'worst' and CONFIG['plugin_params']['use_eg_outer']:
        # Use per-group thresholds from EG-outer directly (ensuring consistency)
        t_group = t_group_star.cpu().numpy().tolist() if hasattr(t_group_star, 'cpu') else t_group_star
        print(f"✅ Using per-group thresholds from EG-outer: {t_group}")
        source_info = "EG-outer per-group thresholds"
    else:
        # Use global threshold from plugin
        t_group = [t_star] * num_groups  # Convert single threshold to per-group format
        print(f"✅ Using global threshold t*={t_star:.3f} for all groups")
        source_info = f'gse_{objective}_plugin'
    
    # 7) Save results
    print("\n🎉 GSE-Balanced Plugin Complete!")
    print(f"α* = [{alpha_star[0]:.4f}, {alpha_star[1]:.4f}]")
    print(f"μ* = [{mu_star[0]:.4f}, {mu_star[1]:.4f}]")
    print(f"Best {objective} error on S2 = {best_score:.4f}")
    
    # Define target coverage by group
    target_cov_by_group = [0.55, 0.45] if num_groups == 2 else [cov_target] * num_groups
    
    # Save checkpoint with optimal parameters
    output_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'alpha': alpha_star,
        'mu': mu_star,
        'class_to_group': class_to_group,
        'num_groups': num_groups,
        'threshold': t_group,  # Using the appropriate thresholds (per-group or global)
        't_group': t_group,   # Per-group thresholds
        'per_group_threshold': objective == 'worst' and CONFIG['plugin_params']['use_eg_outer'],  # Flag based on method
        'target_cov_by_group': target_cov_by_group,
        'best_score': best_score,
        'source': source_info,  # Method used to generate results
        'config': CONFIG,
        'gating_net_state_dict': model.gating_net.state_dict(),
    }
    
    # Add extra information based on method used
    if objective == 'worst' and CONFIG['plugin_params']['use_eg_outer']:
        if 'beta_star' in locals():
            checkpoint['beta'] = beta_star.tolist() if hasattr(beta_star, 'tolist') else beta_star
        if 'eg_hist' in locals():
            checkpoint['eg_history'] = eg_hist
    
    ckpt_path = output_dir / 'gse_balanced_plugin.ckpt'
    torch.save(checkpoint, ckpt_path)
    print(f"💾 Saved checkpoint to {ckpt_path}")

if __name__ == '__main__':
    main()