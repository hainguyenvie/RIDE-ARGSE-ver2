import torch

def estimate_group_acceptance(s_tau, y_true, class_to_group, num_groups):
    """
    Ước tính E_hat[s_tau * 1{y thuộc nhóm G_k}] cho mỗi nhóm k.
    """
    device = s_tau.device
    class_to_group = class_to_group.to(device)
    y_groups = class_to_group[y_true]  # [B]
    
    # One-hot encoding cho nhóm
    group_one_hot = torch.nn.functional.one_hot(y_groups, num_classes=num_groups)
    
    # s_tau * one_hot encoding  
    s_tau_per_group = s_tau.unsqueeze(1) * group_one_hot
    
    # Kỳ vọng: (1/B) * sum(s_tau * 1_{y thuộc nhóm G_k})
    acceptance_expectation_k = torch.mean(s_tau_per_group, dim=0)  # [K]
    
    return acceptance_expectation_k

def primal_dual_step_fixed_point(model, batch, optimizers, loss_fn, params):
    """
    Updated primal-dual step using fixed-point alpha updates instead of SGD.
    This prevents tail collapse by avoiding the destructive gradient feedback loop.
    """
    logits, labels = batch
    device = params['device']
    logits, labels = logits.to(device), labels.to(device)

    for opt in optimizers.values():
        opt.zero_grad()

    # ----- Forward -----
    outputs = model(logits, params['c'], params['tau'], params['class_to_group'])
    # CLAMP để tránh log(0) hoặc phân phối suy biến
    eta_mix = outputs['eta_mix'].clamp(1e-6, 1 - 1e-6)
    w = outputs['w'].clamp_min(1e-6)
    w = w / w.sum(dim=1, keepdim=True)
    s_tau = outputs['s_tau']
    margin = outputs['margin']

    # ----- Loss components (NO constraint loss) -----
    loss_cls = loss_fn(
        eta_mix=eta_mix,
        y_true=labels,
        s_tau=s_tau,
        beta=params['beta'],
        alpha=model.alpha,
        class_to_group=params['class_to_group']
    )

    # Entropy regularization (usually turned off for stability)
    lambda_ent = params.get('lambda_ent', 0.0)
    gating_entropy = -torch.sum(w * (w + 1e-8).log(), dim=1)
    loss_ent = lambda_ent * gating_entropy.mean()

    # Rejection term
    loss_rej = params['c'] * (1.0 - s_tau).mean()

    # Total loss WITHOUT constraint term
    loss_total = loss_cls + loss_rej + loss_ent
    loss_total.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.gating_net.parameters(), max_norm=1.0)
    # Only clip mu, not alpha (alpha will be updated via fixed-point)
    torch.nn.utils.clip_grad_norm_([model.mu], max_norm=1.0)

    # ----- Optimizer steps -----
    optimizers['phi'].step()  # Update gating network
    if 'mu' in optimizers:
        optimizers['mu'].step()  # Update mu (if not frozen)

    # ----- Fixed-point alpha update (replaces SGD) -----
    model.update_alpha_fixed_point(
        s_tau=s_tau,
        labels=labels, 
        class_to_group=params['class_to_group'],
        alpha_clip=(params.get('alpha_clip', 5e-2), 2.0)
    )

    # ----- Post-processing mu -----
    with torch.no_grad():
        # Center mu to avoid drift
        model.mu.sub_(model.mu.mean())
        model.mu.clamp_(-2.0, 2.0)

    # ----- Compute constraint violation for monitoring (no dual update) -----
    with torch.no_grad():
        class_to_group = params['class_to_group']
        sample_groups = class_to_group[labels]
        K = model.num_groups
        B = labels.size(0)
        
        cons_violation = torch.zeros(K, device=device)
        for k in range(K):
            mask = (sample_groups == k)
            if mask.any():
                joint = s_tau[mask].sum() / B     # (1/B) Σ s_tau I{y∈G_k}
                actual_scaled = K * joint
                cons_violation[k] = model.alpha[k] - actual_scaled

    # ----- Statistics -----
    stats = {
        'loss_cls': loss_cls.item(),
        'loss_rej': loss_rej.item(),
        'loss_cons': 0.0,  # No constraint loss anymore
        'loss_ent': loss_ent.item(),
        'loss_total': loss_total.item(),
        'mean_coverage': s_tau.mean().item(),
        'mean_margin': margin.mean().item(),
        'alpha_mean': model.alpha.mean().item(),
        'alpha_std': model.alpha.std().item(),
        'mu_mean': model.mu.mean().item(),
        'mu_std': model.mu.std().item(),
    }
    
    # Add per-group constraint violations for monitoring
    for k in range(K):
        stats[f'cons_viol_{k}'] = cons_violation[k].item()
        
    return stats, cons_violation

def calculate_optimal_c(model, dataloader, target_coverage=0.6):
    """
    Calculate optimal rejection cost c to achieve target coverage.
    Does one forward pass to estimate c = 1 - quantile_q(max_y eta_y).
    """
    model.eval()
    max_probs = []
    
    with torch.no_grad():
        for logits, _ in dataloader:
            logits = logits.to(next(model.parameters()).device)
            # Get mixture probabilities without rejection (c=0, tau=1)
            outputs = model(logits, c=0.0, tau=1.0, class_to_group=torch.zeros(100).long())
            eta_mix = outputs['eta_mix']
            max_prob, _ = eta_mix.max(dim=1)
            max_probs.append(max_prob)
    
    all_max_probs = torch.cat(max_probs)
    optimal_c = 1.0 - torch.quantile(all_max_probs, target_coverage)
    
    return optimal_c.item()

def compute_per_group_acceptance_rates(model, dataloader, c, class_to_group):
    """
    Compute per-group acceptance rates for monitoring.
    """
    model.eval()
    K = model.num_groups
    group_accepts = torch.zeros(K)
    group_totals = torch.zeros(K)
    
    with torch.no_grad():
        for logits, labels in dataloader:
            device = next(model.parameters()).device
            logits, labels = logits.to(device), labels.to(device)
            
            outputs = model(logits, c, tau=1.0, class_to_group=class_to_group)
            s_tau = outputs['s_tau']
            
            sample_groups = class_to_group[labels]
            for k in range(K):
                mask = (sample_groups == k)
                if mask.any():
                    group_accepts[k] += s_tau[mask].sum().cpu()
                    group_totals[k] += mask.sum().cpu()
    
    # Compute rates, handle division by zero
    rates = []
    for k in range(K):
        if group_totals[k] > 0:
            rates.append((group_accepts[k] / group_totals[k]).item())
        else:
            rates.append(0.0)
    
    return rates