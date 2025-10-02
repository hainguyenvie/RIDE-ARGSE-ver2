"""
Per-group threshold (Mondrian) implementation for GSE plugin.
Uses different thresholds t_k for each group based on predicted class group.
"""
import torch

def fit_group_thresholds_from_raw(raw_margins, pred_groups, target_cov_by_group, K):
    """Fit per-group thresholds t_k on ALL samples (not only correct).

    Args:
        raw_margins: [N]
        pred_groups: [N]
        target_cov_by_group: list/array length K of desired coverage per group (τ_k)
        K: number of groups

    Returns:
        t_k: [K] thresholds where t_k = Quantile_{1-τ_k}( m_raw | G_k )
    """
    t_k = torch.zeros(K, dtype=raw_margins.dtype)
    for k in range(K):
        mk = (pred_groups == k)
        if mk.sum() == 0:
            # fallback to global quantile with average τ
            avg_tau = float(sum(target_cov_by_group)) / K
            t_k[k] = torch.quantile(raw_margins, 1.0 - avg_tau)
            print(f"⚠️ No samples predicted for group {k}, using global threshold {t_k[k]:.4f}")
        else:
            # Ensure mask and raw margins are on same device
            mk = mk.to(raw_margins.device)
            t_k[k] = torch.quantile(raw_margins[mk], 1.0 - target_cov_by_group[k])
            print(f"Group {k}: n={mk.sum().item()} τ_k={target_cov_by_group[k]:.3f} -> t_k={t_k[k]:.4f}")
    return t_k

def accept_with_group_thresholds(raw_margins, pred_groups, t_k):
    """
    Accept samples based on per-group thresholds.
    
    Args:
        raw_margins: [N] raw margin scores
        pred_groups: [N] predicted group for each sample
        t_k: [K] threshold for each group
    
    Returns:
        accepted: [N] boolean mask for accepted samples
    """
    return raw_margins >= t_k[pred_groups]