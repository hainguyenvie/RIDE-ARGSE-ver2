# src/models/surrogate_losses.py
import torch
import torch.nn.functional as F

def selective_cls_loss(
    eta_mix: torch.Tensor,
    y_true: torch.Tensor,
    s_tau: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,              # vẫn giữ tham số để tương thích API
    class_to_group: torch.LongTensor,
    kind: str = "ce"
) -> torch.Tensor:
    # nhóm của từng mẫu
    sample_groups = class_to_group[y_true]
    sample_beta   = beta[sample_groups]

    # per-sample classification loss trên LOG-prob
    if kind == "ce":
        per_sample_loss = F.nll_loss(torch.log(eta_mix.clamp(1e-6, 1.0)), y_true, reduction='none')
    elif kind == "one_minus_p":
        p_true = eta_mix.gather(1, y_true.unsqueeze(1)).squeeze()
        per_sample_loss = 1 - p_true
    else:
        raise ValueError(f"Unknown loss kind: {kind}")

    # KHÔNG nhân α ở đây để tránh feedback dương gây nổ
    weighted = sample_beta * per_sample_loss * s_tau
    return weighted.mean()
