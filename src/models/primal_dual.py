import torch

def estimate_group_acceptance(s_tau, y_true, class_to_group, num_groups):
    """
    Ước tính E_hat[s_tau * 1{y thuộc nhóm G_k}] cho mỗi nhóm k.
    Đây là phiên bản đã sửa lỗi, tính toán kỳ vọng một cách chính xác.
    """
    device = s_tau.device
    
    # Đảm bảo class_to_group ở đúng device
    class_to_group = class_to_group.to(device)
    
    y_groups = class_to_group[y_true]  # [B]
    
    # Tạo one-hot encoding cho nhóm của mỗi mẫu
    # Shape: [B, K]
    group_one_hot = torch.nn.functional.one_hot(y_groups, num_classes=num_groups)
    
    # Nhân s_tau với one-hot encoding
    # s_tau.unsqueeze(1) -> [B, 1]
    # group_one_hot -> [B, K]
    # Kết quả -> [B, K], mỗi cột k chứa s_tau nếu mẫu thuộc nhóm k, ngược lại là 0
    s_tau_per_group = s_tau.unsqueeze(1) * group_one_hot
    
    # Ước tính kỳ vọng bằng cách lấy trung bình trên toàn batch
    # Đây chính là (1/B) * sum(s_tau * 1_{y thuộc nhóm G_k})
    acceptance_expectation_k = torch.mean(s_tau_per_group, dim=0)  # Shape: [K]
    
    return acceptance_expectation_k

def primal_dual_step(model, batch, optimizers, loss_fn, params):
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

    # ----- Loss components -----
    loss_cls = loss_fn(
        eta_mix=eta_mix,
        y_true=labels,
        s_tau=s_tau,
        beta=params['beta'],
        alpha=model.alpha,                # sẽ không dùng để nhân như patch 2.1
        class_to_group=params['class_to_group']
    )

    # ENTROPY REG: nên TẮT hoặc PHẠT entropy cao (không tối đa hóa)
    # Gợi ý: tắt ở giai đoạn ổn định hóa
    lambda_ent = params.get('lambda_ent', 0.0)
    gating_entropy = -torch.sum(w * (w + 1e-8).log(), dim=1)
    # Nếu muốn sparse/hard routing: loss_ent = +λ * H(w)
    loss_ent = (+lambda_ent) * gating_entropy.mean()

    # Rejection term
    loss_rej = params['c'] * (1.0 - s_tau).mean()

    # Constraint term: g_k = α_k - K * (1/B) * Σ s_tau * 1{y∈G_k}
    class_to_group = params['class_to_group']
    sample_groups = class_to_group[labels]
    K = model.num_groups
    B = labels.size(0)
    cons_violation = torch.zeros(K, device=device)
    alpha = model.alpha  # có grad

    for k in range(K):
        mask = (sample_groups == k)
        if mask.any():
            joint = s_tau[mask].sum() / B     # (1/B) Σ s_tau I{y∈G_k}
            actual_scaled = K * joint
            cons_violation[k] = alpha[k] - actual_scaled

    loss_cons = (model.Lambda * cons_violation).sum()

    # ----- Total & backward -----
    loss_total = loss_cls + loss_rej + loss_ent + loss_cons
    loss_total.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.gating_net.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_([model.alpha, model.mu], max_norm=1.0)

    # ----- ONE step each (chỉ 1 lần!) -----
    optimizers['phi'].step()
    optimizers['alpha_mu'].step()      # <<<< CHỈ step 1 lần, KHÔNG step lần 2

    # ----- Projection / post-processing (không step thêm sau đó) -----
    with torch.no_grad():
        # center & clamp mu
        model.mu.sub_(model.mu.mean())
        model.mu.clamp_(-2.0, 2.0)

        # normalize & clamp alpha
        alpha_min = params.get('alpha_clip', 5e-2)
        alpha_max = 2.0                # giảm từ 3.0 xuống 2.0 để chắc chắn
        model.alpha.clamp_(min=alpha_min)
        log_alpha = model.alpha.log()
        model.alpha.copy_(torch.exp(log_alpha - log_alpha.mean()))
        model.alpha.clamp_(min=alpha_min, max=alpha_max)

    # ----- Dual update -----
    with torch.no_grad():
        cons_det = torch.zeros(K, device=device)
        for k in range(K):
            mask = (sample_groups == k)
            if mask.any():
                joint = s_tau[mask].sum() / B
                cons_det[k] = model.alpha[k] - K * joint

        rho = params.get('rho', 1e-2)
        model.Lambda.data = (model.Lambda + rho * cons_det).clamp_min(0.0)

    stats = {
        'loss_cls': loss_cls.item(),
        'loss_rej': loss_rej.item(),
        'loss_cons': loss_cons.item(),
        'loss_ent': loss_ent.item(),
        'loss_total': loss_total.item(),
        'mean_coverage': s_tau.mean().item(),
        'mean_margin': margin.mean().item(),
    }
    for k in range(K):
        stats[f'cons_viol_{k}'] = cons_violation[k].item()
    return stats, cons_violation
