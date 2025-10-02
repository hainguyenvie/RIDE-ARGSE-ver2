# src/models/argse.py
import torch
import torch.nn as nn
from .gating import GatingFeatureBuilder, GatingNet
import torch.nn.functional as F

class AR_GSE(nn.Module):
    def __init__(self, num_experts: int, num_classes: int, num_groups: int, gating_feature_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.num_groups = num_groups

        # Gating components
        self.feature_builder = GatingFeatureBuilder()
        self.gating_net = GatingNet(in_dim=gating_feature_dim, num_experts=num_experts)

        # Primal variables (learnable parameters)
        # Initialize alpha to 1 for all groups
        self.alpha = nn.Parameter(torch.full((num_groups,), 1.0))  # Start with reasonable confidence scaling
        # Initialize mu to 0 as per paper (will be centered during training anyway)
        self.mu = nn.Parameter(torch.zeros(num_groups))
        
        # Dual variables (not optimized by SGD, but part of state)
        self.register_buffer('Lambda', torch.zeros(num_groups))
        
        # EMA buffers for stable training
        self.register_buffer('alpha_ema', torch.ones(num_groups))
        self.register_buffer('m_std', torch.tensor(1.0))  # margin std for normalization

    def forward(self, expert_logits, c, tau, class_to_group):
        """
        Full forward pass to get all necessary components for the primal-dual loss.
        
        Args:
            expert_logits (Tensor): Shape [B, E, C]
            c (float): Rejection cost
            tau (float): Temperature for sigmoid
            class_to_group (LongTensor): Shape [C]

        Returns:
            A dictionary of tensors for loss computation.
        """
        # 1. Gating
        gating_features = self.feature_builder(expert_logits)
        gating_raw_weights = self.gating_net(gating_features)
        w = F.softmax(gating_raw_weights, dim=1) # Shape [B, E]

        # 2. Convert expert LOGITS to POSTERIORS (η̃_y(x) should be probabilities)
        # Note: expert_logits are pre-computed calibrated logits from experts
        expert_posteriors = F.softmax(expert_logits, dim=-1)  # [B, E, C]
        
        # 3. Mixture of expert posteriors: η̃_y(x) = Σ_e w^(e)(x) * p^(e)(y|x)
        eta_mix = torch.einsum('be,bec->bc', w, expert_posteriors)  # [B, C]
        eta_mix = torch.clamp(eta_mix, min=1e-8, max=1.0-1e-8)  # Stability + valid probabilities

        # 4. Margin & Acceptance Probability  
        # Chỉ dùng một hàm selective_margin để tránh drift
        raw_margin = self.selective_margin(eta_mix, 0.0, class_to_group)  # c=0 for raw
        margin = raw_margin - c  # Final margin with rejection cost
        
        # Normalize margin by EMA std to help gradient flow
        if self.training:
            # Update margin std EMA during training
            self.m_std.mul_(0.99).add_(0.01 * margin.std().detach())
        m_norm = margin / (self.m_std + 1e-6)
        s_tau = torch.sigmoid(tau * m_norm)

        return {
            'eta_mix': eta_mix,
            's_tau': s_tau,
            'w': w,
            'margin': margin,
            'raw_margin': raw_margin,
        }

    def selective_margin(self, eta_mix, c, class_to_group):
        """
        Calculates the selective margin m(x) = max_score - threshold.
        Unified function to avoid drift between raw_margin and margin calculations.
        
        Args:
            eta_mix: [B, C] mixture probabilities
            c: rejection cost (use 0.0 for raw margin)  
            class_to_group: [C] class to group mapping
            
        Returns:
            margin: [B] selective margin scores
        """
        device = eta_mix.device
        alpha = self.alpha.to(device)
        mu = self.mu.to(device)
        class_to_group = class_to_group.to(device)

        # Score: max_y alpha_g(y) * eta_y 
        score_per_class = alpha[class_to_group] * eta_mix  # [B, C]
        max_score, _ = score_per_class.max(dim=1)  # [B]
        
        # Threshold: Σ_{y'} (1/α_{grp(y')} - μ_{grp(y')}) η̃_{y'}(x) - c
        # Vectorized computation to avoid loops
        coeff = 1.0 / alpha[class_to_group] - mu[class_to_group]  # [C]
        threshold = (coeff.unsqueeze(0) * eta_mix).sum(dim=1) - c  # [B]
        
        # Margin: score - threshold  
        margin = max_score - threshold
        return margin
    
    def update_alpha_fixed_point(self, s_tau, labels, class_to_group, alpha_clip=(5e-2, 2.0)):
        """
        Update alpha using fixed-point matching instead of SGD to avoid tail collapse.
        
        Args:
            s_tau: [B] acceptance probabilities
            labels: [B] ground truth labels  
            class_to_group: [C] class to group mapping
            alpha_clip: (min, max) clipping bounds for alpha
        """
        with torch.no_grad():
            K = self.num_groups
            
            # One-hot encoding for group membership based on LABELS
            group_labels = class_to_group[labels]  # [B] 
            one_hot = torch.nn.functional.one_hot(group_labels, num_classes=K).float()  # [B, K]
            
            # Compute per-group acceptance rate: (1/B) Σ s_tau * 1{y∈G_k}
            joint = (s_tau.unsqueeze(1) * one_hot).mean(dim=0)  # [K]
            alpha_hat = K * joint  # [K] target alpha_k
            
            # EMA update for stability
            self.alpha_ema.mul_(0.9).add_(0.1 * alpha_hat.to(self.alpha_ema.device))
            
            # Project: clamp + normalize geomean=1
            alpha_new = self.alpha_ema.clone()
            alpha_new.clamp_(min=alpha_clip[0], max=alpha_clip[1])
            log_alpha = alpha_new.log()
            alpha_new = torch.exp(log_alpha - log_alpha.mean())
            
            # Update model parameters
            self.alpha.data.copy_(alpha_new)