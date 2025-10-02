# src/models/gating.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingFeatureBuilder:
    """
    Builds scalable, class-count-independent features from expert posteriors/logits.
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_logits: Tensor of shape [B, E, C] (Batch, Experts, Classes)
        
        Returns:
            A feature tensor of shape [B, D] where D is the feature dimension.
        """
        # Ensure input is float32 for stable calculations
        expert_logits = expert_logits.float()
        B, E, C = expert_logits.shape
        
        # Use posteriors for probability-based features
        expert_posteriors = torch.softmax(expert_logits, dim=-1)

        # Feature 1: Entropy of each expert's prediction  [B, E]
        entropy = -torch.sum(expert_posteriors * torch.log(expert_posteriors + 1e-8), dim=-1)

        # Feature 2: Top-k probability mass and residual mass per expert
        topk_vals, _ = torch.topk(expert_posteriors, k=min(self.top_k, expert_posteriors.size(-1)), dim=-1)
        topk_mass = torch.sum(topk_vals, dim=-1)            # [B, E]
        residual_mass = 1.0 - topk_mass                    # [B, E]

        # Feature 3: Expert confidence (max prob) and top1-top2 gap
        max_probs, _ = expert_posteriors.max(dim=-1)        # [B, E]
        if topk_vals.size(-1) >= 2:
            top1 = topk_vals[..., 0]
            top2 = topk_vals[..., 1]
            top_gap = top1 - top2                           # [B, E]
        else:
            top_gap = torch.zeros_like(max_probs)

        # Feature 4: Cosine similarity to ensemble mean posterior (agreement proxy)
        mean_posterior = torch.mean(expert_posteriors, dim=1)        # [B, C]
        cosine_sim = F.cosine_similarity(expert_posteriors, mean_posterior.unsqueeze(1), dim=-1)  # [B, E]

        # Feature 5: KL divergence of each expert to mean posterior (disagreement)
        # KL(p_e || mean_p) = Σ p_e (log p_e - log mean_p)
        kl_to_mean = torch.sum(expert_posteriors * (torch.log(expert_posteriors + 1e-8) - torch.log(mean_posterior.unsqueeze(1) + 1e-8)), dim=-1)  # [B, E]

        # Global (per-sample) features capturing ensemble uncertainty / disagreement
        # Mixture / mean posterior entropy
        mean_entropy = -torch.sum(mean_posterior * torch.log(mean_posterior + 1e-8), dim=-1)  # [B]
        # Mean variance across classes (how much experts disagree on class probs)
        class_var = expert_posteriors.var(dim=1)                     # [B, C]
        mean_class_var = class_var.mean(dim=-1)                      # [B]
        # Std of expert max probabilities (confidence dispersion)
        std_max_conf = max_probs.std(dim=-1)                         # [B]

        # Concatenate per-expert features first (all [B,E])
        per_expert_feats = [entropy, topk_mass, residual_mass, max_probs, top_gap, cosine_sim, kl_to_mean]
        per_expert_concat = torch.cat(per_expert_feats, dim=1)       # [B, 7*E]

        # Concatenate global features (broadcast not needed; just append) -> shape [B, 7E + 3]
        global_feats = torch.stack([mean_entropy, mean_class_var, std_max_conf], dim=1)  # [B, 3]
        features = torch.cat([per_expert_concat, global_feats], dim=1)
        
        return features

class GatingNet(nn.Module):
    """
    A simple MLP that takes gating features and outputs expert weights.
    """
    def __init__(self, in_dim: int, hidden_dims: list = [128, 64], num_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Gating features of shape [B, D]
        
        Returns:
            Expert weights (before softmax) of shape [B, E]
        """
        return self.net(x)