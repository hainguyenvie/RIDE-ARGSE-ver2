# src/models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitAdjustLoss(nn.Module):
    def __init__(self, class_counts, tau=1.0):
        super(LogitAdjustLoss, self).__init__()
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        # Calculate class priors
        priors = class_counts / class_counts.sum()
        self.log_priors = torch.log(priors).unsqueeze(0)
        self.tau = tau

    def forward(self, logits, target):
        # Move log_priors to the same device as logits
        self.log_priors = self.log_priors.to(logits.device)
        
        # Adjust logits
        adjusted_logits = logits + self.tau * self.log_priors
        return F.cross_entropy(adjusted_logits, target)

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts):
        super(BalancedSoftmaxLoss, self).__init__()
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        self.log_priors = torch.log(class_counts).unsqueeze(0)

    def forward(self, logits, target):
        self.log_priors = self.log_priors.to(logits.device)
        
        # Adjust logits before softmax
        adjusted_logits = logits + self.log_priors
        return F.cross_entropy(adjusted_logits, target)