#!/usr/bin/env python3
"""
Test script để validate AR-GSE implementation theo lý thuyết.
"""

import torch
import numpy as np
from pathlib import Path
from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group
from src.data.datasets import get_cifar100_lt_counts

def test_argse_theoretical_consistency():
    """Test các thành phần AR-GSE theo lý thuyết."""
    print("🔍 Testing AR-GSE Theoretical Consistency")
    print("=" * 60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_experts = 3
    num_classes = 100
    num_groups = 2
    batch_size = 4
    gating_feature_dim = 4 * num_experts
    
    # Get class-to-group mapping  
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group(class_counts, K=num_groups, head_ratio=0.5).to(device)
    
    # Create model
    model = AR_GSE(num_experts, num_classes, num_groups, gating_feature_dim).to(device)
    
    # Test data: [B, E, C] logits
    expert_logits = torch.randn(batch_size, num_experts, num_classes).to(device)
    c = 0.2
    tau = 2.0
    
    print(f"✅ Model initialized:")
    print(f"   Experts: {num_experts}, Classes: {num_classes}, Groups: {num_groups}")
    print(f"   Alpha shape: {model.alpha.shape}, Mu shape: {model.mu.shape}")
    print(f"   Lambda shape: {model.Lambda.shape}")
    
    # Forward pass
    outputs = model(expert_logits, c, tau, class_to_group)
    
    print(f"\n🔬 Forward Pass Results:")
    print(f"   eta_mix shape: {outputs['eta_mix'].shape}")
    print(f"   eta_mix sum (should ≈ 1): {outputs['eta_mix'].sum(dim=1)}")
    print(f"   s_tau shape: {outputs['s_tau'].shape}")
    print(f"   w (gating) shape: {outputs['w'].shape}")
    print(f"   w sum (should = 1): {outputs['w'].sum(dim=1)}")
    print(f"   margin shape: {outputs['margin'].shape}")
    
    # Test 1: η̃_y(x) is valid probability distribution
    eta_mix = outputs['eta_mix']
    assert torch.all(eta_mix >= 0), "❌ eta_mix contains negative values!"
    assert torch.all(eta_mix <= 1), "❌ eta_mix contains values > 1!"
    assert torch.allclose(eta_mix.sum(dim=1), torch.ones(batch_size).to(device), atol=1e-6), "❌ eta_mix not normalized!"
    print("✅ Test 1: η̃_y(x) is valid probability distribution")
    
    # Test 2: Gating weights sum to 1
    w = outputs['w']
    assert torch.allclose(w.sum(dim=1), torch.ones(batch_size).to(device), atol=1e-6), "❌ Gating weights don't sum to 1!"
    print("✅ Test 2: Gating weights w_φ(x) sum to 1")
    
    # Test 3: Classification rule h_α(x) = argmax_y α_{grp(y)} * η̃_y(x)
    alpha = model.alpha
    reweighted_scores = alpha[class_to_group] * eta_mix  # [B, C]
    predictions = torch.argmax(reweighted_scores, dim=1)
    print(f"✅ Test 3: Classification rule h_α(x) computed")
    print(f"   Sample predictions: {predictions.cpu().numpy()}")
    
    # Test 4: Margin formula
    # margin = max_y α_{grp(y)} * η̃_y(x) - (Σ_{y'} (1/α_{grp(y')} - μ_{grp(y')}) * η̃_{y'}(x) - c)
    margin_manual = torch.zeros(batch_size).to(device)
    for b in range(batch_size):
        # Max score
        max_score = torch.max(alpha[class_to_group] * eta_mix[b]).item()
        
        # Threshold
        inv_alpha_minus_mu = 1.0 / alpha[class_to_group] - model.mu[class_to_group]
        threshold = torch.sum(inv_alpha_minus_mu * eta_mix[b]).item() - c
        
        margin_manual[b] = max_score - threshold
    
    margin_computed = outputs['margin']
    assert torch.allclose(margin_manual, margin_computed, atol=1e-5), "❌ Margin formula mismatch!"
    print("✅ Test 4: Margin formula m_α,μ(x) is correct")
    
    # Test 5: s_τ(x) = σ(τ * m_α,μ(x))
    s_tau_manual = torch.sigmoid(tau * margin_computed)
    s_tau_computed = outputs['s_tau']
    assert torch.allclose(s_tau_manual, s_tau_computed, atol=1e-6), "❌ s_tau formula mismatch!"
    print("✅ Test 5: Soft acceptance s_τ(x) = σ(τ * m_α,μ(x)) is correct")
    
    print("\n🎯 All theoretical consistency tests passed!")
    print(f"   Sample margin values: {margin_computed.detach().cpu().numpy()}")
    print(f"   Sample s_tau values: {s_tau_computed.detach().cpu().numpy()}")
    
    return True

def test_primal_dual_setup():
    """Test primal-dual loss components."""
    print(f"\n🔧 Testing Primal-Dual Loss Components")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Mock classification loss
    from src.models.surrogate_losses import selective_cls_loss
    
    batch_size = 8
    num_classes = 100
    num_groups = 2
    
    # Mock data
    eta_mix = torch.softmax(torch.randn(batch_size, num_classes), dim=1).to(device)
    y_true = torch.randint(0, num_classes, (batch_size,)).to(device)
    s_tau = torch.sigmoid(torch.randn(batch_size)).to(device)
    beta = torch.tensor([0.6, 0.4]).to(device)  # Group weights
    alpha = torch.tensor([1.2, 0.8]).to(device)  # Alpha parameters
    
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group(class_counts, K=num_groups, head_ratio=0.5).to(device)
    
    # Test classification loss
    loss_cls = selective_cls_loss(eta_mix, y_true, s_tau, beta, alpha, class_to_group, kind="ce")
    print(f"✅ Classification loss computed: {loss_cls.item():.4f}")
    
    # Test entropy on gating weights (not on eta_mix!)
    w = torch.softmax(torch.randn(batch_size, 3), dim=1).to(device)  # Mock gating weights
    gating_entropy = -torch.sum(w * torch.log(w + 1e-8), dim=1)
    loss_ent = -0.01 * gating_entropy.mean()
    print(f"✅ Gating entropy loss: {loss_ent.item():.4f}")
    
    # Test rejection loss
    c = 0.2
    loss_rej = c * (1.0 - s_tau).mean()
    print(f"✅ Rejection loss: {loss_rej.item():.4f}")
    
    print("✅ All loss components computed successfully!")

if __name__ == '__main__':
    test_argse_theoretical_consistency()
    test_primal_dual_setup()
    print(f"\n🎉 All AR-GSE tests passed! Ready for training.")