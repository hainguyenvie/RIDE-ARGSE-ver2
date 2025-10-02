#!/usr/bin/env python3
"""
Quick validation test for AR-GSE selective training optimizations.
Tests core functionality without running full training.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.argse import AR_GSE
from src.models.gating import GatingFeatureBuilder
from src.train.train_gating_only import (
    fit_temperature_scaling, temperature_scale_logits,
    mu_from_lambda_grid, compute_raw_margin
)

def test_enriched_features():
    """Test that enriched gating features work correctly."""
    print("Testing enriched gating features...")
    
    fb = GatingFeatureBuilder()
    dummy_logits = torch.randn(8, 3, 100)  # 8 samples, 3 experts, 100 classes
    features = fb(dummy_logits)
    
    expected_dim = 7 * 3 + 3  # 7 per-expert features * 3 experts + 3 global features
    assert features.shape == (8, expected_dim), f"Expected shape (8, {expected_dim}), got {features.shape}"
    assert not torch.isnan(features).any(), "Features contain NaN values"
    assert torch.isfinite(features).all(), "Features contain infinite values"
    
    print(f"✅ Feature shape: {features.shape} (expected 7*3+3=24)")
    return True

def test_temperature_scaling():
    """Test temperature scaling functionality.""" 
    print("Testing temperature scaling...")
    
    # Mock data
    expert_logits = torch.randn(100, 3, 10)  # 100 samples, 3 experts, 10 classes
    labels = torch.randint(0, 10, (100,))
    expert_names = ['expert_a', 'expert_b', 'expert_c']
    
    # Test temperature fitting
    temperatures = fit_temperature_scaling(expert_logits, labels, expert_names, 'cpu')
    
    assert len(temperatures) == 3, f"Expected 3 temperatures, got {len(temperatures)}"
    for name, temp in temperatures.items():
        assert 0.1 <= temp <= 5.0, f"Temperature {temp} for {name} outside reasonable range"
    
    # Test temperature application
    scaled_logits = temperature_scale_logits(expert_logits, expert_names, temperatures)
    assert scaled_logits.shape == expert_logits.shape, "Scaling changed tensor shape"
    
    print(f"✅ Fitted temperatures: {temperatures}")
    return True

def test_expanded_mu_grid():
    """Test expanded μ grid generation."""
    print("Testing expanded μ grid...")
    
    lambda_grid = [round(x, 2) for x in np.linspace(-2.0, 2.0, 41)]
    mu_candidates = mu_from_lambda_grid(lambda_grid, K=2)
    
    assert len(mu_candidates) == 41, f"Expected 41 μ candidates, got {len(mu_candidates)}"
    
    # Test extreme values
    mu_min = mu_candidates[0]  # Should be [+1.0, -1.0] for λ=-2.0
    mu_max = mu_candidates[-1]  # Should be [-1.0, +1.0] for λ=+2.0
    
    assert torch.allclose(mu_min, torch.tensor([-1.0, 1.0]), atol=1e-3), f"Min μ incorrect: {mu_min}"
    assert torch.allclose(mu_max, torch.tensor([1.0, -1.0]), atol=1e-3), f"Max μ incorrect: {mu_max}"
    
    print(f"✅ μ grid: {len(mu_candidates)} candidates, range λ∈[{min(lambda_grid):.1f}, {max(lambda_grid):.1f}]")
    return True

def test_raw_margin_computation():
    """Test raw margin computation with dummy data."""
    print("Testing raw margin computation...")
    
    batch_size = 16
    num_classes = 10
    K = 2
    
    # Mock mixture posterior
    eta = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
    alpha = torch.tensor([1.1, 0.9])  # Slightly unbalanced
    mu = torch.tensor([0.2, -0.2])    # Small bias
    class_to_group = torch.tensor([0] * 5 + [1] * 5)  # 5 head, 5 tail classes
    
    raw_margins = compute_raw_margin(eta, alpha, mu, class_to_group)
    
    assert raw_margins.shape == (batch_size,), f"Expected shape ({batch_size},), got {raw_margins.shape}"
    assert torch.isfinite(raw_margins).all(), "Raw margins contain non-finite values"
    
    print(f"✅ Raw margins shape: {raw_margins.shape}, range: [{raw_margins.min():.3f}, {raw_margins.max():.3f}]")
    return True

def test_dynamic_feature_dimension():
    """Test dynamic gating feature dimension computation."""
    print("Testing dynamic feature dimension...")
    
    num_experts = 3
    num_classes = 100
    num_groups = 2
    batch_size = 4
    
    # Create dummy model to get feature dimension
    dummy_logits = torch.zeros(batch_size, num_experts, num_classes)
    temp_model = AR_GSE(num_experts, num_classes, num_groups, 1)  # Placeholder dim
    actual_dim = temp_model.feature_builder(dummy_logits).size(-1)
    
    # Create properly sized model
    model = AR_GSE(num_experts, num_classes, num_groups, actual_dim)
    
    # Test forward pass
    features = model.feature_builder(dummy_logits)
    gating_weights = torch.softmax(model.gating_net(features), dim=1)
    
    assert gating_weights.shape == (batch_size, num_experts), f"Gating weights shape mismatch: {gating_weights.shape}"
    assert torch.allclose(gating_weights.sum(dim=1), torch.ones(batch_size)), "Gating weights don't sum to 1"
    
    print(f"✅ Dynamic feature dim: {actual_dim}, gating weights: {gating_weights.shape}")
    return True

def main():
    """Run all validation tests."""
    print("🚀 AR-GSE Optimization Validation Tests\n")
    
    tests = [
        test_enriched_features,
        test_temperature_scaling,
        test_expanded_mu_grid,
        test_raw_margin_computation,
        test_dynamic_feature_dimension,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All optimizations validated successfully!")
        print("\nYou can now run:")
        print("  python -m src.train.train_gating_only --mode selective")
        print("  python -m src.train.gse_balanced_plugin")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())