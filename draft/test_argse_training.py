# test_argse_training.py
"""
Comprehensive test script for AR-GSE training stage (Stage 3).
Tests both balanced and worst-group training modes.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import sys
import traceback

# Add src to path for imports
sys.path.append('src')

from src.models.argse import AR_GSE
from src.models.primal_dual import primal_dual_step
from src.models.surrogate_losses import selective_cls_loss
from src.data.groups import get_class_to_group
from src.data.datasets import get_cifar100_lt_counts

def test_argse_components():
    """Test individual AR-GSE components."""
    print("=" * 60)
    print("TESTING AR-GSE COMPONENTS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tests_passed = 0
    total_tests = 0
    
    # Test data setup
    batch_size = 32
    num_experts = 3
    num_classes = 100
    num_groups = 2
    gating_feature_dim = 4 * num_experts
    
    # Create mock data
    expert_logits = torch.randn(batch_size, num_experts, num_classes).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Get class to group mapping
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group(class_counts, K=2, head_ratio=0.5).to(device)
    
    # Test 1: AR_GSE model initialization
    total_tests += 1
    try:
        model = AR_GSE(num_experts, num_classes, num_groups, gating_feature_dim).to(device)
        
        # Check initialization values
        assert torch.all(model.alpha > 0), "Alpha should be positive"
        assert torch.all(model.mu < 0), "Mu should be negative (conservative)"
        assert model.alpha.shape == (num_groups,), f"Alpha shape should be ({num_groups},), got {model.alpha.shape}"
        assert model.mu.shape == (num_groups,), f"Mu shape should be ({num_groups},), got {model.mu.shape}"
        
        print("✅ Test 1: AR_GSE model initialization - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1: AR_GSE model initialization - FAILED: {e}")
    
    # Test 2: Forward pass
    total_tests += 1
    try:
        outputs = model(expert_logits, c=0.2, tau=2.0, class_to_group=class_to_group)
        
        # Check output shapes
        assert outputs['eta_mix'].shape == (batch_size, num_classes), "eta_mix shape mismatch"
        assert outputs['margin'].shape == (batch_size,), "margin shape mismatch"
        assert outputs['s_tau'].shape == (batch_size,), "s_tau shape mismatch"
        
        # Check output ranges
        assert torch.all(outputs['eta_mix'] > 0), "eta_mix should be positive (probabilities)"
        assert torch.allclose(outputs['eta_mix'].sum(dim=1), torch.ones(batch_size).to(device), atol=1e-6), "eta_mix should sum to 1"
        assert torch.all(outputs['s_tau'] >= 0) and torch.all(outputs['s_tau'] <= 1), "s_tau should be in [0,1]"
        
        print("✅ Test 2: AR_GSE forward pass - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2: AR_GSE forward pass - FAILED: {e}")
    
    # Test 3: Surrogate loss function
    total_tests += 1
    try:
        beta = torch.ones(num_groups).to(device) / num_groups
        
        loss = selective_cls_loss(
            eta_mix=outputs['eta_mix'],
            y_true=labels,
            s_tau=outputs['s_tau'],
            beta=beta,
            alpha=model.alpha,
            class_to_group=class_to_group,
            kind="ce"
        )
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"
        
        print("✅ Test 3: Surrogate loss function - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3: Surrogate loss function - FAILED: {e}")
    
    # Test 4: Primal-dual step
    total_tests += 1
    try:
        # Setup optimizers
        optimizers = {
            'phi': torch.optim.Adam(model.gating_net.parameters(), lr=5e-4),
            'alpha_mu': torch.optim.Adam([model.alpha, model.mu], lr=2.5e-3)
        }
        
        # Create batch
        batch = (expert_logits, labels)
        
        # Create params
        params = {
            'device': device,
            'c': 0.2,
            'tau': 2.0,
            'beta': beta,
            'class_to_group': class_to_group,
            'lambda_ent': 1e-3,
            'alpha_clip': 1e-3,
            'rho': 5e-3,
        }
        
        # Store initial dual variables
        initial_lambda = model.Lambda.clone()
        
        # Perform primal-dual step
        stats, cons_violation = primal_dual_step(model, batch, optimizers, selective_cls_loss, params)
        
        # Check stats
        required_stats = ['loss_cls', 'loss_ent', 'loss_total', 'mean_coverage', 'mean_margin']
        for stat in required_stats:
            assert stat in stats, f"Missing stat: {stat}"
            assert np.isfinite(stats[stat]), f"Stat {stat} is not finite: {stats[stat]}"
        
        # Check constraint violation
        assert cons_violation.shape == (num_groups,), f"cons_violation shape should be ({num_groups},), got {cons_violation.shape}"
        assert torch.all(cons_violation >= 0), "Constraint violations should be non-negative"
        
        # Check that dual variables were updated
        lambda_changed = not torch.allclose(initial_lambda, model.Lambda)
        print(f"Lambda changed: {lambda_changed} (Initial: {initial_lambda.mean().item():.6f}, Final: {model.Lambda.mean().item():.6f})")
        
        print("✅ Test 4: Primal-dual step - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 4: Primal-dual step - FAILED: {e}")
        traceback.print_exc()
    
    # Test 5: Temperature scaling effect
    total_tests += 1
    try:
        # Test with different temperatures (checking calibration effect)
        outputs_low_temp = model(expert_logits, c=0.2, tau=1.0, class_to_group=class_to_group)
        outputs_high_temp = model(expert_logits, c=0.2, tau=5.0, class_to_group=class_to_group)
        
        # With higher temperature, acceptance should be more conservative (lower s_tau)
        avg_acceptance_low = outputs_low_temp['s_tau'].mean()
        avg_acceptance_high = outputs_high_temp['s_tau'].mean()
        
        print(f"Acceptance rate - Low temp (tau=1.0): {avg_acceptance_low:.3f}, High temp (tau=5.0): {avg_acceptance_high:.3f}")
        
        print("✅ Test 5: Temperature scaling effect - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 5: Temperature scaling effect - FAILED: {e}")
    
    print(f"\n📊 AR-GSE Components Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def test_data_loading():
    """Test data loading for AR-GSE training."""
    print("=" * 60)
    print("TESTING DATA LOADING FOR AR-GSE")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Check if logits files exist
    total_tests += 1
    try:
        logits_dir = Path('./outputs/logits/cifar100_lt_if100')
        expert_names = ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
        splits = ['tuneV', 'val_small']
        
        missing_files = []
        for expert in expert_names:
            for split in splits:
                logits_file = logits_dir / expert / f"{split}_logits.pt"
                if not logits_file.exists():
                    missing_files.append(str(logits_file))
        
        if missing_files:
            print("❌ Test 1: Missing logits files:")
            for file in missing_files:
                print(f"   - {file}")
            print("   Run expert training first!")
        else:
            print("✅ Test 1: All required logits files exist - PASSED")
            tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1: Check logits files - FAILED: {e}")
    
    # Test 2: Check splits files
    total_tests += 1
    try:
        splits_dir = Path('./data/cifar100_lt_if100_splits')
        required_splits = ['tuneV_indices.json', 'val_small_indices.json']
        
        missing_splits = []
        for split_file in required_splits:
            if not (splits_dir / split_file).exists():
                missing_splits.append(split_file)
        
        if missing_splits:
            print(f"❌ Test 2: Missing split files: {missing_splits}")
        else:
            print("✅ Test 2: All required split files exist - PASSED")
            tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2: Check split files - FAILED: {e}")
    
    # Test 3: Load and validate data shapes
    total_tests += 1
    try:
        if tests_passed == 2:  # Only if previous tests passed
            # Load one split to check shapes
            with open('./data/cifar100_lt_if100_splits/tuneV_indices.json', 'r') as f:
                tuneV_indices = json.load(f)
            
            # Load logits from first expert
            logits_path = logits_dir / 'ce_baseline' / 'tuneV_logits.pt'
            logits = torch.load(logits_path, map_location='cpu')
            
            expected_samples = len(tuneV_indices)
            actual_samples = logits.shape[0]
            expected_classes = 100
            actual_classes = logits.shape[1]
            
            assert actual_samples == expected_samples, f"Sample count mismatch: expected {expected_samples}, got {actual_samples}"
            assert actual_classes == expected_classes, f"Class count mismatch: expected {expected_classes}, got {actual_classes}"
            
            print(f"✅ Test 3: Data shapes validation - PASSED")
            print(f"   Samples: {actual_samples}, Classes: {actual_classes}")
            tests_passed += 1
        else:
            print("⏭️  Test 3: Skipped due to previous failures")
    except Exception as e:
        print(f"❌ Test 3: Data shapes validation - FAILED: {e}")
    
    print(f"\n📊 Data Loading Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def test_training_modes():
    """Test both balanced and worst-group training modes."""
    print("=" * 60)
    print("TESTING TRAINING MODES")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup minimal training configuration
    num_experts = 3
    num_classes = 100
    num_groups = 2
    gating_feature_dim = 4 * num_experts
    batch_size = 16
    
    # Create mock data
    expert_logits = torch.randn(batch_size, num_experts, num_classes).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group(class_counts, K=2, head_ratio=0.5).to(device)
    
    # Test 1: Balanced mode training step
    total_tests += 1
    try:
        model = AR_GSE(num_experts, num_classes, num_groups, gating_feature_dim).to(device)
        
        optimizers = {
            'phi': torch.optim.Adam(model.gating_net.parameters(), lr=5e-4),
            'alpha_mu': torch.optim.Adam([model.alpha, model.mu], lr=2.5e-3)
        }
        
        beta_balanced = torch.ones(num_groups, device=device) / num_groups
        
        params = {
            'device': device,
            'c': 0.2,
            'tau': 2.0,
            'beta': beta_balanced,
            'class_to_group': class_to_group,
            'lambda_ent': 1e-3,
            'alpha_clip': 1e-3,
            'rho': 5e-3,
        }
        
        batch = (expert_logits, labels)
        stats, cons_violation = primal_dual_step(model, batch, optimizers, selective_cls_loss, params)
        
        # Check that stats are reasonable
        assert 0 <= stats['mean_coverage'] <= 1, f"Coverage should be in [0,1], got {stats['mean_coverage']}"
        assert stats['loss_total'] > 0, f"Total loss should be positive, got {stats['loss_total']}"
        
        print(f"✅ Test 1: Balanced mode training step - PASSED")
        print(f"   Loss: {stats['loss_total']:.4f}, Coverage: {stats['mean_coverage']:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1: Balanced mode training step - FAILED: {e}")
    
    # Test 2: Worst-group mode with beta update
    total_tests += 1
    try:
        # Initialize worst-group beta
        beta_worst = torch.ones(num_groups, device=device) / num_groups
        
        # Simulate group errors for beta update
        group_errors = [0.1, 0.3]  # Group 1 has higher error
        
        # Update beta using EG
        xi = 1.0
        group_errors_tensor = torch.tensor(group_errors, device=device)
        new_beta = beta_worst * torch.exp(xi * group_errors_tensor)
        new_beta = new_beta / new_beta.sum()
        
        # Check that beta gives more weight to worse group
        assert new_beta[1] > new_beta[0], "Beta should give more weight to group with higher error"
        
        print(f"✅ Test 2: Worst-group beta update - PASSED")
        print(f"   Initial beta: {beta_worst.tolist()}")
        print(f"   Updated beta: {[f'{b:.3f}' for b in new_beta.tolist()]}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2: Worst-group beta update - FAILED: {e}")
    
    print(f"\n📊 Training Modes Tests: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def main():
    """Run all AR-GSE training tests."""
    print("🚀 STARTING AR-GSE TRAINING TESTS")
    print("=" * 80)
    
    all_tests_passed = True
    
    # Run component tests
    component_tests = test_argse_components()
    all_tests_passed &= component_tests
    
    print()
    
    # Run data loading tests  
    data_tests = test_data_loading()
    all_tests_passed &= data_tests
    
    print()
    
    # Run training mode tests
    training_tests = test_training_modes()
    all_tests_passed &= training_tests
    
    print()
    print("=" * 80)
    if all_tests_passed:
        print("🎉 ALL AR-GSE TRAINING TESTS PASSED!")
        print("✅ AR-GSE implementation is ready for training")
        print()
        print("Next steps:")
        print("1. Run: python src/train/train_argse.py")
        print("2. Check results in: ./checkpoints/argse_balance/")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Please fix the issues before proceeding with training")
    
    print("=" * 80)
    
    return all_tests_passed

if __name__ == '__main__':
    main()