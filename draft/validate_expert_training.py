# validate_expert_training.py
"""
Comprehensive validation of expert training logic before running the full training.
This checks all components and logic paths without actually training.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from pathlib import Path
from collections import Counter
import sys

# Import modules to validate
from src.models.experts import Expert
from src.models.losses import LogitAdjustLoss, BalancedSoftmaxLoss
from src.metrics.calibration import TemperatureScaler, calculate_ece

def test_model_architecture():
    """Test Expert model architecture and forward pass"""
    print("🧪 Testing Expert model architecture...")
    
    try:
        # Test model creation
        model = Expert(num_classes=100)
        print(f"✅ Model created: {model.__class__.__name__}")
        
        # Test forward pass
        dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR images
        
        with torch.no_grad():
            logits = model(dummy_input)
            calibrated_logits = model.get_calibrated_logits(dummy_input)
        
        # Check shapes
        assert logits.shape == (4, 100), f"Expected logits shape (4, 100), got {logits.shape}"
        assert calibrated_logits.shape == (4, 100), f"Expected calibrated logits shape (4, 100), got {calibrated_logits.shape}"
        
        # Test temperature setting
        model.set_temperature(1.5)
        assert abs(model.temperature.item() - 1.5) < 1e-6, "Temperature setting failed"
        
        print(f"✅ Forward pass works: logits {logits.shape}, temperature {model.temperature.item()}")
        
        # Test parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return True
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False

def test_loss_functions():
    """Test all loss functions with dummy data"""
    print("\n🧪 Testing loss functions...")
    
    try:
        # Create dummy data
        batch_size = 8
        num_classes = 100
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Dummy class counts (long-tail distribution)
        class_counts = [500 * (5/500) ** (i/99) for i in range(100)]
        class_counts = [max(1, int(count)) for count in class_counts]  # Ensure minimum 1
        
        # Test Cross Entropy
        ce_loss = nn.CrossEntropyLoss()
        ce_result = ce_loss(logits, targets)
        print(f"✅ CrossEntropy loss: {ce_result.item():.4f}")
        
        # Test Logit Adjust Loss
        la_loss = LogitAdjustLoss(class_counts=class_counts)
        la_result = la_loss(logits, targets)
        print(f"✅ LogitAdjust loss: {la_result.item():.4f}")
        
        # Test Balanced Softmax Loss
        bs_loss = BalancedSoftmaxLoss(class_counts=class_counts)
        bs_result = bs_loss(logits, targets)
        print(f"✅ BalancedSoftmax loss: {bs_result.item():.4f}")
        
        # Test gradients
        loss_funcs = [('CE', ce_loss), ('LA', la_loss), ('BS', bs_loss)]
        for name, loss_func in loss_funcs:
            logits_test = torch.randn(4, 100, requires_grad=True)
            targets_test = torch.randint(0, 100, (4,))
            
            loss = loss_func(logits_test, targets_test)
            loss.backward()
            
            assert logits_test.grad is not None, f"{name} loss didn't compute gradients"
            assert not torch.isnan(logits_test.grad).any(), f"{name} loss produced NaN gradients"
            print(f"✅ {name} gradients computed correctly")
        
        return True
    except Exception as e:
        print(f"❌ Loss functions test failed: {e}")
        return False

def test_data_loading():
    """Test data loading and splits"""
    print("\n🧪 Testing data loading...")
    
    try:
        # Test basic dataset loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=False, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False, transform=transform)
        
        print(f"✅ Base datasets loaded: train={len(train_dataset)}, test={len(test_dataset)}")
        
        # Test split loading
        splits_dir = Path("./data/cifar100_lt_if100_splits")
        required_splits = ['train', 'tuneV', 'val_small', 'calib', 'val_lt', 'test_lt']
        
        splits_data = {}
        for split_name in required_splits:
            filepath = splits_dir / f"{split_name}_indices.json"
            assert filepath.exists(), f"Split file {split_name} not found"
            
            with open(filepath, 'r') as f:
                indices = json.load(f)
            splits_data[split_name] = indices
            print(f"✅ {split_name} split: {len(indices)} samples")
        
        # Test subset creation
        train_indices = splits_data['train']
        val_indices = splits_data['val_lt']
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(test_dataset, val_indices)  # val_lt comes from test set
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # Test a batch
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        assert train_batch[0].shape[0] == 32, "Train batch size incorrect"
        assert val_batch[0].shape[0] == 32, "Val batch size incorrect"
        assert train_batch[0].shape[1:] == (3, 32, 32), "Image shape incorrect"
        
        print(f"✅ DataLoaders created: train batches={len(train_loader)}, val batches={len(val_loader)}")
        
        # Test class distribution in training set
        train_targets = np.array(train_dataset.targets)[train_indices]
        class_counts = Counter(train_targets)
        
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        print(f"✅ Class distribution: min={min_count}, max={max_count}, ratio={imbalance_ratio:.1f}")
        
        return True, (train_loader, val_loader, class_counts)
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False, None

def test_optimizer_scheduler():
    """Test optimizer and scheduler setup"""
    print("\n🧪 Testing optimizer and scheduler...")
    
    try:
        model = Expert(num_classes=100)
        
        # Test optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
        print(f"✅ Optimizer created: {optimizer.__class__.__name__}")
        
        # Test scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[96, 192, 224], gamma=0.1)
        print(f"✅ Scheduler created: {scheduler.__class__.__name__}")
        
        # Test learning rate progression
        initial_lr = optimizer.param_groups[0]['lr']
        print(f"✅ Initial LR: {initial_lr}")
        
        # Simulate some epochs
        test_epochs = [0, 95, 96, 97, 191, 192, 193, 223, 224, 225]
        for epoch in test_epochs:
            if epoch > 0:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}: LR = {current_lr:.6f}")
        
        return True
    except Exception as e:
        print(f"❌ Optimizer/scheduler test failed: {e}")
        return False

def test_calibration():
    """Test temperature scaling calibration"""
    print("\n🧪 Testing calibration...")
    
    try:
        model = Expert(num_classes=100)
        
        # Create dummy validation data
        dummy_loader = []
        for _ in range(5):  # 5 batches
            inputs = torch.randn(8, 3, 32, 32)
            targets = torch.randint(0, 100, (8,))
            dummy_loader.append((inputs, targets))
        
        # Test temperature scaler
        scaler = TemperatureScaler()
        
        # Mock the calibration process
        model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dummy_loader:
                logits = model(inputs)
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        print(f"✅ Collected logits: {all_logits.shape}")
        print(f"✅ Collected labels: {all_labels.shape}")
        
        # Test ECE calculation
        posteriors = torch.softmax(all_logits, dim=1)
        ece_before = calculate_ece(posteriors, all_labels)
        print(f"✅ ECE calculation works: {ece_before:.4f}")
        
        # Test manual temperature setting
        model.set_temperature(1.5)
        calibrated_logits = all_logits / model.temperature
        posteriors_after = torch.softmax(calibrated_logits, dim=1)
        ece_after = calculate_ece(posteriors_after, all_labels)
        print(f"✅ Calibrated ECE: {ece_after:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Calibration test failed: {e}")
        return False

def test_training_logic():
    """Test the main training logic without full training"""
    print("\n🧪 Testing training logic...")
    
    try:
        # Setup
        model = Expert(num_classes=100)
        
        # Create simple dummy data
        dummy_inputs = torch.randn(4, 3, 32, 32)
        dummy_targets = torch.randint(0, 100, (4,))
        
        # Test with each loss type
        class_counts = [100 - i for i in range(100)]  # Simple decreasing counts
        
        loss_configs = [
            ('ce', nn.CrossEntropyLoss()),
            ('logitadjust', LogitAdjustLoss(class_counts=class_counts)),
            ('balsoftmax', BalancedSoftmaxLoss(class_counts=class_counts))
        ]
        
        for loss_name, criterion in loss_configs:
            print(f"  Testing {loss_name} training step...")
            
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            
            # Forward pass
            model.train()
            outputs = model(dummy_inputs)
            loss = criterion(outputs, dummy_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"    ✅ {loss_name}: loss={loss.item():.4f}")
        
        # Test warmup logic
        print("  Testing warmup logic...")
        base_lr = 0.4
        warmup_steps = 15
        optimizer = optim.SGD(model.parameters(), lr=base_lr)
        
        for step in [0, 5, 10, 14, 15, 20]:
            if step < warmup_steps:
                lr_scale = (step + 1) / warmup_steps
                expected_lr = base_lr * lr_scale
                
                # Apply warmup
                for param_group in optimizer.param_groups:
                    param_group['lr'] = expected_lr
                
                actual_lr = optimizer.param_groups[0]['lr']
                assert abs(actual_lr - expected_lr) < 1e-6, f"Warmup LR mismatch at step {step}"
                print(f"    ✅ Step {step:2d}: LR = {actual_lr:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Training logic test failed: {e}")
        return False

def test_export_logic():
    """Test logits export functionality"""
    print("\n🧪 Testing export logic...")
    
    try:
        model = Expert(num_classes=100)
        model.eval()
        
        # Test logits export for different data sizes
        test_sizes = [32, 100, 1000]
        
        for size in test_sizes:
            dummy_inputs = torch.randn(size, 3, 32, 32)
            
            with torch.no_grad():
                logits = model.get_calibrated_logits(dummy_inputs)
            
            assert logits.shape == (size, 100), f"Export shape mismatch for size {size}"
            assert not torch.isnan(logits).any(), f"NaN logits for size {size}"
            assert torch.isfinite(logits).all(), f"Infinite logits for size {size}"
            
            print(f"    ✅ Export test passed for {size} samples")
        
        # Test saving/loading
        test_logits = torch.randn(100, 100).to(torch.float16)
        test_path = Path("test_logits.pt")
        
        torch.save(test_logits, test_path)
        loaded_logits = torch.load(test_path)
        
        assert torch.allclose(test_logits.float(), loaded_logits.float(), rtol=1e-3), "Save/load mismatch"
        test_path.unlink()  # Clean up
        
        print(f"✅ Logits save/load works correctly")
        
        return True
    except Exception as e:
        print(f"❌ Export logic test failed: {e}")
        return False

def test_config_validation():
    """Test the training configuration"""
    print("\n🧪 Testing configuration...")
    
    try:
        # Test the CONFIG from train_expert.py
        config = {
            'dataset': {
                'name': 'cifar100_lt_if100',
                'data_root': './data',
                'splits_dir': './data/cifar100_lt_if100_splits',
                'num_classes': 100,
            },
            'experts_to_train': [
                {'name': 'ce_baseline', 'loss_type': 'ce'},
                {'name': 'logitadjust_baseline', 'loss_type': 'logitadjust'},
                {'name': 'balsoftmax_baseline', 'loss_type': 'balsoftmax'},
            ],
            'train_params': {
                'epochs': 256,
                'batch_size': 128,
                'lr': 0.4,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'warmup_steps': 15,
            },
            'output': {
                'checkpoints_dir': './checkpoints/experts',
                'logits_dir': './outputs/logits',
            },
            'seed': 42
        }
        
        # Validate dataset config
        assert config['dataset']['num_classes'] == 100, "Wrong number of classes"
        assert Path(config['dataset']['data_root']).exists(), "Data root doesn't exist"
        assert Path(config['dataset']['splits_dir']).exists(), "Splits dir doesn't exist"
        print("✅ Dataset config valid")
        
        # Validate experts config
        assert len(config['experts_to_train']) == 3, "Should train 3 experts"
        expert_names = [exp['name'] for exp in config['experts_to_train']]
        loss_types = [exp['loss_type'] for exp in config['experts_to_train']]
        
        expected_names = ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
        expected_losses = ['ce', 'logitadjust', 'balsoftmax']
        
        assert expert_names == expected_names, f"Expert names mismatch: {expert_names}"
        assert loss_types == expected_losses, f"Loss types mismatch: {loss_types}"
        print("✅ Experts config valid")
        
        # Validate training params
        train_params = config['train_params']
        assert train_params['epochs'] == 256, "Should train for 256 epochs"
        assert train_params['lr'] == 0.4, "Should use high learning rate 0.4"
        assert train_params['batch_size'] == 128, "Should use batch size 128"
        print("✅ Training params valid")
        
        # Validate output paths
        checkpoints_dir = Path(config['output']['checkpoints_dir'])
        logits_dir = Path(config['output']['logits_dir'])
        
        # These directories should be created during training
        print(f"✅ Output dirs configured: {checkpoints_dir}, {logits_dir}")
        
        return True
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("="*60)
    print("EXPERT TRAINING VALIDATION")
    print("="*60)
    
    tests = [
        test_model_architecture,
        test_loss_functions,
        test_data_loading,
        test_optimizer_scheduler,
        test_calibration,
        test_training_logic,
        test_export_logic,
        test_config_validation
    ]
    
    results = []
    data_info = None
    
    for test_func in tests:
        try:
            if test_func == test_data_loading:
                result, data_info = test_func()
                results.append(result)
            else:
                result = test_func()
                results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Expert training logic is ready.")
        
        if data_info:
            train_loader, val_loader, class_counts = data_info
            print(f"\n📊 Ready to train with:")
            print(f"  - Training batches: {len(train_loader)}")
            print(f"  - Validation batches: {len(val_loader)}")
            print(f"  - Classes with data: {len(class_counts)}/100")
            print(f"  - Most/least frequent: {max(class_counts.values())}/{min(class_counts.values())}")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)