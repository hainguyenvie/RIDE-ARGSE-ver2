# test_data_pipeline.py
"""
Comprehensive test of the data preparation pipeline for AR-GSE.
This validates all components needed for the AR-GSE training.
"""

import torch
import torchvision
import numpy as np
import json
from pathlib import Path
from collections import Counter

from src.data.datasets import (
    get_cifar100_lt_counts, 
    generate_longtail_train_set,
    get_train_augmentations,
    get_eval_augmentations,
    get_randaug_train_augmentations
)
from src.data.groups import get_class_to_group
from src.data.splits import create_and_save_splits, create_longtail_val_test_splits

def test_data_loading():
    """Test basic data loading functionality"""
    print("🧪 Testing data loading...")
    
    # Test CIFAR-100 loading
    try:
        train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=False)
        test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
        
        assert len(train_dataset) == 50000, f"Expected 50000 train samples, got {len(train_dataset)}"
        assert len(test_dataset) == 10000, f"Expected 10000 test samples, got {len(test_dataset)}"
        
        print("✅ CIFAR-100 loading works correctly")
        return True
    except Exception as e:
        print(f"❌ CIFAR-100 loading failed: {e}")
        return False

def test_longtail_generation():
    """Test long-tail dataset generation"""
    print("\n🧪 Testing long-tail dataset generation...")
    
    try:
        # Load original dataset
        train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=False)
        
        # Test different imbalance factors
        for imb_factor in [10, 100]:
            print(f"  Testing imbalance factor {imb_factor}...")
            
            # Generate long-tail counts
            expected_counts = get_cifar100_lt_counts(imb_factor)
            assert len(expected_counts) == 100, "Should have counts for 100 classes"
            
            # Check imbalance ratio
            actual_ratio = expected_counts[0] / expected_counts[-1]
            assert abs(actual_ratio - imb_factor) < 1e-10, f"Imbalance ratio mismatch: {actual_ratio} vs {imb_factor}"
            
            # Generate long-tail dataset
            lt_indices, lt_targets = generate_longtail_train_set(train_dataset, imb_factor)
            
            # Verify the distribution
            class_counts = Counter(lt_targets)
            actual_counts = [class_counts.get(i, 0) for i in range(100)]
            
            # Check that we got approximately the right distribution
            correlation = np.corrcoef(expected_counts, actual_counts)[0, 1]
            assert correlation > 0.99, f"Poor correlation with expected distribution: {correlation}"
            
            print(f"    ✅ IF={imb_factor}: {len(lt_indices)} samples, correlation={correlation:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Long-tail generation failed: {e}")
        return False

def test_group_assignment():
    """Test class-to-group assignment"""
    print("\n🧪 Testing group assignment...")
    
    try:
        # Test with different configurations
        class_counts = get_cifar100_lt_counts(100)
        
        # Test K=2 (head/tail)
        class_to_group = get_class_to_group(class_counts, K=2, head_ratio=0.5)
        assert len(class_to_group) == 100, "Should assign all 100 classes"
        assert set(class_to_group.tolist()) == {0, 1}, "Should only have groups 0 and 1"
        
        # Check that head classes (group 0) have higher counts
        head_classes = [i for i in range(100) if class_to_group[i] == 0]
        tail_classes = [i for i in range(100) if class_to_group[i] == 1]
        
        min_head_count = min(class_counts[i] for i in head_classes)
        max_tail_count = max(class_counts[i] for i in tail_classes)
        
        assert min_head_count >= max_tail_count, "Head classes should have more samples than tail classes"
        assert len(head_classes) == 50, f"Should have 50 head classes, got {len(head_classes)}"
        assert len(tail_classes) == 50, f"Should have 50 tail classes, got {len(tail_classes)}"
        
        print("✅ Group assignment works correctly")
        return True
    except Exception as e:
        print(f"❌ Group assignment failed: {e}")
        return False

def test_augmentations():
    """Test data augmentation functions"""
    print("\n🧪 Testing data augmentations...")
    
    try:
        # Test train augmentations
        train_transform = get_train_augmentations()
        eval_transform = get_eval_augmentations()
        
        # Create a dummy PIL image (CIFAR-100 format)
        from PIL import Image
        dummy_image = Image.new('RGB', (32, 32), color='red')
        
        # Test train transform
        train_tensor = train_transform(dummy_image)
        assert train_tensor.shape == (3, 32, 32), f"Expected shape (3,32,32), got {train_tensor.shape}"
        assert train_tensor.dtype == torch.float32, f"Expected float32, got {train_tensor.dtype}"
        
        # Test eval transform
        eval_tensor = eval_transform(dummy_image)
        assert eval_tensor.shape == (3, 32, 32), f"Expected shape (3,32,32), got {eval_tensor.shape}"
        assert eval_tensor.dtype == torch.float32, f"Expected float32, got {eval_tensor.dtype}"
        
        # Test RandAugment (optional)
        try:
            randaug_transform = get_randaug_train_augmentations(num_ops=2, magnitude=9)
            randaug_tensor = randaug_transform(dummy_image)
            assert randaug_tensor.shape == (3, 32, 32), "RandAug should maintain image shape"
            print("✅ RandAugment available and working")
        except ImportError:
            print("⚠️  RandAugment not available (using fallback)")
        
        print("✅ Data augmentations work correctly")
        return True
    except Exception as e:
        print(f"❌ Data augmentations failed: {e}")
        return False

def test_split_functionality():
    """Test data splitting functionality"""
    print("\n🧪 Testing data splitting...")
    
    try:
        # Create dummy data for testing splits
        dummy_indices = np.arange(1000)
        dummy_targets = np.random.randint(0, 100, 1000)
        
        split_ratios = {
            'train': 0.8,
            'tuneV': 0.1,
            'val_small': 0.07,
            'calib': 0.03
        }
        
        output_dir = Path("./test_splits")
        output_dir.mkdir(exist_ok=True)
        
        # Test split creation
        create_and_save_splits(dummy_indices, dummy_targets, split_ratios, output_dir, seed=42)
        
        # Verify splits were created
        total_samples = 0
        for split_name in split_ratios.keys():
            filepath = output_dir / f"{split_name}_indices.json"
            assert filepath.exists(), f"Split file {split_name} not created"
            
            with open(filepath, 'r') as f:
                indices = json.load(f)
                total_samples += len(indices)
        
        assert total_samples == len(dummy_indices), "Split sizes don't sum to original size"
        
        # Clean up test files
        for split_name in split_ratios.keys():
            (output_dir / f"{split_name}_indices.json").unlink()
        output_dir.rmdir()
        
        print("✅ Data splitting works correctly")
        return True
    except Exception as e:
        print(f"❌ Data splitting failed: {e}")
        return False

def test_ar_gse_readiness():
    """Test if data is ready for AR-GSE training"""
    print("\n🧪 Testing AR-GSE readiness...")
    
    try:
        # Check if all required files exist
        split_dir = Path("./data/cifar100_lt_if100_splits")
        required_splits = ['train', 'tuneV', 'val_small', 'calib', 'val_lt', 'test_lt']
        
        for split_name in required_splits:
            filepath = split_dir / f"{split_name}_indices.json"
            assert filepath.exists(), f"Required split {split_name} missing"
        
        # Load and verify split sizes
        split_sizes = {}
        for split_name in required_splits:
            filepath = split_dir / f"{split_name}_indices.json"
            with open(filepath, 'r') as f:
                indices = json.load(f)
                split_sizes[split_name] = len(indices)
        
        # Check AR-GSE requirements
        requirements = {
            'tuneV': 500,    # Minimum for gating training
            'calib': 200,    # Minimum for conformal calibration
            'val_small': 300, # Minimum for validation
            'val_lt': 1000,   # Minimum for evaluation
            'test_lt': 3000   # Minimum for evaluation
        }
        
        for split_name, min_size in requirements.items():
            actual_size = split_sizes[split_name]
            assert actual_size >= min_size, f"{split_name} too small: {actual_size} < {min_size}"
        
        print("✅ Data meets all AR-GSE requirements")
        
        # Print summary
        print("\n📊 Data Summary:")
        for split_name, size in split_sizes.items():
            print(f"  {split_name}: {size} samples")
        
        return True
    except Exception as e:
        print(f"❌ AR-GSE readiness check failed: {e}")
        return False

def test_data_consistency():
    """Test data consistency across the pipeline"""
    print("\n🧪 Testing data consistency...")
    
    try:
        # Load original data
        train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=False)
        test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
        
        # Load splits
        split_dir = Path("./data/cifar100_lt_if100_splits")
        
        # Check training splits consistency
        train_splits = ['train', 'tuneV', 'val_small', 'calib']
        all_train_indices = set()
        
        for split_name in train_splits:
            with open(split_dir / f"{split_name}_indices.json", 'r') as f:
                indices = json.load(f)
                split_indices = set(indices)
                
                # Check no overlap with previous splits
                overlap = all_train_indices.intersection(split_indices)
                assert len(overlap) == 0, f"Overlap found in {split_name}"
                
                # Check indices are valid
                assert all(0 <= idx < len(train_dataset) for idx in indices), f"Invalid indices in {split_name}"
                
                all_train_indices.update(split_indices)
        
        # Check test splits consistency
        test_splits = ['val_lt', 'test_lt']
        all_test_indices = set()
        
        for split_name in test_splits:
            with open(split_dir / f"{split_name}_indices.json", 'r') as f:
                indices = json.load(f)
                split_indices = set(indices)
                
                # Check no overlap
                overlap = all_test_indices.intersection(split_indices)
                assert len(overlap) == 0, f"Overlap found in {split_name}"
                
                # Check indices are valid
                assert all(0 <= idx < len(test_dataset) for idx in indices), f"Invalid indices in {split_name}"
                
                all_test_indices.update(split_indices)
        
        print("✅ Data consistency verified")
        return True
    except Exception as e:
        print(f"❌ Data consistency check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("AR-GSE DATA PIPELINE COMPREHENSIVE TEST")
    print("="*60)
    
    tests = [
        test_data_loading,
        test_longtail_generation,
        test_group_assignment,
        test_augmentations,
        test_split_functionality,
        test_ar_gse_readiness,
        test_data_consistency
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Data pipeline is ready for AR-GSE training.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)