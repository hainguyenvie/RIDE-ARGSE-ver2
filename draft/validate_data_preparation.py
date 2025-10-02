# validate_data_preparation.py
import json
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

from src.data.datasets import get_cifar100_lt_counts
from src.data.groups import get_class_to_group

def load_split_indices(split_dir, split_name):
    """Load indices from JSON file"""
    filepath = split_dir / f"{split_name}_indices.json"
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_split_distribution(indices, targets, split_name, class_to_group=None):
    """Analyze class and group distribution in a split"""
    print(f"\n=== {split_name.upper()} SPLIT ANALYSIS ===")
    print(f"Total samples: {len(indices)}")
    
    # Get targets for this split
    split_targets = targets[indices]
    class_counts = Counter(split_targets)
    
    # Class distribution stats
    print(f"Classes represented: {len(class_counts)}/100")
    print(f"Min samples per class: {min(class_counts.values())}")
    print(f"Max samples per class: {max(class_counts.values())}")
    print(f"Avg samples per class: {np.mean(list(class_counts.values())):.2f}")
    
    # Group analysis if provided
    if class_to_group is not None:
        group_counts = {0: 0, 1: 0}  # head, tail
        for class_id, count in class_counts.items():
            group = class_to_group[class_id].item()
            group_counts[group] += count
            
        total_samples = sum(group_counts.values())
        print(f"Head group samples: {group_counts[0]} ({group_counts[0]/total_samples*100:.1f}%)")
        print(f"Tail group samples: {group_counts[1]} ({group_counts[1]/total_samples*100:.1f}%)")
    
    return class_counts

def validate_long_tail_property(class_counts, expected_counts, tolerance=0.1):
    """Validate that the class distribution follows long-tail property"""
    print(f"\n=== LONG-TAIL PROPERTY VALIDATION ===")
    
    # Check if class distribution matches expected exponential decay
    actual_counts = [class_counts.get(i, 0) for i in range(100)]
    
    # Calculate correlation between expected and actual
    correlation = np.corrcoef(expected_counts, actual_counts)[0, 1]
    print(f"Correlation with expected long-tail distribution: {correlation:.4f}")
    
    # Check imbalance ratio
    max_count = max(actual_counts)
    min_count = min([c for c in actual_counts if c > 0])
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"Actual imbalance ratio: {imbalance_ratio:.2f}")
    
    return correlation > 0.9  # Good correlation threshold

def check_split_overlaps(splits_dict):
    """Check that splits don't overlap"""
    print(f"\n=== SPLIT OVERLAP VALIDATION ===")
    
    split_names = list(splits_dict.keys())
    all_good = True
    
    for i, split1 in enumerate(split_names):
        for j, split2 in enumerate(split_names[i+1:], i+1):
            set1 = set(splits_dict[split1])
            set2 = set(splits_dict[split2])
            overlap = set1.intersection(set2)
            
            if overlap:
                print(f"❌ Overlap found between {split1} and {split2}: {len(overlap)} samples")
                all_good = False
            else:
                print(f"✅ No overlap between {split1} and {split2}")
    
    return all_good

def check_ar_gse_requirements(splits_dict):
    """Check if data meets AR-GSE specific requirements"""
    print(f"\n=== AR-GSE REQUIREMENTS VALIDATION ===")
    
    requirements_met = True
    
    # 1. Check tuneV size (for gating training)
    tuneV_size = len(splits_dict['tuneV'])
    min_tuneV_size = 500  # Minimum for stable gating training
    if tuneV_size >= min_tuneV_size:
        print(f"✅ tuneV size adequate: {tuneV_size} >= {min_tuneV_size}")
    else:
        print(f"❌ tuneV size too small: {tuneV_size} < {min_tuneV_size}")
        requirements_met = False
    
    # 2. Check calib size (for conformal calibration)
    calib_size = len(splits_dict['calib'])
    min_calib_size = 200  # Minimum for conformal calibration
    if calib_size >= min_calib_size:
        print(f"✅ calib size adequate: {calib_size} >= {min_calib_size}")
    else:
        print(f"❌ calib size too small: {calib_size} < {min_calib_size}")
        requirements_met = False
    
    # 3. Check val_small size (for hyperparameter tuning)
    val_small_size = len(splits_dict['val_small'])
    min_val_small_size = 300
    if val_small_size >= min_val_small_size:
        print(f"✅ val_small size adequate: {val_small_size} >= {min_val_small_size}")
    else:
        print(f"❌ val_small size too small: {val_small_size} < {min_val_small_size}")
        requirements_met = False
    
    # 4. Check evaluation splits (val_lt, test_lt)
    val_lt_size = len(splits_dict['val_lt'])
    test_lt_size = len(splits_dict['test_lt'])
    
    if val_lt_size >= 1000:
        print(f"✅ val_lt size adequate: {val_lt_size} >= 1000")
    else:
        print(f"⚠️  val_lt size small but acceptable: {val_lt_size}")
    
    if test_lt_size >= 3000:
        print(f"✅ test_lt size adequate: {test_lt_size} >= 3000")
    else:
        print(f"⚠️  test_lt size small but acceptable: {test_lt_size}")
    
    return requirements_met

def main():
    # Configuration
    IMB_FACTOR = 100
    DATA_ROOT = "./data"
    SPLIT_DIR = Path(f"./data/cifar100_lt_if100_splits")
    
    print("="*60)
    print("DATA PREPARATION VALIDATION FOR AR-GSE")
    print("="*60)
    
    # Load original data
    print("Loading original CIFAR-100 data...")
    cifar100_train = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=False)
    targets = np.array(cifar100_train.targets)
    
    # Load expected class counts and groups
    expected_class_counts = get_cifar100_lt_counts(IMB_FACTOR)
    class_to_group = get_class_to_group(expected_class_counts, K=2, head_ratio=0.5)
    
    # Load all splits
    split_names = ['train', 'tuneV', 'val_small', 'calib', 'val_lt', 'test_lt']
    splits_dict = {}
    
    for split_name in split_names:
        try:
            splits_dict[split_name] = load_split_indices(SPLIT_DIR, split_name)
            print(f"✅ Loaded {split_name}: {len(splits_dict[split_name])} samples")
        except FileNotFoundError:
            print(f"❌ Failed to load {split_name}")
            return
    
    # 1. Analyze each split
    for split_name in split_names:
        if split_name in ['val_lt', 'test_lt']:
            # These come from test set, need different targets
            cifar100_test = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=False)
            test_targets = np.array(cifar100_test.targets)
            analyze_split_distribution(splits_dict[split_name], test_targets, split_name, class_to_group)
        else:
            # These come from train set
            analyze_split_distribution(splits_dict[split_name], targets, split_name, class_to_group)
    
    # 2. Validate long-tail property (only for training splits)
    train_indices = splits_dict['train']
    train_targets = targets[train_indices]
    train_class_counts = Counter(train_targets)
    
    is_long_tail = validate_long_tail_property(train_class_counts, expected_class_counts)
    
    # 3. Check split overlaps (only for splits from same source)
    train_splits = {k: v for k, v in splits_dict.items() if k not in ['val_lt', 'test_lt']}
    overlaps_ok = check_split_overlaps(train_splits)
    
    # 4. Check AR-GSE specific requirements
    requirements_ok = check_ar_gse_requirements(splits_dict)
    
    # Final validation summary
    print(f"\n{'='*60}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    all_checks = [
        ("Long-tail property", is_long_tail),
        ("No split overlaps", overlaps_ok),
        ("AR-GSE requirements", requirements_ok)
    ]
    
    all_passed = True
    for check_name, passed in all_checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall status: {'✅ DATA READY FOR AR-GSE' if all_passed else '❌ ISSUES FOUND'}")
    
    # Generate summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    total_train_samples = sum(len(splits_dict[k]) for k in ['train', 'tuneV', 'val_small', 'calib'])
    print(f"Total training data: {total_train_samples} samples")
    print(f"Training split: {len(splits_dict['train'])} ({len(splits_dict['train'])/total_train_samples*100:.1f}%)")
    print(f"Tuning split: {len(splits_dict['tuneV'])} ({len(splits_dict['tuneV'])/total_train_samples*100:.1f}%)")
    print(f"Validation split: {len(splits_dict['val_small'])} ({len(splits_dict['val_small'])/total_train_samples*100:.1f}%)")
    print(f"Calibration split: {len(splits_dict['calib'])} ({len(splits_dict['calib'])/total_train_samples*100:.1f}%)")
    
    print(f"\nEvaluation data: {len(splits_dict['val_lt']) + len(splits_dict['test_lt'])} samples")
    print(f"Val LT: {len(splits_dict['val_lt'])} samples")
    print(f"Test LT: {len(splits_dict['test_lt'])} samples")

if __name__ == "__main__":
    main()