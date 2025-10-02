#!/usr/bin/env python3
"""
Simple Data Distribution Analyzer - No external dependencies except PyTorch/torchvision
"""

import json
import torch
import torchvision
from pathlib import Path
from collections import Counter

def check_split_indices():
    """Check all split indices for validity."""
    
    print("🔍 Checking split indices validity...")
    
    # Load CIFAR-100 datasets
    print("📥 Loading CIFAR-100 datasets...")
    cifar_train = torchvision.datasets.CIFAR100(root='data', train=True, download=False)
    cifar_test = torchvision.datasets.CIFAR100(root='data', train=False, download=False)
    
    print(f"✅ Train dataset: {len(cifar_train)} samples (indices 0-{len(cifar_train)-1})")
    print(f"✅ Test dataset: {len(cifar_test)} samples (indices 0-{len(cifar_test)-1})")
    
    splits_dir = Path("data/cifar100_lt_if100_splits")
    split_files = {
        'train': 'train_indices.json',
        'val_lt': 'val_lt_indices.json', 
        'test_lt': 'test_lt_indices.json',
        'tuneV': 'tuneV_indices.json',
        'val_small': 'val_small_indices.json',
        'calib': 'calib_indices.json'
    }
    
    for split_name, filename in split_files.items():
        filepath = splits_dir / filename
        if not filepath.exists():
            print(f"❌ Missing: {filename}")
            continue
            
        with open(filepath, 'r') as f:
            indices = json.load(f)
            
        # Check validity
        if split_name == 'train':
            max_valid = len(cifar_train) - 1
            dataset_name = "train"
        else:
            max_valid = len(cifar_test) - 1
            dataset_name = "test"
            
        min_idx = min(indices)
        max_idx = max(indices)
        invalid_count = len([i for i in indices if i > max_valid])
        
        status = "✅" if invalid_count == 0 else "❌"
        print(f"{status} {split_name}: {len(indices)} samples")
        print(f"    Range: {min_idx} - {max_idx} (dataset: {dataset_name}, max_valid: {max_valid})")
        
        if invalid_count > 0:
            print(f"    ⚠️  {invalid_count} invalid indices found!")
            invalid_indices = [i for i in indices if i > max_valid]
            print(f"    Invalid: {invalid_indices[:10]}...")
            
    return cifar_train, cifar_test


def analyze_distribution_simple(split_name, indices, source_dataset, max_valid_idx):
    """Simple distribution analysis."""
    
    print(f"\n📊 Analyzing {split_name}...")
    
    # Filter valid indices
    valid_indices = [i for i in indices if i <= max_valid_idx]
    invalid_count = len(indices) - len(valid_indices)
    
    if invalid_count > 0:
        print(f"⚠️  Skipping {invalid_count} invalid indices")
        
    # Extract labels
    labels = []
    for idx in valid_indices:
        try:
            _, label = source_dataset[idx]
            labels.append(label)
        except IndexError:
            continue
    
    if not labels:
        print("❌ No valid samples found!")
        return None
        
    # Count classes
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate statistics
    counts_list = list(class_counts.values())
    max_count = max(counts_list)
    min_count = min(counts_list)
    imbalance_factor = max_count / min_count if min_count > 0 else float('inf')
    
    # Most/least frequent classes
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   ✅ Valid samples: {total_samples}")
    print(f"   📈 Classes represented: {len(class_counts)}/100")
    print(f"   🔢 Imbalance Factor: {imbalance_factor:.1f}")
    print(f"   📊 Max/Min counts: {max_count}/{min_count}")
    print(f"   🏆 Top 5 classes: {sorted_classes[:5]}")
    print(f"   📉 Bottom 5 classes: {sorted_classes[-5:]}")
    
    return {
        'split_name': split_name,
        'valid_samples': total_samples,
        'invalid_samples': invalid_count,
        'num_classes': len(class_counts),
        'imbalance_factor': imbalance_factor,
        'max_count': max_count,
        'min_count': min_count,
        'class_counts': dict(class_counts),
        'sorted_classes': sorted_classes
    }


def main():
    """Main analysis function."""
    print("CIFAR-100-LT Simple Data Distribution Analyzer")
    print("=" * 60)
    
    # Check split indices validity
    cifar_train, cifar_test = check_split_indices()
    
    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    splits_dir = Path("data/cifar100_lt_if100_splits")
    split_files = {
        'train': 'train_indices.json',
        'val_lt': 'val_lt_indices.json', 
        'test_lt': 'test_lt_indices.json',
        'tuneV': 'tuneV_indices.json',
        'val_small': 'val_small_indices.json',
        'calib': 'calib_indices.json'
    }
    
    results = {}
    
    for split_name, filename in split_files.items():
        filepath = splits_dir / filename
        if not filepath.exists():
            continue
            
        with open(filepath, 'r') as f:
            indices = json.load(f)
            
        # Determine source dataset and max valid index
        if split_name == 'train':
            source_dataset = cifar_train
            max_valid_idx = len(cifar_train) - 1
        else:
            source_dataset = cifar_test
            max_valid_idx = len(cifar_test) - 1
            
        # Analyze distribution
        result = analyze_distribution_simple(split_name, indices, source_dataset, max_valid_idx)
        if result:
            results[split_name] = result
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Split':<12} {'Valid':<8} {'Invalid':<8} {'Classes':<8} {'IF':<8} {'Max':<6} {'Min':<6}")
    print("-" * 60)
    
    for split_name, result in results.items():
        print(f"{split_name:<12} {result['valid_samples']:<8} {result['invalid_samples']:<8} "
              f"{result['num_classes']:<8} {result['imbalance_factor']:<8.1f} "
              f"{result['max_count']:<6} {result['min_count']:<6}")
    
    # Check train split quality
    if 'train' in results:
        train_result = results['train']
        print(f"\n📈 Train Split Quality:")
        print(f"   Expected IF ≈ 100: {'✅' if 90 <= train_result['imbalance_factor'] <= 110 else '❌'} {train_result['imbalance_factor']:.1f}")
        print(f"   All classes present: {'✅' if train_result['num_classes'] == 100 else '❌'} {train_result['num_classes']}/100")
        print(f"   No invalid indices: {'✅' if train_result['invalid_samples'] == 0 else '❌'} {train_result['invalid_samples']} invalid")
    
    print(f"\n🎉 Analysis completed!")
    return results


if __name__ == "__main__":
    results = main()