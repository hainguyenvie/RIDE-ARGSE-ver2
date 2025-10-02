#!/usr/bin/env python3
"""
Enhanced data preparation for CIFAR-100-LT with duplication-based val/test creation.
Implements the methodology:
1. Train: Standard exponential profile long-tail
2. Val/Test: Match train proportions with duplication when needed
3. TuneV: Subset from test to avoid data leakage
"""

import numpy as np
import json
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset

class CIFAR100LTDataset(Dataset):
    """Custom Dataset wrapper for CIFAR-100-LT with flexible indexing."""
    
    def __init__(self, cifar_dataset, indices, transform=None):
        self.cifar_dataset = cifar_dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Map to actual CIFAR index
        cifar_idx = self.indices[idx]
        image, label = self.cifar_dataset[cifar_idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_cifar100_transforms():
    """Get CIFAR-100 transforms following paper specifications."""
    
    # Training transforms (basic augmentation as per Menon et al., 2021a)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
            std=[0.2675, 0.2565, 0.2761]   # CIFAR-100 std
        )
    ])
    
    # Evaluation transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    return train_transform, eval_transform

def get_exponential_counts(num_classes: int = 100, imb_factor: float = 100, max_samples: int = 500) -> List[int]:
    """Generate exponential profile counts for long-tail distribution."""
    counts = []
    for cls_idx in range(num_classes):
        # Exponential decay: n_i = n_max * (IF)^(-i/(C-1))
        num = max_samples * (imb_factor ** (-cls_idx / (num_classes - 1.0)))
        counts.append(max(1, int(num)))
    return counts

def create_longtail_train(cifar_train_dataset, imb_factor: float = 100, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Create long-tail training set using exponential profile."""
    print(f"Creating CIFAR-100-LT training set (IF={imb_factor})...")
    
    np.random.seed(seed)
    targets = np.array(cifar_train_dataset.targets)
    num_classes = 100
    
    # Get target counts
    target_counts = get_exponential_counts(num_classes, imb_factor, 500)
    
    # Sample indices for each class
    train_indices = []
    actual_counts = []
    
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        num_to_sample = min(target_counts[cls], len(cls_indices))
        
        # Random sample without replacement
        sampled = np.random.choice(cls_indices, num_to_sample, replace=False)
        train_indices.extend(sampled.tolist())
        actual_counts.append(num_to_sample)
    
    train_targets = targets[train_indices].tolist()
    
    print(f"  Created training set: {len(train_indices):,} samples")
    print(f"  Head class: {actual_counts[0]} samples")
    print(f"  Tail class: {actual_counts[-1]} samples")
    print(f"  Actual IF: {actual_counts[0] / actual_counts[-1]:.1f}")
    
    return train_indices, train_targets, actual_counts

def create_proportional_test_val_with_duplication(
    cifar_test_dataset, 
    train_class_counts: List[int],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Create val/test sets that match train proportions using duplication when needed.
    
    Args:
        cifar_test_dataset: Original CIFAR-100 test dataset (balanced, 100 per class)
        train_class_counts: Count of samples per class in training set
        val_ratio: Proportion for validation (rest goes to test)
        seed: Random seed
        
    Returns:
        Tuple of (val_indices, val_targets, test_indices, test_targets)
    """
    print("\nCreating proportional val/test sets with duplication...")
    
    np.random.seed(seed)
    test_targets = np.array(cifar_test_dataset.targets)
    num_classes = 100
    
    # Calculate train proportions
    total_train = sum(train_class_counts)
    train_proportions = [count / total_train for count in train_class_counts]
    
    # Target test size (can be larger than original 10k due to duplication)
    target_total_size = 12000  # Slightly larger to accommodate duplication
    
    # Calculate target counts per class in combined val+test
    target_class_counts = []
    duplication_info = []
    
    for cls in range(num_classes):
        # Target count based on train proportion
        target_count = int(round(train_proportions[cls] * target_total_size))
        target_count = max(1, target_count)  # At least 1 sample per class
        
        # Available samples in original test (100 per class)
        available_in_test = 100
        
        if target_count <= available_in_test:
            # No duplication needed
            duplication_factor = 1
            final_count = target_count
        else:
            # Need duplication
            duplication_factor = int(np.ceil(target_count / available_in_test))
            final_count = target_count  # We'll sample exactly what we need
            
        target_class_counts.append(final_count)
        duplication_info.append({
            'class': cls,
            'target': target_count,
            'available': available_in_test,
            'duplication_factor': duplication_factor,
            'final_count': final_count
        })
    
    print(f"  Target combined val+test size: {sum(target_class_counts):,}")
    
    # Create the combined pool with duplication
    combined_indices = []
    combined_targets = []
    
    duplication_stats = {'no_dup': 0, 'duplicated': 0, 'max_dup_factor': 0}
    
    for cls in range(num_classes):
        cls_indices_in_test = np.where(test_targets == cls)[0]
        target_count = target_class_counts[cls]
        available_count = len(cls_indices_in_test)
        
        if target_count <= available_count:
            # Simple sampling, no duplication
            sampled_indices = np.random.choice(cls_indices_in_test, target_count, replace=False)
            duplication_stats['no_dup'] += 1
        else:
            # Need duplication
            duplication_factor = int(np.ceil(target_count / available_count))
            duplication_stats['max_dup_factor'] = max(duplication_stats['max_dup_factor'], duplication_factor)
            duplication_stats['duplicated'] += 1
            
            # Create duplicated pool
            duplicated_pool = np.tile(cls_indices_in_test, duplication_factor)
            
            # Sample exactly what we need
            sampled_indices = np.random.choice(duplicated_pool, target_count, replace=False)
            
        combined_indices.extend(sampled_indices.tolist())
        combined_targets.extend([cls] * target_count)
    
    print("  Duplication stats:")
    print(f"    Classes without duplication: {duplication_stats['no_dup']}")
    print(f"    Classes with duplication: {duplication_stats['duplicated']}")  
    print(f"    Maximum duplication factor: {duplication_stats['max_dup_factor']}")
    
    # Now split into val and test
    combined_indices = np.array(combined_indices)
    combined_targets = np.array(combined_targets)
    
    # Stratified split to maintain proportions in both val and test
    val_indices = []
    test_indices = []
    
    for cls in range(num_classes):
        cls_mask = combined_targets == cls
        cls_combined_indices = combined_indices[cls_mask]
        
        if len(cls_combined_indices) > 0:
            np.random.shuffle(cls_combined_indices)
            n_val = int(round(len(cls_combined_indices) * val_ratio))
            n_val = max(1, n_val)  # At least 1 for val
            n_val = min(n_val, len(cls_combined_indices) - 1)  # Leave at least 1 for test
            
            val_indices.extend(cls_combined_indices[:n_val])
            test_indices.extend(cls_combined_indices[n_val:])
    
    # Get corresponding targets
    val_targets = test_targets[val_indices].tolist()
    test_targets_final = test_targets[test_indices].tolist()
    
    print("  Final splits:")
    print(f"    Validation: {len(val_indices):,} samples")
    print(f"    Test: {len(test_indices):,} samples")
    
    return val_indices, val_targets, test_indices, test_targets_final

def create_tunev_from_test(test_indices: List[int], test_targets: List[int], tunev_ratio: float = 0.15, seed: int = 42) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Create tuneV subset from test set to avoid data leakage.
    
    Args:
        test_indices: Test indices  
        test_targets: Test targets
        tunev_ratio: Ratio of test to use for tuneV (e.g., 0.15 = 15%)
        seed: Random seed
        
    Returns:
        Tuple of (tunev_indices, tunev_targets, remaining_test_indices, remaining_test_targets)
    """
    print(f"\nCreating tuneV from test set ({tunev_ratio:.1%})...")
    
    np.random.seed(seed)
    test_indices = np.array(test_indices)
    test_targets = np.array(test_targets)
    
    tunev_indices = []
    remaining_test_indices = []
    
    # Stratified sampling to maintain proportions
    num_classes = 100
    for cls in range(num_classes):
        cls_mask = test_targets == cls
        cls_test_indices = test_indices[cls_mask]
        
        if len(cls_test_indices) > 0:
            np.random.shuffle(cls_test_indices)
            n_tunev = max(1, int(round(len(cls_test_indices) * tunev_ratio)))
            n_tunev = min(n_tunev, len(cls_test_indices) - 1)  # Leave at least 1 for remaining test
            
            tunev_indices.extend(cls_test_indices[:n_tunev])
            remaining_test_indices.extend(cls_test_indices[n_tunev:])
    
    tunev_targets = test_targets[test_targets == test_targets].tolist()  # Get targets for tunev indices
    remaining_test_targets = test_targets[test_targets == test_targets].tolist()  # Get targets for remaining
    
    # Fix target extraction
    from collections import Counter
    tunev_targets = [test_targets[np.where(test_indices == idx)[0][0]] for idx in tunev_indices]
    remaining_test_targets = [test_targets[np.where(test_indices == idx)[0][0]] for idx in remaining_test_indices]
    
    print(f"  TuneV: {len(tunev_indices):,} samples")
    print(f"  Remaining test: {len(remaining_test_indices):,} samples")
    
    return tunev_indices, tunev_targets, remaining_test_indices, remaining_test_targets

def analyze_distribution(indices: List[int], targets: List[int], name: str, train_counts: Optional[List[int]] = None):
    """Analyze and print distribution statistics."""
    print(f"\n=== {name.upper()} DISTRIBUTION ===")
    
    target_counts = Counter(targets)
    sorted_counts = [target_counts.get(i, 0) for i in range(100)]
    
    total = sum(sorted_counts)
    head_count = sorted_counts[0]
    tail_count = sorted_counts[99]
    
    print(f"Total samples: {total:,}")
    print(f"Head class (0): {head_count} samples ({head_count/total*100:.2f}%)")
    print(f"Tail class (99): {tail_count} samples ({tail_count/total*100:.2f}%)")
    if tail_count > 0:
        print(f"Imbalance factor: {head_count/tail_count:.1f}")
    
    # Group analysis
    groups = {
        'Head (0-9)': sum(sorted_counts[0:10]),
        'Medium (10-49)': sum(sorted_counts[10:50]), 
        'Low (50-89)': sum(sorted_counts[50:90]),
        'Tail (90-99)': sum(sorted_counts[90:100])
    }
    
    print("Distribution by groups:")
    for group_name, group_count in groups.items():
        print(f"  {group_name}: {group_count:,} samples ({group_count/total*100:.1f}%)")
    
    # Compare with train if available
    if train_counts is not None:
        print("\nComparison with train proportions:")
        train_total = sum(train_counts)
        for i in [0, 25, 50, 75, 99]:  # Sample classes
            train_prop = train_counts[i] / train_total
            test_prop = sorted_counts[i] / total
            diff = abs(train_prop - test_prop)
            print(f"  Class {i:2d}: train={train_prop:.4f}, {name.lower()}={test_prop:.4f}, diff={diff:.4f}")

def save_splits_to_json(splits_dict: Dict, output_dir: str):
    """Save all splits to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving splits to {output_dir}...")
    
    for split_name, indices in splits_dict.items():
        filepath = output_path / f"{split_name}_indices.json"
        
        # Convert numpy types to Python native types for JSON serialization
        if hasattr(indices, 'tolist'):
            indices_to_save = indices.tolist()
        elif isinstance(indices, (list, tuple)):
            indices_to_save = [int(idx) if hasattr(idx, 'item') else idx for idx in indices]
        else:
            indices_to_save = list(indices)
            
        with open(filepath, 'w') as f:
            json.dump(indices_to_save, f)
        print(f"  Saved {split_name}: {len(indices_to_save):,} samples")

def create_full_cifar100_lt_splits(
    imb_factor: float = 100,
    output_dir: str = "data/cifar100_lt_if100_splits", 
    val_ratio: float = 0.2,
    tunev_ratio: float = 0.15,
    seed: int = 42
):
    """
    Create complete CIFAR-100-LT dataset splits with the new methodology.
    """
    print("=" * 60)
    print("CREATING CIFAR-100-LT DATASET SPLITS")  
    print("=" * 60)
    
    # Load original CIFAR-100
    print("Loading CIFAR-100 datasets...")
    cifar_train = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=None)
    cifar_test = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=None)
    
    # 1. Create long-tail training set
    train_indices, train_targets, train_counts = create_longtail_train(cifar_train, imb_factor, seed)
    
    # 2. Create val/test with proportional duplication
    val_indices, val_targets, test_indices, test_targets = create_proportional_test_val_with_duplication(
        cifar_test, train_counts, val_ratio, seed
    )
    
    # 3. Create tuneV from test
    tunev_indices, tunev_targets, final_test_indices, final_test_targets = create_tunev_from_test(
        test_indices, test_targets, tunev_ratio, seed + 1
    )
    
    # 4. Analyze all distributions
    analyze_distribution(train_indices, train_targets, "TRAIN")
    analyze_distribution(val_indices, val_targets, "VALIDATION", train_counts)  
    analyze_distribution(final_test_indices, final_test_targets, "TEST", train_counts)
    analyze_distribution(tunev_indices, tunev_targets, "TUNEV", train_counts)
    
    # 5. Save all splits (removed val_small and calib - use val_lt for both!)
    splits = {
        'train_indices': train_indices,
        'val_lt_indices': val_indices,
        'test_lt_indices': final_test_indices, 
        'tuneV_indices': tunev_indices
    }
    
    save_splits_to_json(splits, output_dir)
    
    # 6. Create dataset objects with transforms
    train_transform, eval_transform = get_cifar100_transforms()
    
    datasets = {
        'train': CIFAR100LTDataset(cifar_train, train_indices, train_transform),
        'val': CIFAR100LTDataset(cifar_test, val_indices, eval_transform),
        'test': CIFAR100LTDataset(cifar_test, final_test_indices, eval_transform),
        'tunev': CIFAR100LTDataset(cifar_test, tunev_indices, eval_transform)
    }
    
    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return datasets, splits

if __name__ == "__main__":
    # Create the full dataset
    datasets, splits = create_full_cifar100_lt_splits(
        imb_factor=100,
        output_dir="data/cifar100_lt_if100_splits",
        val_ratio=0.2,
        tunev_ratio=0.15,
        seed=42
    )
    
    print("\nDatasets ready for training:")
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset):,} samples")