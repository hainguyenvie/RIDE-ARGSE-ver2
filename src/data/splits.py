# src/data/splits.py
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from collections import Counter

def create_longtail_val_test_splits(
    cifar100_test_dataset,
    train_class_counts: list,
    output_dir: Path,
    val_size: float = 0.2,
    seed: int = 42
):
    """
    Creates val/test splits with long-tail distribution matching train distribution.
    Follows paper methodology: re-sample original test to match train-LT proportions.
    """
    print("\nCreating long-tail validation and test sets...")
    test_targets = np.array(cifar100_test_dataset.targets)
    num_classes = len(train_class_counts)
    
    # Calculate train-LT class proportions
    total_train_samples = sum(train_class_counts)
    train_proportions = [count / total_train_samples for count in train_class_counts]
    
    # Original test size (CIFAR-100 has 100 samples per class = 10k total)
    original_test_size = len(test_targets)
    
    # Calculate target counts for each class in test-LT
    # We want to match train proportions, but limited by available test samples
    target_test_counts = []
    for i in range(num_classes):
        target_count = int(round(train_proportions[i] * original_test_size))
        # Can't exceed 100 samples per class (original CIFAR test limit)
        target_count = min(target_count, 100)
        target_test_counts.append(target_count)
    
    print(f"Target test-LT distribution:")
    print(f"  Head class: {target_test_counts[0]} samples")
    print(f"  Tail class: {target_test_counts[-1]} samples")
    
    # Re-sample from original test to create test-LT pool
    lt_test_pool_indices = []
    
    for i in range(num_classes):
        class_indices_in_test = np.where(test_targets == i)[0]
        num_samples_to_take = target_test_counts[i]
        
        if num_samples_to_take > 0:
            # Randomly sample without replacement  
            sampled_indices = np.random.choice(
                class_indices_in_test, 
                num_samples_to_take, 
                replace=False
            )
            lt_test_pool_indices.extend(sampled_indices)
        
    lt_test_pool_indices = np.array(lt_test_pool_indices)
    
    print(f"Created LT test pool with {len(lt_test_pool_indices)} samples")
    
    # Split the LT test pool: 20% val, 80% test  
    # Important: Do NOT use stratify here - we want to maintain LT distribution
    np.random.shuffle(lt_test_pool_indices)  # Random shuffle
    n_val = int(round(val_size * len(lt_test_pool_indices)))
    
    val_indices = lt_test_pool_indices[:n_val]
    test_indices = lt_test_pool_indices[n_val:]
    
    splits = {
        'val_lt': val_indices.tolist(),
        'test_lt': test_indices.tolist()
    }
    
    for name, indices in splits.items():
        filepath = output_dir / f"{name}_indices.json"
        print(f"Saving {name} split with {len(indices)} samples to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(indices, f)

def safe_train_test_split(X, y, test_size, random_state, min_samples_per_class=2):
    """
    Performs train_test_split with stratification, falling back to random split if 
    stratification fails due to insufficient samples per class.
    """
    try:
        # Check if we have enough samples per class for stratification
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        
        if min_count < min_samples_per_class:
            print(f"Warning: Some classes have only {min_count} samples, using random split instead of stratified")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        if "too few" in str(e).lower():
            print(f"Warning: Stratification failed ({e}), falling back to random split")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            raise e

def create_and_save_splits(
    lt_indices: np.ndarray,
    lt_targets: np.ndarray,
    split_ratios: Dict[str, float],
    output_dir: Path,
    seed: int = 42
):
    """
    Splits the long-tailed dataset indices into train, tuneV, val_small, calib
    and saves them to JSON files.

    Args:
        lt_indices: Indices of the full long-tailed training set.
        lt_targets: Corresponding targets for stratification.
        split_ratios: Dictionary with ratios for tuneV, val_small, calib.
                      The rest will be the 'train' set.
        output_dir: Directory to save the JSON files.
        seed: Random seed for reproducibility.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, split off the largest part: train
    # The rest will be a temporary set for further splitting
    remaining_ratio = sum(v for k, v in split_ratios.items() if k != 'train')
    
    train_indices, temp_indices, train_targets, temp_targets = safe_train_test_split(
        lt_indices,
        lt_targets,
        test_size=remaining_ratio,
        random_state=seed
    )
    
    # Now split the temp set into tuneV, val_small, calib
    # Ratios need to be renormalized
    tuneV_ratio = split_ratios['tuneV'] / remaining_ratio
    
    tuneV_indices, temp_indices, tuneV_targets, temp_targets = safe_train_test_split(
        temp_indices,
        temp_targets,
        test_size=1.0 - tuneV_ratio,
        random_state=seed
    )
    
    remaining_ratio_2 = split_ratios['val_small'] + split_ratios['calib']
    val_small_ratio = split_ratios['val_small'] / remaining_ratio_2
    
    val_small_indices, calib_indices, _, _ = safe_train_test_split(
        temp_indices,
        temp_targets,
        test_size=1.0 - val_small_ratio,
        random_state=seed
    )
    
    splits = {
        'train': train_indices.tolist(),
        'tuneV': tuneV_indices.tolist(),
        'val_small': val_small_indices.tolist(),
        'calib': calib_indices.tolist()
    }
    
    # Save each split to a separate file
    for name, indices in splits.items():
        filepath = output_dir / f"{name}_indices.json"
        print(f"Saving {name} split with {len(indices)} samples to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(indices, f)
            
    print("\nSplits created and saved successfully!")