#!/usr/bin/env python3
"""
Script to analyze train data distribution and show head/tail proportions.
"""

import numpy as np
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from src.data.datasets import get_cifar100_lt_counts

def load_train_indices(data_dir="data/cifar100_lt_if100_splits"):
    """Load train indices from saved splits."""
    train_file = Path(data_dir) / "train_indices.json"
    
    if train_file.exists():
        with open(train_file, 'r') as f:
            train_indices = json.load(f)
        return np.array(train_indices)
    else:
        print(f"Train indices file not found at {train_file}")
        return None

def analyze_train_distribution(data_dir="data/cifar100_lt_if100_splits"):
    """Analyze the training data distribution."""
    print("=== CIFAR-100-LT Training Data Distribution Analysis ===\n")
    
    # Method 1: From theoretical exponential profile
    print("1. Theoretical Distribution (from exponential profile):")
    theoretical_counts = get_cifar100_lt_counts(imb_factor=100, num_classes=100)
    
    total_theoretical = sum(theoretical_counts)
    head_count = theoretical_counts[0]
    tail_count = theoretical_counts[-1]
    
    print(f"   Total samples: {total_theoretical:,}")
    print(f"   Head class (class 0): {head_count} samples ({head_count/total_theoretical*100:.2f}%)")
    print(f"   Tail class (class 99): {tail_count} samples ({tail_count/total_theoretical*100:.2f}%)")
    print(f"   Imbalance Factor: {head_count/tail_count:.1f}")
    
    # Show distribution by groups
    print("\n   Distribution by class groups:")
    print(f"   Classes 0-9 (head):    {sum(theoretical_counts[0:10]):,} samples ({sum(theoretical_counts[0:10])/total_theoretical*100:.1f}%)")
    print(f"   Classes 10-49 (medium): {sum(theoretical_counts[10:50]):,} samples ({sum(theoretical_counts[10:50])/total_theoretical*100:.1f}%)")
    print(f"   Classes 50-89 (low):    {sum(theoretical_counts[50:90]):,} samples ({sum(theoretical_counts[50:90])/total_theoretical*100:.1f}%)")
    print(f"   Classes 90-99 (tail):   {sum(theoretical_counts[90:100]):,} samples ({sum(theoretical_counts[90:100])/total_theoretical*100:.1f}%)")
    
    # Method 2: From actual saved train indices (if available)
    train_indices = load_train_indices(data_dir)
    
    if train_indices is not None:
        print("\n2. Actual Saved Training Data:")
        print(f"   Loaded {len(train_indices):,} training samples from {data_dir}")
        
        # Load CIFAR-100 to get actual targets
        try:
            import torchvision
            cifar100_train = torchvision.datasets.CIFAR100(
                root='data', train=True, download=False
            )
            all_targets = np.array(cifar100_train.targets)
            train_targets = all_targets[train_indices]
            
            # Count samples per class
            class_counts = Counter(train_targets)
            actual_counts = [class_counts.get(i, 0) for i in range(100)]
            
            total_actual = sum(actual_counts)
            head_actual = actual_counts[0]
            tail_actual = actual_counts[99]
            
            print(f"   Total samples: {total_actual:,}")
            print(f"   Head class (class 0): {head_actual} samples ({head_actual/total_actual*100:.2f}%)")
            print(f"   Tail class (class 99): {tail_actual} samples ({tail_actual/total_actual*100:.2f}%)")
            print(f"   Actual Imbalance Factor: {head_actual/tail_actual:.1f}")
            
            # Show actual distribution by groups
            print(f"\n   Actual distribution by class groups:")
            print(f"   Classes 0-9 (head):    {sum(actual_counts[0:10]):,} samples ({sum(actual_counts[0:10])/total_actual*100:.1f}%)")
            print(f"   Classes 10-49 (medium): {sum(actual_counts[10:50]):,} samples ({sum(actual_counts[10:50])/total_actual*100:.1f}%)")
            print(f"   Classes 50-89 (low):    {sum(actual_counts[50:90]):,} samples ({sum(actual_counts[50:90])/total_actual*100:.1f}%)")
            print(f"   Classes 90-99 (tail):   {sum(actual_counts[90:100]):,} samples ({sum(actual_counts[90:100])/total_actual*100:.1f}%)")
            
            # Compare theoretical vs actual
            print(f"\n3. Comparison (Theoretical vs Actual):")
            print(f"   Total samples: {total_theoretical:,} vs {total_actual:,}")
            print(f"   Head samples: {head_count} vs {head_actual}")  
            print(f"   Tail samples: {tail_count} vs {tail_actual}")
            
            # Show some examples of class counts
            print(f"\n4. Sample of class counts (first 10 and last 10 classes):")
            print("   Class | Theoretical | Actual")
            print("   ------|-------------|--------")
            for i in range(10):
                print(f"   {i:5d} | {theoretical_counts[i]:11d} | {actual_counts[i]:6d}")
            print("   ...   |     ...     |  ...")
            for i in range(90, 100):
                print(f"   {i:5d} | {theoretical_counts[i]:11d} | {actual_counts[i]:6d}")
            
        except Exception as e:
            print(f"   Error loading CIFAR-100 data: {e}")
            print("   Make sure CIFAR-100 data is downloaded in 'data/' directory")
    
    else:
        print(f"\n2. No saved training indices found at {data_dir}")
        print("   Run the data preparation script first to generate training splits")

def plot_distribution():
    """Plot the class distribution."""
    try:
        counts = get_cifar100_lt_counts(imb_factor=100, num_classes=100)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(100), counts, alpha=0.7, color='steelblue')
        plt.xlabel('Class Index (0=Head, 99=Tail)')
        plt.ylabel('Number of Samples')
        plt.title('CIFAR-100-LT Class Distribution (IF=100)')
        plt.yscale('log')  # Log scale to better visualize the exponential decay
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        plt.annotate(f'Head: {counts[0]} samples', 
                    xy=(0, counts[0]), xytext=(10, counts[0]*2),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        plt.annotate(f'Tail: {counts[99]} samples', 
                    xy=(99, counts[99]), xytext=(80, counts[99]*10),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        plt.tight_layout()
        plt.savefig('cifar100_lt_distribution.png', dpi=150, bbox_inches='tight')
        print(f"\n5. Distribution plot saved as 'cifar100_lt_distribution.png'")
        
    except Exception as e:
        print(f"Error creating plot: {e}")

if __name__ == "__main__":
    analyze_train_distribution()
    plot_distribution()