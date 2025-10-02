#!/usr/bin/env python3
"""
Script to analyze the impact of different head ratio and threshold settings
on class grouping for CIFAR-100-LT.
"""

import sys
import torch
sys.path.append('.')
from src.data.datasets import get_cifar100_lt_counts
from src.data.groups import get_class_to_group, get_class_to_group_by_threshold
import matplotlib.pyplot as plt


def analyze_grouping_impact():
    """Analyze how different grouping methods affect class distribution."""
    
    # Get CIFAR-100-LT class counts
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    
    print("="*60)
    print("CIFAR-100-LT Grouping Analysis")
    print("="*60)
    
    print("Dataset statistics:")
    print(f"Total classes: {len(class_counts)}")
    print(f"Max samples per class: {max(class_counts)}")
    print(f"Min samples per class: {min(class_counts)}")
    print(f"Total samples: {sum(class_counts)}")
    
    # 1. Analyze threshold-based grouping
    print("\n1. THRESHOLD-BASED GROUPING:")
    print("-" * 40)
    
    thresholds = [10, 15, 20, 25, 30, 40, 50]
    for threshold in thresholds:
        class_to_group = get_class_to_group_by_threshold(class_counts, threshold)
        head_classes = (class_to_group == 0).sum().item()
        tail_classes = (class_to_group == 1).sum().item()
        
        # Calculate sample distribution
        head_samples = sum(class_counts[i] for i in range(len(class_counts)) if class_to_group[i] == 0)
        tail_samples = sum(class_counts[i] for i in range(len(class_counts)) if class_to_group[i] == 1)
        
        print(f"Threshold {threshold:2d}: {head_classes:2d} head ({head_samples:5d} samples) | "
              f"{tail_classes:2d} tail ({tail_samples:4d} samples) | "
              f"Head ratio: {head_classes/100:.2f}")
    
    # 2. Analyze ratio-based grouping
    print("\n2. RATIO-BASED GROUPING:")
    print("-" * 40)
    
    head_ratios = [0.3, 0.4, 0.5, 0.6, 0.69, 0.7, 0.8]
    for head_ratio in head_ratios:
        class_to_group = get_class_to_group(class_counts, K=2, head_ratio=head_ratio)
        head_classes = (class_to_group == 0).sum().item()
        tail_classes = (class_to_group == 1).sum().item()
        
        # Calculate sample distribution
        head_samples = sum(class_counts[i] for i in range(len(class_counts)) if class_to_group[i] == 0)
        tail_samples = sum(class_counts[i] for i in range(len(class_counts)) if class_to_group[i] == 1)
        
        # Find the actual threshold this ratio corresponds to
        sorted_counts = sorted(class_counts, reverse=True)
        actual_threshold = sorted_counts[head_classes-1] if head_classes > 0 else 0
        
        print(f"Head ratio {head_ratio:.2f}: {head_classes:2d} head ({head_samples:5d} samples) | "
              f"{tail_classes:2d} tail ({tail_samples:4d} samples) | "
              f"Min head samples: {actual_threshold}")
    
    # 3. Show equivalence
    print("\n3. THRESHOLD vs RATIO EQUIVALENCE:")
    print("-" * 40)
    
    # Current setup: threshold=20 vs head_ratio=0.69
    threshold_based = get_class_to_group_by_threshold(class_counts, 20)
    ratio_based = get_class_to_group(class_counts, K=2, head_ratio=0.69)
    
    if torch.equal(threshold_based, ratio_based):
        print("✅ Threshold=20 is EQUIVALENT to head_ratio=0.69")
    else:
        print("❌ Threshold=20 is NOT equivalent to head_ratio=0.69")
        
    print(f"Threshold=20: {(threshold_based==0).sum()} head classes")
    print(f"Ratio=0.69:   {(ratio_based==0).sum()} head classes")
    
    # 4. Recommendations
    print("\n4. RECOMMENDATIONS:")
    print("-" * 40)
    print("• Use threshold=20 for interpretable, rule-based grouping")
    print("• Classes with >20 samples are considered 'head', <=20 are 'tail'")
    print("• This gives a natural division based on sufficient sample size")
    print("• Threshold method is more interpretable than arbitrary ratios")
    print("• Current head_ratio=0.69 was likely chosen to match threshold=20")


def create_grouping_visualization():
    """Create a visualization of class counts and grouping."""
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    
    # Create threshold-based grouping
    class_to_group = get_class_to_group_by_threshold(class_counts, 20)
    
    plt.figure(figsize=(12, 6))
    
    # Plot class counts with colors based on groups
    colors = ['#ff7f0e' if group == 0 else '#1f77b4' for group in class_to_group]
    plt.bar(range(len(class_counts)), class_counts, color=colors, alpha=0.7)
    
    # Add threshold line
    plt.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Threshold = 20')
    
    # Formatting
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title('CIFAR-100-LT Class Distribution with Threshold-based Grouping')
    plt.grid(True, alpha=0.3)
    
    # Create custom legend
    import matplotlib.patches as mpatches
    head_patch = mpatches.Patch(color='#ff7f0e', alpha=0.7, label='Head Classes (>20 samples)')
    tail_patch = mpatches.Patch(color='#1f77b4', alpha=0.7, label='Tail Classes (≤20 samples)')
    threshold_line = plt.Line2D([0], [0], color='red', linestyle='--', label='Threshold = 20')
    
    plt.legend(handles=[head_patch, tail_patch, threshold_line], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('class_grouping_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Visualization saved as 'class_grouping_analysis.png'")


if __name__ == "__main__":
    analyze_grouping_impact()
    create_grouping_visualization()