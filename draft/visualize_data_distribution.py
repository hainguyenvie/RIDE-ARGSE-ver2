# visualize_data_distribution.py
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import torchvision

from src.data.datasets import get_cifar100_lt_counts
from src.data.groups import get_class_to_group

def load_split_indices(split_dir, split_name):
    """Load indices from JSON file"""
    filepath = split_dir / f"{split_name}_indices.json"
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_class_distribution():
    """Visualize class distribution across splits"""
    
    # Configuration
    IMB_FACTOR = 100
    DATA_ROOT = "./data"
    SPLIT_DIR = Path(f"./data/cifar100_lt_if100_splits")
    
    # Load data
    cifar100_train = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=False)
    targets = np.array(cifar100_train.targets)
    
    # Load expected counts and groups
    expected_counts = get_cifar100_lt_counts(IMB_FACTOR)
    class_to_group = get_class_to_group(expected_counts, K=2, head_ratio=0.5)
    
    # Load splits
    train_indices = load_split_indices(SPLIT_DIR, 'train')
    
    # Get actual class distribution
    train_targets = targets[train_indices]
    actual_counts = Counter(train_targets)
    
    # Prepare data for plotting
    classes = list(range(100))
    expected_counts_list = expected_counts
    actual_counts_list = [actual_counts.get(i, 0) for i in classes]
    
    # Identify head/tail classes
    head_classes = [i for i in range(100) if class_to_group[i] == 0]
    tail_classes = [i for i in range(100) if class_to_group[i] == 1]
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Expected vs Actual distribution
    plt.subplot(2, 2, 1)
    plt.plot(classes, expected_counts_list, 'b-', label='Expected (Exponential Decay)', linewidth=2)
    plt.plot(classes, actual_counts_list, 'r-', label='Actual Train Split', linewidth=2, alpha=0.8)
    plt.xlabel('Class Index (sorted by frequency)')
    plt.ylabel('Sample Count')
    plt.title(f'CIFAR-100-LT (IF={IMB_FACTOR}): Expected vs Actual Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Head vs Tail distribution
    plt.subplot(2, 2, 2)
    head_counts = [actual_counts_list[i] for i in head_classes]
    tail_counts = [actual_counts_list[i] for i in tail_classes]
    
    plt.boxplot([head_counts, tail_counts], labels=['Head Classes', 'Tail Classes'])
    plt.ylabel('Sample Count')
    plt.title('Head vs Tail Class Distribution')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative sample percentage
    plt.subplot(2, 2, 3)
    sorted_actual = sorted(actual_counts_list, reverse=True)
    cumulative_samples = np.cumsum(sorted_actual)
    cumulative_percentage = cumulative_samples / cumulative_samples[-1] * 100
    
    plt.plot(range(1, 101), cumulative_percentage, 'g-', linewidth=2)
    plt.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% of data')
    plt.axvline(x=50, color='b', linestyle='--', alpha=0.7, label='Top 50% classes')
    plt.xlabel('Number of Top Classes')
    plt.ylabel('Cumulative Sample Percentage')
    plt.title('Cumulative Distribution of Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Sample distribution histogram
    plt.subplot(2, 2, 4)
    plt.hist(head_counts, bins=20, alpha=0.7, label='Head Classes', color='blue')
    plt.hist(tail_counts, bins=20, alpha=0.7, label='Tail Classes', color='red')
    plt.xlabel('Sample Count')
    plt.ylabel('Number of Classes')
    plt.title('Distribution of Sample Counts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("="*60)
    print("DETAILED DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"Total classes: 100")
    print(f"Head classes (top 50%): {len(head_classes)}")
    print(f"Tail classes (bottom 50%): {len(tail_classes)}")
    
    print(f"\nHead classes statistics:")
    print(f"  Min samples: {min(head_counts)}")
    print(f"  Max samples: {max(head_counts)}")
    print(f"  Mean samples: {np.mean(head_counts):.2f}")
    print(f"  Total samples: {sum(head_counts)} ({sum(head_counts)/sum(actual_counts_list)*100:.1f}%)")
    
    print(f"\nTail classes statistics:")
    print(f"  Min samples: {min(tail_counts)}")
    print(f"  Max samples: {max(tail_counts)}")
    print(f"  Mean samples: {np.mean(tail_counts):.2f}")
    print(f"  Total samples: {sum(tail_counts)} ({sum(tail_counts)/sum(actual_counts_list)*100:.1f}%)")
    
    # Long-tail metrics
    total_samples = sum(actual_counts_list)
    top_10_percent_classes = int(0.1 * 100)
    top_10_percent_samples = sum(sorted(actual_counts_list, reverse=True)[:top_10_percent_classes])
    
    print(f"\nLong-tail metrics:")
    print(f"  Imbalance ratio: {max(actual_counts_list) / min([c for c in actual_counts_list if c > 0]):.2f}")
    print(f"  Top 10% classes contain: {top_10_percent_samples/total_samples*100:.1f}% of data")
    print(f"  Correlation with expected: {np.corrcoef(expected_counts_list, actual_counts_list)[0,1]:.4f}")

def plot_split_comparisons():
    """Compare distributions across different splits"""
    
    # Configuration
    SPLIT_DIR = Path(f"./data/cifar100_lt_if100_splits")
    DATA_ROOT = "./data"
    
    # Load data
    cifar100_train = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=False)
    cifar100_test = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=False)
    train_targets = np.array(cifar100_train.targets)
    test_targets = np.array(cifar100_test.targets)
    
    # Load all training splits
    splits = ['train', 'tuneV', 'val_small', 'calib']
    split_distributions = {}
    
    for split_name in splits:
        indices = load_split_indices(SPLIT_DIR, split_name)
        targets = train_targets[indices]
        counts = Counter(targets)
        split_distributions[split_name] = [counts.get(i, 0) for i in range(100)]
    
    # Load test splits
    test_splits = ['val_lt', 'test_lt']
    for split_name in test_splits:
        indices = load_split_indices(SPLIT_DIR, split_name)
        targets = test_targets[indices]
        counts = Counter(targets)
        split_distributions[split_name] = [counts.get(i, 0) for i in range(100)]
    
    # Create comparison plot
    plt.figure(figsize=(18, 12))
    
    # Plot each split
    for i, (split_name, counts) in enumerate(split_distributions.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(range(100), counts, 'o-', markersize=3, linewidth=1)
        plt.title(f'{split_name.upper()} Split\n({sum(counts)} samples)')
        plt.xlabel('Class Index')
        plt.ylabel('Sample Count')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('split_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print split comparison statistics
    print("\n" + "="*60)
    print("SPLIT COMPARISON STATISTICS")
    print("="*60)
    
    for split_name, counts in split_distributions.items():
        non_zero_counts = [c for c in counts if c > 0]
        print(f"\n{split_name.upper()}:")
        print(f"  Total samples: {sum(counts)}")
        print(f"  Classes represented: {len(non_zero_counts)}/100")
        print(f"  Min/Max samples: {min(non_zero_counts) if non_zero_counts else 0}/{max(counts)}")
        print(f"  Mean samples: {np.mean(counts):.2f}")

if __name__ == "__main__":
    print("Generating distribution visualizations...")
    plot_class_distribution()
    print("\nGenerating split comparisons...")
    plot_split_comparisons()
    print("\nVisualization complete! Check 'data_distribution_analysis.png' and 'split_distributions.png'")