#!/usr/bin/env python3
"""
Simple Dataset Distribution Visualizer (No external dependencies beyond matplotlib)
Visualize CIFAR-100-LT dataset distributions from head to tail classes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import Dict, List


def load_split_indices(splits_dir: str = "data/cifar100_lt_if100_splits") -> Dict[str, List[int]]:
    """Load all dataset split indices."""
    splits_dir = Path(splits_dir)
    datasets = {}
    
    split_files = {
        'train': 'train_indices.json',
        'val_lt': 'val_lt_indices.json', 
        'test_lt': 'test_lt_indices.json',
        'tunev': 'tuneV_indices.json'
    }
    
    print("📂 Loading dataset indices...")
    for split_name, filename in split_files.items():
        file_path = splits_dir / filename
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                indices = json.load(f)
            datasets[split_name] = indices
            print(f"  ✅ {split_name}: {len(indices):,} samples")
        else:
            print(f"  ⚠️  {split_name} not found")
            
    return datasets


def get_class_counts_from_indices(indices: List[int], is_train_split: bool = True) -> Dict[int, int]:
    """Get class counts from dataset indices."""
    import torchvision
    
    # Load appropriate CIFAR-100 split
    cifar_dataset = torchvision.datasets.CIFAR100(
        root="data", train=is_train_split, download=False
    )
    
    # Count classes
    class_counts = Counter()
    for idx in indices:
        if idx < len(cifar_dataset):
            _, label = cifar_dataset[idx]
            class_counts[label] += 1
            
    return dict(class_counts)


def analyze_all_distributions(dataset_indices: Dict[str, List[int]]) -> Dict[str, Dict[int, int]]:
    """Analyze class distributions for all datasets."""
    print("\n🔍 Analyzing class distributions...")
    
    distributions = {}
    
    # Train split (from CIFAR-100 train)
    if 'train' in dataset_indices:
        print("  Analyzing train split...")
        distributions['train'] = get_class_counts_from_indices(
            dataset_indices['train'], is_train_split=True
        )
        
    # Test splits (from CIFAR-100 test)  
    test_splits = ['val_lt', 'test_lt', 'tunev', 'val_small', 'calib']
    for split in test_splits:
        if split in dataset_indices:
            print(f"  Analyzing {split} split...")
            distributions[split] = get_class_counts_from_indices(
                dataset_indices[split], is_train_split=False
            )
            
    return distributions


def create_distribution_plots(distributions: Dict[str, Dict[int, int]]) -> None:
    """Create comprehensive distribution plots."""
    print("\n📊 Creating distribution visualizations...")
    
    # Setup output directory
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Color scheme (updated for 4 datasets only)
    colors = {
        'train': '#1f77b4',      # Blue
        'val_lt': '#ff7f0e',     # Orange  
        'test_lt': '#2ca02c',    # Green
        'tunev': '#d62728'       # Red
    }
    
    # 1. Individual distribution plots (2x2 grid for 4 datasets)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (split_name, class_counts) in enumerate(distributions.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Prepare data (classes 0-99, head to tail)
        classes = list(range(100))
        counts = [class_counts.get(i, 0) for i in classes]
        
        # Create bar plot
        ax.bar(classes, counts, alpha=0.7, color=colors.get(split_name, 'gray'))
        
        # Calculate statistics
        head_count = sum(counts[:33])    # Classes 0-32
        mid_count = sum(counts[33:67])   # Classes 33-66
        tail_count = sum(counts[67:])    # Classes 67-99
        total = sum(counts)
        
        # Styling
        ax.set_title(f'{split_name.upper().replace("_", " ")}\n'
                    f'{total:,} samples', fontweight='bold', fontsize=11)
        ax.set_xlabel('Class Index (Head → Tail)')
        ax.set_ylabel('Sample Count')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add group boundaries
        ax.axvline(x=32.5, color='red', linestyle='--', alpha=0.6, linewidth=1)
        ax.axvline(x=66.5, color='red', linestyle='--', alpha=0.6, linewidth=1)
        
        # Add statistics text
        if total > 0:
            stats_text = f'Head: {head_count:,} ({head_count/total*100:.1f}%)\n'
            stats_text += f'Mid: {mid_count:,} ({mid_count/total*100:.1f}%)\n'
            stats_text += f'Tail: {tail_count:,} ({tail_count/total*100:.1f}%)'
            
            if counts[0] > 0 and counts[99] > 0:
                stats_text += f'\nIF: {counts[0]/counts[99]:.1f}'
                
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Hide unused subplots
    for idx in range(len(distributions), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "individual_distributions.png", dpi=200, bbox_inches='tight')
    print(f"  💾 Saved: {output_dir}/individual_distributions.png")
    plt.close()
    
    # 2. Combined comparison plot
    plt.figure(figsize=(14, 8))
    
    classes = list(range(100))
    
    # Plot each distribution as proportions
    for split_name, class_counts in distributions.items():
        counts = [class_counts.get(i, 0) for i in classes]
        total = sum(counts) if sum(counts) > 0 else 1
        proportions = [c / total for c in counts]
        
        plt.semilogy(classes, proportions, 'o-', alpha=0.8, 
                    color=colors.get(split_name, 'gray'),
                    label=split_name.upper().replace('_', ' '),
                    markersize=3, linewidth=2)
    
    # Add theoretical curve
    theoretical = [500 * (100 ** (-(i/99))) for i in classes]
    theoretical_total = sum(theoretical)
    theoretical_props = [t / theoretical_total for t in theoretical]
    
    plt.semilogy(classes, theoretical_props, 'k--', alpha=0.6, 
                label='Theoretical (IF=100)', linewidth=2)
    
    # Styling
    plt.title('CIFAR-100-LT: All Dataset Distributions Comparison\n(Proportional Scale)', 
              fontweight='bold', fontsize=14)
    plt.xlabel('Class Index (Head → Tail)', fontsize=12)
    plt.ylabel('Proportion (log scale)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add group boundaries and labels
    plt.axvline(x=32.5, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=66.5, color='red', linestyle='--', alpha=0.5)
    
    # Group labels
    plt.text(16, plt.ylim()[1]*0.3, 'HEAD\n(0-32)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    plt.text(49, plt.ylim()[1]*0.3, 'MID\n(33-66)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    plt.text(83, plt.ylim()[1]*0.3, 'TAIL\n(67-99)', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_comparison.png", dpi=200, bbox_inches='tight')
    print(f"  💾 Saved: {output_dir}/combined_comparison.png")
    plt.close()
    
    # 3. Summary statistics table
    create_summary_table(distributions, output_dir)


def create_summary_table(distributions: Dict[str, Dict[int, int]], output_dir: Path) -> None:
    """Create a summary statistics table."""
    print("  📋 Creating summary statistics...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Split', 'Total', 'Head Count', 'Mid Count', 'Tail Count', 
               'Head %', 'Mid %', 'Tail %', 'IF']
    
    for split_name, class_counts in distributions.items():
        counts = [class_counts.get(i, 0) for i in range(100)]
        total = sum(counts)
        
        if total == 0:
            continue
            
        head_count = sum(counts[:33])
        mid_count = sum(counts[33:67])
        tail_count = sum(counts[67:])
        
        imbalance_factor = counts[0] / counts[99] if counts[99] > 0 else 0
        
        row = [
            split_name.upper().replace('_', ' '),
            f"{total:,}",
            f"{head_count:,}",
            f"{mid_count:,}",
            f"{tail_count:,}",
            f"{head_count/total*100:.1f}%",
            f"{mid_count/total*100:.1f}%",
            f"{tail_count/total*100:.1f}%",
            f"{imbalance_factor:.1f}" if imbalance_factor > 0 else "N/A"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    table.auto_set_column_width(col=list(range(len(headers))))
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('CIFAR-100-LT Dataset Statistics Summary', 
              fontweight='bold', fontsize=14, pad=20)
    
    plt.savefig(output_dir / "summary_table.png", dpi=200, bbox_inches='tight')
    print(f"  💾 Saved: {output_dir}/summary_table.png")
    plt.close()


def print_console_summary(distributions: Dict[str, Dict[int, int]]) -> None:
    """Print detailed summary to console."""
    print("\n" + "="*80)
    print("📊 CIFAR-100-LT DATASET DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for split_name, class_counts in distributions.items():
        counts = [class_counts.get(i, 0) for i in range(100)]
        total = sum(counts)
        
        if total == 0:
            continue
        
        head_count = sum(counts[:33])
        mid_count = sum(counts[33:67])
        tail_count = sum(counts[67:])
        
        print(f"\n🎯 {split_name.upper().replace('_', ' ')} DATASET:")
        print(f"   📊 Total samples: {total:,}")
        print(f"   🏆 Head (0-32):   {head_count:,} samples ({head_count/total*100:.2f}%)")
        print(f"   🎯 Mid (33-66):   {mid_count:,} samples ({mid_count/total*100:.2f}%)")
        print(f"   🎭 Tail (67-99):  {tail_count:,} samples ({tail_count/total*100:.2f}%)")
        
        if counts[0] > 0 and counts[99] > 0:
            print(f"   ⚖️  Imbalance Factor: {counts[0]}/{counts[99]} = {counts[0]/counts[99]:.2f}")
        
        # Show sample counts for key classes
        print(f"   📈 Class distribution examples:")
        print(f"      Class 0 (head): {counts[0]} samples")
        print(f"      Class 50 (mid): {counts[50]} samples") 
        print(f"      Class 99 (tail): {counts[99]} samples")
    
    print("\n" + "="*80)


def main():
    """Main function to run complete visualization."""
    print("🎨 CIFAR-100-LT Dataset Distribution Visualizer")
    print("="*60)
    
    try:
        # Load data
        dataset_indices = load_split_indices()
        
        if not dataset_indices:
            print("❌ No dataset indices found!")
            return
        
        # Analyze distributions
        distributions = analyze_all_distributions(dataset_indices)
        
        # Print console summary
        print_console_summary(distributions)
        
        # Create visualizations
        create_distribution_plots(distributions)
        
        print("\n" + "="*60)
        print("✅ Visualization completed successfully!")
        print("📁 Check outputs/visualizations/ for plots:")
        print("   • individual_distributions.png - Each dataset separately")
        print("   • combined_comparison.png - All datasets overlaid")
        print("   • summary_table.png - Statistical summary table")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()