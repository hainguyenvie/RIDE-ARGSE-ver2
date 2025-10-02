#!/usr/bin/env python3
"""
Data Distribution Visualization for CIFAR-100-LT datasets.
Visualize the distribution of all datasets from head to tail classes.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Import our data utilities
from src.data.dataloader_utils import CIFAR100LTDataModule


class DatasetDistributionVisualizer:
    """Comprehensive visualizer for CIFAR-100-LT dataset distributions."""
    
    def __init__(self, splits_dir: str = "data/cifar100_lt_if100_splits"):
        self.splits_dir = Path(splits_dir)
        self.output_dir = Path("outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette for different splits
        self.colors = {
            'train': '#1f77b4',      # Blue
            'val_lt': '#ff7f0e',     # Orange  
            'test_lt': '#2ca02c',    # Green
            'tunev': '#d62728',      # Red
            'val_small': '#9467bd',  # Purple
            'calib': '#8c564b'       # Brown
        }
        
        # Load all dataset indices
        self.dataset_indices = self._load_all_indices()
        
    def _load_all_indices(self) -> Dict[str, List[int]]:
        """Load indices for all available splits."""
        datasets = {}
        
        split_files = {
            'train': 'train_indices.json',
            'val_lt': 'val_lt_indices.json', 
            'test_lt': 'test_lt_indices.json',
            'tunev': 'tuneV_indices.json',
            'val_small': 'val_small_indices.json',
            'calib': 'calib_indices.json'
        }
        
        for split_name, filename in split_files.items():
            file_path = self.splits_dir / filename
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    indices = json.load(f)
                datasets[split_name] = indices
                print(f"✅ Loaded {split_name}: {len(indices):,} samples")
            else:
                print(f"⚠️  {split_name} not found: {file_path}")
                
        return datasets
    
    def _get_class_counts(self, split_name: str, is_train_split: bool = True) -> Dict[int, int]:
        """Get class counts for a specific split."""
        if split_name not in self.dataset_indices:
            return {}
            
        indices = self.dataset_indices[split_name]
        
        # Load CIFAR-100 to get labels
        import torchvision
        if is_train_split:
            cifar_dataset = torchvision.datasets.CIFAR100(
                root="data", train=True, download=False
            )
        else:
            cifar_dataset = torchvision.datasets.CIFAR100(
                root="data", train=False, download=False  
            )
        
        # Count classes from indices
        class_counts = Counter()
        for idx in indices:
            if idx < len(cifar_dataset):
                _, label = cifar_dataset[idx]
                class_counts[label] += 1
                
        return dict(class_counts)
    
    def get_all_class_distributions(self) -> Dict[str, Dict[int, int]]:
        """Get class distributions for all splits."""
        distributions = {}
        
        # Train split comes from CIFAR-100 train set
        if 'train' in self.dataset_indices:
            distributions['train'] = self._get_class_counts('train', is_train_split=True)
            
        # All other splits come from CIFAR-100 test set
        test_splits = ['val_lt', 'test_lt', 'tunev', 'val_small', 'calib']
        for split in test_splits:
            if split in self.dataset_indices:
                distributions[split] = self._get_class_counts(split, is_train_split=False)
                
        return distributions
    
    def create_comprehensive_visualization(self, save_plots: bool = True) -> None:
        """Create comprehensive visualization of all dataset distributions."""
        print("\n🎨 Creating comprehensive dataset visualizations...")
        
        distributions = self.get_all_class_distributions()
        
        if not distributions:
            print("❌ No distributions found!")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Individual distribution plots (2x3 grid)
        for idx, (split_name, class_counts) in enumerate(distributions.items()):
            ax = plt.subplot(4, 2, idx + 1)
            self._plot_single_distribution(ax, split_name, class_counts)
            
        # 2. Combined comparison plot
        ax_combined = plt.subplot(4, 1, 3)
        self._plot_combined_distributions(ax_combined, distributions)
        
        # 3. Statistical comparison
        ax_stats = plt.subplot(4, 1, 4)
        self._plot_statistical_comparison(ax_stats, distributions)
        
        plt.tight_layout()
        
        if save_plots:
            output_file = self.output_dir / "comprehensive_data_distributions.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"💾 Saved comprehensive plot: {output_file}")
            
        plt.show()
        
    def _plot_single_distribution(self, ax, split_name: str, class_counts: Dict[int, int]) -> None:
        """Plot distribution for a single split."""
        classes = list(range(100))
        counts = [class_counts.get(i, 0) for i in classes]
        
        # Create bar plot
        bars = ax.bar(classes, counts, alpha=0.7, color=self.colors.get(split_name, 'gray'))
        
        # Styling
        ax.set_title(f'{split_name.upper().replace("_", " ")} Distribution\n'
                    f'({sum(counts):,} samples)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Class Index (Head → Tail)')
        ax.set_ylabel('Sample Count')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add group boundaries
        ax.axvline(x=32.5, color='red', linestyle='--', alpha=0.5, label='Head|Mid')
        ax.axvline(x=66.5, color='red', linestyle='--', alpha=0.5, label='Mid|Tail')
        
        # Calculate and display key statistics
        head_count = sum(counts[:33])
        mid_count = sum(counts[33:67])
        tail_count = sum(counts[67:])
        total = sum(counts)
        
        if total > 0:
            stats_text = f'H: {head_count:,} ({head_count/total*100:.1f}%)\n'
            stats_text += f'M: {mid_count:,} ({mid_count/total*100:.1f}%)\n' 
            stats_text += f'T: {tail_count:,} ({tail_count/total*100:.1f}%)'
            
            if counts[0] > 0 and counts[99] > 0:
                stats_text += f'\nIF: {counts[0]/counts[99]:.1f}'
                
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
    
    def _plot_combined_distributions(self, ax, distributions: Dict[str, Dict[int, int]]) -> None:
        """Plot all distributions together for comparison."""
        classes = list(range(100))
        
        for split_name, class_counts in distributions.items():
            counts = [class_counts.get(i, 0) for i in classes]
            
            # Normalize to show proportions instead of raw counts
            total = sum(counts) if sum(counts) > 0 else 1
            proportions = [c / total for c in counts]
            
            ax.semilogy(classes, proportions, 'o-', alpha=0.7, 
                       color=self.colors.get(split_name, 'gray'),
                       label=f'{split_name.upper().replace("_", " ")}',
                       markersize=3, linewidth=2)
        
        # Add theoretical exponential curve for reference
        theoretical = [500 * (100 ** (-(i/99))) for i in classes]
        theoretical_total = sum(theoretical)
        theoretical_props = [t / theoretical_total for t in theoretical]
        
        ax.semilogy(classes, theoretical_props, 'k--', alpha=0.5, 
                   label='Theoretical (IF=100)', linewidth=2)
        
        # Styling
        ax.set_title('All Dataset Distributions Comparison\n(Proportional Scale)', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Class Index (Head → Tail)', fontsize=12)
        ax.set_ylabel('Proportion (log scale)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add group boundaries
        ax.axvline(x=32.5, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=66.5, color='red', linestyle='--', alpha=0.5)
        
        # Group labels
        ax.text(16, ax.get_ylim()[1]*0.5, 'HEAD\n(0-32)', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(49, ax.get_ylim()[1]*0.5, 'MID\n(33-66)', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.text(83, ax.get_ylim()[1]*0.5, 'TAIL\n(67-99)', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    def _plot_statistical_comparison(self, ax, distributions: Dict[str, Dict[int, int]]) -> None:
        """Plot statistical comparison between datasets."""
        stats_data = []
        
        for split_name, class_counts in distributions.items():
            counts = [class_counts.get(i, 0) for i in range(100)]
            total = sum(counts)
            
            if total == 0:
                continue
                
            head_count = sum(counts[:33])
            mid_count = sum(counts[33:67])
            tail_count = sum(counts[67:])
            
            # Calculate statistics
            stats = {
                'Split': split_name.upper().replace('_', ' '),
                'Total': total,
                'Head %': head_count / total * 100,
                'Mid %': mid_count / total * 100,
                'Tail %': tail_count / total * 100,
                'IF': counts[0] / counts[99] if counts[99] > 0 else 0
            }
            stats_data.append(stats)
        
        # Create grouped bar chart
        df = pd.DataFrame(stats_data)
        
        if len(df) > 0:
            x = np.arange(len(df))
            width = 0.25
            
            bars1 = ax.bar(x - width, df['Head %'], width, label='Head %', 
                          color='lightgreen', alpha=0.8)
            bars2 = ax.bar(x, df['Mid %'], width, label='Mid %', 
                          color='lightyellow', alpha=0.8)
            bars3 = ax.bar(x + width, df['Tail %'], width, label='Tail %', 
                          color='lightcoral', alpha=0.8)
            
            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Group Distribution Comparison Across Splits', 
                        fontweight='bold', fontsize=14)
            ax.set_xlabel('Dataset Split', fontsize=12)
            ax.set_ylabel('Percentage', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(df['Split'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add table with detailed stats
            table_data = []
            for _, row in df.iterrows():
                table_data.append([
                    row['Split'],
                    f"{row['Total']:,}",
                    f"{row['Head %']:.1f}%",
                    f"{row['Mid %']:.1f}%", 
                    f"{row['Tail %']:.1f}%",
                    f"{row['IF']:.1f}" if row['IF'] > 0 else 'N/A'
                ])
            
            # Add table below the chart
            table = ax.table(cellText=table_data,
                           colLabels=['Split', 'Total', 'Head %', 'Mid %', 'Tail %', 'IF'],
                           cellLoc='center',
                           loc='bottom',
                           bbox=[0, -0.6, 1, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
    
    def save_detailed_report(self, output_file: str = None) -> None:
        """Save detailed statistical report."""
        if output_file is None:
            output_file = self.output_dir / "dataset_distribution_report.json"
        
        print(f"\n📊 Generating detailed distribution report...")
        
        distributions = self.get_all_class_distributions()
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': {},
            'distributions': {},
            'comparisons': {}
        }
        
        # Calculate detailed statistics for each split
        for split_name, class_counts in distributions.items():
            counts = [class_counts.get(i, 0) for i in range(100)]
            total = sum(counts)
            
            if total == 0:
                continue
            
            head_count = sum(counts[:33])
            mid_count = sum(counts[33:67])
            tail_count = sum(counts[67:])
            
            split_stats = {
                'total_samples': total,
                'class_counts': class_counts,
                'group_counts': {
                    'head': head_count,
                    'mid': mid_count,
                    'tail': tail_count
                },
                'group_proportions': {
                    'head': head_count / total,
                    'mid': mid_count / total,
                    'tail': tail_count / total
                },
                'imbalance_factor': counts[0] / counts[99] if counts[99] > 0 else None,
                'statistics': {
                    'mean': np.mean(counts),
                    'std': np.std(counts),
                    'min': np.min(counts),
                    'max': np.max(counts)
                }
            }
            
            report['distributions'][split_name] = split_stats
        
        # Overall summary
        report['summary'] = {
            'total_splits': len(distributions),
            'splits_analyzed': list(distributions.keys()),
            'total_samples_across_splits': sum(
                sum(counts.values()) for counts in distributions.values()
            )
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"💾 Detailed report saved: {output_file}")
        
        return report
    
    def print_summary(self) -> None:
        """Print a concise summary of all datasets."""
        print("\n" + "="*80)
        print("📊 CIFAR-100-LT DATASET DISTRIBUTION SUMMARY")
        print("="*80)
        
        distributions = self.get_all_class_distributions()
        
        for split_name, class_counts in distributions.items():
            counts = [class_counts.get(i, 0) for i in range(100)]
            total = sum(counts)
            
            if total == 0:
                continue
            
            head_count = sum(counts[:33])
            mid_count = sum(counts[33:67])
            tail_count = sum(counts[67:])
            
            print(f"\n🎯 {split_name.upper().replace('_', ' ')}:")
            print(f"   Total: {total:,} samples")
            print(f"   Head (0-32):  {head_count:,} ({head_count/total*100:.1f}%)")
            print(f"   Mid (33-66):  {mid_count:,} ({mid_count/total*100:.1f}%)")
            print(f"   Tail (67-99): {tail_count:,} ({tail_count/total*100:.1f}%)")
            
            if counts[0] > 0 and counts[99] > 0:
                print(f"   IF: {counts[0]/counts[99]:.1f}")
        
        print("\n" + "="*80)


def visualize_all_distributions(
    splits_dir: str = "data/cifar100_lt_if100_splits",
    save_plots: bool = True,
    show_plots: bool = True
) -> DatasetDistributionVisualizer:
    """
    Main function to visualize all dataset distributions.
    
    Args:
        splits_dir: Directory containing the split indices
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots
        
    Returns:
        DatasetDistributionVisualizer instance
    """
    print("🎨 Starting comprehensive dataset distribution visualization...")
    
    # Create visualizer
    visualizer = DatasetDistributionVisualizer(splits_dir)
    
    # Print summary
    visualizer.print_summary()
    
    # Create visualizations
    if show_plots or save_plots:
        # Set matplotlib backend for saving
        if save_plots and not show_plots:
            plt.switch_backend('Agg')
            
        visualizer.create_comprehensive_visualization(save_plots=save_plots)
        
        if not show_plots:
            plt.close('all')
    
    # Save detailed report
    visualizer.save_detailed_report()
    
    print("\n✅ Visualization complete!")
    print(f"📁 Check outputs/visualizations/ for saved plots and reports")
    
    return visualizer


if __name__ == "__main__":
    # Run visualization
    visualizer = visualize_all_distributions(
        save_plots=True,
        show_plots=False  # Set to True if you want to display plots
    )