#!/usr/bin/env python3
"""
Data Distribution Analysis for CIFAR-100-LT Splits
Analyzes class distribution, imbalance factors, and proportion matching across splits.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import pandas as pd
import torchvision
from typing import Dict, List, Tuple

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataDistributionAnalyzer:
    """Comprehensive analyzer for CIFAR-100-LT data distribution."""
    
    def __init__(self, splits_dir: str = "data/cifar100_lt_if100_splits"):
        self.splits_dir = Path(splits_dir)
        self.splits_data = {}
        self.cifar_test = None
        self.cifar_train = None
        
        # Load CIFAR-100 datasets to get true labels
        self._load_cifar_datasets()
        
        # Load all split files
        self._load_splits()
        
    def _load_cifar_datasets(self):
        """Load original CIFAR-100 datasets for label extraction."""
        print("📥 Loading CIFAR-100 datasets...")
        
        # Load without transforms for analysis
        self.cifar_train = torchvision.datasets.CIFAR100(
            root='data', train=True, download=False
        )
        self.cifar_test = torchvision.datasets.CIFAR100(
            root='data', train=False, download=False
        )
        
        print(f"✅ Train: {len(self.cifar_train)} samples")
        print(f"✅ Test: {len(self.cifar_test)} samples")
        
    def _load_splits(self):
        """Load all split index files."""
        print("\n📊 Loading split files...")
        
        split_files = {
            'train': 'train_indices.json',
            'val_lt': 'val_lt_indices.json', 
            'test_lt': 'test_lt_indices.json',
            'tuneV': 'tuneV_indices.json',
            'val_small': 'val_small_indices.json',
            'calib': 'calib_indices.json'
        }
        
        for split_name, filename in split_files.items():
            filepath = self.splits_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    indices = json.load(f)
                self.splits_data[split_name] = indices
                print(f"✅ {split_name}: {len(indices)} samples")
            else:
                print(f"⚠️  Missing: {filename}")
    
    def get_split_labels(self, split_name: str) -> List[int]:
        """Extract labels for a specific split."""
        if split_name not in self.splits_data:
            raise ValueError(f"Split '{split_name}' not found")
            
        indices = self.splits_data[split_name]
        
        # Determine source dataset
        if split_name == 'train':
            source_dataset = self.cifar_train
            max_idx = len(self.cifar_train) - 1
        else:
            source_dataset = self.cifar_test
            max_idx = len(self.cifar_test) - 1
            
        # Extract labels with bounds checking
        labels = []
        invalid_indices = []
        
        for idx in indices:
            if idx > max_idx:
                invalid_indices.append(idx)
                continue
                
            try:
                _, label = source_dataset[idx]
                labels.append(label)
            except IndexError:
                invalid_indices.append(idx)
                continue
        
        if invalid_indices:
            print(f"⚠️  Warning: {len(invalid_indices)} invalid indices in {split_name} split")
            print(f"    Invalid indices: {invalid_indices[:10]}..." if len(invalid_indices) > 10 else f"    Invalid indices: {invalid_indices}")
            print(f"    Dataset size: {len(source_dataset)}, Max valid index: {max_idx}")
            
        return labels
    
    def analyze_split_distribution(self, split_name: str) -> Dict:
        """Analyze class distribution for a specific split."""
        labels = self.get_split_labels(split_name)
        
        # Count classes
        class_counts = Counter(labels)
        
        # Calculate statistics
        total_samples = len(labels)
        num_classes = len(class_counts)
        
        # Sort classes by count (descending)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate proportions
        proportions = {cls: count/total_samples for cls, count in class_counts.items()}
        
        # Imbalance statistics
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_factor = max_count / min_count if min_count > 0 else float('inf')
        
        # Group analysis (head/medium/tail)
        sorted_counts = [count for _, count in sorted_classes]
        head_threshold = len(sorted_counts) // 3
        tail_threshold = 2 * len(sorted_counts) // 3
        
        head_classes = sorted_classes[:head_threshold]
        medium_classes = sorted_classes[head_threshold:tail_threshold]
        tail_classes = sorted_classes[tail_threshold:]
        
        analysis = {
            'split_name': split_name,
            'total_samples': total_samples,
            'num_classes': num_classes,
            'class_counts': dict(class_counts),
            'proportions': proportions,
            'sorted_classes': sorted_classes,
            'imbalance_factor': imbalance_factor,
            'max_count': max_count,
            'min_count': min_count,
            'head_classes': head_classes,
            'medium_classes': medium_classes, 
            'tail_classes': tail_classes,
            'head_avg': np.mean([count for _, count in head_classes]),
            'medium_avg': np.mean([count for _, count in medium_classes]),
            'tail_avg': np.mean([count for _, count in tail_classes])
        }
        
        return analysis
    
    def analyze_all_splits(self) -> Dict:
        """Analyze distribution for all available splits."""
        print("\n🔍 Analyzing all splits...")
        
        all_analysis = {}
        
        for split_name in self.splits_data.keys():
            print(f"\n📊 Analyzing {split_name}...")
            analysis = self.analyze_split_distribution(split_name)
            all_analysis[split_name] = analysis
            
            # Print summary
            print(f"   Total samples: {analysis['total_samples']:,}")
            print(f"   Classes: {analysis['num_classes']}")
            print(f"   IF: {analysis['imbalance_factor']:.1f}")
            print(f"   Max/Min: {analysis['max_count']}/{analysis['min_count']}")
            print(f"   Head/Med/Tail avg: {analysis['head_avg']:.1f}/{analysis['medium_avg']:.1f}/{analysis['tail_avg']:.1f}")
            
        return all_analysis
    
    def compare_proportions(self, reference_split: str = 'train') -> pd.DataFrame:
        """Compare class proportions across splits."""
        print(f"\n🔄 Comparing proportions (reference: {reference_split})...")
        
        # Get reference proportions
        ref_analysis = self.analyze_split_distribution(reference_split)
        ref_proportions = ref_analysis['proportions']
        
        # Compare with other splits
        comparison_data = []
        
        for split_name in self.splits_data.keys():
            if split_name == reference_split:
                continue
                
            split_analysis = self.analyze_split_distribution(split_name)
            split_proportions = split_analysis['proportions']
            
            for class_id in range(100):  # CIFAR-100 has 100 classes
                ref_prop = ref_proportions.get(class_id, 0)
                split_prop = split_proportions.get(class_id, 0)
                diff = abs(ref_prop - split_prop)
                
                comparison_data.append({
                    'class': class_id,
                    'split': split_name,
                    'reference_prop': ref_prop,
                    'split_prop': split_prop,
                    'abs_diff': diff,
                    'rel_diff': diff / ref_prop if ref_prop > 0 else 0
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Summary statistics
        print("\n📈 Proportion matching summary:")
        for split_name in df['split'].unique():
            split_data = df[df['split'] == split_name]
            avg_abs_diff = split_data['abs_diff'].mean()
            max_abs_diff = split_data['abs_diff'].max()
            print(f"   {split_name}: avg_diff={avg_abs_diff:.6f}, max_diff={max_abs_diff:.6f}")
            
        return df
    
    def plot_class_distributions(self, splits_to_plot: List[str] = None, 
                                figsize: Tuple[int, int] = (20, 12)):
        """Plot class distributions for specified splits."""
        if splits_to_plot is None:
            splits_to_plot = list(self.splits_data.keys())
            
        num_splits = len(splits_to_plot)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, split_name in enumerate(splits_to_plot):
            if idx >= len(axes):
                break
                
            analysis = self.analyze_split_distribution(split_name)
            sorted_classes = analysis['sorted_classes']
            
            classes, counts = zip(*sorted_classes)
            
            ax = axes[idx]
            bars = ax.bar(range(len(classes)), counts, alpha=0.7)
            
            # Color coding: head=red, medium=orange, tail=blue
            head_size = len(analysis['head_classes'])
            med_size = len(analysis['medium_classes'])
            
            for i, bar in enumerate(bars):
                if i < head_size:
                    bar.set_color('red')
                elif i < head_size + med_size:
                    bar.set_color('orange') 
                else:
                    bar.set_color('blue')
            
            ax.set_title(f'{split_name.upper()}\nIF={analysis["imbalance_factor"]:.1f}, N={analysis["total_samples"]:,}')
            ax.set_xlabel('Classes (sorted by frequency)')
            ax.set_ylabel('Sample count')
            ax.set_yscale('log')
            
        # Remove empty subplots
        for idx in range(len(splits_to_plot), len(axes)):
            axes[idx].remove()
            
        plt.tight_layout()
        plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_proportion_comparison(self, reference_split: str = 'train',
                                 figsize: Tuple[int, int] = (15, 10)):
        """Plot proportion comparison between splits."""
        df = self.compare_proportions(reference_split)
        
        splits = df['split'].unique()
        num_splits = len(splits)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, split_name in enumerate(splits):
            if idx >= len(axes):
                break
                
            split_data = df[df['split'] == split_name]
            
            ax = axes[idx]
            
            # Scatter plot: reference vs split proportions
            ax.scatter(split_data['reference_prop'], split_data['split_prop'], 
                      alpha=0.6, s=30)
            
            # Perfect match line
            max_prop = max(split_data['reference_prop'].max(), 
                          split_data['split_prop'].max())
            ax.plot([0, max_prop], [0, max_prop], 'r--', alpha=0.8, label='Perfect match')
            
            ax.set_xlabel(f'{reference_split} proportion')
            ax.set_ylabel(f'{split_name} proportion')
            ax.set_title(f'{split_name.upper()} vs {reference_split.upper()}\nCorrelation: {split_data["reference_prop"].corr(split_data["split_prop"]):.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('proportion_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_summary_report(self, save_path: str = 'data_distribution_report.txt'):
        """Generate comprehensive text report."""
        print(f"\n📝 Generating summary report: {save_path}")
        
        all_analysis = self.analyze_all_splits()
        proportion_df = self.compare_proportions('train')
        
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CIFAR-100-LT DATA DISTRIBUTION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall summary
            f.write("OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Total splits analyzed: {len(all_analysis)}\n")
            f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Per-split analysis
            f.write("SPLIT-BY-SPLIT ANALYSIS\n")
            f.write("-"*40 + "\n")
            
            for split_name, analysis in all_analysis.items():
                f.write(f"\n{split_name.upper()}:\n")
                f.write(f"  Total samples: {analysis['total_samples']:,}\n")
                f.write(f"  Number of classes: {analysis['num_classes']}\n")
                f.write(f"  Imbalance Factor: {analysis['imbalance_factor']:.2f}\n")
                f.write(f"  Max count (head): {analysis['max_count']}\n")
                f.write(f"  Min count (tail): {analysis['min_count']}\n")
                f.write(f"  Head classes avg: {analysis['head_avg']:.1f}\n")
                f.write(f"  Medium classes avg: {analysis['medium_avg']:.1f}\n")
                f.write(f"  Tail classes avg: {analysis['tail_avg']:.1f}\n")
                
                # Top and bottom classes
                f.write(f"  Top 5 classes: {analysis['sorted_classes'][:5]}\n")
                f.write(f"  Bottom 5 classes: {analysis['sorted_classes'][-5:]}\n")
            
            # Proportion matching analysis
            f.write("\nPROPORTION MATCHING ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write("(Reference: train split)\n\n")
            
            for split_name in proportion_df['split'].unique():
                split_data = proportion_df[proportion_df['split'] == split_name]
                f.write(f"{split_name.upper()}:\n")
                f.write(f"  Average absolute difference: {split_data['abs_diff'].mean():.6f}\n")
                f.write(f"  Maximum absolute difference: {split_data['abs_diff'].max():.6f}\n")
                f.write(f"  Classes with >0.001 diff: {len(split_data[split_data['abs_diff'] > 0.001])}\n")
                f.write(f"  Correlation with train: {split_data['reference_prop'].corr(split_data['split_prop']):.6f}\n\n")
            
            # Data quality assessment
            f.write("DATA QUALITY ASSESSMENT\n")
            f.write("-"*40 + "\n")
            
            train_analysis = all_analysis.get('train', {})
            if train_analysis:
                f.write(f"✅ Train IF: {train_analysis['imbalance_factor']:.1f} (target: ~100)\n")
            
            for split_name in ['val_lt', 'test_lt', 'tuneV']:
                if split_name in proportion_df['split'].unique():
                    split_data = proportion_df[proportion_df['split'] == split_name]
                    avg_diff = split_data['abs_diff'].mean()
                    status = "✅" if avg_diff < 0.001 else "⚠️"
                    f.write(f"{status} {split_name} proportion match: {avg_diff:.6f} (target: <0.001)\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("Report completed successfully!\n")
        
        print(f"✅ Report saved to: {save_path}")
        
    def run_complete_analysis(self):
        """Run complete analysis pipeline."""
        print("🚀 Starting complete data distribution analysis...\n")
        
        # 1. Analyze all splits
        all_analysis = self.analyze_all_splits()
        
        # 2. Compare proportions
        proportion_df = self.compare_proportions('train')
        
        # 3. Generate plots
        print("\n📊 Generating distribution plots...")
        self.plot_class_distributions()
        
        print("\n📈 Generating proportion comparison plots...")
        self.plot_proportion_comparison('train')
        
        # 4. Generate summary report
        self.generate_summary_report()
        
        print("\n🎉 Complete analysis finished!")
        print("📁 Generated files:")
        print("   - data_distribution_analysis.png")
        print("   - proportion_comparison.png") 
        print("   - data_distribution_report.txt")
        
        return all_analysis, proportion_df


def main():
    """Main analysis function."""
    print("CIFAR-100-LT Data Distribution Analyzer")
    print("="*50)
    
    # Initialize analyzer
    analyzer = DataDistributionAnalyzer()
    
    # Run complete analysis
    all_analysis, proportion_df = analyzer.run_complete_analysis()
    
    return analyzer, all_analysis, proportion_df


if __name__ == "__main__":
    analyzer, all_analysis, proportion_df = main()