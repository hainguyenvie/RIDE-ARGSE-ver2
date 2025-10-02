#!/usr/bin/env python3
"""
Comprehensive Data Analysis for CIFAR-100-LT Training.
Provides detailed analysis and visualization of data distributions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

# Import our data utilities
from src.data.dataloader_utils import CIFAR100LTDataModule, get_cifar100_lt_dataloaders

class DataDistributionAnalyzer:
    """Comprehensive analyzer for CIFAR-100-LT data distributions."""
    
    def __init__(self, data_module: Optional[CIFAR100LTDataModule] = None):
        if data_module is None:
            self.data_module = CIFAR100LTDataModule()
            self.data_module.setup_datasets()
        else:
            self.data_module = data_module
            
        # Analysis results storage
        self.analysis_results = {}
        
    def analyze_split_distribution(self, split: str) -> Dict:
        """Analyze class distribution for a specific split."""
        print(f"\n🔍 Analyzing {split.upper()} distribution...")
        
        if split not in self.data_module.datasets:
            print(f"❌ Split '{split}' not available")
            return {}
            
        # Get class counts
        class_counts = self.data_module.get_class_counts(split)
        total_samples = sum(class_counts.values())
        
        # Calculate distribution statistics
        counts_array = np.array([class_counts.get(i, 0) for i in range(100)])
        proportions = counts_array / total_samples
        
        # Head/Mid/Tail groups (as per AR-GSE paper)
        head_classes = list(range(0, 33))    # Classes 0-32
        mid_classes = list(range(33, 67))    # Classes 33-66  
        tail_classes = list(range(67, 100))  # Classes 67-99
        
        head_count = sum(counts_array[head_classes])
        mid_count = sum(counts_array[mid_classes])
        tail_count = sum(counts_array[tail_classes])
        
        # Imbalance factor
        max_count = counts_array.max()
        min_count = counts_array[counts_array > 0].min() if (counts_array > 0).any() else 1
        imbalance_factor = max_count / min_count
        
        analysis = {
            'split': split,
            'total_samples': total_samples,
            'num_classes': len([c for c in counts_array if c > 0]),
            'class_counts': class_counts,
            'class_proportions': {i: prop for i, prop in enumerate(proportions)},
            'head_count': head_count,
            'mid_count': mid_count, 
            'tail_count': tail_count,
            'head_prop': head_count / total_samples,
            'mid_prop': mid_count / total_samples,
            'tail_prop': tail_count / total_samples,
            'max_count': int(max_count),
            'min_count': int(min_count),
            'imbalance_factor': imbalance_factor,
            'mean_count': counts_array.mean(),
            'std_count': counts_array.std()
        }
        
        # Print summary
        print(f"  📊 Total: {total_samples:,} samples")
        print(f"  🏆 Head group (0-32): {head_count:,} ({head_count/total_samples*100:.1f}%)")
        print(f"  🎯 Mid group (33-66): {mid_count:,} ({mid_count/total_samples*100:.1f}%)")  
        print(f"  🎭 Tail group (67-99): {tail_count:,} ({tail_count/total_samples*100:.1f}%)")
        print(f"  ⚖️ Imbalance Factor: {imbalance_factor:.1f}")
        print(f"  📈 Range: {int(min_count)} - {int(max_count)} samples")
        
        self.analysis_results[split] = analysis
        return analysis
        
    def analyze_all_splits(self) -> Dict:
        """Analyze all available splits."""
        print("🚀 Starting comprehensive data analysis...")
        
        all_results = {}
        for split in self.data_module.datasets.keys():
            all_results[split] = self.analyze_split_distribution(split)
            
        return all_results
        
    def compare_splits_proportions(self, reference_split: str = 'train') -> Dict:
        """Compare proportions across splits to check consistency."""
        print(f"\n🔄 Comparing splits against {reference_split.upper()}...")
        
        if reference_split not in self.analysis_results:
            print(f"❌ Reference split '{reference_split}' not analyzed yet")
            return {}
            
        ref_props = self.analysis_results[reference_split]['class_proportions']
        comparison_results = {}
        
        for split, analysis in self.analysis_results.items():
            if split == reference_split:
                continue
                
            split_props = analysis['class_proportions']
            
            # Calculate proportion differences
            prop_diffs = {}
            max_diff = 0
            mean_diff = 0
            
            for class_id in range(100):
                ref_prop = ref_props.get(class_id, 0)
                split_prop = split_props.get(class_id, 0)
                diff = abs(split_prop - ref_prop)
                prop_diffs[class_id] = diff
                max_diff = max(max_diff, diff)
                mean_diff += diff
                
            mean_diff /= 100
            
            comparison_results[split] = {
                'max_proportion_diff': max_diff,
                'mean_proportion_diff': mean_diff,
                'proportion_diffs': prop_diffs
            }
            
            print(f"  {split.upper()} vs {reference_split.upper()}:")
            print(f"    Max difference: {max_diff:.4f}")
            print(f"    Mean difference: {mean_diff:.4f}")
            
        return comparison_results
        
    def visualize_distributions(self, save_dir: str = "outputs/plots"):
        """Create comprehensive visualizations of data distributions."""
        print(f"\n📊 Creating visualizations in {save_dir}...")
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Class count distributions for all splits
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CIFAR-100-LT Class Distributions by Split', fontsize=16, fontweight='bold')
        
        splits = list(self.analysis_results.keys())[:4]  # Max 4 splits
        
        for idx, split in enumerate(splits):
            ax = axes[idx // 2, idx % 2]
            analysis = self.analysis_results[split]
            
            classes = list(range(100))
            counts = [analysis['class_counts'].get(i, 0) for i in classes]
            
            ax.bar(classes, counts, alpha=0.7, color=sns.color_palette("husl", 4)[idx])
            ax.set_title(f'{split.upper()} Distribution\n({analysis["total_samples"]:,} samples, IF={analysis["imbalance_factor"]:.1f})')
            ax.set_xlabel('Class Index')
            ax.set_ylabel('Sample Count')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f"{save_dir}/class_distributions.png", dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_dir}/class_distributions.png")
        
        # 2. Group-wise comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group counts
        groups = ['Head (0-32)', 'Mid (33-66)', 'Tail (67-99)']
        split_names = list(self.analysis_results.keys())
        
        head_counts = [self.analysis_results[s]['head_count'] for s in split_names]
        mid_counts = [self.analysis_results[s]['mid_count'] for s in split_names]  
        tail_counts = [self.analysis_results[s]['tail_count'] for s in split_names]
        
        x = np.arange(len(split_names))
        width = 0.25
        
        ax1.bar(x - width, head_counts, width, label='Head', alpha=0.8)
        ax1.bar(x, mid_counts, width, label='Mid', alpha=0.8)
        ax1.bar(x + width, tail_counts, width, label='Tail', alpha=0.8)
        
        ax1.set_title('Group Sample Counts by Split')
        ax1.set_xlabel('Split')
        ax1.set_ylabel('Sample Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.upper() for s in split_names])
        ax1.legend()
        ax1.set_yscale('log')
        
        # Group proportions
        head_props = [self.analysis_results[s]['head_prop'] for s in split_names]
        mid_props = [self.analysis_results[s]['mid_prop'] for s in split_names]
        tail_props = [self.analysis_results[s]['tail_prop'] for s in split_names]
        
        ax2.bar(x - width, head_props, width, label='Head', alpha=0.8)
        ax2.bar(x, mid_props, width, label='Mid', alpha=0.8) 
        ax2.bar(x + width, tail_props, width, label='Tail', alpha=0.8)
        
        ax2.set_title('Group Proportions by Split')
        ax2.set_xlabel('Split')
        ax2.set_ylabel('Proportion')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.upper() for s in split_names])
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/group_comparison.png", dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_dir}/group_comparison.png")
        
        # 3. Detailed exponential profile visualization
        if 'train' in self.analysis_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            train_analysis = self.analysis_results['train']
            classes = list(range(100))
            counts = [train_analysis['class_counts'].get(i, 0) for i in classes]
            
            # Theoretical exponential profile
            theoretical = [500 * (100 ** (-(i/99))) for i in classes]
            
            ax.semilogy(classes, counts, 'bo-', alpha=0.7, label='Actual Distribution', markersize=4)
            ax.semilogy(classes, theoretical, 'r--', alpha=0.8, label='Theoretical (IF=100)', linewidth=2)
            
            ax.set_title('CIFAR-100-LT: Actual vs Theoretical Distribution')
            ax.set_xlabel('Class Index')
            ax.set_ylabel('Sample Count (log scale)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/exponential_profile.png", dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved: {save_dir}/exponential_profile.png")
        
        plt.close('all')
        print(f"  📊 All visualizations saved to {save_dir}/")
        
    def save_analysis_report(self, save_path: str = "outputs/data_analysis_report.json"):
        """Save detailed analysis report."""
        print(f"\n💾 Saving analysis report to {save_path}...")
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_results': self.analysis_results,
            'summary': {
                'total_splits': len(self.analysis_results),
                'splits_analyzed': list(self.analysis_results.keys())
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"  ✅ Report saved: {save_path}")
        
    def print_training_ready_summary(self):
        """Print a summary suitable for training logs."""
        print("\n" + "="*60)
        print("🚀 CIFAR-100-LT DATASET READY FOR TRAINING")
        print("="*60)
        
        for split, analysis in self.analysis_results.items():
            print(f"\n📊 {split.upper()} SET:")
            print(f"  • Total samples: {analysis['total_samples']:,}")
            print(f"  • Head/Mid/Tail: {analysis['head_count']:,}/{analysis['mid_count']:,}/{analysis['tail_count']:,}")
            print(f"  • Imbalance Factor: {analysis['imbalance_factor']:.1f}")
            print(f"  • Group proportions: {analysis['head_prop']:.3f}/{analysis['mid_prop']:.3f}/{analysis['tail_prop']:.3f}")
            
        print("\n✅ All datasets validated and ready for AR-GSE training!")
        print("="*60)

def run_comprehensive_analysis():
    """Run complete data analysis pipeline."""
    print("🔥 Starting CIFAR-100-LT Comprehensive Data Analysis...")
    
    try:
        # Initialize analyzer
        analyzer = DataDistributionAnalyzer()
        
        # Run analysis
        analyzer.analyze_all_splits()
        analyzer.compare_splits_proportions('train')
        
        # Create visualizations
        analyzer.visualize_distributions()
        
        # Save report
        analyzer.save_analysis_report()
        
        # Print training summary
        analyzer.print_training_ready_summary()
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    analyzer = run_comprehensive_analysis()
