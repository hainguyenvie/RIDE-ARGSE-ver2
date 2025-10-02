#!/usr/bin/env python3
"""
Advanced Data Distribution Analysis with Visualizations
Creates comprehensive plots and analysis for CIFAR-100-LT splits.
"""

import json
import torchvision
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Configure matplotlib for better output
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class AdvancedDataAnalyzer:
    """Advanced analyzer with comprehensive visualizations."""
    
    def __init__(self):
        self.cifar_train = torchvision.datasets.CIFAR100(root='data', train=True, download=False)
        self.cifar_test = torchvision.datasets.CIFAR100(root='data', train=False, download=False)
        self.splits_dir = Path("data/cifar100_lt_if100_splits")
        self.analyses = {}
        
        # CIFAR-100 class names
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        
    def load_and_analyze(self):
        """Load all splits and perform analysis."""
        
        print("🔍 Loading and analyzing all splits...")
        
        split_configs = {
            'train': self.cifar_train,
            'val_lt': self.cifar_test, 
            'test_lt': self.cifar_test,
            'tuneV': self.cifar_test,
            'val_small': self.cifar_test,
            'calib': self.cifar_test
        }
        
        for split_name, source_dataset in split_configs.items():
            filename = f"{split_name}_indices.json"
            filepath = self.splits_dir / filename
            
            if not filepath.exists():
                print(f"⚠️  Missing: {filename}")
                continue
                
            with open(filepath, 'r') as f:
                indices = json.load(f)
                
            analysis = self._analyze_split(split_name, indices, source_dataset)
            if analysis:
                self.analyses[split_name] = analysis
                print(f"✅ Analyzed {split_name}: {analysis['total_samples']} samples, IF={analysis['imbalance_factor']:.1f}")
        
        return self.analyses
    
    def _analyze_split(self, split_name: str, indices: List[int], source_dataset) -> Optional[Dict]:
        """Analyze a single split."""
        
        if not indices:
            return None
            
        # Extract labels
        labels = []
        for idx in indices:
            try:
                _, label = source_dataset[idx]
                labels.append(label)
            except (IndexError, TypeError):
                continue
        
        if not labels:
            return None
            
        # Count classes
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        # Calculate statistics
        counts_list = list(class_counts.values())
        max_count = max(counts_list)
        min_count = min(counts_list)
        imbalance_factor = max_count / min_count if min_count > 0 else float('inf')
        
        # Calculate proportions
        proportions = {cls: count/total_samples for cls, count in class_counts.items()}
        
        # Sort classes by frequency
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'split_name': split_name,
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'imbalance_factor': imbalance_factor,
            'max_count': max_count,
            'min_count': min_count,
            'class_counts': dict(class_counts),
            'proportions': proportions,
            'sorted_classes': sorted_classes,
            'labels': labels
        }
    
    def plot_class_distributions(self, figsize: Tuple[int, int] = (20, 15)):
        """Plot class distributions for all splits."""
        
        if not self.analyses:
            self.load_and_analyze()
            
        n_splits = len(self.analyses)
        n_cols = 3
        n_rows = (n_splits + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for idx, (split_name, analysis) in enumerate(self.analyses.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]
            
            # Get sorted class counts
            sorted_classes = analysis['sorted_classes']
            classes, counts = zip(*sorted_classes)
            
            # Create bar plot
            bars = ax.bar(range(len(classes)), counts, 
                         color=colors[idx % len(colors)], alpha=0.7)
            
            # Highlight head, medium, tail with different colors
            n_classes = len(classes)
            head_size = n_classes // 3
            tail_start = 2 * n_classes // 3
            
            for i, bar in enumerate(bars):
                if i < head_size:
                    bar.set_color('#FF6B6B')  # Red for head
                elif i < tail_start:
                    bar.set_color('#FFA500')  # Orange for medium
                else:
                    bar.set_color('#4169E1')  # Blue for tail
            
            ax.set_title(f'{split_name.upper()}\\nSamples: {analysis["total_samples"]:,}, IF: {analysis["imbalance_factor"]:.1f}',
                        fontweight='bold')
            ax.set_xlabel('Classes (sorted by frequency)')
            ax.set_ylabel('Sample count')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Add text annotations for extremes
            if len(classes) > 0:
                ax.text(0, max(counts) * 0.8, f'Max: {max(counts)}', fontsize=8)
                ax.text(len(classes)-5, min(counts) * 2, f'Min: {min(counts)}', fontsize=8)
        
        # Remove empty subplots
        for idx in range(len(self.analyses), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].remove()
        
        plt.tight_layout()
        plt.savefig('advanced_class_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_imbalance_comparison(self, figsize: Tuple[int, int] = (16, 10)):
        """Plot imbalance factor comparison and curves."""
        
        if not self.analyses:
            self.load_and_analyze()
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Imbalance Factor Bar Chart
        splits = list(self.analyses.keys())
        ifs = [self.analyses[split]['imbalance_factor'] for split in splits]
        
        bars = ax1.bar(splits, ifs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        ax1.set_ylabel('Imbalance Factor')
        ax1.set_title('Imbalance Factor Comparison', fontweight='bold')
        ax1.set_ylim(0, max(ifs) * 1.1)
        
        # Add value labels on bars
        for bar, if_val in zip(bars, ifs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(ifs)*0.01,
                    f'{if_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # 2. Sample Count Comparison
        sample_counts = [self.analyses[split]['total_samples'] for split in splits]
        
        bars2 = ax2.bar(splits, sample_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        ax2.set_ylabel('Total Samples')
        ax2.set_title('Sample Count Comparison', fontweight='bold')
        
        # Add value labels
        for bar, count in zip(bars2, sample_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(sample_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        # 3. Long-tail curves (for main splits)
        main_splits = ['train', 'test_lt', 'val_lt']
        colors_curve = ['#FF6B6B', '#45B7D1', '#4ECDC4']
        
        for i, split_name in enumerate(main_splits):
            if split_name not in self.analyses:
                continue
                
            analysis = self.analyses[split_name]
            sorted_classes = analysis['sorted_classes']
            
            if sorted_classes:
                _, counts = zip(*sorted_classes)
                ranks = np.arange(1, len(counts) + 1)
                
                ax3.loglog(ranks, counts, 'o-', color=colors_curve[i], 
                          label=f'{split_name} (IF={analysis["imbalance_factor"]:.1f})',
                          markersize=4, linewidth=2)
        
        ax3.set_xlabel('Class Rank')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Long-tail Distribution Curves', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Class Coverage Comparison
        class_coverage = [self.analyses[split]['num_classes'] for split in splits]
        
        bars3 = ax4.bar(splits, class_coverage, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        ax4.set_ylabel('Number of Classes')
        ax4.set_title('Class Coverage Comparison', fontweight='bold')
        ax4.set_ylim(0, 100)
        
        # Add value labels
        for bar, coverage in zip(bars3, class_coverage):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{coverage}', ha='center', va='bottom', fontweight='bold')
        
        # Add reference line at 100
        ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='All classes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('imbalance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_proportion_matching(self, reference='train', figsize: Tuple[int, int] = (16, 12)):
        """Plot proportion matching analysis."""
        
        if not self.analyses:
            self.load_and_analyze()
            
        if reference not in self.analyses:
            print(f"Reference split '{reference}' not found!")
            return None
            
        ref_props = self.analyses[reference]['proportions']
        
        # Compare with other splits
        comparison_splits = [s for s in self.analyses.keys() if s != reference and s in ['val_lt', 'test_lt', 'tuneV']]
        
        if not comparison_splits:
            print("No comparison splits found!")
            return None
            
        n_splits = len(comparison_splits)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for idx, split_name in enumerate(comparison_splits):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            split_props = self.analyses[split_name]['proportions']
            
            # Calculate proportion differences
            classes = list(range(100))
            ref_values = [ref_props.get(c, 0) for c in classes]
            split_values = [split_props.get(c, 0) for c in classes]
            
            # Scatter plot
            ax.scatter(ref_values, split_values, alpha=0.6, s=30, color=colors[idx])
            
            # Perfect match line
            max_prop = max(max(ref_values), max(split_values))
            ax.plot([0, max_prop], [0, max_prop], 'r--', alpha=0.8, label='Perfect match')
            
            # Calculate correlation
            if ref_values and split_values:
                correlation = np.corrcoef(ref_values, split_values)[0, 1]
            else:
                correlation = 0
                
            ax.set_xlabel(f'{reference} proportion')
            ax.set_ylabel(f'{split_name} proportion') 
            ax.set_title(f'{split_name.upper()} vs {reference.upper()}\\nCorrelation: {correlation:.4f}',
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            abs_diffs = [abs(r - s) for r, s in zip(ref_values, split_values)]
            avg_diff = np.mean(abs_diffs)
            max_diff = max(abs_diffs)
            
            ax.text(0.02, 0.98, f'Avg diff: {avg_diff:.6f}\\nMax diff: {max_diff:.6f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove unused subplot
        if len(comparison_splits) < len(axes):
            for idx in range(len(comparison_splits), len(axes)):
                axes[idx].remove()
        
        plt.tight_layout()
        plt.savefig('proportion_matching.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_summary_dashboard(self, figsize: Tuple[int, int] = (20, 16)):
        """Create comprehensive summary dashboard."""
        
        if not self.analyses:
            self.load_and_analyze()
            
        fig = plt.figure(figsize=figsize)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Class distributions (top row)
        for idx, (split_name, analysis) in enumerate(list(self.analyses.items())[:3]):
            ax = fig.add_subplot(gs[0, idx])
            
            sorted_classes = analysis['sorted_classes']
            if sorted_classes:
                classes, counts = zip(*sorted_classes)
                
                bars = ax.bar(range(len(classes)), counts, alpha=0.7)
                
                # Color coding
                n_classes = len(classes)
                head_size = n_classes // 3
                tail_start = 2 * n_classes // 3
                
                for i, bar in enumerate(bars):
                    if i < head_size:
                        bar.set_color('#FF6B6B')
                    elif i < tail_start:
                        bar.set_color('#FFA500')
                    else:
                        bar.set_color('#4169E1')
                
                ax.set_title(f'{split_name.upper()}\\n{analysis["total_samples"]:,} samples',
                           fontsize=10, fontweight='bold')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
        
        # 2. Imbalance factors (middle left)
        ax_if = fig.add_subplot(gs[1, :2])
        splits = list(self.analyses.keys())
        ifs = [self.analyses[split]['imbalance_factor'] for split in splits]
        
        bars = ax_if.bar(splits, ifs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        ax_if.set_ylabel('Imbalance Factor')
        ax_if.set_title('Imbalance Factor Comparison', fontweight='bold')
        
        for bar, if_val in zip(bars, ifs):
            height = bar.get_height()
            ax_if.text(bar.get_x() + bar.get_width()/2., height + max(ifs)*0.01,
                      f'{if_val:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax_if.grid(True, alpha=0.3)
        
        # 3. Sample counts (middle right)
        ax_samples = fig.add_subplot(gs[1, 2:])
        sample_counts = [self.analyses[split]['total_samples'] for split in splits]
        
        bars = ax_samples.bar(splits, sample_counts, 
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        ax_samples.set_ylabel('Total Samples')
        ax_samples.set_title('Sample Count Comparison', fontweight='bold')
        
        for bar, count in zip(bars, sample_counts):
            height = bar.get_height()
            ax_samples.text(bar.get_x() + bar.get_width()/2., height + max(sample_counts)*0.01,
                          f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax_samples.grid(True, alpha=0.3)
        
        # 4. Long-tail curves (bottom left)
        ax_curves = fig.add_subplot(gs[2:, :2])
        main_splits = ['train', 'test_lt', 'val_lt'] 
        colors_curve = ['#FF6B6B', '#45B7D1', '#4ECDC4']
        
        for i, split_name in enumerate(main_splits):
            if split_name not in self.analyses:
                continue
                
            analysis = self.analyses[split_name]
            sorted_classes = analysis['sorted_classes']
            
            if sorted_classes:
                _, counts = zip(*sorted_classes)
                ranks = np.arange(1, len(counts) + 1)
                
                ax_curves.loglog(ranks, counts, 'o-', color=colors_curve[i],
                               label=f'{split_name} (IF={analysis["imbalance_factor"]:.1f})',
                               markersize=3, linewidth=2)
        
        ax_curves.set_xlabel('Class Rank')
        ax_curves.set_ylabel('Sample Count')
        ax_curves.set_title('Long-tail Distribution Curves', fontweight='bold')
        ax_curves.legend()
        ax_curves.grid(True, alpha=0.3)
        
        # 5. Quality summary (bottom right)
        ax_quality = fig.add_subplot(gs[2:, 2:])
        ax_quality.axis('off')
        
        # Create quality summary text
        quality_text = "DATA QUALITY SUMMARY\\n" + "="*40 + "\\n\\n"
        
        if 'train' in self.analyses:
            train_if = self.analyses['train']['imbalance_factor']
            quality_text += f"✅ Train: IF={train_if:.1f} (target ~100)\\n"
        
        for split_name in ['val_lt', 'test_lt', 'tuneV']:
            if split_name in self.analyses:
                analysis = self.analyses[split_name]
                samples = analysis['total_samples']
                classes = analysis['num_classes']
                
                # Quality indicators
                quality_mark = "✅" if samples > 100 and classes >= 80 else "⚠️"
                quality_text += f"{quality_mark} {split_name}: {samples} samples, {classes}/100 classes\\n"
        
        quality_text += "\\n" + "ISSUES DETECTED:\\n" + "-"*20 + "\\n"
        
        # Check for issues
        issues = []
        if 'tuneV' in self.analyses and self.analyses['tuneV']['total_samples'] < 500:
            issues.append("• TuneV split too small for reliable training")
        if 'val_small' in self.analyses and self.analyses['val_small']['num_classes'] < 90:
            issues.append("• Val_small missing many classes")
        if 'calib' in self.analyses and self.analyses['calib']['total_samples'] < 100:
            issues.append("• Calibration set very small")
            
        if issues:
            quality_text += "\\n".join(issues)
        else:
            quality_text += "• No critical issues detected"
        
        ax_quality.text(0, 0.9, quality_text, transform=ax_quality.transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('CIFAR-100-LT Data Distribution Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('data_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_complete_analysis(self):
        """Run complete analysis with all visualizations."""
        
        print("🚀 Starting Advanced Data Distribution Analysis...")
        print("="*60)
        
        # Load and analyze
        self.load_and_analyze()
        
        print(f"\\n📊 Generated {len(self.analyses)} split analyses")
        
        # Create visualizations
        print("\\n🎨 Generating visualizations...")
        
        print("  1. Class distribution plots...")
        self.plot_class_distributions()
        
        print("  2. Imbalance analysis...")
        self.plot_imbalance_comparison()
        
        print("  3. Proportion matching analysis...")
        self.plot_proportion_matching()
        
        print("  4. Summary dashboard...")
        self.create_summary_dashboard()
        
        print("\\n🎉 Complete analysis finished!")
        print("📁 Generated visualization files:")
        print("   - advanced_class_distributions.png")
        print("   - imbalance_analysis.png")
        print("   - proportion_matching.png")
        print("   - data_analysis_dashboard.png")
        
        return self.analyses


def main():
    """Main execution."""
    analyzer = AdvancedDataAnalyzer()
    analyses = analyzer.run_complete_analysis()
    return analyzer, analyses


if __name__ == "__main__":
    analyzer, analyses = main()