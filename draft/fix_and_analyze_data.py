#!/usr/bin/env python3
"""
Fix invalid split indices and create comprehensive data analysis.
"""

import json
import torchvision
from pathlib import Path
from collections import Counter
import numpy as np

def fix_split_indices():
    """Fix invalid indices in split files."""
    
    print("🔧 Fixing invalid split indices...")
    
    # Load datasets to understand size limits  
    cifar_train = torchvision.datasets.CIFAR100(root='data', train=True, download=False)
    cifar_test = torchvision.datasets.CIFAR100(root='data', train=False, download=False)
    
    train_size = len(cifar_train)  # 50000
    test_size = len(cifar_test)    # 10000
    
    print(f"📏 Dataset sizes: Train={train_size}, Test={test_size}")
    
    splits_dir = Path("data/cifar100_lt_if100_splits")
    
    # Define which splits should use which dataset
    split_sources = {
        'train': ('train', train_size - 1),
        'val_lt': ('test', test_size - 1), 
        'test_lt': ('test', test_size - 1),
        'tuneV': ('test', test_size - 1),      # Should be from test!
        'val_small': ('test', test_size - 1),  # Should be from test!
        'calib': ('test', test_size - 1)       # Should be from test!
    }
    
    fixed_files = []
    
    for split_name, (source, max_valid) in split_sources.items():
        filename = f"{split_name}_indices.json"
        filepath = splits_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  Missing: {filename}")
            continue
            
        # Load current indices
        with open(filepath, 'r') as f:
            indices = json.load(f)
            
        # Filter valid indices
        valid_indices = [i for i in indices if i <= max_valid]
        invalid_count = len(indices) - len(valid_indices)
        
        print(f"\n📊 {split_name}:")
        print(f"   Original: {len(indices)} indices")
        print(f"   Valid: {len(valid_indices)} indices") 
        print(f"   Invalid: {invalid_count} indices")
        print(f"   Expected source: {source} dataset (max_idx: {max_valid})")
        
        if invalid_count > 0:
            # Backup original
            backup_path = splits_dir / f"{split_name}_indices_original.json"
            with open(backup_path, 'w') as f:
                json.dump(indices, f)
            print(f"   💾 Backed up original to: {backup_path}")
            
            # Save fixed version
            with open(filepath, 'w') as f:
                json.dump(valid_indices, f)
            print(f"   ✅ Fixed and saved: {filepath}")
            fixed_files.append(split_name)
        else:
            print(f"   ✅ Already valid")
    
    return fixed_files


def analyze_class_distribution(split_name, indices, source_dataset):
    """Analyze class distribution for a split."""
    
    if not indices:
        return None
        
    # Extract labels
    labels = []
    for idx in indices:
        try:
            _, label = source_dataset[idx]
            labels.append(label)
        except IndexError:
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
    
    # Group statistics (head/medium/tail by sample count)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    n_classes = len(sorted_classes)
    
    head_size = n_classes // 3
    tail_start = 2 * n_classes // 3
    
    head_classes = sorted_classes[:head_size]
    medium_classes = sorted_classes[head_size:tail_start]
    tail_classes = sorted_classes[tail_start:]
    
    return {
        'split_name': split_name,
        'total_samples': total_samples,
        'num_classes': n_classes,
        'imbalance_factor': imbalance_factor,
        'max_count': max_count,
        'min_count': min_count,
        'class_counts': dict(class_counts),
        'proportions': proportions,
        'sorted_classes': sorted_classes,
        'head_classes': head_classes,
        'medium_classes': medium_classes,
        'tail_classes': tail_classes,
        'head_avg': np.mean([count for _, count in head_classes]) if head_classes else 0,
        'medium_avg': np.mean([count for _, count in medium_classes]) if medium_classes else 0,
        'tail_avg': np.mean([count for _, count in tail_classes]) if tail_classes else 0,
        'head_total': sum(count for _, count in head_classes),
        'medium_total': sum(count for _, count in medium_classes), 
        'tail_total': sum(count for _, count in tail_classes)
    }


def create_comprehensive_report():
    """Create comprehensive data distribution report."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE DATA DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Load datasets
    cifar_train = torchvision.datasets.CIFAR100(root='data', train=True, download=False)
    cifar_test = torchvision.datasets.CIFAR100(root='data', train=False, download=False)
    
    splits_dir = Path("data/cifar100_lt_if100_splits")
    
    # Analyze all splits
    analyses = {}
    split_configs = {
        'train': cifar_train,
        'val_lt': cifar_test, 
        'test_lt': cifar_test,
        'tuneV': cifar_test,
        'val_small': cifar_test,
        'calib': cifar_test
    }
    
    for split_name, source_dataset in split_configs.items():
        filename = f"{split_name}_indices.json"
        filepath = splits_dir / filename
        
        if not filepath.exists():
            continue
            
        with open(filepath, 'r') as f:
            indices = json.load(f)
            
        analysis = analyze_class_distribution(split_name, indices, source_dataset)
        if analysis:
            analyses[split_name] = analysis
    
    # Print detailed analysis
    for split_name, analysis in analyses.items():
        print(f"\n{'='*50}")
        print(f"📊 {split_name.upper()} SPLIT ANALYSIS")
        print(f"{'='*50}")
        print(f"Total samples: {analysis['total_samples']:,}")
        print(f"Classes represented: {analysis['num_classes']}/100")
        print(f"Imbalance Factor: {analysis['imbalance_factor']:.2f}")
        print(f"Max/Min counts: {analysis['max_count']}/{analysis['min_count']}")
        
        print(f"\nGroup Statistics:")
        print(f"  Head classes ({len(analysis['head_classes'])}): avg={analysis['head_avg']:.1f}, total={analysis['head_total']}")
        print(f"  Medium classes ({len(analysis['medium_classes'])}): avg={analysis['medium_avg']:.1f}, total={analysis['medium_total']}")  
        print(f"  Tail classes ({len(analysis['tail_classes'])}): avg={analysis['tail_avg']:.1f}, total={analysis['tail_total']}")
        
        print(f"\nClass Distribution:")
        print(f"  Most frequent: {analysis['sorted_classes'][:5]}")
        print(f"  Least frequent: {analysis['sorted_classes'][-5:]}")
        
        # Sample proportions
        head_prop = analysis['head_total'] / analysis['total_samples']
        medium_prop = analysis['medium_total'] / analysis['total_samples']
        tail_prop = analysis['tail_total'] / analysis['total_samples']
        
        print(f"\nSample Proportions:")
        print(f"  Head: {head_prop:.1%} ({analysis['head_total']} samples)")
        print(f"  Medium: {medium_prop:.1%} ({analysis['medium_total']} samples)")
        print(f"  Tail: {tail_prop:.1%} ({analysis['tail_total']} samples)")
    
    # Compare proportions between train and test splits
    if 'train' in analyses:
        train_props = analyses['train']['proportions']
        
        print(f"\n{'='*50}")
        print("PROPORTION MATCHING ANALYSIS")
        print(f"{'='*50}")
        print("(Comparing with train split as reference)")
        
        for split_name in ['val_lt', 'test_lt', 'tuneV']:
            if split_name not in analyses:
                continue
                
            split_props = analyses[split_name]['proportions']
            
            # Calculate differences
            diffs = []
            for class_id in range(100):
                train_p = train_props.get(class_id, 0)
                split_p = split_props.get(class_id, 0)
                diff = abs(train_p - split_p)
                diffs.append(diff)
            
            avg_diff = np.mean(diffs)
            max_diff = max(diffs)
            
            # Classes with significant differences
            sig_diffs = len([d for d in diffs if d > 0.001])
            
            print(f"\n{split_name.upper()}:")
            print(f"  Average absolute difference: {avg_diff:.6f}")
            print(f"  Maximum absolute difference: {max_diff:.6f}")
            print(f"  Classes with >0.1% difference: {sig_diffs}/100")
            
            # Quality assessment
            if avg_diff < 0.001:
                quality = "✅ Excellent"
            elif avg_diff < 0.005:
                quality = "🟡 Good"
            else:
                quality = "❌ Poor"
            print(f"  Proportion matching quality: {quality}")
    
    # Data quality summary
    print(f"\n{'='*50}")
    print("DATA QUALITY SUMMARY")
    print(f"{'='*50}")
    
    if 'train' in analyses:
        train_if = analyses['train']['imbalance_factor']
        train_classes = analyses['train']['num_classes']
        print(f"✅ Train split: IF={train_if:.1f} (target ~100), Classes={train_classes}/100")
    
    for split_name in ['val_lt', 'test_lt', 'tuneV', 'val_small', 'calib']:
        if split_name not in analyses:
            print(f"❌ Missing split: {split_name}")
            continue
            
        analysis = analyses[split_name]
        samples = analysis['total_samples']
        classes = analysis['num_classes']
        
        # Quality indicators
        sufficient_samples = "✅" if samples > 100 else "⚠️"
        sufficient_classes = "✅" if classes >= 50 else "⚠️"
        
        print(f"{sufficient_samples}{sufficient_classes} {split_name}: {samples} samples, {classes}/100 classes")
    
    print(f"\n🎉 Analysis completed successfully!")
    
    # Save detailed report
    save_detailed_report(analyses)
    
    return analyses


def save_detailed_report(analyses):
    """Save detailed text report."""
    
    with open('fixed_data_distribution_report.txt', 'w') as f:
        f.write("CIFAR-100-LT FIXED DATA DISTRIBUTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Split':<12} {'Samples':<8} {'Classes':<8} {'IF':<8} {'Max':<6} {'Min':<6}\n")
        f.write("-" * 60 + "\n")
        
        for split_name, analysis in analyses.items():
            f.write(f"{split_name:<12} {analysis['total_samples']:<8} "
                   f"{analysis['num_classes']:<8} {analysis['imbalance_factor']:<8.1f} "
                   f"{analysis['max_count']:<6} {analysis['min_count']:<6}\n")
        
        f.write("\n\nDETAILED ANALYSIS\n")
        f.write("=" * 40 + "\n")
        
        for split_name, analysis in analyses.items():
            f.write(f"\n{split_name.upper()}:\n")
            f.write(f"  Total samples: {analysis['total_samples']:,}\n")
            f.write(f"  Classes: {analysis['num_classes']}/100\n")
            f.write(f"  IF: {analysis['imbalance_factor']:.2f}\n")
            f.write(f"  Head avg: {analysis['head_avg']:.1f}\n")
            f.write(f"  Medium avg: {analysis['medium_avg']:.1f}\n")
            f.write(f"  Tail avg: {analysis['tail_avg']:.1f}\n")
            f.write(f"  Top classes: {analysis['sorted_classes'][:5]}\n")
            f.write(f"  Bottom classes: {analysis['sorted_classes'][-5:]}\n")
    
    print("💾 Detailed report saved to: fixed_data_distribution_report.txt")


def main():
    """Main execution function."""
    
    print("CIFAR-100-LT Data Distribution Fixer & Analyzer")
    print("=" * 65)
    
    # Step 1: Fix invalid indices
    fixed_files = fix_split_indices()
    
    if fixed_files:
        print(f"\n✅ Fixed {len(fixed_files)} split files: {fixed_files}")
    else:
        print(f"\n✅ All split files were already valid")
    
    # Step 2: Comprehensive analysis
    analyses = create_comprehensive_report()
    
    return analyses


if __name__ == "__main__":
    analyses = main()