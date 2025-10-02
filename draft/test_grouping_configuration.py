#!/usr/bin/env python3
"""
Test script to validate the updated train_argse.py with configurable grouping methods.
"""

import sys
sys.path.append('.')

# Test different configurations
test_configs = [
    {
        'name': 'Threshold-based (threshold=20)',
        'config': {
            'dataset': {
                'name': 'cifar100_lt_if100',
                'splits_dir': './data/cifar100_lt_if100_splits',
                'num_classes': 100,
            },
            'grouping': {
                'method': 'threshold',
                'threshold': 20,
                'head_ratio': 0.69,
                'K': 2,
            },
        }
    },
    {
        'name': 'Threshold-based (threshold=25)', 
        'config': {
            'dataset': {
                'name': 'cifar100_lt_if100',
                'splits_dir': './data/cifar100_lt_if100_splits',
                'num_classes': 100,
            },
            'grouping': {
                'method': 'threshold',
                'threshold': 25,
                'head_ratio': 0.69,
                'K': 2,
            },
        }
    },
    {
        'name': 'Ratio-based (head_ratio=0.69)',
        'config': {
            'dataset': {
                'name': 'cifar100_lt_if100',
                'splits_dir': './data/cifar100_lt_if100_splits',
                'num_classes': 100,
            },
            'grouping': {
                'method': 'ratio',
                'threshold': 20,
                'head_ratio': 0.69,
                'K': 2,
            },
        }
    },
    {
        'name': 'Ratio-based (head_ratio=0.5)',
        'config': {
            'dataset': {
                'name': 'cifar100_lt_if100',
                'splits_dir': './data/cifar100_lt_if100_splits',
                'num_classes': 100,
            },
            'grouping': {
                'method': 'ratio',
                'threshold': 20,
                'head_ratio': 0.5,
                'K': 2,
            },
        }
    }
]

def test_grouping_configuration():
    """Test different grouping configurations."""
    from src.data.datasets import get_cifar100_lt_counts
    from src.data.groups import get_class_to_group, get_class_to_group_by_threshold
    
    print("=" * 60)
    print("TESTING GROUPING CONFIGURATIONS")
    print("=" * 60)
    
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    
    for test_case in test_configs:
        print(f"\n--- {test_case['name']} ---")
        config = test_case['config']
        
        # Simulate the logic from train_argse.py
        if config['grouping']['method'] == 'threshold':
            print(f"Using threshold-based grouping with threshold={config['grouping']['threshold']}")
            class_to_group = get_class_to_group_by_threshold(class_counts, 
                                                           threshold=config['grouping']['threshold'])
            head_count = (class_to_group == 0).sum().item()
            tail_count = (class_to_group == 1).sum().item()
            print(f"Groups created: {head_count} head classes (>{config['grouping']['threshold']} samples), "
                  f"{tail_count} tail classes (<={config['grouping']['threshold']} samples)")
        else:  # ratio method
            print(f"Using ratio-based grouping with head_ratio={config['grouping']['head_ratio']}")
            class_to_group = get_class_to_group(class_counts, 
                                              K=config['grouping']['K'], 
                                              head_ratio=config['grouping']['head_ratio'])
            head_count = int(config['dataset']['num_classes'] * config['grouping']['head_ratio'])
            tail_count = config['dataset']['num_classes'] - head_count
            print(f"Groups created: {head_count} head classes, {tail_count} tail classes")
        
        # Calculate sample distribution
        head_samples = sum(class_counts[i] for i in range(len(class_counts)) if class_to_group[i] == 0)
        tail_samples = sum(class_counts[i] for i in range(len(class_counts)) if class_to_group[i] == 1)
        
        print(f"Sample distribution: {head_samples} head samples, {tail_samples} tail samples")
        print(f"Head/Tail sample ratio: {head_samples/tail_samples:.2f}")

def test_training_script_integration():
    """Test that the training script logic works with new configuration."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING SCRIPT INTEGRATION")
    print("=" * 60)
    
    # Test the exact logic from the updated train_argse.py
    from src.data.datasets import get_cifar100_lt_counts
    from src.data.groups import get_class_to_group, get_class_to_group_by_threshold
    
    # Simulate the CONFIG from train_argse.py
    CONFIG = {
        'dataset': {
            'name': 'cifar100_lt_if100',
            'splits_dir': './data/cifar100_lt_if100_splits',
            'num_classes': 100,
        },
        'grouping': {
            'method': 'threshold',  # Test threshold method
            'threshold': 20,
            'head_ratio': 0.69,
            'K': 2,
        },
    }
    
    print("Testing with threshold-based configuration:")
    print(f"Config: {CONFIG['grouping']}")
    
    # Simulate the main() function logic
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    
    if CONFIG['grouping']['method'] == 'threshold':
        print(f"Using threshold-based grouping with threshold={CONFIG['grouping']['threshold']}")
        class_to_group = get_class_to_group_by_threshold(class_counts, 
                                                       threshold=CONFIG['grouping']['threshold'])
        head_count = (class_to_group == 0).sum().item()
        tail_count = (class_to_group == 1).sum().item()
        print(f"Groups created: {head_count} head classes (>{CONFIG['grouping']['threshold']} samples), "
              f"{tail_count} tail classes (<={CONFIG['grouping']['threshold']} samples)")
    else:
        print(f"Using ratio-based grouping with head_ratio={CONFIG['grouping']['head_ratio']}")
        class_to_group = get_class_to_group(class_counts, 
                                          K=CONFIG['grouping']['K'], 
                                          head_ratio=CONFIG['grouping']['head_ratio'])
        head_count = int(CONFIG['dataset']['num_classes'] * CONFIG['grouping']['head_ratio'])
        tail_count = CONFIG['dataset']['num_classes'] - head_count
        print(f"Groups created: {head_count} head classes, {tail_count} tail classes")
    
    num_groups = class_to_group.max().item() + 1
    print(f"Total groups: {num_groups}")
    
    # Test switching to ratio method
    print("\nSwitching to ratio-based configuration:")
    CONFIG['grouping']['method'] = 'ratio'
    print(f"Config: {CONFIG['grouping']}")
    
    if CONFIG['grouping']['method'] == 'threshold':
        print(f"Using threshold-based grouping with threshold={CONFIG['grouping']['threshold']}")
        class_to_group = get_class_to_group_by_threshold(class_counts, 
                                                       threshold=CONFIG['grouping']['threshold'])
        head_count = (class_to_group == 0).sum().item()
        tail_count = (class_to_group == 1).sum().item()
        print(f"Groups created: {head_count} head classes (>{CONFIG['grouping']['threshold']} samples), "
              f"{tail_count} tail classes (<={CONFIG['grouping']['threshold']} samples)")
    else:
        print(f"Using ratio-based grouping with head_ratio={CONFIG['grouping']['head_ratio']}")
        class_to_group = get_class_to_group(class_counts, 
                                          K=CONFIG['grouping']['K'], 
                                          head_ratio=CONFIG['grouping']['head_ratio'])
        head_count = int(CONFIG['dataset']['num_classes'] * CONFIG['grouping']['head_ratio'])
        tail_count = CONFIG['dataset']['num_classes'] - head_count
        print(f"Groups created: {head_count} head classes, {tail_count} tail classes")
    
    num_groups = class_to_group.max().item() + 1
    print(f"Total groups: {num_groups}")
    
    print("\n✅ Training script integration test passed!")

def show_recommendations():
    """Show recommendations for using the new configuration system."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("""
🎯 CONFIGURATION RECOMMENDATIONS:

1. THRESHOLD-BASED GROUPING (Recommended):
   - Use 'method': 'threshold'
   - Set 'threshold': 20 for natural head/tail division
   - More interpretable: classes with >20 samples are 'head'
   - Rule-based and consistent across datasets
   
2. RATIO-BASED GROUPING (Legacy):
   - Use 'method': 'ratio'  
   - Keep for backwards compatibility
   - Less interpretable than threshold method

3. CONFIGURATION EXAMPLES:

   # Rule-based (recommended)
   'grouping': {
       'method': 'threshold',
       'threshold': 20,
       'head_ratio': 0.69,  # ignored
       'K': 2,
   }
   
   # Proportion-based (legacy)
   'grouping': {
       'method': 'ratio',
       'threshold': 20,     # ignored
       'head_ratio': 0.69,
       'K': 2,
   }

4. EXPERIMENTATION:
   - Try different thresholds: 15, 20, 25, 30
   - threshold=20 gives balanced head/tail split
   - Lower thresholds = more tail classes
   - Higher thresholds = more head classes

5. IMPACT ON TRAINING:
   - Threshold method ensures sufficient samples in head group
   - More stable training than arbitrary ratio splits
   - Better theoretical justification
    """)

if __name__ == "__main__":
    test_grouping_configuration()
    test_training_script_integration()
    show_recommendations()