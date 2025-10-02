# analyze_expert_training.py
"""
Detailed analysis of expert training logic to identify potential issues
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np

def analyze_training_config():
    """Analyze the training configuration for potential issues"""
    
    print("🔍 ANALYZING EXPERT TRAINING CONFIGURATION")
    print("="*60)
    
    # CONFIG từ train_expert.py
    CONFIG = {
        'dataset': {
            'name': 'cifar100_lt_if100',
            'data_root': './data', 
            'splits_dir': './data/cifar100_lt_if100_splits',
            'num_classes': 100,
        },
        'experts_to_train': [
            {'name': 'ce_baseline', 'loss_type': 'ce'},
            {'name': 'logitadjust_baseline', 'loss_type': 'logitadjust'},
            {'name': 'balsoftmax_baseline', 'loss_type': 'balsoftmax'},
        ],
        'train_params': {
            'epochs': 256,
            'batch_size': 128,
            'lr': 0.4,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'warmup_steps': 15,
        },
        'output': {
            'checkpoints_dir': './checkpoints/experts',
            'logits_dir': './outputs/logits',
        },
        'seed': 42
    }
    
    issues = []
    warnings = []
    
    # 1. Training Duration Analysis
    print("\n📊 TRAINING DURATION ANALYSIS")
    epochs = CONFIG['train_params']['epochs']
    batch_size = CONFIG['train_params']['batch_size']
    
    # Load data to calculate actual training time
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    with open(splits_dir / 'train_indices.json', 'r') as f:
        train_indices = json.load(f)
    
    num_samples = len(train_indices)
    batches_per_epoch = (num_samples + batch_size - 1) // batch_size
    total_batches = epochs * batches_per_epoch
    
    print(f"  Training samples: {num_samples:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Total epochs: {epochs}")
    print(f"  Total batches: {total_batches:,}")
    
    # Estimate training time (rough)
    estimated_time_per_batch = 0.5  # seconds per batch (rough estimate)
    estimated_total_hours = (total_batches * estimated_time_per_batch) / 3600
    print(f"  Estimated training time per expert: {estimated_total_hours:.1f} hours")
    print(f"  Total for 3 experts: {estimated_total_hours * 3:.1f} hours")
    
    if estimated_total_hours > 12:
        warnings.append(f"Long training time: {estimated_total_hours:.1f}h per expert")
    
    # 2. Learning Rate Analysis
    print("\n📈 LEARNING RATE ANALYSIS")
    lr = CONFIG['train_params']['lr']
    warmup_steps = CONFIG['train_params']['warmup_steps']
    
    print(f"  Base learning rate: {lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Warmup ends after: {warmup_steps / batches_per_epoch:.3f} epochs")
    
    # LR schedule milestones
    milestones = [96, 192, 224]
    gamma = 0.1
    
    print(f"  LR schedule milestones: {milestones}")
    print(f"  LR decay factor: {gamma}")
    
    for i, milestone in enumerate(milestones):
        lr_at_milestone = lr * (gamma ** (i + 1))
        print(f"    Epoch {milestone}: LR = {lr_at_milestone:.6f}")
    
    if lr > 0.1:
        warnings.append(f"High learning rate: {lr} (typical range 0.01-0.1)")
    
    if warmup_steps < batches_per_epoch:
        warnings.append(f"Short warmup: {warmup_steps} steps < 1 epoch ({batches_per_epoch} batches)")
    
    # 3. Memory Analysis
    print("\n💾 MEMORY ANALYSIS")
    
    # Model parameters
    model_params = 472_756  # từ validation test
    param_memory_mb = (model_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    # Activation memory (rough estimate)
    # For ResNet32: input (3,32,32) -> features -> logits (100)
    # Batch size 128, multiple layers with different sizes
    activation_memory_mb = (128 * 64 * 8 * 8 * 4) / (1024 * 1024)  # Rough estimate
    
    # Gradient memory (same as parameters)
    grad_memory_mb = param_memory_mb
    
    # Optimizer state (SGD with momentum)
    optimizer_memory_mb = param_memory_mb  # momentum buffers
    
    total_memory_mb = param_memory_mb + activation_memory_mb + grad_memory_mb + optimizer_memory_mb
    
    print(f"  Model parameters: {model_params:,} ({param_memory_mb:.1f} MB)")
    print(f"  Activation memory: ~{activation_memory_mb:.1f} MB")
    print(f"  Gradient memory: {grad_memory_mb:.1f} MB") 
    print(f"  Optimizer state: {optimizer_memory_mb:.1f} MB")
    print(f"  Total estimated: {total_memory_mb:.1f} MB")
    
    if total_memory_mb > 2000:  # 2GB
        warnings.append(f"High memory usage: {total_memory_mb:.1f} MB")
    
    # 4. Data Analysis
    print("\n📊 DATA ANALYSIS")
    
    # Load validation data info
    with open(splits_dir / 'val_lt_indices.json', 'r') as f:
        val_indices = json.load(f)
    
    print(f"  Training samples: {len(train_indices):,}")
    print(f"  Validation samples: {len(val_indices):,}")
    print(f"  Train/Val ratio: {len(train_indices) / len(val_indices):.1f}")
    
    # Check class imbalance in training data
    import torchvision
    dataset = torchvision.datasets.CIFAR100(root=CONFIG['dataset']['data_root'], train=True, download=False)
    train_targets = np.array(dataset.targets)[train_indices]
    class_counts = Counter(train_targets)
    
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"  Class imbalance ratio: {imbalance_ratio:.1f}")
    print(f"  Classes with <10 samples: {sum(1 for c in class_counts.values() if c < 10)}")
    
    if imbalance_ratio > 50:
        warnings.append(f"Extreme class imbalance: {imbalance_ratio:.1f}")
    
    # 5. Output Analysis
    print("\n📁 OUTPUT ANALYSIS")
    
    experts = CONFIG['experts_to_train']
    checkpoints_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    logits_dir = Path(CONFIG['output']['logits_dir']) / CONFIG['dataset']['name']
    
    print(f"  Number of experts: {len(experts)}")
    print(f"  Checkpoints dir: {checkpoints_dir}")
    print(f"  Logits dir: {logits_dir}")
    
    # Estimate output sizes
    splits = ['train', 'tuneV', 'val_small', 'calib', 'val_lt', 'test_lt']
    total_logits_mb = 0
    
    for expert in experts:
        expert_name = expert['name']
        for split in splits:
            split_file = splits_dir / f"{split}_indices.json"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    indices = json.load(f)
                
                # Logits: samples x 100 classes x 2 bytes (float16)
                logits_mb = (len(indices) * 100 * 2) / (1024 * 1024)
                total_logits_mb += logits_mb
    
    print(f"  Total logits size: {total_logits_mb:.1f} MB")
    
    # Model checkpoints (2 per expert: best + final_calibrated)
    checkpoint_mb = param_memory_mb * 2 * len(experts)
    print(f"  Total checkpoints size: {checkpoint_mb:.1f} MB")
    
    # 6. Critical Issues Check
    print("\n⚠️  POTENTIAL ISSUES")
    
    # Check for critical issues
    if not Path(CONFIG['dataset']['data_root']).exists():
        issues.append("Data root directory doesn't exist")
    
    if not Path(CONFIG['dataset']['splits_dir']).exists():
        issues.append("Splits directory doesn't exist")
    
    # Check CUDA availability
    import torch
    if not torch.cuda.is_available():
        warnings.append("CUDA not available - training will be slow on CPU")
    else:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU memory available: {gpu_memory:.1f} GB")
        if total_memory_mb > gpu_memory * 1024 * 0.8:  # 80% threshold
            warnings.append(f"May exceed GPU memory: {total_memory_mb:.1f}MB vs {gpu_memory:.1f}GB available")
    
    # Warmup issue
    if warmup_steps < 10:
        issues.append(f"Warmup too short: {warmup_steps} steps may not be effective")
    
    # Print summary
    print(f"\n🔍 ANALYSIS SUMMARY")
    print(f"Critical issues: {len(issues)}")
    print(f"Warnings: {len(warnings)}")
    
    if issues:
        print(f"\n❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print(f"\n✅ NO ISSUES FOUND - Ready for training!")
    
    return len(issues) == 0

def analyze_loss_functions():
    """Analyze the loss function implementations"""
    
    print(f"\n🧮 LOSS FUNCTION ANALYSIS")
    print("="*40)
    
    # Test loss functions with real class distribution
    splits_dir = Path("./data/cifar100_lt_if100_splits")
    with open(splits_dir / 'train_indices.json', 'r') as f:
        train_indices = json.load(f)
    
    import torchvision
    dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=False)
    train_targets = np.array(dataset.targets)[train_indices]
    class_counts = [train_targets.tolist().count(i) for i in range(100)]
    
    print(f"Class distribution:")
    print(f"  Min: {min(class_counts)}, Max: {max(class_counts)}")
    print(f"  Mean: {np.mean(class_counts):.1f}, Std: {np.std(class_counts):.1f}")
    
    # Analyze each loss function
    import torch
    from src.models.losses import LogitAdjustLoss, BalancedSoftmaxLoss
    
    # Create dummy data
    batch_size = 128
    num_classes = 100
    logits = torch.randn(batch_size, num_classes)
    
    # Test tail class (class 99) vs head class (class 0)
    targets_head = torch.zeros(batch_size, dtype=torch.long)  # All class 0
    targets_tail = torch.full((batch_size,), 99, dtype=torch.long)  # All class 99
    
    # Cross Entropy
    ce_loss = torch.nn.CrossEntropyLoss()
    ce_head = ce_loss(logits, targets_head).item()
    ce_tail = ce_loss(logits, targets_tail).item()
    
    print(f"\nCross Entropy:")
    print(f"  Head class loss: {ce_head:.4f}")
    print(f"  Tail class loss: {ce_tail:.4f}")
    print(f"  Difference: {abs(ce_head - ce_tail):.4f}")
    
    # Logit Adjust
    la_loss = LogitAdjustLoss(class_counts=class_counts)
    la_head = la_loss(logits, targets_head).item()
    la_tail = la_loss(logits, targets_tail).item()
    
    print(f"\nLogit Adjust:")
    print(f"  Head class loss: {la_head:.4f}")
    print(f"  Tail class loss: {la_tail:.4f}")
    print(f"  Difference: {abs(la_head - la_tail):.4f}")
    
    # Balanced Softmax
    bs_loss = BalancedSoftmaxLoss(class_counts=class_counts)
    bs_head = bs_loss(logits, targets_head).item()
    bs_tail = bs_loss(logits, targets_tail).item()
    
    print(f"\nBalanced Softmax:")
    print(f"  Head class loss: {bs_head:.4f}")
    print(f"  Tail class loss: {bs_tail:.4f}")
    print(f"  Difference: {abs(bs_head - bs_tail):.4f}")
    
    # Analysis
    print(f"\n📊 Analysis:")
    print(f"  LogitAdjust vs CE: {la_head - ce_head:.4f} (head), {la_tail - ce_tail:.4f} (tail)")
    print(f"  BalancedSoftmax vs CE: {bs_head - ce_head:.4f} (head), {bs_tail - ce_tail:.4f} (tail)")
    
    # The goal is that re-balancing methods should penalize head classes more
    # and be more lenient on tail classes

if __name__ == "__main__":
    success = analyze_training_config()
    analyze_loss_functions()
    
    if success:
        print(f"\n🎉 EXPERT TRAINING READY!")
    else:
        print(f"\n⚠️  PLEASE FIX ISSUES BEFORE TRAINING")