#!/usr/bin/env python3
"""
Quick test of the expert training with CIFARResNet-32.
Tests one epoch to validate the training pipeline.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Import our custom modules
from src.models.experts import Expert
from src.models.losses import LogitAdjustLoss, BalancedSoftmaxLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def quick_training_test():
    """Test training for a few batches to validate the pipeline."""
    print("🧪 Quick Training Pipeline Test")
    print("="*40)
    
    # Simple transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Load a small subset of CIFAR-100
    dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use only first 512 samples for quick test
    small_dataset = Subset(dataset, list(range(512)))
    train_loader = DataLoader(small_dataset, batch_size=32, shuffle=True)
    
    print(f"  Device: {DEVICE}")
    print(f"  Dataset size: {len(small_dataset)} samples")
    print(f"  Batches: {len(train_loader)}")
    
    # Test all expert configurations
    expert_configs = {
        'ce': {'dropout_rate': 0.0, 'loss_type': 'ce'},
        'logitadjust': {'dropout_rate': 0.1, 'loss_type': 'logitadjust'}, 
        'balsoftmax': {'dropout_rate': 0.1, 'loss_type': 'balsoftmax'}
    }
    
    for expert_name, config in expert_configs.items():
        print(f"\n📋 Testing {expert_name.upper()} Expert:")
        
        # Create model
        model = Expert(
            num_classes=100,
            backbone_name='cifar_resnet32',
            dropout_rate=config['dropout_rate'],
            init_weights=True
        ).to(DEVICE)
        
        # Create loss function
        if config['loss_type'] == 'ce':
            criterion = nn.CrossEntropyLoss()
        else:
            # Get class counts for rebalanced losses
            targets = [dataset.targets[i] for i in range(512)]
            class_counts = [targets.count(i) for i in range(100)]
            
            if config['loss_type'] == 'logitadjust':
                criterion = LogitAdjustLoss(class_counts=class_counts)
            else:  # balsoftmax
                criterion = BalancedSoftmaxLoss(class_counts=class_counts)
        
        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        
        # Model summary
        model.summary()
        
        # Train for 3 batches
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            if i >= 3:  # Only 3 batches for quick test
                break
                
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print(f"    Batch {i+1}: Loss = {loss.item():.4f}")
        
        avg_loss = running_loss / 3
        print(f"    ✅ Average loss: {avg_loss:.4f}")
        
        # Test calibrated inference
        model.eval()
        with torch.no_grad():
            sample_input, sample_target = next(iter(train_loader))
            sample_input = sample_input[:4].to(DEVICE)  # First 4 samples
            
            regular_logits = model(sample_input)
            calibrated_logits = model.get_calibrated_logits(sample_input)
            features = model.get_features(sample_input)
            
            print(f"    Regular logits shape: {regular_logits.shape}")
            print(f"    Calibrated logits shape: {calibrated_logits.shape}")
            print(f"    Features shape: {features.shape}")
    
    print("\n✅ All expert training tests passed!")
    print("The CIFARResNet-32 architecture is ready for full training.")

if __name__ == "__main__":
    quick_training_test()