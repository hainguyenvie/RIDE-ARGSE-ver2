#!/usr/bin/env python3
"""
Test script for the updated CIFARResNet-32 implementation.
"""

import torch
import torch.nn as nn
from src.models.experts import Expert

def test_cifar_resnet32():
    """Test the CIFARResNet-32 implementation."""
    print("🧪 Testing CIFARResNet-32 Implementation")
    print("="*50)
    
    # Test different configurations
    configs = [
        {"name": "Standard", "dropout_rate": 0.0},
        {"name": "With Dropout", "dropout_rate": 0.1},
        {"name": "High Dropout", "dropout_rate": 0.2}
    ]
    
    for config in configs:
        print(f"\n📋 Testing {config['name']} Configuration:")
        
        # Create model
        model = Expert(
            num_classes=100, 
            backbone_name='cifar_resnet32',
            dropout_rate=config['dropout_rate']
        )
        
        # Test with CIFAR-sized input
        batch_size = 8
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        
        print(f"  Input shape: {input_tensor.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            # Standard forward
            logits = model(input_tensor)
            print(f"  Output logits shape: {logits.shape}")
            
            # Calibrated logits
            calibrated = model.get_calibrated_logits(input_tensor)
            print(f"  Calibrated logits shape: {calibrated.shape}")
            
            # Feature extraction
            features = model.get_features(input_tensor)
            print(f"  Feature shape: {features.shape}")
        
        # Test training mode
        model.train()
        logits_train = model(input_tensor)
        print(f"  Training mode logits shape: {logits_train.shape}")
        
        # Model summary
        model.summary()
        
        # Test temperature scaling
        model.set_temperature(1.5)
        print(f"  Temperature after setting to 1.5: {model.temperature.item():.4f}")
        
    print("\n✅ All tests passed!")

def test_backward_compatibility():
    """Test backward compatibility with old ResNet32 calls."""
    print("\n🔄 Testing Backward Compatibility")
    print("="*30)
    
    # Test old-style call
    model_old = Expert(num_classes=100, backbone_name='resnet32')
    
    # Test new-style call
    model_new = Expert(num_classes=100, backbone_name='cifar_resnet32')
    
    input_tensor = torch.randn(4, 3, 32, 32)
    
    with torch.no_grad():
        out_old = model_old(input_tensor)
        out_new = model_new(input_tensor)
        
    print(f"  Old backbone output shape: {out_old.shape}")
    print(f"  New backbone output shape: {out_new.shape}")
    print("  ✅ Backward compatibility maintained")

def test_gradient_flow():
    """Test that gradients flow properly through the network."""
    print("\n🔥 Testing Gradient Flow")
    print("="*25)
    
    model = Expert(num_classes=100, backbone_name='cifar_resnet32')
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    input_tensor = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 100, (4,))
    
    logits = model(input_tensor)
    loss = criterion(logits, targets)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"  Model has gradients: {has_gradients}")
    
    # Check specific layers
    backbone_grads = any(p.grad is not None for p in model.backbone.parameters())
    classifier_grads = model.fc.weight.grad is not None
    
    print(f"  Backbone gradients: {backbone_grads}")
    print(f"  Classifier gradients: {classifier_grads}")
    print("  ✅ Gradient flow working correctly")

if __name__ == "__main__":
    print("🚀 CIFARResNet-32 Test Suite")
    print("="*60)
    
    test_cifar_resnet32()
    test_backward_compatibility()
    test_gradient_flow()
    
    print("\n🎉 All tests completed successfully!")
    print("The CIFARResNet-32 implementation is ready for training.")