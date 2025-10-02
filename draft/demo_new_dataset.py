#!/usr/bin/env python3
"""
Demo script showing how to use the new CIFAR-100-LT dataset in training.
Shows usage for both expert training and AR-GSE training.
"""

from src.data.dataloader_utils import (
    get_expert_training_dataloaders,
    get_argse_training_dataloaders, 
    get_cifar100_lt_dataloaders,
    CIFAR100LTDataModule
)

def demo_expert_training():
    """Demo: How to use dataloaders for expert training."""
    print("=" * 60)
    print("DEMO: EXPERT TRAINING DATALOADERS")
    print("=" * 60)
    
    # Get dataloaders for expert training
    train_loader, val_loader = get_expert_training_dataloaders(batch_size=128)
    
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Training batches per epoch: {len(train_loader)}")
    
    # Show a few batches
    print("\nSample training batches:")
    for i, (images, labels) in enumerate(train_loader):
        if i >= 3:  # Just show first 3 batches
            break
        print(f"  Batch {i+1}: {images.shape} | Labels range: {labels.min()}-{labels.max()}")
        
    print("\nReady for expert training!")

def demo_argse_training():
    """Demo: How to use dataloaders for AR-GSE training."""
    print("\n" + "=" * 60)
    print("DEMO: AR-GSE TRAINING DATALOADERS")
    print("=" * 60)
    
    # Get dataloaders for AR-GSE training  
    tunev_loader, val_loader, test_loader = get_argse_training_dataloaders(batch_size=64)
    
    print(f"TuneV samples (main training): {len(tunev_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"TuneV batches per epoch: {len(tunev_loader)}")
    
    # Show class distribution in tuneV
    data_module = CIFAR100LTDataModule(batch_size=64)
    data_module.setup_datasets()
    
    train_counts = data_module.get_train_class_counts_list()
    tunev_counts = data_module.get_class_counts('tunev')
    
    print(f"\nClass distribution comparison:")
    print(f"  Train head class: {train_counts[0]} samples")
    print(f"  TuneV head class: {tunev_counts.get(0, 0)} samples") 
    print(f"  Train tail class: {train_counts[99]} samples")
    print(f"  TuneV tail class: {tunev_counts.get(99, 0)} samples")
    
    print("\nReady for AR-GSE training!")

def demo_full_pipeline():
    """Demo: Complete training pipeline."""
    print("\n" + "=" * 60)
    print("DEMO: COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    
    # Get all dataloaders at once
    train_loader, val_loader, test_loader, tunev_loader = get_cifar100_lt_dataloaders(
        batch_size=128, num_workers=2
    )
    
    print("Available dataloaders:")
    print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"  Val: {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")  
    print(f"  Test: {len(test_loader.dataset):,} samples ({len(test_loader)} batches)")
    print(f"  TuneV: {len(tunev_loader.dataset):,} samples ({len(tunev_loader)} batches)")
    
    print(f"\nRecommended training workflow:")
    print(f"  1. Train experts using train_loader + val_loader")
    print(f"  2. Train AR-GSE using tunev_loader + val_loader")  
    print(f"  3. Final evaluation using test_loader")
    print(f"  4. Val/Test both have matching long-tail distribution!")

def demo_data_analysis():
    """Demo: Data analysis and statistics."""
    print("\n" + "=" * 60)
    print("DEMO: DATA ANALYSIS")
    print("=" * 60)
    
    data_module = CIFAR100LTDataModule()
    data_module.setup_datasets()
    
    # Get detailed statistics
    data_module.print_dataset_stats()
    
    # Show proportion matching
    train_counts = data_module.get_class_counts('train')
    test_counts = data_module.get_class_counts('test')
    val_counts = data_module.get_class_counts('val')
    
    print(f"\nProportion matching verification:")
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    total_val = sum(val_counts.values())
    
    print(f"Sample of proportion matching:")
    for cls in [0, 25, 50, 75, 99]:
        train_prop = train_counts.get(cls, 0) / total_train
        test_prop = test_counts.get(cls, 0) / total_test
        val_prop = val_counts.get(cls, 0) / total_val
        
        print(f"  Class {cls:2d}: train={train_prop:.4f}, test={test_prop:.4f}, val={val_prop:.4f}")
    
    # Show duplication effectiveness
    print(f"\nDuplication effectiveness:")
    print(f"  Original CIFAR test: 10,000 samples (100 per class)")
    print(f"  Our val+test: {total_val + total_test:,} samples")
    print(f"  Tail classes now have sufficient representation!")

def demo_transforms():
    """Demo: Data transforms and augmentations."""
    print("\n" + "=" * 60) 
    print("DEMO: DATA TRANSFORMS")
    print("=" * 60)
    
    from src.data.enhanced_datasets import get_cifar100_transforms
    
    train_transform, eval_transform = get_cifar100_transforms()
    
    print("Training transforms (with augmentation):")
    for transform in train_transform.transforms:
        print(f"  - {transform}")
        
    print(f"\nEvaluation transforms (no augmentation):")
    for transform in eval_transform.transforms:
        print(f"  - {transform}")
        
    print(f"\nFollows paper specification:")
    print(f"  - Basic augmentations only (no ColorJitter/RandAugment)")
    print(f"  - Standard CIFAR-100 normalization")
    print(f"  - RandomCrop + RandomHorizontalFlip for training")

if __name__ == "__main__":
    demo_expert_training()
    demo_argse_training() 
    demo_full_pipeline()
    demo_data_analysis()
    demo_transforms()
    
    print(f"\n" + "=" * 60)
    print("ALL DEMOS COMPLETED!")
    print("Dataset is ready for AR-GSE experiments 🚀")
    print("=" * 60)