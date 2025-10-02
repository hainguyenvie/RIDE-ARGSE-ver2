#!/usr/bin/env python3
"""
DataLoader utilities for CIFAR-100-LT training with AR-GSE.
Provides easy-to-use functions for loading datasets in training scripts.
"""

from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
from src.data.enhanced_datasets import CIFAR100LTDataset, get_cifar100_transforms

class CIFAR100LTDataModule:
    """
    DataModule for CIFAR-100-LT that handles all data loading needs.
    Compatible with expert training and AR-GSE training.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        splits_dir: str = "data/cifar100_lt_if100_splits", 
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.data_dir = data_dir
        self.splits_dir = splits_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Load transforms
        self.train_transform, self.eval_transform = get_cifar100_transforms()
        
        # Load CIFAR-100 datasets
        self.cifar_train = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=None
        )
        self.cifar_test = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=None  
        )
        
        # Initialize datasets
        self.datasets = {}
        self.dataloaders = {}
        
    def load_indices(self, split_name: str) -> list:
        """Load indices for a specific split."""
        indices_file = Path(self.splits_dir) / f"{split_name}_indices.json"
        
        if not indices_file.exists():
            raise FileNotFoundError(f"Split file not found: {indices_file}")
            
        with open(indices_file, 'r') as f:
            indices = json.load(f)
            
        return indices
        
    def setup_datasets(self):
        """Setup all datasets from saved splits."""
        print("Setting up CIFAR-100-LT datasets...")
        
        # Define split mappings (removed val_small and calib)
        splits = {
            'train': ('train_indices', self.cifar_train, self.train_transform),
            'val': ('val_lt_indices', self.cifar_test, self.eval_transform),
            'test': ('test_lt_indices', self.cifar_test, self.eval_transform),
            'tunev': ('tuneV_indices', self.cifar_test, self.eval_transform)
        }
        
        for split_key, (indices_key, base_dataset, transform) in splits.items():
            try:
                indices = self.load_indices(indices_key)
                self.datasets[split_key] = CIFAR100LTDataset(
                    base_dataset, indices, transform
                )
                print(f"  {split_key}: {len(indices):,} samples")
                
            except FileNotFoundError as e:
                print(f"  Warning: {split_key} not found, skipping...")
                continue
                
        print("Datasets setup complete!")
        
    def get_dataloader(
        self, 
        split: str, 
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        **kwargs
    ) -> DataLoader:
        """Get DataLoader for a specific split."""
        
        if split not in self.datasets:
            raise ValueError(f"Split '{split}' not available. Available splits: {list(self.datasets.keys())}")
            
        # Default shuffle behavior
        if shuffle is None:
            shuffle = (split == 'train')
            
        # Use provided batch_size or default
        bs = batch_size if batch_size is not None else self.batch_size
        
        # Merge kwargs with defaults
        dataloader_kwargs = {
            'batch_size': bs,
            'shuffle': shuffle,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': (split == 'train')  # Drop last batch only for training
        }
        dataloader_kwargs.update(kwargs)
        
        return DataLoader(self.datasets[split], **dataloader_kwargs)
    
    def get_all_dataloaders(self) -> Dict[str, DataLoader]:
        """Get all available dataloaders."""
        dataloaders = {}
        
        for split in self.datasets.keys():
            dataloaders[split] = self.get_dataloader(split)
            
        return dataloaders
    
    def get_class_counts(self, split: str = 'train') -> Dict[int, int]:
        """Get class distribution for a split."""
        if split not in self.datasets:
            raise ValueError(f"Split '{split}' not available")
            
        from collections import Counter
        
        # Get all targets from the dataset
        targets = []
        dataset = self.datasets[split]
        for i in range(len(dataset)):
            _, target = dataset[i]
            targets.append(target)
            
        return dict(Counter(targets))
    
    def get_train_class_counts_list(self) -> list:
        """Get training class counts as a list (for compatibility)."""
        class_counts = self.get_class_counts('train')
        return [class_counts.get(i, 0) for i in range(100)]
    
    def print_dataset_stats(self):
        """Print statistics for all available datasets."""
        print("\n=== DATASET STATISTICS ===")
        
        for split_name in self.datasets.keys():
            dataset = self.datasets[split_name]
            class_counts = self.get_class_counts(split_name)
            
            total = len(dataset)
            head_count = class_counts.get(0, 0)
            tail_count = class_counts.get(99, 0)
            
            print(f"\n{split_name.upper()}:")
            print(f"  Total samples: {total:,}")
            print(f"  Head class: {head_count} ({head_count/total*100:.1f}%)")
            print(f"  Tail class: {tail_count} ({tail_count/total*100:.1f}%)")
            if tail_count > 0:
                print(f"  Imbalance factor: {head_count/tail_count:.1f}")

# Convenience functions for easy use in training scripts

def get_cifar100_lt_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "data",
    splits_dir: str = "data/cifar100_lt_if100_splits"
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Quick function to get train, val, test, tunev dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, tunev_loader)
    """
    
    data_module = CIFAR100LTDataModule(
        data_dir=data_dir,
        splits_dir=splits_dir,
        batch_size=batch_size, 
        num_workers=num_workers
    )
    
    data_module.setup_datasets()
    
    train_loader = data_module.get_dataloader('train')
    val_loader = data_module.get_dataloader('val') 
    test_loader = data_module.get_dataloader('test')
    tunev_loader = data_module.get_dataloader('tunev')
    
    return train_loader, val_loader, test_loader, tunev_loader

def get_expert_training_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get dataloaders for expert training (train + val_lt for realistic validation).
    
    Returns:
        Tuple of (train_loader, val_loader) 
    """
    
    data_module = CIFAR100LTDataModule(batch_size=batch_size, num_workers=num_workers)
    data_module.setup_datasets()
    
    train_loader = data_module.get_dataloader('train')
    val_loader = data_module.get_dataloader('val')  # Use val_lt for realistic validation
    
    return train_loader, val_loader

def get_calibration_dataloader(
    batch_size: int = 128,
    num_workers: int = 4
) -> DataLoader:
    """
    Get dataloader for temperature scaling calibration.
    Uses val_lt dataset for realistic long-tail calibration.
    
    Returns:
        DataLoader for calibration
    """
    
    data_module = CIFAR100LTDataModule(batch_size=batch_size, num_workers=num_workers)
    data_module.setup_datasets()
    
    # Use val (val_lt) for calibration - realistic long-tail distribution
    calib_loader = data_module.get_dataloader('val')
    
    return calib_loader

def get_argse_training_dataloaders(
    batch_size: int = 64,  # Smaller batch for AR-GSE
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get dataloaders for AR-GSE training.
    
    Returns:
        Tuple of (tunev_loader, val_loader, test_loader)
    """
    
    data_module = CIFAR100LTDataModule(batch_size=batch_size, num_workers=num_workers)
    data_module.setup_datasets()
    
    tunev_loader = data_module.get_dataloader('tunev')  # Main training for AR-GSE
    val_loader = data_module.get_dataloader('val')      # Validation 
    test_loader = data_module.get_dataloader('test')    # Final evaluation
    
    return tunev_loader, val_loader, test_loader

# Example usage functions
def test_dataloaders():
    """Test function to verify dataloaders work correctly."""
    print("Testing CIFAR-100-LT DataLoaders...")
    
    try:
        # Test full setup
        data_module = CIFAR100LTDataModule(batch_size=32, num_workers=0)  # num_workers=0 for testing
        data_module.setup_datasets()
        data_module.print_dataset_stats()
        
        # Test a few batches
        print("\nTesting batches:")
        for split in ['train', 'val', 'test', 'tunev']:
            if split in data_module.datasets:
                loader = data_module.get_dataloader(split)
                batch = next(iter(loader))
                images, labels = batch
                print(f"  {split}: batch_size={images.shape[0]}, image_shape={images.shape[1:]}")
        
        print("\nDataLoaders test completed successfully!")
        
    except Exception as e:
        print(f"Error testing dataloaders: {e}")
        raise

if __name__ == "__main__":
    test_dataloaders()