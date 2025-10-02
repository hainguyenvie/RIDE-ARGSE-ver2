# prepare_cifar100_lt.py
import json
from sklearn.model_selection import train_test_split
import torch
import torchvision
import numpy as np
from pathlib import Path
import collections

# Make sure src is in the python path or run as a module
from src.data.datasets import generate_longtail_train_set, get_cifar100_lt_counts
## THAY ĐỔI: Import thêm hàm mới
from src.data.splits import create_and_save_splits, create_longtail_val_test_splits
from src.data.groups import get_class_to_group

def main():
    # --- Configuration ---
    SEED = 42
    IMB_FACTOR = 100
    DATA_ROOT = "./data"
    OUTPUT_DIR = Path(f"./data/cifar100_lt_if{IMB_FACTOR}_splits")
    
    # Ratios cho các split lấy từ tập train (vẫn giữ nguyên)
    SPLIT_RATIOS = {
        'train': 0.8,
        'tuneV': 0.08,
        'val_small': 0.06,
        'calib': 0.03
    }
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- M1.1: Load original data and create LT train set ---
    print("Step 1: Loading original CIFAR-100 and creating long-tail training set...")
    cifar100_train = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True)
    cifar100_test = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True)

    lt_indices, lt_targets = generate_longtail_train_set(cifar100_train, IMB_FACTOR)
    
    print(f"Total samples in new long-tail train set: {len(lt_indices)}")
    
    # --- M1.2: Create and save splits from the LT train set ---
    print("\nStep 2: Splitting the LT train set and saving indices...")
    # Logic này vẫn giữ nguyên để tạo ra các tập cần thiết cho AR-GSE
    total_prop = sum(SPLIT_RATIOS.values()) # 0.8 + 0.08 + 0.06 + 0.03 = 0.97
    num_total_samples = int(len(lt_indices) * total_prop)
    
    # Lấy một tập con từ lt_indices để chia
    subset_indices, _, subset_targets, _ = train_test_split(
        lt_indices, lt_targets, train_size=num_total_samples, random_state=SEED, stratify=lt_targets
    )
    
    # Chuẩn hóa lại tỉ lệ để chia tập con
    split_ratios_norm = {name: ratio / total_prop for name, ratio in SPLIT_RATIOS.items()}
    create_and_save_splits(subset_indices, subset_targets, split_ratios_norm, OUTPUT_DIR, SEED)

    # --- M1.3: Define groups ---
    print("\nStep 3: Defining class groups (Head/Tail)...")
    # Chúng ta cần train_class_counts cho bước tiếp theo
    train_class_counts = get_cifar100_lt_counts(IMB_FACTOR)
    class_to_group = get_class_to_group(train_class_counts, K=2, head_ratio=0.5)

    ## THAY ĐỔI: Thêm bước M1.4 để tạo val/test long-tail
    # --- M1.4: Create Long-tail Validation and Test splits from original test set ---
    print("\nStep 4: Creating long-tail validation and test sets based on baseline paper...")
    create_longtail_val_test_splits(
        cifar100_test_dataset=cifar100_test,
        train_class_counts=train_class_counts,
        output_dir=OUTPUT_DIR,
        val_size=0.2, # 20% cho validation như trong paper
        seed=SEED
    )

    # --- DoD Check: Print stats ---
    print("\n--- DoD CHECK (UPDATED) ---")
    print(f"Imbalance Factor: {IMB_FACTOR}")
    print(f"Most frequent train class samples: {train_class_counts[0]}")
    print(f"Least frequent train class samples: {train_class_counts[-1]}")
    
    # Verify split sizes
    print("\nSplit sizes:")
    ## THAY ĐỔI: Thêm các split mới vào danh sách kiểm tra
    splits_to_verify = ['train', 'tuneV', 'val_small', 'calib', 'val_lt', 'test_lt']
    for split_name in splits_to_verify:
        filepath = OUTPUT_DIR / f"{split_name}_indices.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                indices = json.load(f)
                print(f"- {split_name}: {len(indices)} samples")
        else:
            print(f"- {split_name}: File not found!")
    
    # Verify group distribution
    head_classes = (class_to_group == 0).sum().item()
    tail_classes = (class_to_group == 1).sum().item()
    print(f"\nGroup distribution: {head_classes} head classes, {tail_classes} tail classes.")
    
    lt_target_counts = collections.Counter(lt_targets)
    head_samples = sum(lt_target_counts[i] for i in range(100) if class_to_group[i] == 0)
    tail_samples = sum(lt_target_counts[i] for i in range(100) if class_to_group[i] == 1)
    print(f"Total samples in original LT train set -> Head: {head_samples}, Tail: {tail_samples}")
    print("\nMilestone M1 (updated) is complete!")

if __name__ == '__main__':
    main()