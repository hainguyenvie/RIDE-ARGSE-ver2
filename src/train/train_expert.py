import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import sys

# Make RIDE code importable
_ride_root = Path(__file__).resolve().parents[2] / 'RIDE-LongTailRecognition'
sys.path.append(str(_ride_root))

# RIDE imports (now from RIDE repository root)
from model.model import ResNet32Model
from utils import rename_parallel_state_dict, load_state_dict as ride_load_state_dict

# --- CONFIGURATION (RIDE-aligned) ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'data_root': './data',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'ride': {
        'config_path': 'RIDE-LongTailRecognition/configs/config_imbalance_cifar100_ride.json',
        'num_experts': 3,
        'reduce_dimension': 1
    },
    'export': {
        'individual_experts': True
    },
    'output': {
        'logits_dir': './outputs/logits',
        'ride_save_root': 'RIDE-LongTailRecognition/saved/models'
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _run_ride_training():
    """Invoke official RIDE trainer with the CIFAR100-LT config and 3 experts."""
    cmd = [
        'python', 'RIDE-LongTailRecognition/train.py',
        '-c', CONFIG['ride']['config_path'],
        '--reduce_dimension', str(CONFIG['ride']['reduce_dimension']),
        '--num_experts', str(CONFIG['ride']['num_experts'])
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def _find_latest_ride_best_checkpoint() -> Path:
    """Locate the latest model_best.pth produced by RIDE under saved/models."""
    root = Path(CONFIG['output']['ride_save_root'])
    if not root.exists():
        raise FileNotFoundError(f"RIDE save root not found: {root}")
    candidates = list(root.glob('**/model_best.pth'))
    if not candidates:
        raise FileNotFoundError("No RIDE model_best.pth found. Ensure training finished.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _build_ride_model(num_classes: int, num_experts: int) -> torch.nn.Module:
    model = ResNet32Model(num_classes=num_classes, reduce_dimension=bool(CONFIG['ride']['reduce_dimension']), num_experts=num_experts, use_norm=True)
    return model


def _load_ride_weights(model: torch.nn.Module, ckpt_path: Path):
    state = torch.load(ckpt_path, map_location='cpu')
    sd = state['state_dict'] if 'state_dict' in state else state
    rename_parallel_state_dict(sd)
    ride_load_state_dict(model, sd)


def _export_ride_logits(model: torch.nn.Module, expert_names, num_experts: int):
    model = model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    splits_info = [
        {'name': 'train', 'dataset_type': 'train', 'file': 'train_indices.json'},
        {'name': 'tuneV', 'dataset_type': 'train', 'file': 'tuneV_indices.json'},
        {'name': 'val_lt', 'dataset_type': 'test', 'file': 'val_lt_indices.json'},
        {'name': 'test_lt', 'dataset_type': 'test', 'file': 'test_lt_indices.json'},
    ]

    for split_info in splits_info:
        indices_path = splits_dir / split_info['file']
        if not indices_path.exists():
            print(f"  Warning: {indices_path.name} not found, skipping {split_info['name']}")
            continue

        if split_info['dataset_type'] == 'train':
            base_dataset = torchvision.datasets.CIFAR100(root=CONFIG['dataset']['data_root'], train=True, transform=transform)
        else:
            base_dataset = torchvision.datasets.CIFAR100(root=CONFIG['dataset']['data_root'], train=False, transform=transform)

        indices = json.loads(indices_path.read_text())
        subset = Subset(base_dataset, indices)
        loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=4)

        # Collect per-expert logits
        collected = [list() for _ in range(num_experts)]
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=f"Exporting {split_info['name']}"):
                inputs = inputs.to(DEVICE)
                # Ensure we use all experts when exporting
                if hasattr(model, 'backbone') and hasattr(model.backbone, 'num_experts'):
                    try:
                        model.backbone.use_experts = list(range(model.backbone.num_experts))
                    except Exception:
                        pass
                _ = model(inputs)  # forward populates model.backbone.logits (list of [B,C])
                # Stack to [E, B, C]
                per_exp = torch.stack(model.backbone.logits, dim=0)  # [E, B, C]
                for e in range(num_experts):
                    collected[e].append(per_exp[e].cpu())

        # Save per expert
        for e in range(num_experts):
            out_dir = Path(CONFIG['output']['logits_dir']) / CONFIG['dataset']['name'] / expert_names[e]
            out_dir.mkdir(parents=True, exist_ok=True)
            tensor = torch.cat(collected[e], dim=0).to(torch.float16)
            torch.save(tensor, out_dir / f"{split_info['name']}_logits.pt")
        print(f"  ✅ {split_info['name']}: {len(indices):,} samples → saved {num_experts} experts")


def main():
    print("🚀 Training RIDE experts and exporting per-expert logits")
    print(f"Device: {DEVICE}")
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])

    # 1) Train via official RIDE script
    _run_ride_training()

    # 2) Find the latest best checkpoint
    best_ckpt = _find_latest_ride_best_checkpoint()
    print(f"📁 Using checkpoint: {best_ckpt}")

    # 3) Build and load model
    num_experts = CONFIG['ride']['num_experts']
    model = _build_ride_model(CONFIG['dataset']['num_classes'], num_experts)
    _load_ride_weights(model, best_ckpt)

    # 4) Export per-expert logits with consistent names for downstream pipeline
    expert_names = [f"ride_expert_{i}" for i in range(num_experts)]
    _export_ride_logits(model, expert_names, num_experts)
    print("✅ Completed RIDE expert export")


if __name__ == '__main__':
    main()