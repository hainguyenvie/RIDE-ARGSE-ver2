#!/usr/bin/env python3
"""
Main training script for AR-GSE experts.

Usage:
    # Train all experts
    python train_experts_main.py
    
    # Train specific expert
    python train_experts_main.py --expert ce
    python train_experts_main.py --expert logitadjust  
    python train_experts_main.py --expert balsoftmax
"""

import sys
import argparse
from pathlib import Path
from src.train.train_expert import main
from src.train.train_expert import train_single_expert
from src.train.train_expert import EXPERT_CONFIGS

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description="Train AR-GSE Expert Models")
    parser.add_argument(
        '--expert', 
        type=str, 
        choices=list(EXPERT_CONFIGS.keys()), 
        help="Train specific expert (ce/logitadjust/balsoftmax). If not specified, trains all experts."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help="Override number of epochs"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print("🚀 AR-GSE Expert Training")
    print("=" * 50)
    
    if args.expert:
        print(f"Training single expert: {args.expert}")
        if args.epochs:
            # Override epochs in config temporarily
            original_epochs = EXPERT_CONFIGS[args.expert]['epochs']
            EXPERT_CONFIGS[args.expert]['epochs'] = args.epochs
            print(f"Overriding epochs: {original_epochs} -> {args.epochs}")
            
        train_single_expert(args.expert)
    else:
        print("Training all experts...")
        main()
    
    print("\n✅ Training completed!")