# run_argse_training.py
"""
Wrapper script to run AR-GSE training with proper path setup.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Change to project root directory
os.chdir(Path(__file__).parent)

# Now import and run the training
from train.train_argse import main

if __name__ == '__main__':
    print("🚀 Starting AR-GSE Training...")
    print("=" * 60)
    main()