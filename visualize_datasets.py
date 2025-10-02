#!/usr/bin/env python3
"""
Wrapper script to visualize CIFAR-100-LT dataset distributions.
Run from project root directory.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run
try:
    from src.visualize.simple_visualizer import main
    main()
except Exception as e:
    print(f"❌ Error running visualization: {e}")
    import traceback
    traceback.print_exc()