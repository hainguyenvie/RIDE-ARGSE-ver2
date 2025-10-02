"""
Test script for GSE Worst-group with EG-outer optimization.
Run this to enable worst-group optimization with EG-outer.
"""

import sys
sys.path.append('.')

from src.train.gse_balanced_plugin import main

if __name__ == '__main__':
    # Modify configuration for worst-group EG-outer
    from src.train.gse_balanced_plugin import CONFIG
    
    # Enable worst-group optimization with improved EG-outer
    CONFIG['plugin_params']['objective'] = 'worst'
    CONFIG['plugin_params']['use_eg_outer'] = True
    CONFIG['plugin_params']['eg_outer_T'] = 30         # More iterations
    CONFIG['plugin_params']['eg_outer_xi'] = 0.2       # Reduced step size for stability
    
    # Update output directory
    CONFIG['output']['checkpoints_dir'] = './checkpoints/argse_worst_eg/'
    
    print("🚀 Running GSE Worst-Group with EG-Outer optimization...")
    print(f"   EG outer iterations: {CONFIG['plugin_params']['eg_outer_T']}")
    print(f"   EG step size: {CONFIG['plugin_params']['eg_outer_xi']}")
    
    main()