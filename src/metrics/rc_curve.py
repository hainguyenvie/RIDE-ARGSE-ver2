# src/metrics/rc_curve.py
import torch
import numpy as np
import pandas as pd
from .selective_metrics import calculate_selective_errors

def generate_rc_curve(margins, preds, labels, class_to_group, num_groups, num_points=101):
    """
    Generates data points for the Risk-Coverage curve.
    """
    # Sort samples by margin, descending. Higher margin = higher confidence.
    sorted_indices = torch.argsort(margins, descending=True)

    rc_data = []
    total_samples = len(labels)
    
    # Sweep through different coverage levels
    for i in range(1, num_points + 1):
        coverage_target = i / num_points
        num_to_accept = int(total_samples * coverage_target)
        
        if num_to_accept == 0:
            rc_data.append({'coverage': 0, 'balanced_error': 1.0, 'worst_error': 1.0})
            continue

        accepted_mask = torch.zeros_like(labels, dtype=torch.bool)
        accepted_mask[sorted_indices[:num_to_accept]] = True
        
        metrics = calculate_selective_errors(preds, labels, accepted_mask, class_to_group, num_groups)
        rc_data.append({
            'coverage': metrics['coverage'], 
            'balanced_error': metrics['balanced_error'], 
            'worst_error': metrics['worst_error']
        })

    return pd.DataFrame(rc_data)

def generate_rc_curve_from_02(margins, preds, labels, class_to_group, num_groups, num_points=81):
    """
    Generates data points for the Risk-Coverage curve from coverage 0.2 to 1.0.
    More focused on the practical selective classification range.
    
    Args:
        margins: Confidence margins for each sample
        preds: Predicted labels  
        labels: True labels
        class_to_group: Mapping from class to group
        num_groups: Number of groups
        num_points: Number of points to sample (default 81 for 0.2 to 1.0 with 0.01 step)
    
    Returns:
        DataFrame with coverage, balanced_error, worst_error columns
    """
    # Sort samples by margin, descending. Higher margin = higher confidence.
    sorted_indices = torch.argsort(margins, descending=True)

    rc_data = []
    total_samples = len(labels)
    
    # Sweep through coverage levels from 0.2 to 1.0
    coverage_min = 0.2
    coverage_max = 1.0
    
    for i in range(num_points):
        coverage_target = coverage_min + (coverage_max - coverage_min) * i / (num_points - 1)
        num_to_accept = int(total_samples * coverage_target)
        
        if num_to_accept == 0:
            rc_data.append({'coverage': coverage_target, 'balanced_error': 1.0, 'worst_error': 1.0})
            continue

        # Ensure we don't exceed total samples
        num_to_accept = min(num_to_accept, total_samples)
        
        accepted_mask = torch.zeros_like(labels, dtype=torch.bool)
        accepted_mask[sorted_indices[:num_to_accept]] = True
        
        metrics = calculate_selective_errors(preds, labels, accepted_mask, class_to_group, num_groups)
        rc_data.append({
            'coverage': metrics['coverage'], 
            'balanced_error': metrics['balanced_error'], 
            'worst_error': metrics['worst_error']
        })

    return pd.DataFrame(rc_data)

def calculate_aurc(rc_dataframe, risk_key='balanced_error'):
    """Calculates the Area Under the Risk-Coverage Curve."""
    if rc_dataframe.empty:
        return 1.0
    
    coverages = rc_dataframe['coverage'].values
    risks = rc_dataframe[risk_key].values
    
    # Use trapezoidal rule for integration
    return np.trapz(risks, coverages)

def calculate_aurc_from_02(rc_dataframe, risk_key='balanced_error'):
    """
    Calculates the Area Under the Risk-Coverage Curve from coverage 0.2 to 1.0.
    This is more focused on the practical selective classification range.
    
    Args:
        rc_dataframe: DataFrame with coverage and risk columns
        risk_key: Which risk metric to use ('balanced_error' or 'worst_error')
    
    Returns:
        AURC value normalized by the coverage range (0.8)
    """
    if rc_dataframe.empty:
        return 1.0
    
    # Filter for coverage >= 0.2
    filtered_df = rc_dataframe[rc_dataframe['coverage'] >= 0.2].copy()
    
    if filtered_df.empty:
        return 1.0
    
    coverages = filtered_df['coverage'].values
    risks = filtered_df[risk_key].values
    
    # Use trapezoidal rule for integration
    aurc_raw = np.trapz(risks, coverages)
    
    # Normalize by coverage range (1.0 - 0.2 = 0.8) to make it comparable
    coverage_range = coverages.max() - coverages.min()
    if coverage_range > 0:
        aurc_normalized = aurc_raw / coverage_range
    else:
        aurc_normalized = risks[0] if len(risks) > 0 else 1.0
    
    return aurc_normalized