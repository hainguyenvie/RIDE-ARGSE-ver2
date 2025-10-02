# src/metrics/selective_metrics.py
import torch
import numpy as np

def calculate_selective_errors(preds, labels, accepted_mask, class_to_group, num_groups):
    """
    Calculates various selective errors on the accepted subset of data.
    """
    num_accepted = accepted_mask.sum().item()
    total_samples = len(labels)
    coverage = num_accepted / total_samples if total_samples > 0 else 0

    if num_accepted == 0:
        return {
            'coverage': 0,
            'balanced_error': 1.0,
            'worst_error': 1.0,
            'group_errors': [1.0] * num_groups,
            'overall_error': 1.0,
        }

    accepted_preds = preds[accepted_mask]
    accepted_labels = labels[accepted_mask]
    
    accepted_groups = class_to_group[accepted_labels]
    group_errors = []
    for k in range(num_groups):
        group_mask = (accepted_groups == k)
        if group_mask.sum() == 0:
            # If a group has no accepted samples, its conditional error is undefined.
            # We can assign max error (1.0) or NaN. Max error is common for worst-case analysis.
            group_errors.append(1.0)
            continue
        
        correct_in_group = (accepted_preds[group_mask] == accepted_labels[group_mask]).sum().item()
        total_in_group = group_mask.sum().item()
        accuracy_in_group = correct_in_group / total_in_group
        group_errors.append(1.0 - accuracy_in_group)

    overall_error = 1.0 - (accepted_preds == accepted_labels).sum().item() / num_accepted
    
    return {
        'coverage': coverage,
        'balanced_error': np.mean(group_errors),
        'worst_error': np.max(group_errors),
        'group_errors': group_errors,
        'overall_error': overall_error,
    }