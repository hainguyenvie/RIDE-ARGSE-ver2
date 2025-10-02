#!/usr/bin/env python3
"""
Test script to verify data creation follows paper methodology correctly.
"""

import numpy as np
from src.data.datasets import get_cifar100_lt_counts

def test_exponential_profile():
    """Test the exponential profile implementation."""
    print("=== Testing Exponential Profile ===")
    
    # Test with IF=100 (standard)
    counts = get_cifar100_lt_counts(imb_factor=100, num_classes=100)
    
    print(f"Head class (class 0): {counts[0]} samples")
    print(f"Tail class (class 99): {counts[-1]} samples")
    print(f"Actual imbalance factor: {counts[0] / counts[-1]:.1f}")
    print("Expected: 100.0")
    
    # Verify exponential decay
    expected_tail = 500 * (100 ** (-99/99))  # Should be 500/100 = 5
    print(f"Expected tail count: {expected_tail:.1f}")
    print(f"Actual tail count: {counts[-1]}")
    
    # Test few middle classes
    for i in [25, 50, 75]:
        expected = 500 * (100 ** (-i/99))
        actual = counts[i]
        print(f"Class {i}: expected={expected:.1f}, actual={actual}")

def test_proportions_matching():
    """Test that val/test proportions match train proportions."""
    print("\n=== Testing Proportions Matching ===")
    
    # Simulate train-LT counts
    train_counts = get_cifar100_lt_counts(imb_factor=100)
    total_train = sum(train_counts)
    train_props = [count / total_train for count in train_counts]
    
    print(f"Train-LT total samples: {total_train}")
    print(f"Train head proportion: {train_props[0]:.4f}")
    print(f"Train tail proportion: {train_props[-1]:.4f}")
    
    # Simulate test-LT creation
    original_test_size = 10000  # CIFAR-100 test
    target_test_counts = []
    
    for i in range(100):
        target_count = int(round(train_props[i] * original_test_size))
        target_count = min(target_count, 100)  # CIFAR test limit
        target_test_counts.append(target_count)
    
    total_test_lt = sum(target_test_counts)
    test_props = [count / total_test_lt for count in target_test_counts]
    
    print(f"Test-LT total samples: {total_test_lt}")
    print(f"Test head proportion: {test_props[0]:.4f}")
    print(f"Test tail proportion: {test_props[-1]:.4f}")
    
    # Check how well proportions match
    prop_diff = [abs(train_props[i] - test_props[i]) for i in range(100)]
    print(f"Max proportion difference: {max(prop_diff):.4f}")
    print(f"Mean proportion difference: {np.mean(prop_diff):.4f}")

if __name__ == "__main__":
    test_exponential_profile()
    test_proportions_matching()