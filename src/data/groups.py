# src/data/groups.py
import torch
import numpy as np
from typing import List

def get_class_to_group_by_threshold(
    class_counts: List[int], 
    threshold: int = 20
) -> torch.LongTensor:
    """
    Maps class indices to groups based on a sample count threshold.
    Classes with sample count > threshold are assigned to group 0 (head),
    classes with sample count <= threshold are assigned to group 1 (tail).

    Args:
        class_counts: A list of sample counts for each class.
        threshold: The sample count threshold for head/tail division.

    Returns:
        A LongTensor of shape [C] where each element is the group index for that class.
        Group 0: Head classes (count > threshold)
        Group 1: Tail classes (count <= threshold)
    """
    num_classes = len(class_counts)
    class_to_group = torch.zeros(num_classes, dtype=torch.long)
    
    for class_idx, count in enumerate(class_counts):
        if count > threshold:
            class_to_group[class_idx] = 0  # Head group
        else:
            class_to_group[class_idx] = 1  # Tail group
            
    return class_to_group

def get_class_to_group(
    class_counts: List[int], 
    K: int = 2, 
    head_ratio: float = 0.5
) -> torch.LongTensor:
    """
    Maps class indices to group indices based on class frequency.

    Args:
        class_counts: A list of sample counts for each class.
        K: Number of groups. For K=2, it's head/tail.
        head_ratio: The proportion of classes to be considered 'head' classes.
                    Only used when K=2.

    Returns:
        A LongTensor of shape [C] where each element is the group index for that class.
    """
    num_classes = len(class_counts)
    class_to_group = torch.zeros(num_classes, dtype=torch.long)
    
    sorted_indices = np.argsort(-np.array(class_counts)) # Sort descending

    if K == 2:
        num_head = int(num_classes * head_ratio)
        head_classes = sorted_indices[:num_head]
        # Group 0: Head, Group 1: Tail
        class_to_group[head_classes] = 0
        tail_classes = sorted_indices[num_head:]
        class_to_group[tail_classes] = 1
    else:
        # General case for K > 2 using quantiles
        group_size = num_classes // K
        for i in range(K):
            start = i * group_size
            end = (i + 1) * group_size if i < K - 1 else num_classes
            group_classes = sorted_indices[start:end]
            class_to_group[group_classes] = i
            
    return class_to_group