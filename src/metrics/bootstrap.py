# src/metrics/bootstrap.py
import numpy as np

def bootstrap_ci(data, metric_func, n_bootstraps=1000, ci=0.95):
    """
    Computes bootstrap confidence interval for a given metric.
    
    Args:
        data: A tuple/list of arrays (e.g., (margins, preds, labels))
        metric_func: A function that takes the data arrays and returns a scalar metric.
    """
    n_samples = len(data[0])
    bootstrap_scores = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Create bootstrap sample
        resampled_data = [d[indices] for d in data]
        
        score = metric_func(*resampled_data)
        bootstrap_scores.append(score)
        
    lower_bound = np.percentile(bootstrap_scores, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrap_scores, (1 + ci) / 2 * 100)
    
    return np.mean(bootstrap_scores), lower_bound, upper_bound