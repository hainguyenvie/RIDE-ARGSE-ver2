# src/train/eval_gse_plugin.py
"""
Evaluation script for GSE-Balanced plugin results.
Loads optimal (α*, μ*) and evaluates on test set.
"""
import torch
import torchvision
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom modules
from src.models.argse import AR_GSE
from src.metrics.selective_metrics import calculate_selective_errors
from src.metrics.rc_curve import generate_rc_curve, generate_rc_curve_from_02, calculate_aurc, calculate_aurc_from_02
from src.metrics.calibration import calculate_ece
from src.metrics.bootstrap import bootstrap_ci

# Import plugin functions
from src.train.gse_balanced_plugin import compute_margin

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'grouping': {
        'threshold': 20,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits',
    },
    'eval_params': {
        'coverage_points': [0.7, 0.8, 0.9],
        'bootstrap_n': 1000,
    },
    'plugin_checkpoint': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
    'output_dir': './results_worst_eg_improved/cifar100_lt_if100',
    'seed': 42
}

def load_test_data():
    """Load test logits and labels."""
    logits_root = Path(CONFIG['experts']['logits_dir']) / CONFIG['dataset']['name']
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    
    with open(splits_dir / 'test_lt_indices.json', 'r') as f:
        test_indices = json.load(f)
    num_test_samples = len(test_indices)
    
    # Load expert logits for test set
    num_experts = len(CONFIG['experts']['names'])
    stacked_logits = torch.zeros(num_test_samples, num_experts, CONFIG['dataset']['num_classes'])
    
    for i, expert_name in enumerate(CONFIG['experts']['names']):
        logits_path = logits_root / expert_name / "test_lt_logits.pt"
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits file not found: {logits_path}")
        stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu', weights_only=False)
    
    # Load test labels
    full_test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    test_labels = torch.tensor(np.array(full_test_dataset.targets)[test_indices])
    
    return stacked_logits, test_labels

def get_mixture_posteriors(model, logits):
    """Get mixture posteriors η̃(x) from expert logits."""
    model.eval()
    with torch.no_grad():
        logits = logits.to(DEVICE)
        
        # Get expert posteriors
        expert_posteriors = torch.softmax(logits, dim=-1)  # [B, E, C]
        
        # Get gating weights
        gating_features = model.feature_builder(logits)
        gating_weights = torch.softmax(model.gating_net(gating_features), dim=1)  # [B, E]
        
        # Mixture posteriors
        eta_mix = torch.einsum('be,bec->bc', gating_weights, expert_posteriors)  # [B, C]
        
    return eta_mix.cpu()

def analyze_group_performance(eta_mix, preds, labels, accepted, alpha, mu, threshold, class_to_group, K):
    """
    Analyze detailed per-group performance metrics.
    """
    print("\n" + "="*50)
    print("DETAILED GROUP-WISE ANALYSIS")
    print("="*50)
    
    # Ensure all tensors are on same device
    device = eta_mix.device
    class_to_group = class_to_group.to(device)
    y_groups = class_to_group[labels]
    
    for k in range(K):
        group_name = "Head" if k == 0 else "Tail"
        group_mask = (y_groups == k)
        group_size = group_mask.sum().item()
        
        if group_size == 0:
            continue
            
        # Coverage and error for this group
        group_accepted = accepted[group_mask]
        group_coverage = group_accepted.float().mean().item()
        
        # TPR/FPR analysis for this group
        group_preds_all = preds[group_mask]
        group_labels_all = labels[group_mask]
        group_correct = (group_preds_all == group_labels_all)
        
        # True Positive Rate (TPR): fraction of correct predictions that are accepted
        correct_accepted = group_accepted & group_correct
        tpr = correct_accepted.sum().item() / group_correct.sum().item() if group_correct.sum() > 0 else 0.0
        
        # False Positive Rate (FPR): fraction of incorrect predictions that are accepted  
        incorrect_accepted = group_accepted & (~group_correct)
        fpr = incorrect_accepted.sum().item() / (~group_correct).sum().item() if (~group_correct).sum() > 0 else 0.0
        
        if group_accepted.sum() > 0:
            group_preds = preds[group_mask & accepted]
            group_labels = labels[group_mask & accepted]
            group_accuracy = (group_preds == group_labels).float().mean().item()
            group_error = 1.0 - group_accuracy
        else:
            group_error = 1.0
            
        # Raw margin statistics for this group
        raw_margins = compute_margin(eta_mix[group_mask], alpha, mu, 0.0, class_to_group)
        margin_mean = raw_margins.mean().item()
        margin_std = raw_margins.std().item()
        margin_min = raw_margins.min().item()
        margin_max = raw_margins.max().item()
        
        print(f"\n{group_name} Group (k={k}):")
        print(f"  • Size: {group_size} samples")
        print(f"  • Coverage: {group_coverage:.3f}")
        print(f"  • Error: {group_error:.3f}")
        print(f"  • TPR (correct accepted): {tpr:.3f}")
        print(f"  • FPR (incorrect accepted): {fpr:.3f}")
        print(f"  • α_k: {alpha[k]:.3f}")
        print(f"  • μ_k: {mu[k]:.3f}")
        # Show the threshold for this group
        group_threshold_val = threshold[k] if isinstance(threshold, (list, torch.Tensor)) and len(threshold) > k else threshold
        print(f"  • τ_k: {group_threshold_val:.3f}")
        print(f"  • Raw margin stats: μ={margin_mean:.3f}, σ={margin_std:.3f}, range=[{margin_min:.3f}, {margin_max:.3f}]")
        
        # Check separation quality
        accepted_margins = raw_margins[group_accepted]
        rejected_margins = raw_margins[~group_accepted]
        
        if len(accepted_margins) > 0 and len(rejected_margins) > 0:
            separation = accepted_margins.min().item() - rejected_margins.max().item()
            # Use the appropriate threshold for this group
            group_threshold = threshold[k] if isinstance(threshold, (list, torch.Tensor)) and len(threshold) > k else threshold
            overlap_ratio = (rejected_margins > group_threshold).sum().item() / len(rejected_margins)
            print(f"  • Margin separation: {separation:.3f}")
            print(f"  • Overlap ratio: {overlap_ratio:.3f}")
    
    print("\n" + "="*50)


def selective_risk_from_mask(preds, labels, accepted_mask, c_cost, class_to_group, K, kind="balanced"):
    """
    Compute selective risk with rejection cost c_cost.
    Risk = error_rate * coverage + c_cost * (1 - coverage) for each group
    """
    y = labels
    g = class_to_group[y]
    
    if kind == "balanced":
        vals = []
        for k in range(K):
            mk = (g == k)
            if mk.sum() == 0:
                vals.append(c_cost)  # No samples in group k
                continue
                
            acc_k = accepted_mask[mk].float().mean().item()
            
            if accepted_mask[mk].sum() == 0:
                err_k = 1.0  # No accepted samples, assume error = 1.0
            else:
                err_k = (preds[mk & accepted_mask] != y[mk & accepted_mask]).float().mean().item()
            
            risk_k = err_k * acc_k + c_cost * (1.0 - acc_k)
            vals.append(risk_k)
        return float(np.mean(vals))
    else:  # worst-group risk
        worst = 0.0
        for k in range(K):
            mk = (g == k)
            if mk.sum() == 0:
                worst = max(worst, c_cost)
                continue
                
            acc_k = accepted_mask[mk].float().mean().item()
            
            if accepted_mask[mk].sum() == 0:
                err_k = 1.0
            else:
                err_k = (preds[mk & accepted_mask] != y[mk & accepted_mask]).float().mean().item()
            
            risk_k = err_k * acc_k + c_cost * (1.0 - acc_k)
            worst = max(worst, risk_k)
        return worst

def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("=== GSE-Balanced Plugin Evaluation ===")
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load plugin checkpoint
    plugin_ckpt_path = Path(CONFIG['plugin_checkpoint'])
    if not plugin_ckpt_path.exists():
        raise FileNotFoundError(f"Plugin checkpoint not found: {plugin_ckpt_path}")
    
    print(f"📂 Loading plugin checkpoint: {plugin_ckpt_path}")
    checkpoint = torch.load(plugin_ckpt_path, map_location=DEVICE, weights_only=False)
    
    alpha_star = checkpoint['alpha'].to(DEVICE)
    mu_star = checkpoint['mu'].to(DEVICE)
    class_to_group = checkpoint['class_to_group'].to(DEVICE)
    num_groups = checkpoint['num_groups']
    plugin_threshold = checkpoint.get('threshold', checkpoint.get('c'))  # Backward compatibility
    
    print("✅ Loaded optimal parameters:")
    print(f"   α* = [{alpha_star[0]:.4f}, {alpha_star[1]:.4f}]")
    print(f"   μ* = [{mu_star[0]:.4f}, {mu_star[1]:.4f}]")
    
    # Handle both single threshold and per-group thresholds
    if isinstance(plugin_threshold, list):
        print(f"   per-group thresholds t* = {plugin_threshold}")
    else:
        print(f"   raw-margin threshold t* = {plugin_threshold:.3f}")
    if 'best_score' in checkpoint:
        print(f"   Best S2 score = {checkpoint['best_score']:.4f}")
    if 'source' in checkpoint:
        print(f"   Source: {checkpoint['source']}")
    if 'improvement' in checkpoint:
        print(f"   Expected improvement: {checkpoint['improvement']:.1f}%")
    
    # 2. Set up model with optimal parameters
    num_experts = len(CONFIG['experts']['names'])
    
    # Compute dynamic gating feature dimension (enriched features)
    with torch.no_grad():
        dummy_logits = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        temp_model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, 1).to(DEVICE)
        gating_feature_dim = temp_model.feature_builder(dummy_logits).size(-1)
        del temp_model
    print(f"✅ Dynamic gating feature dim: {gating_feature_dim}")
    
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    
    # Load gating network weights with dimension compatibility check
    if 'gating_net_state_dict' in checkpoint:
        saved_state = checkpoint['gating_net_state_dict']
        current_state = model.gating_net.state_dict()
        
        compatible = True
        for key in saved_state.keys():
            if key in current_state and saved_state[key].shape != current_state[key].shape:
                print(f"⚠️  Dimension mismatch for {key}: saved {saved_state[key].shape} vs current {current_state[key].shape}")
                compatible = False
        
        if compatible:
            model.gating_net.load_state_dict(saved_state)
            print("✅ Gating network weights loaded successfully")
        else:
            print("❌ Gating checkpoint incompatible with enriched features. Using random weights.")
    else:
        print("⚠️ No gating network weights found in checkpoint")
    
    # Set optimal α*, μ*
    with torch.no_grad():
        model.alpha.copy_(alpha_star)
        model.mu.copy_(mu_star)
    
    print("✅ Model configured with optimal parameters")
    
    # 3. Load test data
    print("📊 Loading test data...")
    test_logits, test_labels = load_test_data()
    num_test_samples = len(test_labels)
    print(f"✅ Loaded {num_test_samples} test samples")
    
    # 4. Get test predictions
    print("🔮 Computing test predictions...")
    eta_mix = get_mixture_posteriors(model, test_logits)
    
    # Ensure all tensors are on CPU for consistent computation
    alpha_star_cpu = alpha_star.cpu()
    mu_star_cpu = mu_star.cpu()
    class_to_group_cpu = class_to_group.cpu()
    
    # Compute raw margins and predictions
    margins_raw = compute_margin(eta_mix, alpha_star_cpu, mu_star_cpu, 0.0, class_to_group_cpu)
    preds = (alpha_star_cpu[class_to_group_cpu] * eta_mix).argmax(dim=1)
    
    # 🔧 FIXED: Use per-group thresholds with GROUND-TRUTH groups (not predicted groups)
    t_group = checkpoint.get('t_group', None)
    if t_group is not None:
        # Convert to tensor if it's a list and keep original list for display
        if isinstance(t_group, list):
            t_group_list = t_group
            t_group = torch.tensor(t_group)
        else:
            t_group_list = t_group.tolist()
        
        # Use GROUND-TRUTH groups (class_to_group[true_labels]) instead of prediction groups  
        y_groups = class_to_group_cpu[test_labels]  # Ground-truth groups
        
        # Per-sample threshold based on ground-truth group
        thresholds_per_sample = torch.tensor([t_group[g].item() for g in y_groups])
        accepted = margins_raw >= thresholds_per_sample
        
        print(f"✅ Using per-group thresholds with GROUND-TRUTH groups: {t_group_list}")
        print(f"✅ Test coverage: {accepted.float().mean():.3f}")
        
        # Per-group coverage breakdown
        for k in range(len(t_group_list)):
            group_mask = (y_groups == k)
            if group_mask.sum() > 0:
                group_cov = accepted[group_mask].float().mean().item()
                group_name = "head" if k == 0 else "tail" 
                print(f"   📊 {group_name} (group {k}): coverage={group_cov:.3f}, threshold={t_group_list[k]:.3f}")
            
    else:
        accepted = margins_raw >= plugin_threshold
        print(f"✅ Using global threshold: {plugin_threshold:.3f}")
        print(f"✅ Test coverage: {accepted.float().mean():.3f}")
    
    # 5. Calculate metrics
    print("📈 Calculating metrics...")
    results = {}
    
    # 5.1 RC Curve and AURC (using raw margins for fair comparison)
    rc_df = generate_rc_curve(margins_raw, preds, test_labels, class_to_group_cpu, num_groups)
    rc_df.to_csv(output_dir / 'rc_curve.csv', index=False)
    
    aurc_bal = calculate_aurc(rc_df, 'balanced_error')
    aurc_wst = calculate_aurc(rc_df, 'worst_error')
    results['aurc_balanced'] = aurc_bal
    results['aurc_worst'] = aurc_wst
    print(f"AURC (Balanced): {aurc_bal:.4f}, AURC (Worst): {aurc_wst:.4f}")
    
    # 5.2 RC Curve 0.2-1.0 range
    rc_df_02 = generate_rc_curve_from_02(margins_raw, preds, test_labels, class_to_group_cpu, num_groups)
    rc_df_02.to_csv(output_dir / 'rc_curve_02_10.csv', index=False)
    
    aurc_bal_02 = calculate_aurc_from_02(rc_df_02, 'balanced_error')
    aurc_wst_02 = calculate_aurc_from_02(rc_df_02, 'worst_error')
    results['aurc_balanced_02_10'] = aurc_bal_02
    results['aurc_worst_02_10'] = aurc_wst_02
    print(f"AURC 0.2-1.0 (Balanced): {aurc_bal_02:.4f}, AURC 0.2-1.0 (Worst): {aurc_wst_02:.4f}")
    
    # 5.3 Bootstrap CI for AURC
    def aurc_metric_func(m, p, labels):
        rc_df_boot = generate_rc_curve(m, p, labels, class_to_group_cpu, num_groups, num_points=51)
        return calculate_aurc(rc_df_boot, 'balanced_error')

    mean_aurc, lower, upper = bootstrap_ci(
        (margins_raw, preds, test_labels), aurc_metric_func, n_bootstraps=CONFIG['eval_params']['bootstrap_n']
    )
    results['aurc_balanced_bootstrap'] = {'mean': mean_aurc, '95ci_lower': lower, '95ci_upper': upper}
    print(f"AURC Bootstrap 95% CI: [{lower:.4f}, {upper:.4f}]")
    
    # 5.4 Metrics at fixed coverage points (using raw margins)
    results_at_coverage = {}
    for cov_target in CONFIG['eval_params']['coverage_points']:
        thr_cov = torch.quantile(margins_raw, 1.0 - cov_target)
        accepted_mask = margins_raw >= thr_cov   # >= giúp bền vững khi có ties
        
        metrics = calculate_selective_errors(preds, test_labels, accepted_mask, class_to_group_cpu, num_groups)
        results_at_coverage[f'cov_{cov_target}'] = metrics
        print(f"Metrics @ {metrics['coverage']:.2f} coverage: "
              f"Bal.Err={metrics['balanced_error']:.4f}, Worst.Err={metrics['worst_error']:.4f}")
    results['metrics_at_coverage'] = results_at_coverage
    
    # 5.5 Plugin-specific metrics (at optimal threshold)
    plugin_metrics = calculate_selective_errors(preds, test_labels, accepted, class_to_group_cpu, num_groups)
    results['plugin_metrics_at_threshold'] = plugin_metrics
    
    # Better messaging based on threshold type used
    t_group = checkpoint.get('t_group', None)
    if t_group is not None:
        # Convert to tensor if it's a list
        if isinstance(t_group, list):
            t_group_list = t_group
            t_group = torch.tensor(t_group)
        else:
            t_group_list = t_group.tolist()
            
        print(f"Plugin metrics @ per-group thresholds {[f'{t:.3f}' for t in t_group_list]}: "
              f"Coverage={plugin_metrics['coverage']:.3f}, "
              f"Bal.Err={plugin_metrics['balanced_error']:.4f}, Worst.Err={plugin_metrics['worst_error']:.4f}")
    else:
        print(f"Plugin metrics @ global threshold t*={plugin_threshold:.3f}: "
              f"Coverage={plugin_metrics['coverage']:.3f}, "
              f"Bal.Err={plugin_metrics['balanced_error']:.4f}, Worst.Err={plugin_metrics['worst_error']:.4f}")
    
    # 5.5a Detailed Group-wise Analysis
    threshold_param = t_group if t_group is not None else plugin_threshold
    analyze_group_performance(eta_mix, preds, test_labels, accepted,
                             alpha_star_cpu, mu_star_cpu, threshold_param, class_to_group_cpu, num_groups)
    
    # 5.6 ECE
    ece = calculate_ece(eta_mix, test_labels)
    results['ece'] = ece
    print(f"ECE: {ece:.4f}")
    
    # 5.7 Selective risk with different rejection costs (for comparison)
    print("\nSelective Risk with different rejection costs:")
    selective_risks = {}
    for c_cost in [0.3, 0.5, 0.7]:
        bal_risk = selective_risk_from_mask(preds, test_labels, accepted, c_cost,
                                            class_to_group_cpu, num_groups, "balanced")
        worst_risk = selective_risk_from_mask(preds, test_labels, accepted, c_cost,
                                              class_to_group_cpu, num_groups, "worst")
        selective_risks[f'cost_{c_cost}'] = {'balanced': bal_risk, 'worst': worst_risk}
        print(f"  Cost c={c_cost:.2f}: Balanced={bal_risk:.4f}, Worst={worst_risk:.4f}")
    results['selective_risks'] = selective_risks
    
    # 6. Save results
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"💾 Saved metrics to {output_dir / 'metrics.json'}")
    
    # 7. Plot RC curves
    plt.figure(figsize=(12, 5))
    
    # Full range
    plt.subplot(1, 2, 1)
    plt.plot(rc_df['coverage'], rc_df['balanced_error'], label='Balanced Error', linewidth=2)
    plt.plot(rc_df['coverage'], rc_df['worst_error'], label='Worst-Group Error', linestyle='--', linewidth=2)
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error)')
    plt.title('GSE-Balanced Plugin RC Curve (Full Range)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Focused range
    plt.subplot(1, 2, 2)
    plt.plot(rc_df_02['coverage'], rc_df_02['balanced_error'], label='Balanced Error', linewidth=2)
    plt.plot(rc_df_02['coverage'], rc_df_02['worst_error'], label='Worst-Group Error', linestyle='--', linewidth=2)
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error)')
    plt.title('GSE-Balanced Plugin RC Curve (0.2-1.0)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rc_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'rc_curve_comparison.pdf', bbox_inches='tight')
    print(f"📊 Saved RC curve plots to {output_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("GSE-BALANCED PLUGIN EVALUATION SUMMARY")
    print("="*60)
    print(f"Dataset: {CONFIG['dataset']['name']}")
    print(f"Test samples: {num_test_samples}")
    print(f"Optimal parameters: α*={alpha_star.cpu().tolist()}, μ*={mu_star.cpu().tolist()}")
    
    # Handle threshold display
    t_group = checkpoint.get('t_group', None)
    if t_group is not None:
        if isinstance(t_group, list):
            t_group_display = [f'{t:.3f}' for t in t_group]
        else:
            t_group_display = [f'{t:.3f}' for t in t_group.tolist()]
        print(f"Per-group thresholds (fitted on S1): t* = {t_group_display}")
    else:
        print(f"Raw-margin threshold (fitted on S1): t* = {plugin_threshold:.3f}")
    
    print()
    print("Key Results:")
    print(f"• AURC (Balanced): {aurc_bal:.4f}")
    print(f"• AURC (Worst): {aurc_wst:.4f}") 
    
    if t_group is not None:
        print(f"• Plugin @ per-group thresholds: Coverage={plugin_metrics['coverage']:.3f}, "
              f"Bal.Err={plugin_metrics['balanced_error']:.4f}")
    else:
        print(f"• Plugin @ t*={plugin_threshold:.3f}: Coverage={plugin_metrics['coverage']:.3f}, "
              f"Bal.Err={plugin_metrics['balanced_error']:.4f}")
    
    print(f"• ECE: {ece:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()