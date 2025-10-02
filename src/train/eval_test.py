import torch
import torchvision
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Import our custom modules
from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group_by_threshold
from src.data.datasets import get_cifar100_lt_counts
from src.metrics.selective_metrics import calculate_selective_errors
from src.metrics.rc_curve import generate_rc_curve, generate_rc_curve_from_02, calculate_aurc, calculate_aurc_from_02
from src.metrics.calibration import calculate_ece
from src.metrics.bootstrap import bootstrap_ci

# --- CONFIGURATION (sẽ được thay thế bằng Hydra) ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100', ## CẬP NHẬT: Tên dataset nhất quán
        'splits_dir': './data/cifar100_lt_if100_splits', ## CẬP NHẬT: Đường dẫn tới split
        'num_classes': 100,
    },
    'model_name': 'argse_worst', # 'argse_worst' or 'argse_balanced' ## CẬP NHẬT: Tên model
    'checkpoint_path': './checkpoints/argse_worst/cifar100_lt_if100/argse_worst.ckpt', ## CẬP NHẬT: Đường dẫn checkpoint
    'experts': {
        # Cập nhật tên expert cho khớp với M2 mới
        'names': ['ce_baseline','logitadjust_baseline', 'balsoftmax_baseline'], #, 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits',
    },
    'eval_params': {
        'coverage_points': [0.7, 0.8, 0.9],
        'bootstrap_n': 1000,
        'bootstrap_ci': 0.95,
    },
    'output_dir': './results_worst/cifar100_lt_if100', ## CẬP NHẬT: Đường dẫn output
    'seed': 42
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- MAIN EVALUATION SCRIPT ---
def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    output_dir = Path(CONFIG['output_dir']) / CONFIG['model_name']
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Group Info
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    ## CẬP NHẬT: Sử dụng hàm get_class_to_group_by_threshold với threshold thay vì head_ratio
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=20)
    num_groups = class_to_group.max().item() + 1
    
    # 2. Load Model
    num_experts = len(CONFIG['experts']['names'])
    gating_feature_dim = 4 * num_experts
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    model.load_state_dict(torch.load(CONFIG['checkpoint_path'], map_location=DEVICE))
    model.eval()
    print(f"Loaded model from {CONFIG['checkpoint_path']}")

    # 3. Load Test Data (Logits and Labels for the LONG-TAIL test set)
    print("Loading long-tail test logits and labels...")
    logits_root = Path(CONFIG['experts']['logits_dir']) / CONFIG['dataset']['name']
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    
    with open(splits_dir / 'test_lt_indices.json', 'r') as f:
        test_indices = json.load(f)
    num_test_samples = len(test_indices)

    stacked_logits = torch.zeros(num_test_samples, num_experts, CONFIG['dataset']['num_classes'])
    for i, expert_name in enumerate(CONFIG['experts']['names']):
        logits_path = logits_root / expert_name / "test_lt_logits.pt"
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits file not found for expert '{expert_name}': {logits_path}. Please run M2 expert training first.")
        stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu')
    
    full_test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    test_labels = torch.tensor(np.array(full_test_dataset.targets)[test_indices])

    print(f"Successfully loaded {num_test_samples} samples for long-tail test set.")

    # 4. Get Model Predictions on Test Set
    with torch.no_grad():
        # Chuyển class_to_group sang GPU chỉ cho bước forward này
        outputs = model(stacked_logits.to(DEVICE), c=0.7, tau=10.0, class_to_group=class_to_group.to(DEVICE))
    
    margins_point = outputs['margin'].cpu()        # margin with c subtracted (for point metrics)
    margins_raw = outputs['raw_margin'].cpu()      # raw margin without c (for RC curve)
    eta_mix = outputs['eta_mix'].cpu()
    
    # Use re-weighted prediction: h_α(x) = argmax_y α_{grp(y)} * η̃_y(x)
    alpha = model.alpha.cpu()
    class_to_group_cpu = class_to_group.cpu()
    reweighted_scores = alpha[class_to_group_cpu] * eta_mix  # [N, C]
    _, preds = torch.max(reweighted_scores, 1)

    # 5. Calculate All Metrics (class_to_group ở đây là phiên bản CPU, sẽ không có lỗi)
    results = {}
    print("\nCalculating metrics...")

    # 5.1 RC Curve and AURC (Full range 0.0-1.0) - Using RAW MARGIN for fair comparison with paper
    rc_df = generate_rc_curve(margins_raw, preds, test_labels, class_to_group, num_groups)
    rc_df.to_csv(output_dir / 'rc_curve.csv', index=False)
    print(f"Saved RC curve data to {output_dir / 'rc_curve.csv'}")
    
    aurc_bal = calculate_aurc(rc_df, 'balanced_error')
    aurc_wst = calculate_aurc(rc_df, 'worst_error')
    results['aurc_balanced'] = aurc_bal
    results['aurc_worst'] = aurc_wst
    print(f"AURC (Balanced): {aurc_bal:.4f}, AURC (Worst): {aurc_wst:.4f}")

    # 5.1.1 RC Curve and AURC (Focused range 0.2-1.0) - Using RAW MARGIN
    rc_df_02 = generate_rc_curve_from_02(margins_raw, preds, test_labels, class_to_group, num_groups)
    rc_df_02.to_csv(output_dir / 'rc_curve_02_10.csv', index=False)
    print(f"Saved RC curve data (0.2-1.0) to {output_dir / 'rc_curve_02_10.csv'}")
    
    aurc_bal_02 = calculate_aurc_from_02(rc_df_02, 'balanced_error')
    aurc_wst_02 = calculate_aurc_from_02(rc_df_02, 'worst_error')
    results['aurc_balanced_02_10'] = aurc_bal_02
    results['aurc_worst_02_10'] = aurc_wst_02
    print(f"AURC 0.2-1.0 (Balanced): {aurc_bal_02:.4f}, AURC 0.2-1.0 (Worst): {aurc_wst_02:.4f}")

    # 5.2 Bootstrap CI for AURC (Balanced, Full range) - Using RAW MARGIN
    def aurc_metric_func(m, p, labels):
        rc_df_boot = generate_rc_curve(m, p, labels, class_to_group, num_groups, num_points=51)
        return calculate_aurc(rc_df_boot, 'balanced_error')

    mean_aurc, lower, upper = bootstrap_ci((margins_raw, preds, test_labels), aurc_metric_func, n_bootstraps=CONFIG['eval_params']['bootstrap_n'])
    results['aurc_balanced_bootstrap'] = {'mean': mean_aurc, '95ci_lower': lower, '95ci_upper': upper}
    print(f"AURC (Balanced) Bootstrap 95% CI: [{lower:.4f}, {upper:.4f}]")

    # 5.2.1 Bootstrap CI for AURC (Balanced, 0.2-1.0 range) - Using RAW MARGIN
    def aurc_metric_func_02(m, p, labels):
        rc_df_boot = generate_rc_curve_from_02(m, p, labels, class_to_group, num_groups, num_points=41)
        return calculate_aurc_from_02(rc_df_boot, 'balanced_error')

    mean_aurc_02, lower_02, upper_02 = bootstrap_ci((margins_raw, preds, test_labels), aurc_metric_func_02, n_bootstraps=CONFIG['eval_params']['bootstrap_n'])
    results['aurc_balanced_02_10_bootstrap'] = {'mean': mean_aurc_02, '95ci_lower': lower_02, '95ci_upper': upper_02}
    print(f"AURC 0.2-1.0 (Balanced) Bootstrap 95% CI: [{lower_02:.4f}, {upper_02:.4f}]")

    # 5.3 Metrics @ Fixed Coverage - Using RAW MARGIN for fair comparison
    results_at_coverage = {}
    for cov_target in CONFIG['eval_params']['coverage_points']:
        threshold = torch.quantile(margins_raw, 1.0 - cov_target)
        accepted_mask = margins_raw >= threshold
        
        metrics = calculate_selective_errors(preds, test_labels, accepted_mask, class_to_group, num_groups)
        results_at_coverage[f'cov_{cov_target}'] = metrics
        print(f"Metrics @ {metrics['coverage']:.2f} coverage: Bal. Err={metrics['balanced_error']:.4f}, Worst Err={metrics['worst_error']:.4f}")
    results['metrics_at_coverage'] = results_at_coverage

    # 5.4 Point metrics at training c (AR-GSE specific evaluation)
    accepted_mask_point = margins_point >= 0   # reject rule already includes c
    metrics_point = calculate_selective_errors(preds, test_labels, accepted_mask_point, class_to_group, num_groups)
    results['metrics_point_c_train'] = metrics_point
    print("\n=== AR-GSE Point Metrics @ c_train ===")
    print(f"Coverage={metrics_point['coverage']:.3f}, "
          f"Bal.Err={metrics_point['balanced_error']:.4f}, "
          f"Worst.Err={metrics_point['worst_error']:.4f}")

    # 5.5 Calibration (ECE)
    ece = calculate_ece(eta_mix, test_labels)
    results['ece'] = ece
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # 6. Save Results
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved all metrics to {output_dir / 'metrics.json'}")

    # 7. Plot and Save RC Curve (Full range 0.0-1.0)
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Full range (0.0-1.0)
    plt.subplot(1, 2, 1)
    plt.plot(rc_df['coverage'], rc_df['balanced_error'], label='Balanced Error')
    plt.plot(rc_df['coverage'], rc_df['worst_error'], label='Worst-Group Error', linestyle='--')
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error)')
    plt.title(f'Risk-Coverage Curve (Full Range)\n{CONFIG["model_name"]}')
    plt.grid(True, linestyle=':')
    plt.legend()
    
    # Plot 2: Focused range (0.2-1.0)
    plt.subplot(1, 2, 2)
    plt.plot(rc_df_02['coverage'], rc_df_02['balanced_error'], label='Balanced Error')
    plt.plot(rc_df_02['coverage'], rc_df_02['worst_error'], label='Worst-Group Error', linestyle='--')
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error)')
    plt.title(f'Risk-Coverage Curve (0.2-1.0)\n{CONFIG["model_name"]}')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.xlim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rc_curve_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved RC curve comparison plot to {output_dir / 'rc_curve_comparison.png'}")
    
    # Also save individual plots for backward compatibility
    plt.figure()
    plt.plot(rc_df['coverage'], rc_df['balanced_error'], label='Balanced Error')
    plt.plot(rc_df['coverage'], rc_df['worst_error'], label='Worst-Group Error', linestyle='--')
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error)')
    plt.title(f'Risk-Coverage Curve for {CONFIG["model_name"]}')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig(output_dir / 'rc_curve.png')
    print(f"Saved RC curve plot to {output_dir / 'rc_curve.png'}")
    
    # Save focused range plot separately
    plt.figure()
    plt.plot(rc_df_02['coverage'], rc_df_02['balanced_error'], label='Balanced Error')
    plt.plot(rc_df_02['coverage'], rc_df_02['worst_error'], label='Worst-Group Error', linestyle='--')
    plt.xlabel('Coverage')
    plt.ylabel('Selective Risk (Error)')
    plt.title(f'Risk-Coverage Curve (0.2-1.0) for {CONFIG["model_name"]}')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.xlim(0.2, 1.0)
    plt.savefig(output_dir / 'rc_curve_02_10.png')
    print(f"Saved RC curve (0.2-1.0) plot to {output_dir / 'rc_curve_02_10.png'}")
    
    # Summary of evaluation approach
    print("\n" + "="*60)
    print("EVALUATION SUMMARY:")
    print("="*60)
    print("1. RC Curve/AURC Metrics (for comparison with paper):")
    print("   - Uses RAW MARGIN (without c) to scan thresholds")
    print("   - Comparable to original paper evaluation protocol")
    print("   - Reports full range (0.0-1.0) and focused range (0.2-1.0)")
    print()
    print("2. Point Metrics @ c_train (AR-GSE specific):")
    print("   - Uses FINAL MARGIN (raw margin - c) with fixed threshold=0")
    print("   - Shows performance at the specific reject cost used in training")
    print("   - Highlights AR-GSE's ability to optimize for a target trade-off")
    print("="*60)

if __name__ == '__main__':
    main()