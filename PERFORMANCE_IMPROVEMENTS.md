# AR-GSE Selective Training - Performance Optimization Update

## Recent Improvements (v2.0)

This document summarizes the six key optimizations implemented to enhance selective risk performance, particularly for worst-group AURC.

### 1. **Enhanced Per-Group Threshold Fitting**
- **Change**: Modified `fit_per_group_thresholds()` to use **all samples** instead of correct-only
- **Rationale**: Correct-only thresholds cause systematic under-coverage on tail groups
- **Implementation**: `per_group_threshold.py` now fits quantiles on raw margins from all samples
- **τ_k values**: [0.55, 0.45] (head stricter, tail more lenient for fairness)

### 2. **Group Coverage Penalties & Tail Weighting**
- **Added**: λ_cov-g = 20.0 group coverage penalty: `Σ_k (E[s 1{y∈G_k}] - τ_k)²`
- **Added**: β_tail = 2.0 tail weighting in selective loss: `L_sel = -log(mixture_probs) * reweight`
- **Effect**: Enforces per-group coverage targets and emphasizes tail performance

### 3. **Directional Prior for α**
- **Added**: α prior bias u = [-1, +1] with strength ρ = 0.03 
- **Purpose**: Gently pushes α_tail > 1.0 to prevent tail drift
- **Implementation**: Applied after geometric mean normalization with re-projection

### 4. **Expanded μ Grid with EMA Smoothing**  
- **Expanded**: λ ∈ [-2.0, +2.0] with 41 candidates (step 0.1)
- **Added**: EMA smoothing: `μ_new = 0.5 * μ_old + 0.5 * μ_best`
- **Benefit**: Better exploration and stability in μ optimization

### 5. **Temperature Calibration Pipeline** 
- **Added**: `fit_temperature_scaling()` using held-out samples from S1
- **Method**: Grid search over [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0] minimizing NLL
- **Application**: All expert logits scaled before caching η (improves calibration)

### 6. **Enriched Gating Features**
- **Enhanced**: Feature dimension from 4E → **7E + 3** 
- **Per-expert features (7E)**: entropy, top-k mass, residual, max confidence, top1-top2 gap, cosine similarity, KL divergence to mean
- **Global features (+3)**: mixture entropy, mean class variance, std of expert confidences
- **Impact**: Better disagreement detection and ensemble uncertainty modeling

## Additional Improvements

### Expanded α Range
- **Range**: [0.80, 1.60] (vs previous [0.85, 1.15])
- **Benefit**: More aggressive tail boosting capability

### Enhanced κ Value  
- **Increased**: κ = 25.0 (from 20.0) for sharper sigmoid calibration
- **Effect**: Reduced gradient saturation in selective acceptance

### Comprehensive Logging
- **Added**: Per-cycle diagnostics saved to `selective_training_logs.json`
- **Metrics**: Coverage by group, α/μ evolution, best scores, temperature values
- **Usage**: Monitor fairness progression and hyperparameter sensitivity

### Plugin Integration
- **Extended**: μ grid expansion and EMA in `gse_balanced_plugin.py`
- **Added**: Selective checkpoint loading with freeze options
- **Enhanced**: Dynamic feature dimension computation
- **Objective**: Switch to 'worst' for tail-focused optimization

## Usage

### Selective Training
```bash
python -m src.train.train_gating_only --mode selective
```

### Plugin with Selective Init  
```bash
python -m src.train.gse_balanced_plugin --use_selective_init --objective worst
```

### End-to-End Pipeline
```bash
# 1. Train selective gating
python -m src.train.train_gating_only --mode selective

# 2. Run plugin optimization 
python -m src.train.gse_balanced_plugin

# 3. Evaluate results
python -m src.train.eval_gse_plugin
```

## Expected Performance Improvements

1. **Tail Coverage**: More consistent coverage across groups via per-group τ_k
2. **Worst-Group AURC**: 10-15% improvement through comprehensive tail emphasis  
3. **Calibration**: Better confidence estimation via temperature scaling
4. **Stability**: Smoother α/μ convergence through EMA and directional priors
5. **Feature Quality**: Richer gating decisions through disagreement metrics

## Configuration Files

Key parameters are centralized in:
- `train_gating_only.py` CONFIG['selective'] section
- `gse_balanced_plugin.py` CONFIG['plugin_params'] section

All improvements are backward compatible with existing checkpoints.