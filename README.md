## End-to-End AR-GSE Selective + Plugin Flow

1. Generate CIFAR100-LT splits & logits (experts already trained).
2. (Optional) Calibrate experts → set temperatures in `CONFIG['selective']['temperature']` inside `train_gating_only.py`.
3. Selective gating training (learn α, μ, t jointly with gating):
```powershell
python -m src.train.train_gating_only --mode selective
```
Produces: `checkpoints/gating_pretrained/<dataset>/gating_selective.ckpt`.
4. Plugin refinement using selective init (set flags in `gse_balanced_plugin.py`):
```python
plugin_params = {
	 'use_selective_init': True,
	 'freeze_alpha': False,
	 'freeze_mu': False,
}
```
Run:
```powershell
python -m src.train.gse_balanced_plugin
```
5. Evaluate:
```powershell
python -m src.train.eval_gse_plugin
```
### Raw-Margin Decision Rule
Accept if:
```text
m_raw(x) = max_y α_{g(y)} η_y(x) - Σ_y (1/α_{g(y)} - μ_{g(y)}) η_y(x) > t
```
### Selective Loss Summary
| Term    | Meaning                                                |
|---------|--------------------------------------------------------|
| L_sel   | Selective CE normalized by soft coverage               |
| L_cov   | (E[s]-τ)^2 global coverage penalty                     |
| L_cov-g | Group coverage balancing (optional)                    |
| L_H     | Entropy of gating weights (encourage moderate spread)  |
| L_GA    | KL(w || π_{g(y)}) group-aware prior                    |
Tune via `CONFIG['selective']`.
### Recommended Starting Hyperparameters
```text
τ=0.70–0.75, κ=20,
λ_cov=20–30, λ_cov_g=5–15 (enable if worst-group high),
λ_H=0.01, λ_GA=0.05,
StageA=5 epochs, Cycles=6, epochs_per_cycle=3
```
### Troubleshooting
| Issue | Action |
|-------|--------|
| Coverage too low | Increase λ_cov or lower τ |
| s(x) all small | Reduce κ or λ_cov; inspect raw margin distribution |
| Worst-group error high | Enable λ_cov_g, enlarge λ grid, increase α_steps |
| Gating collapse to one expert | Increase λ_H or decrease κ |
| Gating too uniform | Decrease λ_H or slightly raise κ |
### Quick Commands
```powershell
# Pretrain gating only
python -m src.train.train_gating_only --mode pretrain

# Selective gating
python -m src.train.train_gating_only --mode selective

# Plugin refinement (balanced / worst depending on config)
python -m src.train.gse_balanced_plugin

# Evaluation
python -m src.train.eval_gse_plugin
```
# AR-GSE Gemini Pipeline

## Data Preparation

```bash
python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"
```

## Train Experts

```bash
python -m src.train.train_expert
```

## Train AR-GSE (original variant)

```bash
python -m src.train.train_argse
```

## Evaluate AR-GSE

```bash
python -m src.train.eval_argse
```

## Plugin: Balanced + Per-Group Thresholds

```bash
python -m src.train.train_gating_only --mode pretrain
python -m src.train.gse_balanced_plugin
python -m src.train.eval_gse_plugin
```

## Plugin: Worst-Group (EG-Outer) + Per-Group Thresholds

```bash
python -m src.train.train_gating_only --mode pretrain
python run_gse_worst_eg.py   # EG-outer optimization
python -m src.train.eval_gse_plugin
```

## End-to-End AR-GSE Selective + Plugin Flow

1. Generate CIFAR100-LT splits & logits (experts already trained).
2. (Optional) Calibrate experts → set temperatures in `CONFIG['selective']['temperature']` inside `train_gating_only.py`.
3. Selective gating training (learn α, μ, t jointly with gating):

```powershell
python -m src.train.train_gating_only --mode selective
```

Produces: `checkpoints/gating_pretrained/<dataset>/gating_selective.ckpt`.

4. Plugin refinement using selective init (set flags in `gse_balanced_plugin.py`):

```python
plugin_params = {
	 'use_selective_init': True,
	 'freeze_alpha': False,
	 'freeze_mu': False,
}
```

Run:

```powershell
python -m src.train.gse_balanced_plugin
```

5. Evaluate:

```powershell
python -m src.train.eval_gse_plugin
```

### Raw-Margin Decision Rule

Accept if:

```text
m_raw(x) = max_y α_{g(y)} η_y(x) - Σ_y (1/α_{g(y)} - μ_{g(y)}) η_y(x) > t
```

### Selective Loss Summary
| Term | Meaning |
|------|---------|
| L_sel | Selective CE normalized by soft coverage |
| L_cov | (E[s]-τ)^2 global coverage penalty |
| L_cov-g | Group coverage balancing (optional) |
| L_H | Entropy of gating weights (encourage moderate spread) |
| L_GA | KL(w || π_{g(y)}) group-aware prior |

Tune via `CONFIG['selective']`.

### Recommended Starting Hyperparameters
```text
τ=0.70–0.75, κ=20,
λ_cov=20–30, λ_cov_g=5–15 (enable if worst-group high),
λ_H=0.01, λ_GA=0.05,
StageA=5 epochs, Cycles=6, epochs_per_cycle=3
```

### Troubleshooting
| Issue | Action |
|-------|--------|
| Coverage too low | Increase λ_cov or lower τ |
| s(x) all small | Reduce κ or λ_cov; inspect raw margin distribution |
| Worst-group error high | Enable λ_cov_g, enlarge λ grid, increase α_steps |
| Gating collapse to one expert | Increase λ_H or decrease κ |
| Gating too uniform | Decrease λ_H or slightly raise κ |

### Quick Commands
```powershell
# Pretrain gating only
python -m src.train.train_gating_only --mode pretrain

# Selective gating
python -m src.train.train_gating_only --mode selective

# Plugin refinement (balanced / worst depending on config)
python -m src.train.gse_balanced_plugin

# Evaluation
python -m src.train.eval_gse_plugin
```