# Quick training script with all improvements (PowerShell version)

Write-Host "🚀 GSE Plugin Enhanced Training Pipeline" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Step 1: Enhanced Gating Training
Write-Host "`nStep 1: Training enhanced gating network..." -ForegroundColor Yellow
python -m src.train.train_gating_only
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Gating training failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Gating training complete" -ForegroundColor Green

# Step 2: Choose optimization method
Write-Host "`nStep 2: Choose optimization method:" -ForegroundColor Yellow
Write-Host "1) Balanced optimization (standard)"
Write-Host "2) Worst-group with EG-outer (new!)"
$choice = Read-Host "Enter choice (1 or 2)"

switch ($choice) {
    "1" {
        Write-Host "Running standard balanced optimization..."
        python -m src.train.gse_balanced_plugin
        $method = "balanced"
    }
    "2" {
        Write-Host "Running worst-group EG-outer optimization..."
        python run_gse_worst_eg.py
        $method = "worst-group EG-outer"
    }
    default {
        Write-Host "Invalid choice. Running balanced optimization..."
        python -m src.train.gse_balanced_plugin
        $method = "balanced"
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Plugin optimization failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Plugin optimization ($method) complete" -ForegroundColor Green

# Step 3: Evaluation
Write-Host "`nStep 3: Evaluating results..." -ForegroundColor Yellow
python -m src.train.eval_gse_plugin
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Evaluation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n🎉 Complete! All improvements applied:" -ForegroundColor Green
Write-Host "   ✅ tuneV-based gating training" -ForegroundColor Green
Write-Host "   ✅ Blended alpha updates" -ForegroundColor Green
Write-Host "   ✅ Adaptive lambda grid" -ForegroundColor Green
Write-Host "   ✅ Per-group thresholds (Mondrian)" -ForegroundColor Green
if ($method -eq "worst-group EG-outer") {
    Write-Host "   ✅ EG-outer worst-group optimization" -ForegroundColor Green
}
Write-Host "   ✅ Enhanced evaluation" -ForegroundColor Green