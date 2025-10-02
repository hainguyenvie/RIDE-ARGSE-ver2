#!/bin/bash
# Quick training script with all improvements

echo "🚀 GSE Plugin Enhanced Training Pipeline"
echo "======================================="

# Step 1: Enhanced Gating Training
echo "Step 1: Training enhanced gating network..."
python -m src.train.train_gating_only
if [ $? -ne 0 ]; then
    echo "❌ Gating training failed!"
    exit 1
fi
echo "✅ Gating training complete"

# Step 2: Choose optimization method
echo ""
echo "Step 2: Choose optimization method:"
echo "1) Balanced optimization (standard)"
echo "2) Worst-group with EG-outer (new!)"
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo "Running standard balanced optimization..."
    python -m src.train.gse_balanced_plugin
    method="balanced"
elif [ "$choice" = "2" ]; then
    echo "Running worst-group EG-outer optimization..."
    python run_gse_worst_eg.py
    method="worst-group EG-outer"
else
    echo "Invalid choice. Running balanced optimization..."
    python -m src.train.gse_balanced_plugin
    method="balanced"
fi

if [ $? -ne 0 ]; then
    echo "❌ Plugin optimization failed!"
    exit 1
fi
echo "✅ Plugin optimization ($method) complete"

# Step 3: Evaluation
echo ""
echo "Step 3: Evaluating results..."
python -m src.train.eval_gse_plugin
if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

echo ""
echo "🎉 Complete! All improvements applied:"
echo "   ✅ tuneV-based gating training"
echo "   ✅ Blended alpha updates"
echo "   ✅ Adaptive lambda grid"
echo "   ✅ Per-group thresholds (Mondrian)"
if [ "$method" = "worst-group EG-outer" ]; then
    echo "   ✅ EG-outer worst-group optimization"
fi
echo "   ✅ Enhanced evaluation"