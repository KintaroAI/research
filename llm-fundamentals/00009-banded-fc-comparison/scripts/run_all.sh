#!/bin/bash
# Experiment 9: Run all banded sparsity comparison experiments
# Usage: ./run_all.sh [train_banded_path] [model_path]

set -e

TRAIN_BIN="${1:-../dev/train_banded}"
MODEL="${2:-../dev/model.bin}"

# Common parameters
STEPS=5000
BATCH=16
SEQ=256
VAL_EVERY=500
SAMPLE_EVERY=1000

# Create directories
mkdir -p logs checkpoints

echo "=== Experiment 9: Banded Sparsity FC Comparison ==="
echo "Train binary: $TRAIN_BIN"
echo "Model: $MODEL"
echo ""

# Function to run a single experiment
run_exp() {
    local name=$1
    local fc1_bw=$2
    local fc2_bw=$3
    
    echo ">>> Running: $name (FC1=$fc1_bw, FC2=$fc2_bw)"
    $TRAIN_BIN \
        -1 $fc1_bw -2 $fc2_bw \
        -e $MODEL \
        -c checkpoints/ckpt_${name}.bin \
        -n $STEPS -b $BATCH -t $SEQ \
        -v $VAL_EVERY -s $SAMPLE_EVERY \
        -o logs/log_${name}.txt
    echo ">>> Completed: $name"
    echo ""
}

# 1. Baseline (Dense)
run_exp "baseline" 0 0

# 2. FC1 Only
run_exp "fc1_bw1024" 1024 0
run_exp "fc1_bw512" 512 0
run_exp "fc1_bw256" 256 0

# 3. FC2 Only
run_exp "fc2_bw1024" 0 1024
run_exp "fc2_bw512" 0 512
run_exp "fc2_bw256" 0 256

# 4. Both FC1+FC2
run_exp "both_bw1024" 1024 1024
run_exp "both_bw512" 512 512
run_exp "both_bw256" 256 256

echo "=== All experiments completed ==="
echo "Logs saved to: logs/"
echo "Checkpoints saved to: checkpoints/"
