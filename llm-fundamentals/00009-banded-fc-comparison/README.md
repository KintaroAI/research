# Experiment 9: Banded Sparsity — FC1 vs FC2 vs FC1+FC2 Comparison

## Overview

This experiment compares the effect of applying banded sparsity to different MLP layers in a GPT-2 transformer:

| Configuration | Description |
|---------------|-------------|
| **Baseline** | Dense (no sparsity) — 100% of weights active |
| **FC1 only** | Banded sparsity on up-projection (C → 4C) |
| **FC2 only** | Banded sparsity on down-projection (4C → C) |
| **FC1+FC2** | Banded sparsity on both MLP layers |

Each sparse configuration tested with bandwidths: **1024**, **512**, **256**

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Channels (C) | TODO |
| Layers (L) | TODO |
| Heads (NH) | TODO |
| Vocab Size | TODO |
| Parameters | TODO |

## Density by Bandwidth

| Bandwidth | FC1 Density | FC2 Density | Combined MLP Density |
|-----------|-------------|-------------|----------------------|
| 1024 | ~XX% | ~XX% | ~XX% |
| 512 | ~XX% | ~XX% | ~XX% |
| 256 | ~XX% | ~XX% | ~XX% |

---

## Results: Validation Loss at Each Checkpoint

### Bandwidth = 1024

| Step | Baseline | FC1-1024 | FC2-1024 | FC1+FC2-1024 |
|------|----------|----------|----------|--------------|
| 500 | - | - | - | - |
| 1000 | - | - | - | - |
| 1500 | - | - | - | - |
| 2000 | - | - | - | - |
| 2500 | - | - | - | - |
| 3000 | - | - | - | - |
| 3500 | - | - | - | - |
| 4000 | - | - | - | - |
| 4500 | - | - | - | - |
| 5000 | - | - | - | - |

### Bandwidth = 512

| Step | Baseline | FC1-512 | FC2-512 | FC1+FC2-512 |
|------|----------|---------|---------|-------------|
| 500 | - | - | - | - |
| 1000 | - | - | - | - |
| 1500 | - | - | - | - |
| 2000 | - | - | - | - |
| 2500 | - | - | - | - |
| 3000 | - | - | - | - |
| 3500 | - | - | - | - |
| 4000 | - | - | - | - |
| 4500 | - | - | - | - |
| 5000 | - | - | - | - |

### Bandwidth = 256

| Step | Baseline | FC1-256 | FC2-256 | FC1+FC2-256 |
|------|----------|---------|---------|-------------|
| 500 | - | - | - | - |
| 1000 | - | - | - | - |
| 1500 | - | - | - | - |
| 2000 | - | - | - | - |
| 2500 | - | - | - | - |
| 3000 | - | - | - | - |
| 3500 | - | - | - | - |
| 4000 | - | - | - | - |
| 4500 | - | - | - | - |
| 5000 | - | - | - | - |

---

## Summary: Final Validation Loss (Step 5000)

| Configuration | Bandwidth | Density | Final Val Loss | Δ vs Baseline |
|---------------|-----------|---------|----------------|---------------|
| Baseline | - | 100% | - | - |
| FC1 only | 1024 | ~XX% | - | - |
| FC1 only | 512 | ~XX% | - | - |
| FC1 only | 256 | ~XX% | - | - |
| FC2 only | 1024 | ~XX% | - | - |
| FC2 only | 512 | ~XX% | - | - |
| FC2 only | 256 | ~XX% | - | - |
| FC1+FC2 | 1024 | ~XX% | - | - |
| FC1+FC2 | 512 | ~XX% | - | - |
| FC1+FC2 | 256 | ~XX% | - | - |

---

## Analysis

### Key Questions

1. **FC1 vs FC2 tolerance**: Which layer handles sparsity better?
   - TODO

2. **Additive degradation**: Does FC1+FC2 = FC1 + FC2 degradation?
   - TODO

3. **Bandwidth sweet spot**: What's the minimum bandwidth with acceptable loss?
   - TODO

### Observations

- TODO: Add observations after experiments complete

### Conclusions

- TODO: Add conclusions after experiments complete

---

## Reproduction

```bash
# Build
cd ../dev && make train_banded

# Run all experiments
cd ../00009-banded-fc-comparison
chmod +x scripts/run_all.sh
./scripts/run_all.sh ../dev/train_banded ../dev/model.bin
```

## Files

```
00009-banded-fc-comparison/
├── README.md           # This file (results)
├── PLAN.md             # Experiment plan
├── logs/               # Training logs
│   ├── log_baseline.txt
│   ├── log_fc1_bw*.txt
│   ├── log_fc2_bw*.txt
│   └── log_both_bw*.txt
├── checkpoints/        # Model checkpoints
└── scripts/
    └── run_all.sh      # Automation script
```
