# Experiment 9: Banded Sparsity FC1 vs FC2 vs FC1+FC2 Comparison

## Objective
Compare the effect of banded sparsity on different MLP layers:
- **Baseline**: Dense (no sparsity)
- **FC1 only**: Banded sparsity on up-projection (C → 4C)
- **FC2 only**: Banded sparsity on down-projection (4C → C)
- **FC1+FC2**: Banded sparsity on both layers

Test each sparse configuration with bandwidths: **1024, 512, 256**

## Experiment Matrix

| Config | FC1 BW | FC2 BW | Log File | Checkpoint |
|--------|--------|--------|----------|------------|
| Baseline | 0 | 0 | `log_baseline.txt` | `ckpt_baseline.bin` |
| FC1-1024 | 1024 | 0 | `log_fc1_bw1024.txt` | `ckpt_fc1_bw1024.bin` |
| FC1-512 | 512 | 0 | `log_fc1_bw512.txt` | `ckpt_fc1_bw512.bin` |
| FC1-256 | 256 | 0 | `log_fc1_bw256.txt` | `ckpt_fc1_bw256.bin` |
| FC2-1024 | 0 | 1024 | `log_fc2_bw1024.txt` | `ckpt_fc2_bw1024.bin` |
| FC2-512 | 0 | 512 | `log_fc2_bw512.txt` | `ckpt_fc2_bw512.bin` |
| FC2-256 | 0 | 256 | `log_fc2_bw256.txt` | `ckpt_fc2_bw256.bin` |
| FC1+FC2-1024 | 1024 | 1024 | `log_both_bw1024.txt` | `ckpt_both_bw1024.bin` |
| FC1+FC2-512 | 512 | 512 | `log_both_bw512.txt` | `ckpt_both_bw512.bin` |
| FC1+FC2-256 | 256 | 256 | `log_both_bw256.txt` | `ckpt_both_bw256.bin` |

**Total: 10 runs**

## Directory Structure

```
00009-banded-fc-comparison/
├── PLAN.md                 # This file
├── README.md               # Results summary with tables
├── logs/
│   ├── log_baseline.txt
│   ├── log_fc1_bw1024.txt
│   ├── log_fc1_bw512.txt
│   ├── log_fc1_bw256.txt
│   ├── log_fc2_bw1024.txt
│   ├── log_fc2_bw512.txt
│   ├── log_fc2_bw256.txt
│   ├── log_both_bw1024.txt
│   ├── log_both_bw512.txt
│   └── log_both_bw256.txt
├── checkpoints/
│   ├── ckpt_baseline.bin
│   ├── ckpt_fc1_bw1024.bin
│   ├── ... (etc)
└── scripts/
    └── run_all.sh          # Script to run all experiments
```

## Training Parameters (Common)

```
-e model.bin              # Initial model
-n 5000                   # 5000 steps (allows 500-step intervals: 500,1000,1500,2000,2500,3000,3500,4000,4500,5000)
-b 16                     # Batch size
-t 256                    # Sequence length
-v 500                    # Validate every 500 steps
-s 1000                   # Sample every 1000 steps
-k 500                    # Checkpoint every 500 steps (optional)
```

## Commands to Execute

### 1. Baseline (Dense)
```bash
./train_banded -1 0 -2 0 \
    -e model.bin -c checkpoints/ckpt_baseline.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_baseline.txt
```

### 2. FC1 Only (3 runs)
```bash
# FC1 bandwidth=1024
./train_banded -1 1024 -2 0 \
    -e model.bin -c checkpoints/ckpt_fc1_bw1024.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_fc1_bw1024.txt

# FC1 bandwidth=512
./train_banded -1 512 -2 0 \
    -e model.bin -c checkpoints/ckpt_fc1_bw512.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_fc1_bw512.txt

# FC1 bandwidth=256
./train_banded -1 256 -2 0 \
    -e model.bin -c checkpoints/ckpt_fc1_bw256.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_fc1_bw256.txt
```

### 3. FC2 Only (3 runs)
```bash
# FC2 bandwidth=1024
./train_banded -1 0 -2 1024 \
    -e model.bin -c checkpoints/ckpt_fc2_bw1024.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_fc2_bw1024.txt

# FC2 bandwidth=512
./train_banded -1 0 -2 512 \
    -e model.bin -c checkpoints/ckpt_fc2_bw512.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_fc2_bw512.txt

# FC2 bandwidth=256
./train_banded -1 0 -2 256 \
    -e model.bin -c checkpoints/ckpt_fc2_bw256.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_fc2_bw256.txt
```

### 4. FC1+FC2 Both (3 runs)
```bash
# Both bandwidth=1024
./train_banded -1 1024 -2 1024 \
    -e model.bin -c checkpoints/ckpt_both_bw1024.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_both_bw1024.txt

# Both bandwidth=512
./train_banded -1 512 -2 512 \
    -e model.bin -c checkpoints/ckpt_both_bw512.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_both_bw512.txt

# Both bandwidth=256
./train_banded -1 256 -2 256 \
    -e model.bin -c checkpoints/ckpt_both_bw256.bin \
    -n 5000 -b 16 -t 256 -v 500 -s 1000 \
    -o logs/log_both_bw256.txt
```

## README Template Structure

The README.md should contain:

### 1. Overview Section
- Experiment description
- Model architecture (C, L, NH from model.bin)
- Density percentages for each bandwidth

### 2. Results Tables (one per bandwidth)

#### Bandwidth 1024
| Step | Baseline | FC1-1024 | FC2-1024 | FC1+FC2-1024 |
|------|----------|----------|----------|--------------|
| 500  | X.XXX    | X.XXX    | X.XXX    | X.XXX        |
| 1000 | X.XXX    | X.XXX    | X.XXX    | X.XXX        |
| 1500 | X.XXX    | X.XXX    | X.XXX    | X.XXX        |
| 2000 | X.XXX    | X.XXX    | X.XXX    | X.XXX        |
| 2500 | X.XXX    | X.XXX    | X.XXX    | X.XXX        |
| 3000 | X.XXX    | X.XXX    | X.XXX    | X.XXX        |
| 3500 | X.XXX    | X.XXX    | X.XXX    | X.XXX        |
| 4000 | X.XXX    | X.XXX    | X.XXX    | X.XXX        |
| 4500 | X.XXX    | X.XXX    | X.XXX    | X.XXX        |
| 5000 | X.XXX    | X.XXX    | X.XXX    | X.XXX        |

#### Bandwidth 512
(same format)

#### Bandwidth 256
(same format)

### 3. Summary Comparison Table
| Config | BW | Density | Final Val Loss | Δ vs Baseline |
|--------|-----|---------|----------------|---------------|
| Baseline | - | 100% | X.XXX | - |
| FC1 | 1024 | ~XX% | X.XXX | +X.XX% |
| FC1 | 512 | ~XX% | X.XXX | +X.XX% |
| ... | ... | ... | ... | ... |

### 4. Analysis
- Which layer (FC1 vs FC2) tolerates sparsity better?
- Does combining FC1+FC2 have additive degradation?
- Sweet spot bandwidth for each configuration

## Log Parsing

Extract val loss from logs using:
```bash
grep "^s:" logs/log_*.txt | grep "tel:"
# Format: s:500 tel:X.XXXX
```

## Estimated Runtime
- ~10 runs × ~30 min each = ~5 hours total (GPU dependent)

## Pre-requisites
1. Build `train_banded` in `../dev/`
2. Ensure `model.bin` exists (freshly initialized model)
3. Ensure data is prepared (`TinyStories_train.bin`, `TinyStories_val.bin`)
4. Create directories: `mkdir -p logs checkpoints`

## Post-Processing Steps
1. Run all experiments
2. Parse logs to extract val loss at each 500-step interval
3. Populate README.md tables
4. Generate comparison charts (optional)
5. Write analysis conclusions
