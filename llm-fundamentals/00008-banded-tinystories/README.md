# Experiment 00008: Banded Sparsity on TinyStories

**Goal:** Test diagonal/banded sparsity pattern on the FC1 (MLP up-projection) layer to reduce compute while maintaining model quality on a real dataset.

## Results Summary

| Bandwidth | FC1 Density | FC1 Sparsity | Final Val Loss | vs Dense |
|-----------|-------------|--------------|----------------|----------|
| Dense | 100% | 0% | ~1.85 | baseline |
| 1024 | 43.8% | 56.2% | **1.88** | +1.6% |
| 512 | 23.5% | 76.5% | **~2.1** | +13.5% |
| 256 | 12.2% | 87.8% | **2.19** | +18.4% |

**Key finding:** There's a sweet spot around 25-45% FC1 density where quality remains high. Below ~25% density, the model starts generating incoherent text.

### Sample Outputs by Sparsity

**Dense (100%):**
> Once upon a time, there was a boy named Timmy. Timmy had a toy car that he loved to drive around his car all day long. One day, his dad invited him to his park to ride his bike.

**bw1024 (43.8%):**
> Once upon a time, there was a small cat named a strange bird named Jane. Jane loved to balance around her white bed, trying to move her teddy bear to the roof.

**bw512 (23.5%):**
> Tim loved to play with Jimmy and explore are the world needs soft. Tim went to the park and got tired. At the day, there was a cool day, happy girl.

**bw256 (12.2%):**
> Pancaser was not far. Mimi said, "Hiunny, I am the biggest and I am a banana. I have a asked Buddy. What is something?"

Text coherence degrades significantly below ~25% density.

## Concept

In a transformer MLP, the up-projection (FC1) maps `C → 4*C` channels. Instead of fully connecting all inputs to all outputs, we use a **banded pattern**:

```
For input i ∈ [0, C):
  j_center = i × (OC / C)
  Connect to outputs j where |j - j_center| ≤ bandwidth/2
```

This creates a diagonal band of connections:

```
Output j:  0 1 2 3 4 5 6 7 8 9 ...
Input i:
  0       [█ █ █ █ · · · · · · ...]
  1       [· █ █ █ █ █ · · · · ...]
  2       [· · · █ █ █ █ █ · · ...]
  ...
```

### Why This Works

- **Locality hypothesis:** Adjacent input features naturally map to adjacent hidden features
- **Gradual mixing:** Information spreads across the diagonal rather than instantly mixing
- **MLP structure:** FC1 up-projects features; diagonal preserves local structure before FC2 mixes

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Channels (C) | 512 |
| Layers (L) | 8 |
| Heads (NH) | 8 |
| Block size | 512 |
| Parameters | 51.2M |

## Dataset

- **TinyStories** (Eldan & Li, 2023)
- Training: 898M tokens
- Validation: 47M tokens
- Tokenizer: GPT-2 (50257 vocab)

## Training Details

```bash
# Banded training (specify bandwidth with -w)
./train_banded -e model.bin -c checkpoint_bw1024.bin -w 1024 -n 3000 -b 16 -t 512 -v 500 -s 1000
./train_banded -e model.bin -c checkpoint_bw512.bin -w 512 -n 3000 -b 16 -t 512 -v 500 -s 1000
./train_banded -e model.bin -c checkpoint_bw256.bin -w 256 -n 3000 -b 16 -t 512 -v 500 -s 1000

# Dense baseline (disable banding with -d 1)
./train_banded -e model.bin -c checkpoint_dense.bin -d 1 -n 3000 -b 16 -t 512 -v 500 -s 1000
```

| Setting | Value |
|---------|-------|
| Batch size | 16 |
| Sequence length | 512 |
| Learning rate | 3e-4 |
| Steps | 3000 |
| Throughput | ~121k tok/s |
| Hardware | RTX 4090 |

### Training Progression

**bw1024 (43.8% density):**
| Step | Val Loss |
|------|----------|
| 0 | 10.93 |
| 500 | 3.34 |
| 1000 | 2.56 |
| 1500 | 2.25 |
| 2000 | 2.07 |
| 2500 | 1.96 |
| 3000 | **1.88** |

**bw256 (12.2% density):**
| Step | Val Loss |
|------|----------|
| 0 | 10.93 |
| 500 | 4.14 |
| 1000 | 3.15 |
| 1500 | 2.72 |
| 2000 | 2.45 |
| 2500 | 2.30 |
| 3000 | **2.19** |

More aggressive sparsity converges slower and to a worse final loss.

## Bandwidth vs Density

For C=512, OC=2048 (FC1):

| Bandwidth | Density | Sparsity | Quality Impact |
|-----------|---------|----------|----------------|
| 2048 | 100% | 0% | Baseline |
| 1024 | 43.8% | 56.2% | Minimal (~2% loss) |
| 512 | 23.5% | 76.5% | Moderate (~14% loss) |
| 256 | 12.2% | 87.8% | Significant (~18% loss) |

## Implementation

1. **Mask creation:** Binary mask `(L, 4*C, C)` based on band formula
2. **Forward pass:** Weights pre-masked, standard matmul
3. **Backward pass:** Mask applied to weight gradients
4. **Optimizer step:** Re-mask after AdamW update

**Note:** Initial weights are NOT zeroed at start (causes numerical instability). Instead, the mask is applied to gradients during backward and weights after optimizer step. After step 1, masked weights become zero and stay zero.

This is mask-based sparsity (same memory as dense). True sparse kernels would also reduce memory/compute.

### Verification

Sparsity was verified by analyzing saved checkpoints:
```
FC1 zeros (bw1024): 4,715,520/8,388,608 (56.2% sparse) ✓
FC1 zeros (dense):  0/8,388,608 (0.0% sparse) ✓
```

## Files

```
src/
  train_gpt2_fp32_banded.cu  # Training with FC1 band mask
  generate_banded.cu         # Inference
  llmc/                      # Utility headers

data/tinystories/            # Dataset (not in git)
  TinyStories_train.bin      # 1.8GB
  TinyStories_val.bin        # 95MB

checkpoint_banded.bin        # bw1024 (195MB)
checkpoint_dense.bin         # Dense baseline (195MB)
checkpoint_bw512.bin         # bw512 (195MB)
checkpoint_bw256.bin         # bw256 (195MB)
```

## Quick Start

```bash
# Build
make all

# Download and tokenize TinyStories
python tokenize_tinystories.py

# Create model
python create_model.py

# Train with different bandwidths
./train_banded -e model.bin -c checkpoint.bin -w 1024 -n 3000 -b 16 -t 512

# Generate samples
./generate_banded -e checkpoint.bin -n 100
```

## Conclusions

1. **Sweet spot at ~25-45% density:** bw1024 (43.8%) achieves 98% of dense quality
2. **Diminishing returns below 25%:** bw512 (23.5%) shows noticeable degradation
3. **Failure below 15%:** bw256 (12.2%) produces incoherent text
4. **Locality hypothesis validated:** Diagonal connections preserve meaningful structure in FC1

## Future Work

- [x] ~~Try smaller bandwidths~~ (Done: bw512, bw256)
- [ ] Apply to FC2 (down-projection) as well
- [ ] Implement true sparse CUDA kernels for actual speedup
- [ ] Compare with other sparsity patterns (random, learned)
- [ ] Scale to larger models
- [ ] Try adaptive bandwidth per layer

## References

- TinyStories: [Eldan & Li, 2023](https://arxiv.org/abs/2305.07759)
- llm.c: [Karpathy](https://github.com/karpathy/llm.c)
