# Experiment 00007: Banded Sparsity on MLP

**Goal:** Test diagonal/banded sparsity pattern on the FC1 (MLP up-projection) layer to reduce compute while maintaining model quality.

## Concept

In a transformer MLP, the up-projection (FC1) maps `C → 4*C` channels. Instead of fully connecting all inputs to all outputs, we use a **banded pattern**:

```
For input i ∈ [0, C):
  j_center = i × (4*C / C) = i × 4
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

### Why This Might Work

- **Locality hypothesis:** Adjacent input features may naturally map to adjacent hidden features
- **Gradual mixing:** Information spreads across the diagonal rather than instantly mixing everything
- **Significant compute reduction:** With bandwidth=256 on C=128→512, we get ~44% density (56% reduction)

## Model

Same tiny GPT-2 as experiment 6:
- C = 128 channels
- L = 4 layers
- NH = 4 heads
- 7.3M parameters

## Files

```
src/
  train_gpt2_fp32.cu         # Original dense training
  train_gpt2_fp32_banded.cu  # Training with FC1 band mask
  generate.cu                # Original inference
  generate_banded.cu         # Inference (loads banded checkpoint)
  llmc/                      # Utility headers
```

## Quick Start

```bash
# Build everything
make all

# Prepare data (downloads TinyStories ~1.5GB)
make setup

# Create initial model
make model

# Run comparison experiment
make experiment
```

## Manual Usage

### Training with Banded Sparsity

```bash
./train_banded -e model.bin -c checkpoint_banded.bin -w 256 -n 2000

# Options:
#   -w <bandwidth>  Band width (default: 256)
#   -d 1            Disable banding (run dense for comparison)
#   -n <steps>      Training steps
#   -b <batch>      Batch size
#   -t <seqlen>     Sequence length
#   -o <file>       Log file for plotting
```

### Generate from Banded Checkpoint

```bash
./generate_banded -e checkpoint_banded.bin -n 128
```

## Bandwidth vs Density

For model with C=128, 4*C=512:

| Bandwidth | Density | Connections/Layer |
|-----------|---------|-------------------|
| 512       | 100%    | 65,536 (dense)    |
| 256       | ~44%    | 28,768            |
| 128       | ~22%    | 14,336            |
| 64        | ~11%    | 7,168             |

## Implementation Details

1. **Mask creation:** At init, create binary mask `(L, 4*C, C)` based on band formula
2. **Forward pass:** Weights are already masked, matmul works normally
3. **Backward pass:** Apply mask to weight gradients
4. **Optimizer step:** Re-apply mask after AdamW update

This is Option A (mask-based) — same memory footprint as dense, but validates the concept. Option B (true sparse kernel) would reduce memory and compute.

## Expected Results

- **Random data:** Loss should be similar (no learnable structure)
- **TinyStories:** Hypothesis is ~5-10% loss increase acceptable for ~50% compute reduction

## Next Steps

If banded FC1 works:
1. Try smaller bandwidth (more aggressive sparsity)
2. Apply to FC2 (down-projection) as well
3. Implement true sparse kernel (Option B) for actual speedup
4. Try learned sparsity patterns instead of fixed diagonal
