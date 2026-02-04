# Experiment 00003: Full C Training Chain with llm.c

**Date:** 2026-02-03  
**Status:** ✅ Complete (CPU) | 🔜 GPU pending (needs CUDA toolkit)

## Objective

Train an LLM entirely in C (no Python in the training loop) using Karpathy's llm.c.

## Results

| Metric | Value |
|--------|-------|
| Model | GPT-2 124M (pretrained, finetuning) |
| Training | Pure C with OpenMP |
| Val loss (start) | 5.33 |
| Val loss (end) | 4.29 |
| Steps | 40 |
| Time/step | ~3.2s (CPU, 32 threads) |
| Total time | ~2.5 min |

## What This Proves

**The entire training loop runs in C** — no Python, no PyTorch, no frameworks. Just:
- C code for forward/backward passes
- Manual gradient computation
- AdamW optimizer implemented in C
- OpenMP for parallelization

## Architecture

Using pretrained GPT-2 124M weights, finetuning on Shakespeare:
```
max_seq_len: 1024
vocab_size: 50257
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124,475,904
```

## Sample Output

**Before training (step 0):**
```
I am Senior Mayor:
foradeed, for stay to be there, if forsen
I sent you, you know no good him, journey,
with his wits and young ladies; for God's sake, then I can't answer.
```

**After training (step 40):**
```
Being barren walked:
Who, thou, unty'd;
If thou enlight'd well, thou laderestly gaily, award
Dilppour to damning death:
God bids sure that deafness '
But the life is curdly.
That tost pardon him, how about
```

The output becomes more Shakespeare-like (archaic English, verse structure, "thou").

## How to Reproduce

```bash
# Clone llm.c
git clone https://github.com/karpathy/llm.c.git
cd llm.c

# Download starter pack (pretrained weights + tokenized Shakespeare)
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh

# Build CPU version
make train_gpt2

# Train! (uses 32 threads)
OMP_NUM_THREADS=32 ./train_gpt2
```

## Comparison with Previous Experiments

| Experiment | Training | Inference | Params | Language |
|------------|----------|-----------|--------|----------|
| 00001 nanoGPT | Python/PyTorch | Python | 210K | Python |
| 00002 llama2.c | Python/PyTorch | **C** | 870K | Py→C |
| **00003 llm.c** | **C** | **C** | 124M | **Pure C** |

## Next Steps

1. **Install CUDA toolkit** for GPU training (~100x faster)
2. Train from scratch (not finetuning) on custom data
3. Export trained model for llama2.c inference
4. Compare training speed: Python vs C vs CUDA

## Notes

- CPU training is viable for small experiments/debugging
- For real training, GPU is essential (CUDA version runs ~100x faster)
- The C codebase is remarkably clean (~1000 lines for core training)
- OpenMP parallelization scales well with thread count

## Attribution

Based on Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c)
