# Experiment 00005: From-Scratch GPU Training

**Date:** 2026-02-03  
**Status:** ✅ Complete

## Objective

Train an LLM from random initialization entirely in C/CUDA — no Python in training loop.

## Results

| Metric | Start | End |
|--------|-------|-----|
| Train loss | 10.92 | 3.0 |
| Val loss | 10.93 | 5.33 |
| Steps | 931 |
| Time | ~26 sec |
| Throughput | 581K tok/s |

## Model

Created small GPT-2 style model from scratch:
```
Parameters: 7.3M
Layers: 4
Heads: 4
Embedding dim: 128
Vocab: 50257 (GPT-2 tokenizer)
Sequence length: 256
```

## Sample Generation Progress

**Step 0 (random init):**
```
aced performTH peacMadeacc myselfimmWhere Valentine...
```

**Step 300:**
```
L R changed'd as you:
What cannot say IET:
And for closed of thy heart!
```

**Step 600:**
```
Messundy may
O nature, ruin, not.
PRINCE EDWARD:
Yes, sir, if which I Lans
```

**Step 931 (final):**
```
GLOUCESTER:
There; therefore are the time.
LADY GREY:
I do wed's sake, to prove it not,
To see you hope so easily won to instruct straight
```

## Files Created

- `create_small_model.py` — Creates 7.3M param model in llm.c format
- `expand_data.py` — Repeats training data for more epochs
- `train_gpt2_fp32_custom.cu` — Modified trainer with -e model flag

## How to Reproduce

```bash
# 1. Create small model from scratch
python create_small_model.py

# 2. Expand training data (50x repeats)
python expand_data.py

# 3. Compile custom trainer
nvcc --threads=0 -t=0 --use_fast_math -std=c++17 -O3 \
    train_gpt2_fp32_custom.cu -lcublas -lcublasLt -lnvidia-ml \
    -o train_gpt2fp32cu_custom

# 4. Train!
./train_gpt2fp32cu_custom \
    -e small_gpt_scratch.bin \
    -i dev/data/tinyshakespeare/tiny_shakespeare_train_50x.bin \
    -t 256 -b 64 -l 0.001
```

## Key Achievements

1. **Full C/CUDA training** — Python only used for model init, training is pure CUDA
2. **From scratch** — No pretrained weights, random initialization
3. **Fast** — 28ms per step, 581K tokens/second
4. **Learns structure** — Model learns Shakespeare format (character names, dialogue)

## Observations

- Val loss plateaus around 5.3 — model overfits on repeated data
- Needs larger/more diverse dataset for better generalization
- 7.3M params is quite small but learns basic patterns
- GPU training ~34x faster than CPU

## Attribution

Based on Karpathy's [llm.c](https://github.com/karpathy/llm.c)
