# Experiment 00002: Shakespeare LLM with llama2.c

**Date:** 2026-02-03  
**Status:** ✅ Complete

## Objective

Train a small Llama2-architecture model on Shakespeare text and run inference using Karpathy's pure C implementation (llama2.c).

## Results

| Metric | Value |
|--------|-------|
| Model params | ~870K |
| Train loss (final) | 1.79 |
| Val loss (final) | 2.60 |
| Training iterations | 5000 |
| Inference speed (C) | **1401 tok/s** |

## Model Architecture

```
dim = 128
n_layers = 4
n_heads = 4
n_kv_heads = 4
vocab_size = 512 (custom BPE)
max_seq_len = 256
dropout = 0.1
```

## Training Setup

- **Dataset:** Tiny Shakespeare (~1.1M chars, 624K tokens after BPE)
- **Tokenizer:** Custom SentencePiece BPE (512 vocab)
- **Train/Val split:** 90/10
- **Batch size:** 32
- **Learning rate:** 1e-3 (with warmup + cosine decay)
- **GPU:** RTX 4090

## Sample Output

```
./run model.bin -z tokenizer.bin -t 0.8 -n 200 -i "HAMLET:"

HAMLET:
Here is he that bless yourselves,
And you'll burn again. He's as heaven would hear
When the coast, let they go: for what lie,
When I do call me from them, but,
So called in a rather, let him go
That wants but ever fought, which he gave
The duke is call'd to be the field: come on,
There is no time to tell him of the day...
```

## Files

```
00002-llama2c-shakespeare/
├── README.md                 # This file
├── docs/
│   └── training_log.txt      # Training output
├── scripts/
│   ├── shakespeare.py        # Data prep (copy from llama2.c)
│   └── train_shakespeare.py  # Training script (copy from llama2.c)
└── artifacts/                # Model files (not committed - too large)
    ├── model.bin             # ~3.5MB llama2.c format
    ├── tokenizer.bin         # ~6KB tokenizer for C
    ├── tokenizer.model       # ~7KB SentencePiece model
    └── ckpt.pt               # ~10MB PyTorch checkpoint
```

**Note:** Large files (model.bin, ckpt.pt) are not committed. Regenerate by running training.

## How to Reproduce

```bash
# 1. Prepare data
cd ~/llama2.c
python shakespeare.py download
python shakespeare.py train_vocab --vocab_size=512
python shakespeare.py pretokenize --vocab_size=512

# 2. Train
python train_shakespeare.py

# 3. Inference
./run out_shakespeare/model.bin -z out_shakespeare/tokenizer.bin -t 0.8 -n 200 -i "HAMLET:"
```

## Key Learnings

1. **Small models can learn structure** - Even 870K params captures Shakespeare's dialogue format (character names, colons, verse structure)
2. **Custom tokenizers matter** - 512 vocab BPE trained on Shakespeare is much more efficient than using 32K Llama vocab
3. **C inference is blazing fast** - 1400+ tok/s on CPU, ~100x faster than Python
4. **Val loss plateaus early** - Overfitting on small dataset; train loss kept dropping but val loss stabilized ~2.60

## Next Steps

- Try larger model (more layers/dims) with regularization
- Add temperature sampling comparison
- Quantize to int8 for even faster inference
- Compare with our first experiment (char-level GPT)

## Attribution

Based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c)
