# Experiment: Karpathy GPT Local

**Date:** 2026-02-03  
**Status:** ✅ Complete  
**Author:** Kin

## Attribution

This experiment is based on **Andrej Karpathy's** excellent educational material:

- **Video:** ["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY) (YouTube, Jan 2023)
- **Notebook:** [Google Colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-)
- **Repository:** [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- **Course:** [Zero to Hero](https://karpathy.ai/zero-to-hero.html)

All credit for the original implementation goes to Andrej Karpathy.

## Goal

Run the Karpathy GPT tutorial locally on our hardware to:
1. Verify our environment works for ML training
2. Understand the basic transformer architecture hands-on
3. Establish a baseline for future experiments

## Hypothesis

The tutorial model (~210K parameters) should train in minutes on our RTX 4090 and produce Shakespeare-like text.

## Method

### Approach
1. Convert Colab notebook to standalone Python script
2. Install PyTorch with CUDA support
3. Download Tiny Shakespeare dataset
4. Train the model
5. Generate sample outputs

### Setup
- Hardware: RTX 4090 (24GB VRAM), 128 cores, 251GB RAM
- Software: Python 3.12, PyTorch 2.10.0 with CUDA 12
- Dataset: Tiny Shakespeare (~1.1MB, 1,115,394 characters)

### Configuration
```yaml
# Model
n_embd: 64
n_head: 4
n_layer: 4
block_size: 32
vocab_size: 65  # character-level

# Training
batch_size: 16
max_iters: 5000
learning_rate: 1e-3
dropout: 0.0
```

## How to Run

```bash
cd ~/research/llm-fundamentals/00001-karpathy-gpt-local
source venv/bin/activate
python src/train.py
```

## Results

### Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Parameters | 209,729 (0.21M) | |
| Initial train loss | 4.4116 | Random initialization |
| Final train loss | 1.6667 | After 5000 steps |
| Final val loss | 1.8243 | Slight overfitting expected |
| Training time | ~3 minutes | On RTX 4090 |

### Loss Curve Summary

| Step | Train Loss | Val Loss |
|------|------------|----------|
| 0 | 4.41 | 4.40 |
| 1000 | 2.10 | 2.13 |
| 2000 | 1.88 | 2.00 |
| 3000 | 1.80 | 1.92 |
| 4000 | 1.71 | 1.86 |
| 4999 | 1.67 | 1.82 |

### Sample Output

```
And they bride will to lovest made the was toe.
Stir-day mead, and bartht he us hath be?
Fediless, enjrice, you, not, where
When whom tofting Back my would but
With ensent, will is that Glost and the news!
Fere me, lesing that this me; crients!
Or news hithy mount, us.
But and gods, bettle, demety?

KING RIARD HENRY VI:
So thou strong in him, whose mower;
See the danterty af so;
And his live, I male of while Prive my of.

HENRY BOLINGS:
You ards become and to die courtear tear repts
Infortuce th
```

## Analysis

The model successfully learned:
- **Character-level patterns**: Valid English letter combinations
- **Structural patterns**: Character names (KING, HENRY), colons after names, line breaks
- **Vocabulary**: Shakespeare-ish words and fragments

The output is nonsensical but structurally plausible — exactly what's expected from a tiny 210K parameter character-level model. It captured the "vibe" of Shakespeare without understanding meaning.

### Observations

1. **Loss plateau**: Val loss stopped improving much after step ~3500 (1.89 → 1.82)
2. **Overfitting**: Gap between train (1.67) and val (1.82) loss indicates mild overfitting
3. **Fast training**: 5000 steps took only ~3 minutes on GPU

## Conclusions

1. ✅ **Environment verified**: PyTorch + CUDA working correctly
2. ✅ **Baseline established**: We can train transformer models locally
3. ✅ **Understanding gained**: Walked through attention, embeddings, layer norm, etc.

The tiny model demonstrates the core mechanics but lacks:
- Enough capacity to learn semantics
- Subword tokenization for efficiency
- Larger context windows

## Next Steps

- [ ] Visualize attention patterns to see what the model "looks at"
- [ ] Scale up: more layers, larger embeddings, longer context
- [ ] Try subword tokenization (BPE) instead of character-level
- [ ] Connect attention visualization to thalamus-cortex sorting theory

## Notes

This is the "hello world" of GPT training. The fact that 210K parameters can produce Shakespeare-ish structure from just character prediction is remarkable — it shows how much pattern recognition emerges from next-token prediction alone.
