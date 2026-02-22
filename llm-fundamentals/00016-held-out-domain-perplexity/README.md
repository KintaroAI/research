# Experiment: Held-Out Domain Perplexity (Eval Roadmap 1a)

**Date:** 2026-02-22
**Status:** Complete

## Goal

Implement and validate held-out domain perplexity evaluation: train on TinyStories,
measure perplexity on text from different distributions (TinyShakespeare, WikiText-2).
This tests whether architectural variants generalize beyond the training domain,
using existing eval-only mode (`-n 0 -v 1 -j <heldout.bin>`) with no C changes.

## Hypothesis

A TinyStories-trained model should achieve low perplexity on in-domain data but
substantially higher perplexity on out-of-domain text. The gap size reflects how
distribution-specific the learned representations are. Architecture variants that
narrow this gap demonstrate better cross-domain generalization.

## Method

### Infrastructure

1. **`prepare_heldout.py`** — downloads and tokenizes held-out corpora into .bin
   format (same as `prepare_data.py`: magic=20240520, version=1, uint16 tokens)
2. **EVAL.md** — Phase 1a protocol added
3. **EVAL_ROADMAP.md** — item 1a marked done

### Held-out domains

| Domain | Source | Tokens | File |
|--------|--------|--------|------|
| TinyShakespeare | Karpathy's char-rnn repo (GitHub) | 338,026 | `data/heldout/shakespeare.bin` |
| WikiText-2 test | HuggingFace datasets API (Salesforce/wikitext) | 283,286 | `data/heldout/wikitext2.bin` |

Shakespeare is treated as a single document. WikiText-2 is split into documents
on double newlines, each separated by EOT.

### Training setup

- Model: 4-layer, 4-head, 128-dim GPT-2 (7.3M params, `tinystories` preset)
- Data: full TinyStories (906M train tokens, 19M val tokens)
- Training: B=64, T=256, lr=3e-4, 5000 steps (~82M tokens seen), seed 42
- No weight decay, beta2=0.999

### Eval protocol

Eval-only mode with `-n 0 -v 1`. Point `-j` at each held-out .bin file:

```bash
./train -e checkpoint_tinystories.bin -n 0 -v 1 -t 256 -b 64 \
    -i data/tinystories/TinyStories_train.bin \
    -j data/heldout/shakespeare.bin
```

The first `val loss` line printed (before any training steps) is the eval result.
Note: `-n 0` doesn't skip training entirely — it falls through to using
`train_num_batches` from the `-i` file. The eval-only result is the step-0 val
loss. For small held-out files this completes quickly regardless.

## Results

| Domain | Val Loss | Perplexity |
|--------|----------|------------|
| TinyStories (in-domain) | **2.045** | **7.7** |
| TinyShakespeare | **7.714** | **2,237** |
| WikiText-2 | **9.215** | **10,030** |

### Training curve (milestones)

| Step | Train Loss | Val Loss (TinyStories) |
|------|-----------|------------------------|
| 0 | — | 10.831 |
| 500 | 3.698 | 3.721 |
| 1000 | 2.959 | 3.006 |
| 2000 | 2.505 | 2.495 |
| 3000 | 2.257 | 2.268 |
| 4000 | 2.021 | 2.130 |
| 5000 | 2.036 | 2.045 |

Train/val gap is small (0.026 at step 5000) — the model is not overfitting on
906M tokens after only 5000 steps (~82M tokens seen, <10% of one epoch).

## Analysis

The perplexity hierarchy is clear and expected:

1. **TinyStories (7.7)** — in-domain, the model has learned this distribution well.
   Simple children's stories with limited vocabulary and repetitive structure.

2. **Shakespeare (2,237)** — 3.8x higher loss than in-domain. Archaic English,
   verse structure, specialized vocabulary ("thou", "hath", "wherefore") are all
   far from children's stories. But it's still English prose, so some transfer
   happens — much better than random (perplexity ~50K for vocab size 50257).

3. **WikiText-2 (10,030)** — 4.5x higher loss than in-domain, worst of the three.
   Wikipedia covers diverse topics (science, history, geography) with technical
   vocabulary, complex sentence structure, and factual content. The distribution
   mismatch is largest here.

The gap between in-domain and held-out perplexity provides a clear metric for
comparing architectures: variants with smaller gaps have learned more transferable
representations.

### Generation sample

The model generates coherent TinyStories-style text:

> Once upon a time, in a big forest, there lived an owl. The owl lived in a
> safe tree in the woods near a big tree. Every day, the owl would talk to his
> paws high cheer to the animals. The rabbit always listened to his voice and
> he cared for the questions.
> One day, a little rabbit gathered by the tree and decided to help...

Simple vocabulary, story structure with characters, and a moral — characteristic
of the TinyStories distribution.

## Conclusions

- Held-out domain perplexity eval works end-to-end with existing infrastructure
- No C changes needed — eval-only mode (`-n 0 -v 1 -j <file>`) is sufficient
- Clear perplexity separation across domains provides a useful generalization metric
- `prepare_heldout.py` successfully downloads and tokenizes both corpora
- The protocol is documented in EVAL.md Phase 1a, ready for architecture comparisons

## Next Steps

- [ ] Run held-out eval on banded sparsity checkpoints — compare cross-domain gap
- [ ] Add more held-out domains (Simple Wikipedia, code, etc.)
- [ ] Train longer (full epoch) for lower in-domain loss, re-measure gaps
- [ ] Implement Phase 1b (compression ratio / bits-per-byte) using these same files
