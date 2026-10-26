# Task 00034: Is the Gap the Data? (experiment vm-00028)

**Date:** 2026-10-26
**Status:** Complete (results in vm-00028)

## Context

- vm-00026/27: the universal model in a text lifetime reaches 1.77–1.80 nats per letter on Tiny Shakespeare's val
  split, the classical GPT of the same size 1.615. One-hot letters (an embedding table) learn faster but end in the same
  place, so the input map is not the cause.
- The two were trained on different data. GPT: 8 windows of 512 letters per update, each from a random letter offset
  of the train split (about 1M distinct windows). The lifetime: per tick, the newest window of its stream (the last 8
  pages, sliding by one page of 64 letters) and 7 chunks drawn uniformly from long-term memory, each chunk 8 pages
  aligned to the stream's chunk boundaries (one chunk per 512 letters read; the alignment shifts by 334 letters per pass
  over the 1,003,854 train letters). B's train loss 1.57 against val 1.80.

## The test, both directions

1. **The classical GPT on the lifetime's data** (`reference/char_gpt.py --data lifetime`): the same model, but its
   batches built as the lifetime builds them: a stream from a seeded offset advancing 64 letters per update; the
   newest window (the last 512 letters read); 7 chunks drawn uniformly from the chunks completed at least 11 pages
   ago (the gate's delay); before the first chunk, windows from the last 256 pages read (the short-term memory).
   Memory never evicts within 100k updates (12,500 chunks fit). If GPT falls to about 1.8, the data explains the gap.
2. **The universal model on GPT's data** (`train_text_offline.py`, new): the unchanged model (arm B's config), 8
   windows per update of 8 consecutive pages from random letter offsets of the train split, moves and drives 0, each
   window a fresh context; the GPU update as in a lifetime. If it reaches about 1.62, the same conclusion from the
   other side.

Both 100k updates, seeds 1–3, scored as vm-00026 (the same 208 val windows). Checkpoints every 10k for the universal
model (eval_predict.py), GPT's val every 5k.

## Verification

- `--data lifetime`: the windows are those of a simulated lifetime (the newest window ends at the stream position;
  chunks are aligned to the stream and complete before the delay; the counts per update).
- `train_text_offline.py`: a window's pages are 8 consecutive pages of the train split from its offset; a short run's
  loss falls; its checkpoints load in eval_predict.py.

## Results (vm-00028)

- GPT on a lifetime's batches: best 1.70–1.72 (random windows: 1.615), overfitting sooner.
- The universal model on random windows: best 1.78–1.79 (its lifetime: 1.80). The data is not why it is behind; the
  0.17 nats per letter at equal data is in the model, and shows as slow learning. Next: an ablation ladder in the
  PyTorch reference.
