# vm-00026: The Universal Model Reads Shakespeare

**Date:** 2026-10-26
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00026`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00026/`
**Task plan:** [`../tasks/00032-text-world.md`](../tasks/00032-text-world.md)

## Goal

Measure the universal model's prediction on data with a known reference: Tiny Shakespeare, drawn by a text world as grey
pages (a letter is a brightness), read by the unchanged model, against a classical character-level GPT of the same size
on exactly the same held-out letters. Two ways to put letters into tokens: 64 letters in one token, as 64 pixels of a
patch (A), or one letter per token (B).

## Hypothesis

1. With one letter per token (B), the universal model comes close to the classical GPT: the output (256 bins, every
   grey level a bin centre) is a softmax over letters, as GPT's.
2. With 64 letters in one token (A), it is clearly worse: a token's 64 letters are predicted in parallel, each without
   the others; the loss should rise with the position within the token.

## Method

- **The text world** (task 00032): Tiny Shakespeare (65 distinct characters), train = the first 90%, val = the last
  10%. Letter k at grey level round(k · 255 / 64). Each step shows the next page of 64 letters; drives 0, moves ignored.
- **The universal model, unchanged** (d 128, 2 layers, 4 heads, feed-forward 512, ctx 8 steps = 512 letters,
  `bins = 256`), in a lifetime (`lifetime.py`, memory U, one update per tick of 8 windows: the newest and 7 replayed),
  100k ticks, seeds 1–3:

| Arm | Page | Tokens per step | Letters per token | Parameters |
|---|---|---|---|---|
| A | 8 × 8 px, one pixel per letter, one 8 × 8 patch | 3 (body, patch, move) | 64 | 2.52M (the output layer: 64 values × 256 bins) |
| B | 16 × 16 px, a 2 × 2 block per letter, 2 × 2 patches | 66 | 1 | 539k |

- **The classical reference** (`reference/char_gpt.py`): the same Transformer code, widths, initialisation, optimizer
  (AdamW 3e-4, gradient clip 0.5), context of 512 letters, an embedding table in and a softmax over the 65 letters out
  (479k parameters); offline, 8 random windows of 512 train letters per update, 100k updates, seeds 1–3.
- **Scoring** (`eval_predict.py`, `char_gpt.evaluate`): the val split read by 16 evenly spaced readers, 208 windows of
  512 letters (106k letters, all of val), each window a fresh context; nats per letter.
- Equal training: 100k updates of 8 × 512 letters each (about 400 passes over the train split) for every run.
- Baselines: the letter frequencies alone (unigram entropy) 3.31 nats; the previous letter alone (bigram) 2.45.

## Log

### 2026-10-26
- **Implemented** the text world, `eval_predict.py`, `reference/char_gpt.py` (task 00032).
- **Ran** the 9 runs 15:30–15:50; scoring until 15:55.

## Results

### R1. Val nats per letter

| Run | 10k | 20k | 30k | 50k | 70k | 100k | best |
|---|---|---|---|---|---|---|---|
| GPT s1 | 1.679 | 1.635 | 1.617 | 1.658 | 1.709 | 1.800 | **1.617** (30k) |
| GPT s2 | 1.685 | 1.615 | 1.634 | 1.704 | 1.778 | 1.920 | **1.615** (20k) |
| GPT s3 | 1.667 | 1.625 | 1.622 | 1.666 | 1.757 | 1.874 | **1.615** (25k) |
| B s1 | 2.401 | 2.100 | 1.984 | 1.888 | 1.851 | 1.823 | **1.804** (90k) |
| B s2 | 2.391 | 2.073 | 1.939 | 1.863 | 1.837 | 1.796 | **1.796** (100k) |
| B s3 | 2.397 | 2.025 | 1.943 | 1.824 | 1.837 | 1.803 | **1.803** (100k) |
| A s1 | 3.654 | 3.489 | 3.434 | 3.398 | 3.372 | 3.374 | 3.372 (70k) |
| A s2 | 3.620 | 3.459 | 3.474 | 3.430 | 3.381 | 3.377 | 3.376 (80k) |
| A s3 | 3.700 | 3.495 | 3.454 | 3.403 | 3.381 | 3.369 | 3.369 (100k) |

Train loss at 90–100k: B 1.56–1.59 (val 1.80–1.82); A 3.26 (val 3.37).

### R2. Where in the window and in the token

- **By page in the window** (8 pages of 64 letters): the first page is the hardest for both (GPT at its best 1.65–1.67
  against 1.57–1.65 later; B 1.88–1.91 against 1.75–1.86): both use context beyond 64 letters.
- **By position within A's token** (64 letters): flat, 3.29–3.38 at every position, the first letter (which directly
  follows the previous token's last letter) included.

## Analysis

- **One letter per token works, about 0.18 nats per letter behind the classical GPT** (hypothesis 1, partly): B's best
  is 1.80 against GPT's 1.615. It learns more slowly (2.03–2.10 at 20k against 1.615–1.635) and, unlike GPT, does not
  overfit within 100k updates (GPT's val rises from 1.62 at 20–30k to 1.80–1.92 at 100k; B's is still falling or flat at 90–100k).
  At 100k the two are equal only because GPT has overfit.
- **The likely cause of B's gap is the input, not the output.** The output is exactly a softmax over letters, as
  GPT's. The input is not: `W_in` is linear, so a letter enters as code × one vector (its 4 equal values summed), and
  all 65 letters lie on one line in the embedding space; the network must untangle them with its nonlinear layers before
  it can use them. GPT's embedding table gives every letter its own vector from the start.
- **64 letters in one token learn almost nothing beyond letter frequencies** (hypothesis 2, more than expected): A's
  3.37 is close to the unigram entropy (3.31) and far from even the previous letter's 2.45. The loss is flat over the
  token's positions, so A does not even predict the first letter of a token from the letter just before it. The
  parallel prediction alone would have kept the first positions near 2.45 nats; the flat curve says that A cannot
  *read* its context: 64 letters, each a scalar code, are mixed linearly by `W_in` into one 128-dimensional vector,
  from which 2 layers cannot recover the letters. The parallel output is a second cost on top.
- In vision the same input map works because pixels are brightness values whose linear mixture means something
  (an average brightness, an edge); letter codes mixed linearly mean nothing.

## Conclusions

1. **The universal model can model text:** unchanged, one letter per token, it comes within 0.18 nats per letter of
   a classical GPT of the same size (1.80 against 1.615), with no sign of overfitting.
2. **The limitation it shows is the linear input map.** It is the same for every world: values whose linear mixture
   means nothing (letter codes; in vision, perhaps the cursor against the background) must be untangled by later
   layers. The fix belongs in the model for all worlds alike: the learned embedding selection of ARCHITECTURE.md §9 (a
   small network reads a token's values and chooses among learned embeddings, as a text LLM's table).
3. **Packing many independent symbols into one token fails** with this input map: A predicts only letter frequencies.

## Next Steps

- The learned embedding selection (ARCHITECTURE.md §9), then rerun A and B and the circle world: one change for every
  world. Expected: B closes most of its gap to GPT, A starts reading its context.
- A one-hot control (a letter as 65 values, each 0 or 1), which makes `W_in` an embedding table: it isolates the input
  map's cost without changing the model.
- Letters as glyphs (ARCHITECTURE.md §9): text read as vision.
