# vm-00034: Self-Forcing in the Text World

**Date:** 2026-10-27
**Status:** Complete
**Source:** `exp/vm-00034` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00034/`
**Task plan:** [`../tasks/00041-text-self-forcing.md`](../tasks/00041-text-self-forcing.md)

## Goal

Self-forcing (task 00039, vm-00032: some replayed windows get the model's own predictions as the inputs of their last
steps, the targets stay real) kept the circle world's dream truer. Does the same rule, unchanged, work in the text
world, where the universal model predicts and generates as well as a classical GPT (vm-00030, vm-00033)?

**Expectation, stated before the runs:** it hurts. In the circle world the next frame is mostly determined by the world
and the real moves, so a dreamed frame is an estimate of the real one. In text the future is not determined: a dreamed
page is another plausible text that the real next letters do not follow from, so the lesson becomes "recent context
does not predict what comes next" (scheduled sampling's known inconsistency for language models).

## Method

- **Runs:** `configs/T_text_B.json` (one letter per token, bin embeddings, memory U), 100k ticks, seeds 1–3, with
  self-forcing: of each update's 8 windows (the newest + 7 replayed), 2 replayed ones get 1–3 dreamed pages (uniform),
  J = 2 Jacobi iterations per page, `pair` decoding. Baseline: vm-00030's B with bin embeddings (the same without).
  62–72 ticks/s (the baseline 135).
- **Prediction:** val nats per letter every 10k (`eval_predict.py --env-set split=val --env-set tile=true`).
- **Generation** (`reference/text_dream.py`, vm-00033's test, new `--temp`): seed 1 at 30k (both arms' best), 64 val
  contexts, 4 pages (256 letters) each; greedy continuation against the truth; samples scored by vm-00033's classical
  GPT (`exp_00033/001_GPTlife30k_s1`) and by the model itself; temperatures 1, 0.8, 0.6.

## Log

### 2026-10-27
- Seeds 1–2 on the 4090, seed 3 on the 3090 (seed 3 slowed by generation tests on the same GPU).
- Generation at temperature 1 looked broken (samples 2.67 nats per letter under GPT); added `--temp` to tell a flatter
  distribution from worse context use.

## Results

### R1. Val nats per letter

| Run | 10k | 20k | 30k | 40k | 50k | 100k | best |
|---|---|---|---|---|---|---|---|
| baseline s1 (vm-00030) | 1.884 | 1.734 | 1.718 | | 1.791 | 1.972 | 1.718 (30k) |
| **self-forcing s1** | 1.873 | 1.720 | **1.703** | 1.715 | 1.742 | 1.903 | **1.703** (30k) |
| baseline s2 | 1.958 | 1.736 | 1.749 | | 1.790 | 2.019 | 1.736 (20k) |
| **self-forcing s2** | 1.865 | 1.710 | **1.704** | 1.756 | 1.769 | 1.949 | **1.704** (30k) |
| baseline s3 | 1.967 | 1.751 | 1.745 | | 1.766 | 1.940 | 1.745 (30k) |
| **self-forcing s3** | 1.870 | 1.751 | 1.716 | **1.714** | 1.742 | 1.910 | **1.714** (40k) |
| classical GPT, a lifetime's batches (vm-00028) | 2.04–2.23 | 1.76–1.79 | 1.70–1.72 | | 1.77–1.81 | 1.94–1.99 | 1.696–1.716 |

Best: 1.703–1.714 (mean 1.707) against 1.718–1.745 (mean 1.733); at 100k 1.903–1.949 (mean 1.921) against
1.940–2.019 (mean 1.977). Better or equal at every checkpoint in every seed.

### R2. Generation, seed 1 at 30k (nats per letter; real continuation under GPT 1.683)

| | real under itself | greedy letter 1 right | samples T=1 under GPT (under itself) | T=0.8 | T=0.6 |
|---|---|---|---|---|---|
| baseline | 1.668 | 0.47 | 1.673 (1.251) | 1.463 (1.101) | 1.313 (0.984) |
| self-forcing | **1.646** | 0.38 | 2.672 (1.670) | 1.734 (1.249) | 1.552 (1.111) |
| GPT | 1.683 | 0.52 | 1.426 (own) | 1.250 | 1.103 |

GPT's samples under the model: baseline 1.779, self-forcing 1.761 (T=1).

At matched sharpness (the samples equally likely under the model that made them): self-forcing at 0.8 against the
baseline at 1.0 (1.249 / 1.251 under itself): 1.734 against 1.673 under GPT; at 0.6 against 0.8 (1.111 / 1.101):
1.552 against 1.463.

### R3. Text (vm-00033's second context: *"…Was it not to refresh the mind of man / After his studi"*)

- Self-forcing, greedy: *"es of the power to the world / The presseth him that the court in a thousand / The seem of the
  worse the shall be about the state"*
- Self-forcing, sampled (T=1): *"us day to hell: / I' fear metlers and show's done. Ah, cousin, learn, / For no report,
  bewith good face son, yond,"*

## Analysis

- **The expectation was wrong for prediction.** Self-forcing predicts real text better at every checkpoint of every
  seed (best 1.707 against 1.733 on average, now level with the classical GPT on the same data, 1.696–1.716), and
  overfits less late in life (1.92 against 1.98 at 100k). The likeliest reading: a dreamed page is a noisy version of
  the context, and predicting the real next letters from it is a denoising regularizer against memorizing a corpus that
  the memory replays over and over (100k ticks of a 1M-letter corpus; the overfitting after 30k is the replay's).
- **It was right for generation.** The free-running text is worse. Most of the gap at temperature 1 is a flatter
  distribution (its samples score 1.67 under itself against the baseline's 1.25: sampling lands in the tails more often
  and the errors compound), but not all: at matched sharpness its samples are still 0.06–0.09 nats per letter less
  likely to the GPT, and its greedy first letter is right less often (0.38 against 0.47). The model learned to lean
  less on recent context, which helps when the context is real and noisy-to-predict-from, and hurts when it must
  continue its own text.
- **The same rule in two worlds:** in the circle world the dream got truer and one-step prediction no worse
  (vm-00032); in text one-step prediction got better and the dream worse. The difference is the world: where the
  future is determined by the past and the moves, correcting a dreamed input toward the real target is consistent;
  where it is not, the target does not follow from the dreamed input and the lesson is "trust the context less".

## Conclusions

1. Self-forcing is a regularizer of prediction in the text world (val 1.707 against 1.733, less overfitting), with
   no text-specific change.
2. It does not improve the text dream: its samples are worse even at matched sharpness. As a default it would trade
   generation for prediction in text and behaviour for nothing in the circle world (vm-00032); it stays off.

## Next Steps

- A universal refinement that keeps the dreamed inputs consistent with the targets in both worlds: e.g. target the
  dreamed steps' own continuation (the model's prediction from its dream) for the stochastic part, or dream only
  where the model is confident (low predicted entropy), which in text is mostly within words.
- Dropout or input noise as a cheaper control for the regularization effect.
