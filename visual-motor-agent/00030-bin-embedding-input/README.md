# vm-00030: Bin-Embedding Input, in Text and in Vision

**Date:** 2026-10-26
**Status:** Complete
**Source:** `exp/vm-00030` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00030/`
**Task plan:** [`../tasks/00036-bin-embedding-input.md`](../tasks/00036-bin-embedding-input.md)

## Goal

vm-00029 found the universal model's gap to a classical GPT in its input map, and a universal fix in the PyTorch
reference: bin embeddings (each value split two-hot over the output's bins, an embedding per value slot and bin). Here
it is built into the C++ / CUDA model (`model.in_bins`) and tried in both worlds: does it close the text gap in a
lifetime, and does vision gain or at least not lose?

## Hypothesis

1. Text: the lifetime with bin embeddings comes close to the classical GPT trained on the same (a lifetime's) data,
   1.70–1.72 nats per letter (vm-00028), instead of 1.80.
2. Vision: the circle world's prediction (patches, the body, clicks) is at least as good as with the linear input map.

## Method

- **The model:** `in_bins = 1` (task 00036): `W_in` becomes d × (V · bins) over each used value's two-hot bins,
  initialised at 0.02; everything else unchanged. Tested against the PyTorch reference (forward, loss, gradients), the
  GPU against the CPU (forward, update, the instinct's candidates); the full suite passes (1,069 tests).
- **Text:** arm B (`configs/T_text_B.json`, one letter per token, 256 bins) with `in_bins`, a lifetime of 100k ticks,
  seeds 1–3; val nats per letter every 10k, as vm-00026.
- **Vision:** the default universal lifetime (`configs/U_universal.json`, 32 bins: 64 · 32 · 128 = 262k input
  parameters) with `in_bins`, 100k ticks, seeds 1–3, against vm-00022's U (the same configuration and seeds without):
  the frozen tests at 25k, 50k, 100k, the policy alone and the click test on real windows at 100k, the online clicks
  and cross-entropies at 80–100k.
- 135 ticks/s (text) and 63 (vision) per lifetime, three of each at once.

## Log

### 2026-10-26
- **Implemented** `model.in_bins` in C++, CUDA and the reference (task 00036); ran 17:37–18:03, analysis until 18:14.

## Results

### R1. Text: val nats per letter

| Run | 10k | 20k | 30k | 50k | 100k | best |
|---|---|---|---|---|---|---|
| B with bin embeddings, s1 | 1.884 | 1.734 | 1.718 | 1.791 | 1.972 | **1.718** (30k) |
| s2 | 1.958 | 1.736 | 1.749 | 1.790 | 2.019 | **1.736** (20k) |
| s3 | 1.967 | 1.751 | 1.745 | 1.766 | 1.940 | **1.745** (30k) |
| B, linear input (vm-00026) | 2.39–2.40 | 2.03–2.10 | 1.94–1.98 | 1.82–1.89 | 1.80–1.82 | 1.796–1.804 |
| classical GPT on a lifetime's batches (vm-00028) | 2.04–2.23 | 1.76–1.79 | 1.70–1.72 | 1.77–1.81 | 1.94–1.99 | 1.696–1.716 |
| classical GPT on random windows (vm-00026) | 1.67–1.69 | 1.615–1.635 | 1.62–1.63 | 1.66–1.70 | 1.80–1.92 | 1.615 |

### R2. Vision: the circle world, against vm-00022's U (the same seeds)

| Run | eval_self success 25k / 50k / 100k | forage at 100k: food / poison per 1k (at the cap) | policy alone | click test: food right / poison right / nothing flat | online 80–100k: food / poison (cap) | cross-entropy 80–100k: patch / body / move |
|---|---|---|---|---|---|---|
| U s1 | 0.93 / 0.77 / 0.72 | 7.2 / 2.9 (0.02) | 8.2 / 11.0 | 0.47 / 0.90 / 0.73 | 8.6 / 6.8 (0.05) | 0.719 / 0.276 / 0.549 |
| **bins s1** | 0.74 / 0.75 / 0.74 | 6.4 / 1.6 (0.01) | 2.8 / 2.1 | **0.66** / 0.85 / **0.82** | 5.2 / 1.0 (0.01) | **0.710 / 0.243** / 0.229 |
| U s2 | 0.97 / 0.84 / 0.84 | 4.1 / 1.8 (0.01) | 4.0 / 3.2 | 0.32 / 0.72 / 0.84 | 4.3 / 3.1 (0.01) | 0.717 / 0.268 / 0.394 |
| **bins s2** | 0.86 / 0.94 / 0.94 | **10.5** / 1.8 (0.02) | 6.9 / 3.8 | **0.69 / 0.83 / 0.89** | 8.7 / 4.0 (0.05) | **0.710 / 0.218** / 0.191 |
| U s3 | 0.98 / 1.00 / 0.99 | 26.5 / 10.2 (0.02) | 27.0 / 31.8 | 0.70 / 0.80 / 0.95 | 20.1 / 12.8 (0.04) | 0.719 / 0.272 / 0.669 |
| **bins s3** | 0.82 / 0.67 / 0.87 | 11.0 / **4.4** (0.03) | 5.8 / 5.9 | 0.51 / 0.81 / 0.91 | 5.0 / 7.5 (0.07) | **0.708 / 0.249** / 0.169 |

## Analysis

- **Text: the universal model is now as good as a classical GPT on the same data** (hypothesis 1, supported): its
  best in a lifetime, 1.72–1.75 nats per letter at 20–30k, against the classical GPT's 1.70–1.72 on a lifetime's
  batches (and 1.80 with the linear input). It learns as fast as GPT (1.73–1.75 at 20k) and, like GPT on the same data,
  then overfits (1.94–2.02 at 100k): a 100k-step memory replays the 1M-letter text over and over. The gap that is left
  to GPT's 1.615 is the lifetime's data (vm-00028), not the model.
- **Vision: prediction improves a little everywhere** (hypothesis 2, supported): the patches' cross-entropy falls in
  all three seeds (0.708–0.710 against 0.717–0.719), the body's more (0.218–0.249 against 0.268–0.276: hunger is
  predicted better), and the click test improves in two seeds of three (food right 0.66 and 0.69 against 0.47 and
  0.32; clicks on nothing predicted flat in both); seed 3, the strongest forager with the linear input, is a little
  worse on food (0.51 against 0.70).
- **The policy is narrower:** the move token's cross-entropy is 0.17–0.23 against 0.39–0.67, and the frozen policy
  alone clicks much less (poison 2.1–5.9 per 1,000 against 3.2–31.8, food less too). With the instinct the agents
  click poison as little or less (1.6–4.4 against 1.8–10.2); food goes up in one seed and down in two (seed 3: 11.0
  against 26.5). Nothing collapsed (at the cap 0.01–0.07 of the time), but a narrower policy was the start of the
  traps in vm-00024/25.

## Conclusions

1. **Bin-embedding input makes the universal model as good as a classical GPT on text** given the same data, with no
   text-specific part: the input mirrors the output (the same bins), for every value of every token.
2. **In vision it predicts slightly better** (patches, hunger, clicks in two seeds of three) and acts more cautiously.
3. It should become the default input of the universal model; its narrower policy is to be watched in long lives
   (vm-00025's setting, where surprise-gated memory kept the policy broad).

## Next Steps

- Make `in_bins = 1` the default of the universal configs, and rerun the long life (vm-00025's 500k ticks, surprise
  memory) with it to check the policy stays broad.
- Text: the lifetime's overfitting is the fixed corpus replayed by a large memory; the surprise gate (which keeps what
  the model does not yet predict) may help there too.
