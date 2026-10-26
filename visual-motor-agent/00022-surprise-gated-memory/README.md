# vm-00022: Surprise-Gated Memory for the Universal Model

**Date:** 2026-10-25
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00022`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00022/`
**Task plan:** [`../tasks/00028-token-surprise-memory.md`](../tasks/00028-token-surprise-memory.md)

## Goal

Since vm-00019 the universal model has kept every chunk of its life in long-term memory. Here a chunk is kept only
when it was surprising, by one rule for every token: a token's cross-entropy relative to its own slot's recent mean,
the step's largest such ratio, the chunk's largest step. Is a surprising quarter of the experience better than a
random quarter, and than all of it?

## Hypothesis

1. The surprising chunks hold the rare events: clicks on food and poison (the body token's surprise) more often than a
   random quarter.
2. With them, the model predicts the click outcomes better than with a random quarter of the same size.
3. Against keeping everything (4× the memory), a surprising quarter is at least as good.

## Method

`configs/U_universal.json` (the default universal model and its world, vm-00021), 100k ticks, seeds 1–3, three arms:

| Arm | Gate | Long-term memory at 100k |
|---|---|---|
| U | every chunk (vm-00021) | 12,498 chunks: the whole life |
| S | `store_surprise=0.25`, `store_base=0`: the most surprising quarter of recent chunks | 3,404–3,423 |
| U25 | `store_base=0.25`: a random quarter | 3,119–3,229 |

All arms keep the first 1,000 steps (warm-up). The 4090 ran five lifetimes (81 ticks/s each), the 3090 four (56).
Measures: the online clicks, the memory's composition, the frozen forage test with and without the instinct at 25k,
50k and 100k, and the click test on real windows (task 00027) at 100k.

## Log

### 2026-10-25
- **Implemented** the per-token surprise (task 00028): the loss kernel keeps each token's cross-entropy; `lifetime.py`
  reads the newest windows' last step one tick later, after acting (no extra wait for the GPU). Checked against the
  PyTorch reference; with a trained model the body token's cross-entropy at the click step itself is 1.53 (food) and
  3.13 (poison) against 0.36 elsewhere.
- **Ran** the 9 lifetimes 22:58–23:28; analysis until 23:50.

## Results

### R1. What the gate kept

- **Relief chunks (a food click in the chunk or the 11 steps after) in long-term memory at 100k:** S 0.48 / 0.28 /
  0.41, U25 0.24 / 0.07 / 0.09, U 0.16 / 0.08 / 0.33 (seeds 1 / 2 / 3).
- **The token that made a stored chunk surprising:** the move token (the prediction of the agent's own move) for
  1,319–1,963 chunks, the body token (the drives after the move: clicks) for 577–1,069, a patch for 701–891.

### R2. Online, 80–100k ticks

| Arm | food / poison clicks per 1k (seeds 1 / 2 / 3) | time at the hunger cap | move cross-entropy |
|---|---|---|---|
| U | 8.6/6.8, 4.3/3.1, 20.1/12.8 | 0.05, 0.01, 0.04 | 0.55, 0.39, 0.67 |
| U25 | 10.6/29.9, 1.6/1.8, 1.8/77.5 | 0.31, 0.63, 0.91 | 0.10, 0.13, 0.04 |
| S | 24.6/22.8, 8.2/7.5, 17.6/29.6 | 0.12, 0.19, 0.22 | 0.22, 0.19, 0.18 |

### R3. Frozen at 100k: the forage test (with the instinct) and the click test on real windows

| Arm | food / poison per 1k | at the cap | policy alone: food / poison | food right | poison right | nothing flat | predicted poison effect |
|---|---|---|---|---|---|---|---|
| U s1 | 7.2 / 2.9 | 0.02 | 8.2 / 11.0 | 0.47 | 0.90 | 0.73 | +0.17 |
| U s2 | 4.1 / 1.8 | 0.01 | 4.0 / 3.2 | 0.32 | 0.72 | 0.84 | +0.10 |
| U s3 | 26.5 / 10.2 | 0.02 | 27.0 / 31.8 | 0.70 | 0.80 | 0.95 | +0.15 |
| U25 s1 | 20.7 / 35.2 | 0.23 | 22.4 / 39.9 | 0.32 | 0.54 | 0.77 | +0.03 |
| U25 s2 | 7.2 / 8.4 | 0.15 | 6.4 / 6.9 | 0.29 | 0.65 | 0.57 | +0.11 |
| U25 s3 | 22.2 / 32.1 | 0.28 | 16.4 / 25.9 | 0.28 | 0.57 | 0.66 | +0.08 |
| S s1 | 29.1 / 17.9 | 0.09 | 22.9 / 28.2 | 0.66 | 0.78 | 0.87 | +0.20 |
| S s2 | 21.2 / 21.7 | 0.13 | 15.3 / 18.2 | 0.72 | 0.66 | 0.86 | +0.10 |
| S s3 | 28.2 / 29.1 | 0.17 | 22.9 / 30.2 | 0.74 | 0.73 | 0.80 | +0.11 |

(Right: the share of felt clicks whose predicted effect has the right sign; flat: clicks on nothing predicted within
0.05. The true poison effect is about +0.47.)

## Analysis

- **Surprise keeps the rare events** (hypothesis 1, supported): S's memory holds 2–5× the relief chunks of a random
  quarter (0.28–0.48 against 0.07–0.24), more than even the whole life in every seed (U: 0.08–0.33). Most of its picks
  are moves the model did not expect of itself; the rest are clicks (the body token) and patches.
- **At the same size, surprise beats chance on every measure** (hypothesis 2, supported in all three seeds): food
  predicted right 0.66–0.74 against 0.28–0.32, poison 0.66–0.78 against 0.54–0.65, clicks on nothing flat 0.80–0.87
  against 0.57–0.77; the frozen agent eats more and is less often at the cap (0.09–0.17 against 0.15–0.28).
- **A small random memory narrows the policy and traps the agent** at the hunger cap: replayed 4× as often, its
  chunks teach the model its own habits until its move distribution collapses (cross-entropy 0.04–0.13); online, U25
  spent 31–91% of its last 20k steps at the cap, and one seed clicked poison 77 times per 1,000 steps. S's memory of the
  same size keeps the unexpected moves, and its policy stays broader (0.18–0.22).
- **Against keeping everything, it is a trade** (hypothesis 3, partly): S predicts food better than U (0.66–0.74
  against 0.32–0.70) and poison slightly worse (0.66–0.78 against 0.72–0.90), and behaves worse: it clicks both kinds far more
  (poison 18–29 per 1k against 2–10) and is more often at the cap (0.09–0.17 against 0.01–0.02). U's memory is 4× larger
  and its policy the broadest (0.39–0.67).

## Conclusions

1. **The model's own surprise, one rule for every token, is a good salience signal:** at equal size it beats a random
   memory in prediction and behaviour, in all three seeds, and it collects the events that matter (clicks) without
   being told what they are.
2. **Memory size matters through replay:** the same lifetime with a quarter of the memory, replayed 4× as often,
   collapses the policy. Surprise counters that only in part.
3. **Keeping everything is still the better default at 100k ticks** for behaviour. Where memory must be small (a long
   life with a fixed memory), surprise is the better rule.

## Next Steps

- Surprise gating with a share that keeps the memory at the size of U's (e.g. a long lifetime that fills the memory),
  so that S and U differ only in content.
- A guard against policy collapse that applies to every token (an entropy floor on the predicted distributions).
