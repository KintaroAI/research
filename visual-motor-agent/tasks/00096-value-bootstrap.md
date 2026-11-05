# Task 00096: A Value That Reaches Past Four Ticks (experiment vm-00085)

**Date:** 2026-11-04
**Status:** In progress (running since 2026-11-04 22:25)

## Context

- **K4 (task 00082, vm-00071):** the hunger a move is scored on, and the move token learns to predict, is the mean of the real hunger over the next 4 ticks, labelled in hindsight.
  - It gave about 1.7 times K1's food and a lead over a blind walk with a perfect grab (+29.6 ± 15.4 per 1k, frozen).
  - Aim still fades past about 16 px.
  - K16 reached a little further, lost aim up close and all but stopped seeing poison (+6.83 ± 1.39). A longer window of real futures lets in the noise of every later touch.
- **Nothing beyond the window is visible to the label:**
  - vm-00082: no chain of moves at any context length in one transformer;
  - vm-00081 and vm-00083: hidden food is not pursued;
  - vm-00084 (running): bigger models predict their own move far better, but more parameters cannot change what the target says.
- **The owner (2026-11-04)** asked how game-playing RL agents learn from scratch and agreed to test what they have and this agent lacks: a bootstrapped value. The value of a state is trained toward what happens next plus the model's own estimate of the value after it, so credit spreads back through time one step per round of learning, with no fixed window.

## Design

### The label: K4 plus a bootstrapped tail

The value of the move at tick j is the exponentially weighted mean of all future hunger. Here h_t is the hunger after step t (`ndrv[..., 0]`), and V̂ is the model's own prediction:

```
V_j = (1 - γ) · Σ_{i ≥ 0} γ^i h_{j+i}
    = (1 - γ) · Σ_{i < n} γ^i h_{j+i}  +  γ^n · V̂_{j+n}
```

- The first n ticks are real, as in K4. Everything after them is the model's own estimate made at tick j + n: its predicted value for the move it actually took then.
- **The range stays 0–1:** like K4's, the label is a weighted mean of hungers. Bins, the two-hot loss and the instinct are all unchanged (the owner's preference: one normalized value range).
- **No new token.** K4 already replaces the next body's hunger target with a hindsight label, which the move token predicts and the instinct reads. The value is a different label in that same slot.
  - The inputs keep each step's real hunger, as in K4.
  - Nothing is specific to a world.
  - A separate value token next to a real next-hunger prediction was considered and set aside: it changes the token layout, the CUDA kernels and every tool, for no gain in what is tested.
- **Where V̂ comes from:** acting already computes, for every body, the predicted label of the move it chose (`uni_act(..., per_body=True)`). The prediction of the transformer whose move the body made is stored per tick in memory (`vh`); if a rule mixes the moves, it is the mean over the transformers.
  - Under the gain rule that is the lowest of the transformers' predictions, an optimistic choice, as Q-learning's max is. Reading 2 measures how far it biases the value.
  - The label of step j is computed when step j + n is known (lag n, one tick more than K4's k − 1).
  - The label is computed once and kept, like K4's.
  - It costs no extra forward pass.
- **The target lags.**
  - For the newest window, the bootstrap is at most n ticks old.
  - For a window replayed from long-term memory, it is as old as the chunk. This plays the part of DQN's target network, but its age is uneven.
  - Reading 2 checks for the failures this can cause. If they appear, phase 2 refreshes `vh` from each update's own predictions of the windows it trains on (a readback of the move token's expected hunger per step).
- **An episode end, if a world has one:** no tail, and the real part renormalized, as `future_labels` does. The default touch world never resets.

### Settings

- `online.value_gamma = γ`: 0 (the default) means off, with K4's labels unchanged.
- `online.value_n = n`: real ticks before the tail (default: `online.future_k`).
- Both need `online.future_k ≥ 1`, the universal model and no self-forcing, as K4 does.

### Arms

The default model in every arm: four quadrant transformers, the gain rule, log summaries, the ring, the egocentric touch world, `online.future_k = 4`, six seeds, 60k ticks.

| Arm | γ | n | Reach, about 1 / (1 − γ) ticks |
|---|---|---|---|
| K4 (control, rerun on the same card and build) | off | — | 4 |
| V90 | 0.9 | 4 | 10 |
| V97 | 0.97 | 4 | 33 |

- 18 lives. The 4090 is out, so all of them run on the 3090. vm-00083's 18 default lives on two cards took 1.5 h each, so expect about 3 h on one card.
- **The instinct's tolerance stays 0.005 in every arm.**
  - A meal lowers every later hunger and hunger rises slowly afterward, so a meal should move V by about as much as it moves K4's mean.
  - The spread of the candidates' predicted labels can only be measured on trained models, so it is reported by the frozen value test, with the share of ticks the instinct overrides in the metrics, and read as a caveat rather than tuned. In vm-00073's K4 s1 the median spread is .0016, already below the tolerance.

## Measures

1. **Lived:** food, poison and mean hunger over 40–60k, each arm against K4, paired by seed; food per 10k ticks.
2. **Frozen, `eval_hunger.py chain`:**
   - seen, blind, and blind with a perfect grab;
   - the lead over blind with a grab;
   - aim at the food by the gap to its edge, weighted by ticks at 12–24 px and past 24 px.
3. **Frozen, the value itself (`eval_hunger.py value`, new):**
   - the frozen agent lives 3,000 ticks on 16 bodies;
   - each tick's predicted label for the move taken is compared with the realized discounted mean of the hunger that followed, cut where γ^i < 1e-3, ticks too near the end dropped;
   - reported: the mean bias and the correlation, per run.
   - For K4 the same comparison against its own 4-tick mean is the reference.
4. **Training:** the mean label and the mean predicted label in the metrics (`value_label`, `value_pred`), to see any drift during the life.

## The readings, set before the results

1. **Plumbing:**
   - A unit test: on a recorded stretch, the label with the tail replaced by the realized future equals the discounted mean computed directly (to 1e-6). With `online.value_gamma = 0`, the labels are K4's, array for array.
   - A 400-tick life of every arm runs, logs `value_label` and `value_pred` within 0–1, and is no more than 5% slower than K4.
2. **The value holds:** in every V90 and V97 seed, the frozen bias of the predicted value against the realized one is at most twice K4's mean bias against its own label in absolute value, and the correlation is above .5.
   - Failing either in any seed is read as drift, and phase 2 is run before readings 3–6 are interpreted.
   - Set after a smoke test of the tool and before any V life: the first bar was .05, but hunger here is about .025 on average, so .05 would pass anything. vm-00073's K4 s1 (400 ticks, 4 bodies): bias −.010, correlation .84.
3. **Reach grows (the main one):** in V90 or V97, aim at 12–24 px is above K4's by more than its interval.
4. **A chain pays:** in V90 or V97, the lead over blind with a perfect grab is above K4's by more than its interval.
5. **More food:** V90's or V97's lived food at 40–60k is above K4's by more than its interval.
6. **Poison is still seen:** neither V arm's lived poison is above K4's by more than its interval (K16's failure).

## Implementation

- [x] `online.py`:
  - `value_labels(ndrv0, vh, done, gamma, n)`;
  - a `vh` field per tick in short- and long-term memory;
  - goal mode `value`, with lag n.
- [x] `lifetime.py`:
  - store the mean over transformers of the per-body predicted label of the move taken;
  - the settings and their checks;
  - `value_label` and `value_pred` in the metrics.
- [x] Config defaults: `online.value_gamma = 0`, `online.value_n = None` (meaning `future_k`), in `DEFAULT_ONLINE`.
- [x] `tests/test_value_labels.py`: the direct sum, an episode end, `future` mode unchanged, and the labels a gated memory hands to training (the newest window, stored chunks) against the same sum on the stream.
- [x] `eval_hunger.py value`: frozen calibration.
- [x] Plumbing lives (600 ticks per arm, beside vm-00084's lives): the label and prediction within 0–1, the same speed as K4.
- [x] `00085-value-bootstrap/`: README, Makefile (`runs`, `frozen`, `show`).
- [ ] Runs, frozen tests, write-up.

## Verification

- The unit tests above, and `make pytest` unchanged.
- `online.value_gamma = 0` reproduces K4's labels exactly on a recorded stretch. Lives on the GPU are not bitwise reproducible, so whole lives are compared by their arrays of labels, not by their outcomes.
- The plumbing lives' metrics: the label and the prediction within 0–1, no NaN, speed.

## Follow-ups, if the value works

- vm-00083's hiding schedule (G4) with a value: is hidden food pursued once the label reaches past the moment it vanishes?
- vm-00082 and vm-00084's single transformer with a value: does a chain appear where size and context gave none?
- Phase 2 (a fresh bootstrap on replay) against phase 1, if reading 2 shows drift.
