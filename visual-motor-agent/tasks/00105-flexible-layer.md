# Task 00105: The Flexible Layer, W = S + F (experiment vm-00094)

**Date:** 2026-11-16
**Status:** Proposed. The owner approves the design before any code is written.

## Context

- **No memory past the context so far, in this world** (128 × 128, 2 foods, a 64 × 64 window):
  - **vm-00091:** the value agent (C8) aims at unseen food above its blind baseline only for its 8-tick context.
  - **vm-00092:** a 32-tick context did not help, and cost half the food.
  - **vm-00093 (task 00104):** a 64-number state written every tick and read back 8 ticks later did not help either. Trained through the link, the state collapsed to a constant in every seed. Untrained, it varied but carried nothing the agent used.
  - The outcome set in task 00104: next is the flexible layer F, the other memory the plan names.
- **The plan** (ARCHITECTURE.md, "Two layers of weights", the owner, 2026-11-03):
  - W = S + F. F is trained fast and decays toward 0 by a factor λ each tick (λ = .99: a half-life of about 70 ticks); S is the slow, lasting layer.
  - F "may hold 'the food is over there' for tens of ticks."
  - A few lines on the GPU's parameter buffers, no change to the model, and no new token that could collapse.
  - The order set there: F with a fixed share first; consolidation driven by the value's surprise later.

## Design

### The two layers

- **The forward pass reads W = S + F,** as now from one buffer P, which always holds S + F.
- **S learns as now:** the same AdamW step on the same batch (the newest window and 7 replayed). Its step changes P.
- **F learns from the newest window only,** the experience of the last ticks, on every tick:
  - F ← λ F − k · lr · g_new / (√v̂ + ε). g_new is the gradient of the newest window's loss alone (one more pass over one window).
  - v̂ is S's Adam second moment, so F's step is k times the size of an Adam step, per parameter.
  - P changes by F's change. Replayed windows do not train F, so it holds the recent past, not the life.
- **No consolidation (η = 0):** S already learns from everything, so F is a pure short-lived trace on top of an agent that learns as C8 does.
- **Settings:** λ = .99 (half-life 69 ticks), k = 10; all parameters have an F. One arm, so that the comparison with C8 is clean. A second strength (k = 3 or 30) can follow if this one moves anything.
- **Cost:** one pass over one window per tick (about an eighth of an update) and one buffer the size of the parameters.

### Measuring memory while learning

- **A frozen test cannot see F:** a checkpoint does not learn, so its F stops changing. Memory has to be measured in the life.
- **`lifetime.py` gets the chain's memory measure in life (scoring only):**
  - the aim of each move at the nearest unseen food, by ticks since food was last in view, without the ticks after a meal;
  - against the same moves in a blind world, one more world stepped with the same moves, as `eval_hunger.py chain` does.
  - It is logged per 1k ticks as counts and sums, so any stretch of the life can be pooled.
- **C8 is rerun (C8r):** vm-00091's lives lack the measure. C8r also tells whether the measure in life agrees with the frozen one.

### Arms

| Arm | Label | Model | F | Seeds |
|---|---|---|---|---|
| C8r = vm-00091's VW, rerun | value, γ 0.9, n 4, fresh tail | 4 quarters × 8 layers, tolerance 0.002, context 8 | none | 1–12 |
| F | the same | the same | λ .99, k 10, from the newest window | 1–12 |

- **The world and everything else are C8's:** `--config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 --set online.value_gamma=0.9 --set online.value_refresh=1`, plus `--set online.fast=0.99 --set online.fast_k=10` for F.
- **Cards:** 24 lives, seeds 1–7 on the 4090 and 8–12 on the 3090 as vm-00091 (about 4–5 h; F's extra pass is about an eighth), to be confirmed by a timing.

## Code

1. **`online.fast = λ`, `online.fast_k`** (`lifetime.py`, `cuda_lib.py`, `vm_cuda.cu`):
   - the F buffer;
   - after the batch's update, the newest window's gradient and F's step (a new kernel: F's change added to P, S's Adam moments read, not written);
   - `fast = 0` byte-identical to today.
2. **The memory measure in life:** the blind world stepped with the same moves; per-1k counts and sums by bin, with and without meals (as `chain`).
3. **Checkpoints keep P = S + F,** as the life had it, and F alone in a second file: `ckpt_*.bin` acts as the agent was, and S = P − F can be recovered.
4. **Tests:**
   - F's step against a numpy implementation of the rule;
   - F decays as λ^t with no gradient;
   - P = S + F after any number of steps;
   - `fast = 0` unchanged;
   - the measure in life against `eval_hunger.py chain`'s on the same moves (one shared run).

## Measures

- **Lived, 40–60k:** food, poison, hunger, overrides, move CE, smoothness; F against C8r, paired by seed. vm-00091's C8 for reference.
- **Memory in life, 40–60k:** the aim at unseen food by ticks since the last sight (1–8, 9–16, 17–32, 33–64, 65+), seen − blind, without meals; F against C8r, paired by seed.
- **F's size over the life:** |F| / |S| per tensor, logged every 1k ticks.
- **Frozen tests as before** (chain, value, reach) for continuity. They show S + F as it stood at 60k, frozen.

## The readings, set before the results

1. **Plumbing:** the tests pass; all 24 lives run 60k ticks without errors; F stays bounded (|F| / |S| below 0.1 at 60k in every seed).
2. **Memory in life (the main one):** F's aim at unseen food at 9–16 or 17–32 ticks (seen − blind, without meals, 40–60k) is above C8r's, paired over 12 seeds, by more than its interval, in at least one of the two bins.
3. **Memory pays:** F's lived food at 40–60k is above C8r's by more than its interval.
4. **No harm:** F's lived poison is not above C8r's by more than its interval, and its patch CE is not worse by more than its interval.
5. **The measure in life agrees with the frozen one:** C8r's lived memory is positive at 1–8 ticks and about zero past 8, as C8's frozen.

**What follows:**
- **If 2 holds:** fast weights carry the food's place. Next is their reach (λ .995) and the value's surprise as the consolidation, as planned; reading 3 decides whether it is used.
- **If 2 fails and 4 holds:** a trace of the last ticks in the weights, at this strength, does not hold where the food was. Next is one more strength (k 30) before the idea is set aside; a world that needs memory more (hidden food, vm-00081) is then the place to look.
- **If 4 fails:** F's step is too large for this learner. Halve k and rerun before reading 2.

## Implementation

- [ ] The owner approves this design.
- [ ] Code 1–3, with the tests (4) passing; `fast = 0` byte-identical.
- [ ] Timing; then the runs, the frozen tests, the write-up and videos.

## Verification

- Every run's `info.json` has `online.fast` and `online.fast_k` by arm, and C8's settings otherwise.
- C8r's lived food and poison are within the range of vm-00091's VW (the same settings, a rerun).
- The measure in life is computed by the same code path that the frozen chain uses for its bins.
