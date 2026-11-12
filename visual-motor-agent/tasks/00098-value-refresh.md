# Task 00098: A Value Whose Tail Is Fresh and Shared (experiment vm-00087)

**Date:** 2026-11-11
**Status:** Complete (runs 2026-11-12 03:18 to 04:27; written up 2026-11-12)

## Context

- **vm-00085 (task 00096):** with γ 0.9 the bootstrapped value aims further, but it is optimistic. It predicts 30–55% less hunger than follows (K4's label: 14–25%), and reading 2 failed, so readings 3–6 are provisional. Its tail V̂ had two faults, both named in the plan:
  1. **A minimum.** V̂ is the prediction of the winning transformer, stored while acting. The instinct picks the lowest of 16 candidates and the gain rule the best of four transformers, so V̂ is biased low, as Q-learning's max is biased high, and the label learns the bias back.
  2. **Stale.** A label is computed once, when its chunk is written. A chunk replayed 50k ticks later still carries the tail of the model that wrote it.
- **The owner (2026-11-11)** agreed to phase 2 as the pre-set response to reading 2: the mean of the four as the tail, refreshed on replay.

## Design

### One change that fixes both: recompute the tail at every update

- `online.value_refresh = 1` (with `online.value_gamma > 0`). Before every update, the tail of every label in the batch is recomputed from the current models:
  - The label of step j stays (1 − γ) Σ_{i<n} γ^i h_{j+i} + γ^n V̂_{j+n}.
  - The real part and the tail's weight (γ^n, or 0 where an episode ends first) are fixed when the window is made. In long-term memory they are kept with the chunk in place of the label.
  - **V̂_{j+n} is fresh:** a forward pass of each transformer, with no gradient, over the window that starts n steps later. The move token's predicted next hunger at each step is the predicted label of the move actually made there. V̂ is the mean of the four transformers' predictions.
- **This fixes the minimum:** every transformer scores the same real move, so nothing is chosen for being low. The instinct still chooses moves by the lowest predicted label, but the label no longer feeds that minimum back.
- **This fixes the staleness:** the tail is always the current models', like a DQN whose target network is synced every step. With hunger in 0–1 and γ 0.9, divergence is unlikely; the frozen bias test would show it.
- **What it needs in memory:** the bootstrap window for a chunk's last n steps lies past the chunk.
  - With refresh, long-term memory keeps each chunk with the n steps after it (L + n = 12 steps, not 8).
  - Those steps are already in short-term memory when the gate decides a chunk (it waits D ≥ n steps).
  - Cost: half again the chunk store, about 0.2 GB per life with the default 100k-step replay. RAM is not short (230 GB free).
- **What it costs in time:** one forward pass per transformer over the batch (8 windows of 8 steps) per update, about a third of an update. Expected: 10–20% slower lives.
- **The stored `vh`** (the winner's prediction while acting) is still logged as `value_pred` but no longer enters any label.
- **Refused for simplicity** (they change what a transformer sees, or which update it takes): `online.off`, `online.clog`, `online.noise_in`, `env.delays`, the C++ path, the stream memory.

### Arms

The 2-layer default, as in vm-00085: four quadrant transformers, gain, log summaries, the ring, the egocentric touch world, `online.future_k = 4`, six seeds, 60k ticks.

| Arm | Label | Tail |
|---|---|---|
| K4 (vm-00085, not rerun) | mean hunger over the next 4 ticks | none |
| V90 (vm-00085, not rerun) | value, γ 0.9, n 4 | the winner's prediction, stored |
| **V90R** | value, γ 0.9, n 4 | the four's mean prediction of the move made, fresh at every update |

- **6 lives on the RTX 3090** under MPS, once 00021's seed-2 runs there have ended: the card vm-00085's controls ran on.
- If vm-00086 favours 8 layers, an 8-layer V90R follows.

## Measures

As vm-00085: lived food, poison, hunger, overrides and move cross-entropy over 40–60k; frozen `eval_hunger.py chain` and `eval_hunger.py value`, every arm paired by seed with K4 and V90R with V90.

## The readings, set before the results

1. **Plumbing:**
   - Unit tests: with refresh on and a model that predicts the stored `vh`, the labels equal vm-00085's to 1e-6. The real part and tail weight equal a direct sum, episode ends included. A replayed chunk's bootstrap window is the n steps after its own start.
   - The forward pass does not disturb acting: the ring's moves are the same with and without it.
   - A 600-tick life logs the value within 0–1 and is at most 25% slower than V90.
2. **The value holds (the main one):** in every V90R seed, abs bias ≤ .0141 (vm-00085's bar, twice K4's mean) and correlation > .5. Also, V90R's abs bias is below V90's, paired, by more than its interval.
3. **Reach is kept:** V90R's aim at 12–24 px is above K4's by more than its interval.
4. **A chain pays:** V90R's lead over blind with a perfect grab is above K4's by more than its interval.
5. **More food:** V90R's lived food at 40–60k is above K4's by more than its interval.
6. **Poison is still seen:** V90R's lived poison is not above K4's by more than its interval.

## Implementation

- [x] `online.py`:
  - `value_parts(ndrv0, done, gamma, n)` → (real part, tail weight), with `value_labels` rewritten on it;
  - `ChunkStore(..., extra=n)` keeps L + n steps per chunk and the real part and tail weight in place of the label;
  - `GatedMemory` with `value_refresh`: the windows carry `pr_real`, `tw` and the bootstrap window (`b_img`, `b_drv`, `b_act`, `b_es`), from short-term and long-term memory alike.
- [x] `lifetime.py`:
  - the setting and its refusals;
  - before the update, V̂ from every transformer's `forward_uni(..., next_body=True)` on the bootstrap windows, their mean, and `pr` = real + tw · V̂;
  - the metric `value_boot` (the mean fresh V̂).
- [x] `DEFAULT_ONLINE`: `value_refresh = 0`.
- [x] `tests/test_value_labels.py`: the tests above (`test_value_parts`, `test_refresh_windows`, `test_refresh_off_unchanged`); `tests/test_ring.py::test_a_forward_pass_between_ticks_leaves_the_ring_alone` (bitwise).
- [x] Plumbing lives (600 ticks on the 3090 shared with 00021): V90R 1.51 ticks/s against V90's 1.69 (11% slower), `value_boot` .37–.47; `00087-value-refresh/` (README, Makefile); the queue (scratchpad `q87.sh`, after 00021's seed-2 runs).
- [x] Runs, frozen tests, write-up.

## Verification

- `make pytest`, unchanged.
- With refresh off, the labels are vm-00085's, array for array.
