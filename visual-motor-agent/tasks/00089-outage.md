# Task 00089: Outage, Transformers Switched Off at Random (experiment vm-00078)

**Date:** 2026-11-03
**Status:** Complete

## Context

- The owner (2026-11-03): "what if we have N transformers and we randomly turn off some of them in the way that at any going moment roughly N/4 is not working".
- Later the same mechanism may save resources, spread the load and encourage specialization. "If it was off - it was off": a transformer that comes back does not catch up, and its context has gaps.
- The owner's decisions:
  - 16 transformers.
  - Off means the input still works, but the transformer gives no output and does not learn.
  - Restoration is only measured for now: nothing new predicts or feeds the off region. No neighbour ring, no restored inputs, no change to targets, layout or backprop.
  - No context rebuild on return. The ring holds "nothing seen at this tick" without a forward pass; training mirrors the gaps.

## Design

**The switch:** `online.off = q` (the share of time each transformer is off, 0 = off by default, a life exactly as before) and `online.off_span` (the mean off spell in ticks, 32).
- Each transformer runs its own two-state chain from its own generator, so the other draws of a life do not change.
- An off spell ends with probability 1 / off_span per tick; an on transformer turns off with probability q / ((1 - q) off_span), so it is off q of the time. At q = .25: 1/32 and 1/96.
- If every transformer would be off, the lowest one stays on.
- The chain runs from tick 1, with no warm-up.

**An off transformer:**
- **Acting:** no forward pass, no move proposal, no ring step. The gain rule picks among the live transformers.
- **Learning:** no update.
- **Unchanged:** the world, every frame, patch and summary thumbnail, and memory, which stores every step.

**The ring (acting):** one slot per tick, with a seen flag.
- A skipped tick only advances the write position and marks its slot unseen. No compute.
- On return, the missed ticks are skipped at once: `uni_ring_skip(n)`, at most ctx slots of work.
- Attention (the ring softmax and the instinct's candidates) ignores unseen slots. A slot's age stays its distance in ticks, so a gap reads as a gap.
- Birth is unchanged: nothing before the first step.

**Training mirrors it:**
- Memory stores, for each tick, which transformers were on: a bit mask.
- In a replayed window, transformer k's missed steps are not keys for its seen steps, and they carry no loss. The window stays 8 consecutive ticks, so ages are real ticks.
- A seen step's move token still predicts the real next body, even if k missed that next tick.
- The loss keeps its normalization over all tokens, so a window with fewer seen steps gives k less data, not more weight per token.
- A missed step's own rows attend as before and carry no loss. Nothing reads them, so no gradient reaches them.

**Arms:**

| Arm | Transformers | Off |
|---|---|---|
| R0 | 16 | none |
| R1 | 16 | .25 |

- Both: K4, log summaries, the default world, 60k ticks, six seeds.
- vm-00073's K4, 4 quadrants, beside them for scale, not rerun.
- A short timing first decides how many lives run at once.

## Measures

1. **Lived:** food, poison and aim, R1 against R0. The share of transformers live per window.
2. **Frozen:** each final model at test off shares 0, .25 and .5: how gracefully it degrades.
3. **Restoration from what exists, scored only:** live transformers' predictions of the summary thumbnails over an off transformer's region, against the real thumbnails. References: the region's owner's prediction when on, and the copy baseline (the frame before, shifted by the move).
4. **Win shares** of the transformers.

## The readings, set before the results

1. **Plumbing:**
   - With every transformer on, the step mask and the ring flags change nothing, bitwise.
   - Loss and gradients with a step mask equal the reference's.
   - A missed step's frames change nothing in the loss or gradients.
   - A ring with gaps equals a streaming reference whose ages are tick distances.
   - `online.off = 0` is the old life.
   **Met:** 13 tests in `tests/test_outage.py`; 831 GPU tests and 96 lifetime and memory tests unchanged.
2. **Graceful:** R1's lived food over the last 20k ticks is at least 75% of R0's. **Met:** 100%, 114.6 against 114.9.
3. **Practice pays:** frozen at off .25, R1 is above R0 on food by more than its interval: a model that lived with outages copes with them better. **Not met:** +.17 ± 36.6; outages barely hurt either arm (food 120 at off 0, 118 at off .5).
4. **No cost when whole:** frozen at off 0, R1 is level with R0 within its interval. **Met:** −.07 ± 35.1.

## Implementation

- [x] CUDA: a step mask for the uploaded batch (`vmc_set_step_mask`), honoured in the attention softmax (keys) and the universal loss (rows); cleared by the next batch
- [x] CUDA: ring seen flags, `vmc_uni_ring_skip`, honoured in the ring softmax and the candidates
- [x] `reference/torch_reference.py`: `step_mask` in `forward_uni` and `uni_loss`
- [x] `cuda_lib.py`: `set_step_mask`, `uni_ring_skip`, `update_rollout(step_mask=...)`
- [x] `online.py`: `off`, `off_span`; the on mask in memory
- [x] `lifetime.py`: the schedule, acting and updates of the live transformers only, the masks, metrics
- [x] `tests/test_outage.py`
- [x] Frozen tools (`eval_outage.py`): the switch at test time; the restoration score
- [x] Timing, `00078-outage/`, runs, write-up; `eval_outage.py alone` added after the results

## Verification

Readings 1. A short life with 16 transformers and q = .25 on each GPU.
