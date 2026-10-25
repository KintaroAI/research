# vm-00018: Additive Poison, and One Model in Many Worlds

**Date:** 2026-10-25
**Status:** In Progress
**Source:** *tagged on completion as `exp/vm-00018`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00018/`
**Task plan:** [`../tasks/00022-additive-poison-bodies.md`](../tasks/00022-additive-poison-bodies.md)

## Goal

Two follow-ups to vm-00017, both about whether the prediction works:

1. **Poison that adds** +0.5 to hunger instead of setting it to 1, so that hunger rarely sits at its cap, where a
   poison click leaves no trace.
2. **One model in many worlds:** 32 bodies early in life (vm-00016's best basic agent) give the model many more poison
   and food clicks to learn from. A one-body arm with the same batch size separates more data from bigger batches.

## Hypothesis

1. With additive poison and 100k ticks, the 8×8 patch model's drives head predicts the poison click (+0.5) as well as
   the food click, and stops predicting relief for a click on nothing.
2. Parallel bodies learn the click outcomes sooner (by 10k–20k ticks).
3. The gain comes from the extra experience, not the batch size: Q1B (one body, batch 39) learns no faster than Q1.

## Method

- **World:** vm-00017's, with `env.hazard_click_add=0.5` instead of `env.hazard_click=1` (task 00022).
- **Model:** vm-00017's PP (8×8 patch tokens, move loss detached, hidden policy layer, memory U), `lifetime.py`,
  100k ticks, seeds 1–3, on both GPUs.
- **Arms:** Q1 (one body, batch 8), Q32 (32 bodies for 10k ticks, halving every 5k, one from tick 30k; batch 39 → 8),
  Q1B (one body, batch 39 throughout).
- **Measures:** `analyze_hazards.py click` (world: click minus stay −0.301 food, +0.5 poison, 0 background; the eaten
  circle's redness) and the frozen tests at 0, 10k, 20k, 40k, 60k, 100k; online food hits and poison clicks (body 0, all
  bodies) and hunger at the cap.

## Log

### 2026-10-25
- **Implemented** `env.hazard_click_add` (test: adds after relief, clips at 1, off by default) and poison-click counts
  in `online.py` / `lifetime.py`. Smoke run (8 bodies, 1k ticks): about 10 poison clicks per 1,000 body-steps; the
  click test's truth for poison is 0.801 (0.3 + 0.5 + 0.001).
