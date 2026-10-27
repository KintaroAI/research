# Task 00037: Bin-Embedding Input by Default, and a Long Life With It (experiment vm-00031)

**Date:** 2026-10-26
**Status:** Complete (results in vm-00031)

## Context

- vm-00030: bin-embedding input (`model.in_bins`) makes the universal model as good as a classical GPT on text and
  predicts slightly better in the circle world, but its policy is narrower (the move token's cross-entropy 0.17–0.23
  against 0.39–0.67 at 100k). In vm-00024/25 a narrowing policy was how traps began; in vm-00025 the surprise-gated
  memory kept the policy broad over 500k ticks (0.36–0.51) and no lifetime was trapped.

## Change

- `in_bins = 1` in every universal config (`U_universal.json`, `T_text_A/B/C.json`): the default input from now on.
  Earlier experiments keep theirs through their tags.

## Experiment vm-00031

vm-00025's arm S with bin embeddings: `configs/U_universal.json` (now with `in_bins`), `online.replay = 100,000`,
`store_surprise = 0.25`, `store_base = 0`, 500k ticks, seeds 1–3, against vm-00025's S (the same without). And
vm-00025's arm U (every chunk, the most recent 100k steps) with bin embeddings, seeds 1–3: the arm that collapsed
(is it trapped sooner, or later?).

Measures (as vm-00025): online food and poison clicks and time at the hunger cap per 20k window; the move token's
cross-entropy over the life; the memory's relief share; frozen at 200k and 500k: eval_self, the forage test (with the
instinct and the policy alone at 500k), the click test on real windows.

## Results (vm-00031)

- S (surprise-gated memory): no life trapped in 500k ticks; seeds 2 and 3 forage more than any vm-00025 life (food
  19–25 per 1,000 at the end, at the cap 0.00–0.06); seed 1 started badly (at the cap up to 0.7 of a window before
  140k) and recovered. The move token's cross-entropy is about half vm-00025's and still widens over the life
  (0.12–0.14 → 0.19–0.23). Prediction better (patch cross-entropy 0.708–0.711 against 0.717–0.718; click test at 500k
  as good or better).
- U (the most recent steps): two lives of three trapped (s2 from about 200k, sooner than any vm-00025 life; s3 from
  about 340k), one never; the trapped models are damaged in fresh worlds too.
- `in_bins = 1` stays the default; surprise gating should become the default memory.
