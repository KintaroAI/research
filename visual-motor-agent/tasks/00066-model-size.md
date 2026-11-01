# Task 00066: Bigger and Smaller Transformers (experiment vm-00058)

**Date:** 2026-10-31
**Status:** Complete

## Context

- The default model (`configs/U_default.json`, vm-00054's QNA) has four transformers of d = 128, 2 layers, 4 heads, d_ff = 512: 929,282
  parameters per `info.json`. That size has never been tested in the piece design.
- The owner's question: what do more parameters and heads give, and what does less cost? Two arms, three seeds each. Selection rules
  (the list in `../ARCHITECTURE.md`) wait.

## Design

Width scales by 2 each way, heads with it, so a head stays 32 wide. Layers stay 2, so only width changes. No code changes.

| arm | d_model | n_head | d_ff | parameters |
|---|---|---|---|---|
| **S** | 64 | 2 | 256 | 367,362 (0.40×) |
| default | 128 | 4 | 512 | 929,282 |
| **B** | 256 | 8 | 1024 | 2,642,946 (2.84×) |

The input and output tables grow with d, not d², so doubling the width gives under 4×.

## Experiment vm-00058

- `configs/U_default.json` with the sizes above, seeds 1–3, 60k ticks, all six on the RTX 4090 with MPS.
- **Control:** QNA's seeds 1–3 from vm-00054 (the same settings, an older build), and its eight seeds from vm-00055 for the spread (34.7 ± 10.2 food).
- **Measure:** food and poison per 1k over the last 20k ticks; patch, body and move CE; ticks/s; parameters. `turn_cost` stays 0.
- **Reading:** seeds of one size span 14–59 food, so three seeds show only a large effect. A difference inside that spread says nothing.

## Results

Write-up: [`../00058-model-size/README.md`](../00058-model-size/README.md). Last 20k ticks, food / poison per 1k: S 37.5 / 3.2, default 40.5 / 4.0, B 33.4 / 3.7, with
seeds from 13 to 59. Prediction follows size (patch CE 0.805, 0.760, 0.737; move CE 0.889, 0.251, 0.115), foraging does not.
