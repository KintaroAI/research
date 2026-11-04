# vm-00082: Context Length

**Date:** 2026-11-03
**Status:** Queued after vm-00081's lives
**Source:** `dev/` (not yet tagged)
**Outputs:** `~/data/research/visual-motor-agent/exp_00082/`
**Task plan:** [`../tasks/00093-context-length.md`](../tasks/00093-context-length.md)

## Goal

The owner's question: does a longer context help? One transformer over the whole small world, with a context of 8, 12, 16, 24 or 32 ticks, everything else the default. Each final model is also tested with hidden food (vm-00081's world), to see whether the longer context lets it remember longer. The design and readings set before the results are in the task plan.

## Method

- **Arms:** C8, C12, C16, C24, C32: one transformer (`model.band_n = 0`, `online.pieces = 0`, 87 tokens per step), `model.ctx` 8 to 32, K4, the default 64 x 64 world, six seeds, 60k ticks; seeds 1–3 on the 4090, 4–6 on the 3090.
- **Reference:** vm-00073's K4, 4 transformers at ctx 8, not rerun.
- **A longer context also means more tokens per update:** the replayed windows are ctx steps long, so C32 trains on 4 times C8's tokens.
- **Frozen:** `eval_hunger.py chain`; `eval_hidden.py aim` with the food hidden after 4 frames and after 1.
- **Checks passed before the start:** 300-tick lives at ctx 8 and 32 (36.7 and 12.5 ticks/s with the GPU shared), and both frozen tools on a one-transformer model.
