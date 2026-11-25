# vm-00112: Dreamer in the Food World with Our Horizon

**Date:** 2026-11-24
**Status:** Planned (queued after vm-00110's frozen tests)
**Source:** `exp/vm-00112` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00112/`
**Task plan:** [`../tasks/00125-dreamer-short-horizon.md`](../tasks/00125-dreamer-short-horizon.md)

## Goal

The owner, 2026-11-24: "for now we need to do #1 (0.997) and #2 - make both to be 3 lives".

- **vm-00108:** in the food world DreamerV3 reaches food (44.8 per 1k at 40–60k) but never learns to avoid poison (22.9 per 1k; our TT 8.4).
- **Two candidate causes:** its horizon (discount .997, about 330 ticks, against our .95), or a world model that blurs the poison's checker away.

## Method

- **DG:** vm-00108's DR in the food world (train ratio 512, its defaults) with `discount = .95` (`run_dreamer.py --dset discount=0.95`). Nothing else changed.
- **Seeds 1–3, 60k ticks**, on the RTX 4090 under MPS; paired with vm-00108's DR food seeds 1–3, and against TT.
- **The reconstruction check** (`dev/baselines/dreamer_recon.py`): DR's and DG's world models at 60k, the decoded frame beside the real one, the checker's contrast inside poison against food.

## The readings, set before the results (3 seeds: descriptive)

1. **Is it the horizon?** DG's poison at 40–60k against DR's: down to TT's level says yes.
2. **Is it the look?** Whether the reconstruction keeps the checker.
3. **Food:** DG's against DR's and TT's.

## Results

(pending)
