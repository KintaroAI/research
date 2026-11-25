# Task 00125: Dreamer in the Food World with Our Horizon (experiment vm-00112)

**Date:** 2026-11-24
**Status:** Complete (vm-00112: neither the horizon nor the look; the world model sees poison and predicts its cost, Dreamer's acting stops at a third of touches poison). The owner, 2026-11-24: "for now we need to do #1 (0.997) and #2 - make both to be 3 lives"

## Context

- **vm-00108:** in the food world DreamerV3 learns to reach food (44.8 per 1k at 40–60k) but not to avoid poison: its poison stays at about 23 per 1k from 20k on, the level of its first 10k (TT 8.4). In Breakout it plays nearly perfectly.
- **Two candidate causes:**
  - **The horizon:** Dreamer discounts by .997 (a horizon of about 330 ticks); food is plentiful, so a poison's cost (.5 of hunger against a meal's +1) may weigh little against the food behind it. Our agent uses .95.
  - **What its world model sees:** food and poison differ only in look (plain against checkered circles, a few pixels across in the 64 × 64 view); its reconstruction may blur the checker away.

## The arm

- **DG:** vm-00108's DR in the food world (`configs/U_agent.json`'s world, train ratio 512, its defaults) with `discount = 0.95` (`run_dreamer.py --dset discount=0.95`). Nothing else changed.
- **Seeds 1–3, 60k ticks**, on the RTX 4090 under MPS (3 at once). Paired with vm-00108's DR food seeds 1–3, and against TT.

## Measures

- Food and poison per 1k by 10k lived, against DR and TT, paired by seed.
- **The reconstruction check** (`dev/baselines/dreamer_recon.py`): DR's and DG's world models at 60k on frames from a new life in the same world: the decoded frame beside the real one, and the decoded checker contrast inside poison against food (from the world's ground truth, for scoring only). Does the model keep the poison's look?

## The readings, set before the results (3 seeds: descriptive)

1. **Is it the horizon?** DG's poison at 40–60k against DR's: down to TT's level says yes.
2. **Is it the look?** The reconstruction keeps the checker (food and poison distinguishable in the decoded frame) or not.
3. **Food:** DG's food against DR's and TT's.

## Implementation

- [x] `00112-dreamer-short-horizon/` (README, Makefile: vm-00108's `make dreamer` for the food world with `--dset discount=0.95`, arm DG, seeds 1–3 on GPU 1).
- [x] `dev/baselines/dreamer_recon.py` (also the reward head at food and poison ticks).
- [x] The runs, analysis, write-up.
