# Task 00060: Circles That Never Overlap, and a Council of Pieces (experiment vm-00052)

**Date:** 2026-10-30
**Status:** Complete

## Context

- vm-00051 (task 00059): four quadrants with the winner by gain (QG) make 42.8 food and 4.8 poison per 1k ticks against
  one model's 19.0 and 10.5. The owner asked whether poison still gets through as a "win", and how to reduce it.
- `eval_pieces.py --touch` on QG's frozen 60k checkpoints, 339 poison touches, three seeds:
  - **The winner's curse.** The piece that made the move foresaw the rise (> 0.1) in about 1% of the touches (one model,
    F: 30–69%). Gain picks the lowest predicted hunger, so it favours the piece that does not see the poison. Only
    another piece foresaw it in 60%; nobody in 40%. Another piece's move would have missed the poison in 78%.
  - **Food on the same tick:** 26% of the touches (food clears hunger, poison then adds 0.5, so hunger above 0.5 makes
    this a real predicted win; below it, a loss the model may not foresee).
  - **The owner's scenario, poison soon after food:** no enrichment; touches by ticks since food follow all ticks.
  - **High hunger:** 14% of the touches start above 0.5, where poison is cheap (it caps at 1).
- `--rule council` (every piece scores every piece's move, the lowest mean is made), frozen, on those same weights:
  36.8 food, 1.7 poison, against gain's 43.0 and 4.7. `veto` (the lowest worst score): 33.4 and 1.4.
- A frozen rule does not say what learning under it does: vm-00051's QR showed that the rule shapes the learning.

## Design

- **`env.no_overlap`** (default off; needs `wrap` and circular hazards; `world/circle_world.py`): no two circles
  overlap. The general rule, not one for opposites.
  - A new circle (reset, respawn) is resampled until it clears the others (`_clear`), with the existing 1000 tries.
  - After the circles move (`_resolve_overlaps`): two moving circles swap the normal parts of their velocities, equal
    masses, and each is pushed out by half; a moving circle meets still food: it reflects and is pushed out by all of
    it. Dead food takes no part. Off: no new draws from any generator, so old worlds repeat exactly.
- **`online.piece_rule = council`, `veto`** (lifetime.py, GPU, instinct on): `pieces_move` has every piece score every
  piece's move, right after its pass (`uni_candidates`), a matrix [scorer, body, mover]; the lowest mean or lowest
  worst is made whole. One function for `lifetime.py`, `eval_pieces.py` and `forage_video.py`.

## Verification

- `tests/test_no_overlap.py`: no overlap at any tick of a random walk, for the torus distance; food never moves;
  hazards keep their speed against food and their total speed against each other; respawns clear; never food and
  poison on one tick; flag off: identical frames and hazards to before, tick by tick.
- `tests/test_pieces_lifetime.py`: council and veto choose as computed by hand, and run a lifetime.

## Experiment vm-00052

- As vm-00051's QG: `configs/U_touch_ego_c2.json`, `uni_pyr = 3`, `sum_thumb = 1`, quadrants, seeds 1–3, 60k ticks.
- **QN:** gain, `no_overlap`. **QC:** council, the old world. **QCN:** council, `no_overlap`. Control: QG.
- **Measure:** food and poison per 1k ticks over 40k–60k; frozen aim, move size and `--touch`.
- Hypotheses: no_overlap removes the same-tick touches, up to a quarter of the poison; council keeps most of the food
  with about a third of the poison, and may learn better or worse than frozen shows.

## Results

Write-up: [`../00052-no-overlap-council/README.md`](../00052-no-overlap-council/README.md). Over the last 20k ticks: QG 42.8
food, 4.8 poison; QN 43.5, 3.5; QC 29.3, 1.9; QCN 34.9, 2.1. No overlap takes poison down 27% at no cost; council halves it
again at 20-32% of the food, and its paths are twice as smooth.
