# Task 00043: The Touch World, Egocentric (experiment vm-00036)

**Date:** 2026-10-27
**Status:** Complete (results in vm-00036)

## Context

- In the touch world (task 00042, vm-00035) the agent collects food and avoids poison unrewarded; self-forcing with
  the 4-step `pair` lookahead in life (TSF4) cut poison to 3.0 per 1k frozen (TB 10.7) at ~20 food, because its dream
  keeps the cursor (2.5–3.6 px off after 16 steps against 6.3–14.6).
- The world's view was allocentric: the cursor moves over a fixed scene. With `env.egocentric` (vm-00011, needs
  `wrap`) the cursor stays drawn at the center and a move shifts the whole scene. In vm-00012 the egocentric view
  raised the planner's foraging 4× (135–142 against 28–33 hits per 1k).
- Hypothesis (the user's): egocentric, poison contacts drop further. Contact is always at the same place in the view
  (the center), so "a poison circle reaches the center" is one thing to learn, not one per position, and the dream
  cannot lose the cursor. Against: every move shifts every patch, so the frame is harder to predict.

## World

- `configs/U_touch_ego.json`: `U_touch.json` with `egocentric = true` (the dynamics are unchanged: hit tests are in
  world coordinates; only the view differs). The chance numbers are the same as vm-00035's (`forage_chance.py` does not
  look).
- Note: the poison's checkerboard is in view coordinates (`hazard_pattern = checker`), so as the scene shifts the
  checker pattern stays fixed to the view grid while the circle's outline moves (as in the allocentric world for a
  moving poison).

## Runs (100k ticks, seeds 1–3)

- **ESF4:** the egocentric touch world, self-forcing (8 windows, 1–3 steps, J = 2) and the 4-step pair lookahead in
  life (vm-00035's TSF4 setting). Seeds 1–2 on the 4090, seed 3 on the 3090 after EB.
- **EB:** the egocentric touch world, no self-forcing, 1-step instinct (vm-00035's TB setting), on the 3090 first.
- Compared with vm-00035's TB and TSF4 (same seeds, same frozen worlds).

## Measures

- In life: food and poison per 1k by quarter of the life.
- Frozen at 100k (`online.py --eval-tests forage`, 16 worlds × 1,000 steps): instinct off, 1-step, 4-step mean, 4-step
  pair.
- The dream curve (`eval_dream.py`): pixels exactly right and hunger error after 16 dreamed steps. The cursor distance
  is uninformative egocentric (the cursor is always drawn at the center); the scene's shift is what the dream must get
  right, so the pixel share carries it.

## Video

- `forage_video.py` with egocentric worlds: the path and touch rings drawn in the moving view (positions relative to
  the current cursor, wrapped); `dream_video.py` the same for its touch rings.
- ESF4 s3 (the same seed) vs vm-00035's TSF4 s3, side by side, 1,000 steps, the 4-step pair lookahead, the same world; dream
  videos of ESF4 and EB.

## Verification

- World test: egocentric with touch: the cursor drawn at the center, contact food relieves and respawns, contact poison
  applies +0.5 and is replaced away from the cursor (as allocentric).
- The video scripts unchanged for allocentric runs (vm-00035's compare video's counts reproduce).

## Results (vm-00036)

- Implemented and tested (`tests/test_world_ext.py`: egocentric touch events identical to allocentric for the same seed
  and moves, the cursor drawn at the center); the video scripts draw paths and touches relative to the centered
  cursor.
- Frozen at 100k, the 4-step pair lookahead: ESF4 43.7 food / 2.6 poison per 1k (allocentric TSF4 19.7 / 3.0); EB
  (no self-forcing) 44.4 / 4.1 (TB 19.9 / 10.7). Every egocentric arm forages ~2× its allocentric twin, even with the
  instinct off. In life ESF4 ends at 35–38 food and 1.6–2.5 poison per 1k (TSF4 12–21 / 3.6–5.2).
- The dream: hunger error after 16 steps 0.014–0.028 (ESF4), 0.062–0.103 (EB); the egocentric dream shifts the scene
  but empties after ~100 dreamed steps. Video: ESF4 s3 vs TSF4 s3 in the same world, 41 / 1 against 23 / 2.
