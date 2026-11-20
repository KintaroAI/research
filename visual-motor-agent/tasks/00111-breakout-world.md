# Task 00111: A Breakout World (experiment vm-00102)

**Date:** 2026-11-20
**Status:** Proposed (the owner, 2026-11-20, on "arkanoid or breakout type of game - do you think our model can learn to play it?": "Yes write task and I like your plan"; the arms are settled after vm-00099, task 00110)

## Context

- **The owner's question:** can the default agent learn a Breakout-like game?
- **What fits:**
  - the game's events map onto the body values the agent already has: the paddle hitting the ball is a meal (hunger to 0), losing the ball a poison (+0.5 hunger);
  - the meal label (task 00108), stored or re-labeled (task 00109), needs no new machinery;
  - the whole court is in view, unlike the 64 × 64 window of `U_view` where the label flattened past about 8 ticks;
  - the control is one axis (left, still, right), and moving toward the ball's x already catches most balls when the paddle is fast compared with the ball.
- **What will be hard** (from vm-00096 to vm-00098):
  - a miss is decided long before it happens (a ball crosses the court in 30 to 60 ticks), and the meal label is exact only out to about 6 ticks;
  - one tick of paddle movement changes the outcome little, so the instinct's candidates hardly differ (the inertia world's problem);
  - the ball is one small moving object among many still pixels;
  - with three useful moves the model's own move prediction will often be sure of one, and the instinct's 16 candidates are then one move (vm-00098 R8);
  - a run of misses would sit at the hunger cap, where a poison costs nothing unless the full dose is counted (task 00110).
- **Scale:** DQN needed about 10 million frames to play Breakout well; a life here is 60k ticks. The aim is the basic skill, keeping the paddle under the ball, not clearing walls.

## The world (`world/breakout_world.py`, `env.kind = breakout`)

- **The court** in the model's frame (`model.img` 64, grey):
  - walls left, right and at the top; the paddle a bar at the bottom (12 × 2 px to start);
  - the ball a 2 × 2 px square; no bricks in stage 1.
- **The move:** the move's dx moves the paddle by dx × `paddle_speed` px (clipped to the court). dy and the click are ignored, as the universal model's move token is the same in every world.
- **The ball:**
  - moves `ball_speed` px a tick, bounces off the walls;
  - off the paddle its horizontal speed depends on where it hit, as in the game, so that aiming is possible.
- **Events and the body:**
  - a paddle hit is a meal: hunger to 0;
  - a miss (the ball past the paddle) is a poison: hunger +0.5, with `hunger_cut` as in `CircleWorld` (task 00110);
  - the ball then comes back from the top at a random place and angle after a short pause;
  - hunger drifts +.001 a tick as now.
- **The world never resets** (a continual world, as all the default agent's worlds). The `info` keys `lifetime.py` reads: `terminal_image`, `terminal_drives`, `success` (a hit), `hunger_cut`, `gt_hazard_click` (a miss); and for scoring only `gt_ball`, `gt_paddle`.
- **`configs/U_breakout.json`:** the default agent's model (C8's: 4 quarter transformers of 8 layers, tolerance .002, context 8, the ring) in this world.

## The plan in stages

1. **Stage 1, the easy court** (this task):
   - a wide paddle, a slow ball (about 1 px a tick), the paddle three times as fast; no bricks;
   - first a check that the frozen model predicts the ball's path (the dream test, `eval_dream.py`, on the ball's position);
   - two walkers for scale: a random paddle, and a scripted tracker that moves toward the ball's x.
2. **Stage 1's arms** (12 seeds each, on the RTX 3090), the label as vm-00099 leaves it:
   - **K1:** the one-tick instinct;
   - **KH:** the instinct's horizon, each candidate held for 4 imagined ticks (task 00107; `online.ring = 0`);
   - the owner's action repeat (decide every 4 ticks and hold the move; ARCHITECTURE.md section 9) as a third arm if it is built by then.
3. **Stage 2, if stage 1 works:** narrower paddle, faster ball, bricks as a second source of meals.

## Measures

- **Lived** by 5k and at 40–60k: hits and misses per 1k ticks, the hit share (hits over hits and misses), mean hunger, overrides, move CE.
- **Frozen** (`eval_breakout.py`, new): hits, misses and hit share against the two walkers; the label by ticks to the next hit and to the next miss; how different the instinct's candidates are (`eval_candidates.py`); the ball's predicted path (dream).

## The readings, set before the results

1. **Plumbing:** the world's tests (bounces, hits, misses, the body's events, `hunger_cut`, the paddle at the walls, determinism by seed); all lives run 60k ticks.
2. **It sees the ball:** frozen, the dream keeps the ball within 2 px of the real one for 4 ticks.
3. **It plays:** K1's hit share at 40–60k is above the random paddle's, paired by seed.
4. **Lasting choices help:** KH − K1 hit share at 40–60k, paired by seed (and the action-repeat arm if run).
5. **Against the tracker** (descriptive): the best arm's hit share as a share of the scripted tracker's.

**What follows:**
- **If 3 holds:** stage 2.
- **If 2 fails:** the ball is not seen well enough; a bigger ball or the frame's resolution comes first.
- **If 2 holds and 3 fails:** the label's horizon is the limit; next is a longer horizon or the label's back-propagation (re-labeled against stored).

## Implementation

- [ ] `world/breakout_world.py` (`BreakoutConfig`, `BreakoutWorld`), `make_config` / `make_world` for `kind = breakout`; `tests/test_breakout_world.py`.
- [ ] `configs/U_breakout.json`.
- [ ] `eval_breakout.py` (frozen play, walkers, the label's profile); `compare_video.py` / `forage_video.py` support for the world (the court as the view; no window).
- [ ] Timing; `00102-breakout/` (README, Makefile); the runs, frozen tests, write-up, videos.

## Verification

- The model's information hygiene as in every world: the frame, the body (hunger, restlessness, the label) and the previous move only; `gt_*` for scoring only.
- The world's frames and events repeat exactly for a seed and the same moves (tested).
