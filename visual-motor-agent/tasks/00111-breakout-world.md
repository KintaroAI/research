# Task 00111: A Breakout World for the Default Agent (experiment vm-00103)

**Date:** 2026-11-20, revised 2026-11-22
**Status:** Complete (vm-00103, 2026-11-22: the default agent learns to play, hit share .68 lived at 50–60k and .64 frozen at 60k against a random paddle's .24, all 12 seeds). The owner, 2026-11-22: "lets do breakout on our default model - the goal is to have new world without changing the model itself (only custom wiring)"; first proposed 2026-11-20, on "arkanoid or breakout type of game - do you think our model can learn to play it?": "Yes write task and I like your plan")

## Context

- **The owner's question:** can the default agent learn a Breakout-like game?
- **The goal (2026-11-22): a new world, the model unchanged.**
  - The agent is the default, `configs/U_agent.json` (vm-00101's TT): the same model, label, instinct and learning.
  - Only the wiring is new: what the frame shows, what the move does, and which events feed and hurt the body.
  - No new model code, no new training rule, no world-specific path in `lifetime.py`.
- **What fits:**
  - the game's events map onto the body values the agent already has: the paddle hitting the ball is a meal (hunger to 0), losing the ball a poison (+0.5 hunger, the full dose counted, task 00110);
  - the meal label (task 00108), re-labeled at every replay (task 00112), needs no new machinery;
  - the whole court is in view, unlike the 64 × 64 window of `U_view`;
  - the control is one axis, and moving toward the ball's x catches most balls when the paddle is fast compared with the ball.
- **What will be hard:**
  - a miss is decided long before it happens: a ball takes about 40–50 ticks to cross the court, and the meal label is right only out to about 10 ticks (vm-00101);
  - the ball is one 2 × 2 px object among still pixels;
  - one push of the paddle changes the outcome little until the ball is near.
- **Scale:** DQN needed about 10 million frames to play Breakout well; a life here is 60k ticks. The aim is the basic skill, keeping the paddle under the ball, not clearing walls.

## The world (`world/breakout_world.py`, `env.kind = breakout`)

- **The court:** 64 × 64 px, grey, the model's frame as it is (`model.img` 64, the same 4 quarter transformers of 32 × 32 px).
  - Walls 2 px wide on the left, right and top; the background dark as `CircleWorld`'s.
  - The paddle: a bar 12 × 2 px near the bottom (its top at row 60).
  - The ball: a 2 × 2 px square.
  - No bricks in stage 1.
- **The move (the wiring of the move token, unchanged):**
  - its dx moves the paddle by dx × `paddle_speed` px (4.5), clipped to the court;
  - its dy and click do nothing, as a car ignores the click. The model still predicts and sends all three.
- **The ball:**
  - moves `ball_speed` px a tick (1.5), with sub-pixel positions, drawn at the nearest pixel;
  - bounces off the walls and the top;
  - off the paddle, it leaves at an angle set by where it hit (up to 60° from vertical at the paddle's ends), so aiming is possible.
- **Events and the body (the same two body values as `CircleWorld`):**
  - **hunger** drifts +`h_rate` (.001) a tick;
  - **a hit is a meal:** hunger to 0 (`h_relief` 1);
  - **a miss is a poison:** the ball passes below the paddle, hunger +`miss_add` (.5), and the part the cap at 1 cuts off is reported as `hunger_cut` (task 00110);
  - after a miss the ball is gone for `pause` ticks (8), then comes back from just under the top wall, at a random place, falling at a random angle within 45° of vertical;
  - **restlessness** follows `CircleWorld`'s rule (+`b_rate` a tick, relieved by the paddle's movement within a movement budget), so the body token means what it means in the default world.
- **The world never resets** (continual, as all the default agent's worlds).
- **Info for the loop** (as `CircleWorld`'s):
  - `terminal_image`, `terminal_drives`;
  - `success` (a hit), `gt_hazard_click` (a miss: the loop's poison count), `hunger_cut`;
  - `timeout`, `click` and the `ep_*` keys (empty in a continual world).
- **Ground truth, for scoring only:** `gt_ball`, `gt_ball_vel`, `gt_paddle`, `gt_ball_live`.
- **`configs/U_breakout.json`:** `U_agent.json` with the `env` block replaced by the Breakout world's. The `model`, `online` and `ppo` blocks are the default agent's, unchanged.

## The run: vm-00103

- **One arm, BD:** the default agent in the Breakout world, 12 seeds, 60k ticks.
  - 6 lives per card: seeds 1–6 on the RTX 3090, 7–12 on the RTX 4090, with MPS if the daemon is up.
  - Checkpoints every 10k; the video and frozen tests from them.
  - The instinct's horizon (KH, task 00107) and action repeat, arms of the first plan, are left out. They change how the agent acts, and the question now is the default agent unchanged.
- **Walkers for scale** (`breakout_walkers.py`, no model, the same world and seeds):
  - **random:** a random push each tick, held for 6 ticks (as `world_strip`'s random moves);
  - **still:** the paddle never moves;
  - **tracker:** pushes toward the ball's x (a script reading `gt_ball`; the best simple play).

## Measures

- **Lived**, by 5k and at 40–60k:
  - hits and misses per 1k ticks, and the hit share (hits over hits and misses);
  - mean hunger, overrides, move CE, prediction loss.
- **Frozen** at 10k to 60k (`lifetime.py --init`, `ppo.lr = 0`, as vm-00102's frozen test): 2,000 ticks in new worlds (seed 100 + s), the same measures.
- **The walkers:** the same measures over 20k ticks per seed.
- **A video** of a frozen life at 60k: the frame as the model sees it.
  - `online.frame_record = N` saves body 0's first N frames; it generalizes vm-00102's `panel_record` to any world, and it is the only change to `lifetime.py`.

## The readings, set before the results

1. **Plumbing:**
   - the world's tests pass: bounces, hits, misses, the body's events, `hunger_cut`, the paddle at the walls, the pause, determinism by seed;
   - all 12 lives run 60k ticks with the default agent's settings unchanged (checked against `U_agent.json`).
2. **It plays:** BD's hit share at 40–60k is above the random walker's, paired by seed.
3. **It plays well (descriptive):** BD's hit share as a share of the tracker's, lived and frozen.
4. **The label** (descriptive): mean hunger and hits per 1k over the life, as the default world's curves.

**What follows:**
- **If 2 holds:** stage 2 (a narrower paddle, a faster ball, bricks as a second source of meals).
- **If 2 fails:**
  - check whether the model predicts the ball at all (its prediction of the ball's patches), and whether the label tells moves apart near a miss;
  - then the label's horizon (task 00114) or the instinct's horizon (task 00107) before a new world.

## Implementation

- [x] `world/breakout_world.py` (`BreakoutConfig`, `BreakoutWorld`), `make_config` / `make_world` for `kind = breakout`.
- [x] `tests/test_breakout_world.py`.
- [x] `configs/U_breakout.json`.
- [x] `online.frame_record` in `lifetime.py` and `online.py`'s defaults.
- [x] `breakout_walkers.py`.
- [x] A short life from `U_breakout.json`: it runs, and its model and online config equal `U_agent.json`'s.
- [x] Timing; `00103-breakout/` (README, Makefile); the runs, frozen tests, video, write-up.

## Verification

- **Information hygiene** as in every world: the model receives the frame, the body (hunger, restlessness, the label) and the previous move only; `gt_*` for scoring only.
- **Determinism:** the world's frames and events repeat exactly for a seed and the same moves (tested).
- **The model unchanged:** the resolved `model`, `online` and `ppo` config of a Breakout life equal a default life's (tested by diff, as for `U_agent.json`).
