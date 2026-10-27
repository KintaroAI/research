# Task 00042: The Touch World — Collect and Avoid by Moving Alone (experiment vm-00035)

**Date:** 2026-10-27
**Status:** In progress

## Context

- In the circle world food and poison act only on a click inside them. The agent (vm-00031 S, seed 2) clicks on ~23% of
  its steps wherever it is: it avoids clicking poison (P(click | poison) 0.09) but barely seeks food (0.29 against 0.23
  elsewhere). Foraging there mixes two skills: getting to a circle and clicking it.
- Self-forcing (vm-00032) keeps the dreamed cursor 3.6–5.2 px from the real one after 16 steps (against 7.6–18.6), but
  behaviour did not improve, and the 4-step lookahead gained no more with it.
- A world where contact alone counts makes the question purely one of movement: does the agent steer into food and
  around poison, and does a truer dream (self-forcing) let the lookahead do it better?

## World (`env.touch`, default false = unchanged)

- **Food acts on contact:** the cursor inside the alive circle after the move is a success (relief `h_relief`), with or
  without a click; the circle respawns elsewhere (`respawn`), never on the cursor.
- **Poison acts on contact:** the cursor inside a hazard after the move applies `hazard_click` / `hazard_click_add` (no
  click needed; `gt_hazard_click` is set: "a poison event"), and that hazard disappears: a new one (new radius,
  position, direction, from the hazard generator) appears elsewhere, never on the cursor. Hazards stay 2.
- The click still exists in the action (the model predicts it as any other value); it does nothing.
- **Smaller circles:** food radius 8–11 (was 12–16); poison keeps 5–8.
- Config `configs/U_touch.json`: `U_universal.json` with `touch = true`, `radius_min = 8`, `radius_max = 11`.

## Runs (100k ticks, seeds 1–3, the instinct on: 16 candidates, 1 step)

- **TB:** the touch world, no self-forcing.
- **TSF:** the touch world with self-forcing (8 windows, 1–3 steps, J = 2, pair; vm-00032's setting).
- **TSF4** (if the frozen 4-step pair lookahead beats the 1-step instinct): TSF with the 4-step pair lookahead in life
  (`instinct_h = 4`, `instinct_decode = pair`).

## Measures

- In life: food and poison touches per 1k ticks over the life.
- Frozen at 100k, 16 fresh worlds × 1,000 steps (`online.py --eval-tests forage`): food and poison per 1k with the
  instinct off (the move token's own policy), the 1-step instinct, the 4-step lookahead with mean and with pair
  decoding.
- Chance: a random walk (moves uniform in [-1, 1]²) in the same worlds: food and poison per 1k, and their ratio
  (circle areas and motion set it).
- The dream curve (`eval_dream.py`) at 100k for TB and TSF (cursor distance after 16 dreamed steps).

## Verification

- World tests: contact without a click relieves hunger and respawns the food; contact with a hazard applies the poison
  and moves the hazard away from the cursor (radius within range); with `touch = false` the world is unchanged (a
  random action sequence gives the same frames and drives as before the change); state round trip with `touch`.
