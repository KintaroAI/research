# Task 00107: The Instinct's Horizon in the Inertia World (experiment vm-00096)

**Date:** 2026-11-17
**Status:** Complete (2026-11-18; experiment vm-00096, `00096-instinct-horizon/`; the owner, 2026-11-17: "go ahead with the instinct horizon experiment")

## Context

- **Where vm-00095 (task 00106) left this:** the default agent (C8) learns the world with inertia poorly from birth. Reading 2 failed: IB's lived food at 40–60k was .41 of C8r's (19.9 against 48.4), against a bar of .5.
- **The plan's rule for that outcome:** "the model does not learn to forage with inertia. Its instinct looks one move ahead and its hunger label 4 ticks, while a push pays only over several ticks. Next is the instinct's horizon (`online.instinct_h`) or the value's."
- **What vm-00095 suggested:**
  - The instinct overrides the model's own move in 58–70% of ticks (C8r 27%), and more as the life goes on.
  - Its candidates differ by one push, which moves the body's next position by at most 1 px per axis. Their predicted labels spread 5 times as widely as C8r's, in proportion to the labels.
- **The horizon:** each of the instinct's 16 candidate moves is held for H ticks, the ticks after the first imagined by the model, and scored by the predicted label after the last (task 00029, `online.uni_lookahead`). With H = 4 = k, a candidate is a course: 4 ticks of one push take the body from rest to full speed. The candidates then differ by what the body would actually do.

## What has to change

- **The lookahead for pieces:**
  - `uni_lookahead` was written for one model that predicts the whole frame. Since task 00057 each of the 4 quarter transformers predicts only its own 32 × 32 block, so the lookahead fails ("could not broadcast (16, 32, 32, 3) into (16, 64, 64, 3)").
  - The fix: an imagined step takes the piece's own block from its prediction, and the rest of the frame stays as the current frame (`agent_lib.band_piece` gives the block). Each piece still scores its own candidates, and the body's move is chosen by the run's rule, as now.
- **No ring:**
  - The ring (task 00062, acting step by step on a key and value cache) refuses `instinct_h > 1` (`lifetime.py`). The horizon arm therefore acts on full windows (`online.ring = 0`).
  - The two are not the same: in the ring a cached step keeps what it read when new (`tests/test_ring.py`, "the accepted difference"). So the control is the same agent without the ring and with H = 1. Timing: without the ring a single life is as fast (27.3 ticks/s, against 26.9 with it).
- **The frozen tests act on the ring only** (`eval_hunger.Life`). They need the full-window path that `forage_video` already has (the last ctx steps) for runs with `online.ring = 0`.

## Arms (all in the inertia world from birth, `env.inertia = 4`, C8's settings otherwise)

| Arm | Ring | Instinct's horizon | Seeds |
|---|---|---|---|
| IB (vm-00095, linked, not rerun) | yes | 1 | 1–12 |
| **IR** | no | 1 | 1–12 |
| **IH** | no | 4 | 1–12 |

- **IH vs IR is the test.** IR vs IB shows what dropping the ring alone does.
- **Settings:** `--config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 --set online.value_gamma=0.9 --set online.value_refresh=1 --set env.inertia=4 --set online.ring=0`, plus `--set online.instinct_h=4` for IH.
- **The imagination's settings** (`instinct_jacobi`, `instinct_decode`) were set by the timing below. The rule set beforehand: J = 2 and `pair` decoding (task 00038: sharp, and exact for two-hot values) unless 60k ticks would take more than about 10 h, then J = 1.
- **Timing (2026-11-17):**
  - Single lives on the 4090 in the inertia world: H = 1 at 27.3 ticks/s without the ring (26.9 with it); H = 4 at 7.6 with J = 2 (mean or pair decoding alike) and 11.2 with J = 1.
  - The planned layout, 24 lives under MPS for 300 ticks: IR 2.1 ticks/s per life on the 4090 and 2.5 on the 3090, IH with J = 2 at 1.4 and 1.6. That comes to about 10.8 h on the 4090.
  - **So, by the rule, IH runs J = 1 with pair decoding:** `--set online.instinct_h=4 --set online.instinct_jacobi=1 --set online.instinct_decode=pair`, about 8 h. Neither setting does anything at H = 1.
- **Cards:** seeds 1–8 of both arms on the 4090, 9–12 on the 3090.

## Measures

- **Lived, 40–60k and by 5k:** food, poison, mean hunger, instinct overrides, move CE, patch CE, body CE, smoothness. Paired by seed: IH − IR, IR − IB, IH − IB.
- **Frozen in the inertia world**, each agent acting as it lived (IR and IH on full windows, with their own horizon), 1,000 ticks × 16 bodies:
  - `chain` and `move` (speed, braking, aim of the move and of the motion by the gap to food and poison, and now the mean hunger), for both arms.
  - `value` and `reach` for IR only. Under H = 4 the predicted label of the move taken is the label after 4 imagined ticks, not comparable with H = 1's. `reach` scores one-push candidates, while IH's are 4-tick courses.
  - Why the trim: a frozen tick of IH costs about 0.75 s for 16 bodies (the lookahead for 256 candidate windows a pass), so 1,500 ticks would take about 18 minutes per run per test.

## The readings, set before the results

1. **Plumbing:** the lookahead and Life tests pass; all 24 new lives run 60k ticks without errors; every run's `info.json` has its arm's settings.
2. **The horizon helps the inertia world:** IH's lived food at 40–60k is above IR's by more than its interval, paired by seed, and its lived mean hunger is not higher.
3. **Enough to forage:** IH's lived food at 40–60k is at least .5 of C8r's (vm-00094, 48.4): vm-00095's reading 2 bar.
4. **The ring alone** (descriptive): IR − IB at 40–60k in food, poison and hunger.
5. **How it moves** (descriptive, frozen): does IH brake before food, slow near it, or keep away from poison more than IR?

**What follows:**
- **If 2 holds:** the horizon is what this world needs. Next is its cost (H = 4 in the world without inertia, against C8r), and the switch (IS) with it.
- **If 2 fails:** a course of 4 ticks is not enough, or the imagination is too poor to score it. Next is the tolerance scaled to the labels (IB with `instinct_tol` .01), the presumed cause from vm-00095, or the value's horizon.
- **Either way** (added 2026-11-17 during the runs, the owner): **action repeat**, deciding every 4 ticks and holding the move, as DQN did (ARCHITECTURE.md section 9). It separates committing to a course from imagining it: if it matches IH, the commitment helped, not the imagination.

## Implementation

- [x] `online.uni_lookahead`: pieces (band_n ≥ 2) write their own block of the imagined frame; the rest stays the current frame (`uni_frame(cfg, pv, grid)` takes a block's grid of patches).
- [x] `eval_hunger.Life`: runs with `online.ring = 0` act on full windows (the last ctx steps), with their own horizon (no clog, outage, view or state there). `eval_hunger.py move` reports the mean hunger too.
- [x] `tests/test_lookahead.py` (40 pass): a piece's lookahead at H = 1 equals its one-step reading; at H = 2 and 4 the real steps are untouched and each imagined frame is the current frame with only the piece's block replaced; `uni_act` with a horizon runs for a piece; `Life` without the ring hands `uni_act` the window of its last ctx steps (frames in order, the moves made, the current move zero, the episode start).
- [x] Timing (24 lives on both cards); `00096-instinct-horizon/` (README, Makefile). The runs ran 2026-11-17 14:19 to 2026-11-18 00:05, the frozen tests until 04:23. Analysis, write-up and videos (`videos/vm96_s5`, `vm96_s12`) are done.

## Results (summary; full report in the experiment README)

- **Readings:**
  1. Plumbing: **met.**
  2. The horizon helps: **met.** IH − IR at 40–60k: food +8.5 ± 3.6 (12 of 12), poison −5.5 ± 2.2, hunger −.093 ± .067 (lower in 10).
  3. Enough to forage: **met.** IH's food is .55 of C8r's; IR .38, IB .41.
  4. The ring alone (descriptive): IR − IB food −1.7 ± 1.5, poison +2.7 ± 1.7.
  5. How it moves (descriptive, frozen): IH's motion aims at food better and from farther (+.12 within 8 px, +.15 at 16–32 px, 12 of 12), with no more braking and no slowing for food.
- **The gap opens after 30k and grows** (+4.8 at 40–50k, +12.1 at 50–60k). IH is still climbing at 60k (33.5 food per 1k at 55–60k).
- **Frozen in the inertia world,** IH is the best agent yet: 34.2 food, 8.1 poison, hunger .130.
- **What follows (set above):** reading 2 held, so next is the horizon's cost without inertia (H = 4 against C8r) and the switch with it; action repeat (ARCHITECTURE.md section 9) tests whether the commitment alone does it.

## Verification

- [x] Every run's `info.json` has `env.inertia` 4, `online.ring` 0, `online.instinct_h` 1 (IR) or 4 with J 1 and pair decoding (IH), and C8's settings otherwise (all 24 checked).
- [x] IR's lived numbers are close to vm-00095's IB (the same agent and world, with the ring): food 18.2 against 19.9 (−1.7 ± 1.5), poison 17.1 against 14.3 (+2.7 ± 1.7). The ring helps a little; this is reading 4.
