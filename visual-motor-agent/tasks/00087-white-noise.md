# Task 00087: White Noise, Where Does It Break (experiment vm-00076)

**Date:** 2026-11-02
**Status:** Complete (stages 1 and 2 in vm-00076, stage 3 in vm-00077)

## Context

- The owner (2026-11-02): "we want to run several lives with white noise over images of different level to see at which point it breaks". "im ok with no clean truth".
- ARCHITECTURE section 13 logs the idea: learning should suffer from both a white ceiling and white noise, the two ends of the spectrum.
- It differs from clogged channels (vm-00074): there the noise is on the input of one transformer with a clean target to restore. Here it is in the world: input, targets and memory all carry it, and all four transformers see the same noise. Part of the world simply cannot be predicted.

## Design

**`env.noise = q`:** on each tick each pixel of the frame is replaced by a random brightness with probability q, fresh every tick, from its own generator, so that `noise = 0` is the world as it was, draw for draw. q = 0 is the control, q = 1 leaves nothing of the world in the frame. The food, poison, hunger and move are real whatever the frame shows. `env.clean_frame()` gives the frame without noise, for scoring only.

**Stage 1, find the region:** K4 with log summaries, the default world, 60k ticks, q = .05, .1, .25, .5, .75, .9, .95, 1, seeds 1–3, 24 lives, N5 to N100. The two lowest were added for the owner's prediction, below. The control is vm-00073's K4, seeds 1–3, not rerun. The levels bunch at the top because averaging over the roughly 250 pixels of a circle should cancel much of the noise. Seeds 1 and 2 on the 4090, seed 3 on the 3090.

**Stage 2, zoom in:** six seeds at two or three levels around the drop, chosen after stage 1.

**Stage 3, later:** noise on the input only, with clean targets, at the break point: are noisy targets what breaks learning?

## Measures

- **Lead over a blind walk:** frozen `eval_hunger.py chain`, food seen minus food blind, and for poison blind minus seen. Control, K4: about 57 food.
- **Half broken:** the level where the lead falls to half the control's. **Fully broken:** the lead is zero within its interval.
- **Cross-tests:** each final model at each test level, `eval_noise.py chain` with `--levels`, giving a table of train level by test level, and `eval_noise.py error`: prediction against the clean frame, on what holds something.
- **Tokens:** `uni_ce_body`, `uni_ce_patch`, `uni_ce_move` from each life, against the level.

## The owner's prediction

"I would suspect it will climb up better and better with noise and then down - so there will be some sort of maximum there." vm-00074 gives a reason to expect it: damage in training cut poison to about a third on clean input and made moves steadier, and the lower level did at least as well on food. Against it: there the targets stayed clean, here they carry the noise, so nothing is restored and unpredictable targets are added. Two curves can show a maximum, and they need not agree:

- **Lived in it:** each arm at its own level.
- **Raised on it:** each arm tested on clean frames. Noise while acting also blocks the view, which hides a benefit of training, so a maximum may show here and not in the first.

## The three ways to fail

| Failure | Sign |
|---|---|
| It cannot see: the signal is gone | a model raised on clean frames also fails when tested at that level |
| It cannot learn: raised on noise, stunted | a model raised on noise does badly even when tested on clean frames |
| It is drowning: patch tokens with unpredictable targets swamp the body token | the body's cross-entropy rises with the level, and the model's food at low test levels falls with the level it was raised on, while vision of the clean frame through `error` holds |

## The readings, set before the results

1. **Plumbing:** `noise = 0` is the old world draw for draw; the share of replaced pixels is q; q = 1 keeps nothing of the frame; the noise is fresh each tick; memory and targets carry it. **Met:** 6 unit tests, 5,000 ticks draw for draw.
2. **The floor:** at q = 1 the lead over blind is zero within its interval, three seeds. **Met:** food −.56 ± 1.40, poison −.43 ± 4.63.
3. **Poison first:** the lead on poison falls to half at a lower level than the lead on food. **Met:** lived, about .3 against .5; frozen at its own level, the same way but narrower (.37 and .45 of the lead left at .5).
4. **Which failure:** named from the cross-tests, at the level where food is half broken. **Read: it cannot see, in part.** At .5 the clean-raised model gets nothing, the model raised at .5 keeps half (+30.9 ± 5.9 over K4) and none does better; drowning is ruled out (body CE .26 to .29, clean vision worse with the level raised on); the sign of not learning shows but is adaptation, not stunting.
5. **The maximum (the owner's prediction):** **Not met** in stage 1: lived .1 75.4 against 73.5; tested clean no arm above K4; clean poison at .1 −.85 ± 2.06, lower in all three seeds, the only hint. on either curve, an arm above the control by more than its interval, with higher levels below that arm. Food, poison and aim are read separately. With three seeds the interval on food is very wide, about ±60 going by the spread of C10 against K4 in vm-00074, so stage 1 can only suggest a peak in food, and stage 2 repeats any peak with six seeds. Poison and aim are tighter.

## Implementation

- [x] `world/circle_world.py`: `noise`, its generator, `clean_frame`
- [x] `tests/test_noise.py`
- [x] `eval_hunger.py`: `chain` at a test level; `eval_noise.py`
- [x] `00076-white-noise/`: README, Makefile
- [x] Short life on the GPU
- [x] Stage 1 runs and frozen tests, write-up
- [x] Stage 2 (2026-11-03): N10 seeds 4-6, frozen at 0 and .1. Six seeds against K4 tested clean: poison +1.04 ± 2.51, food −13.3 ± 17.0, aim −.048 ± .056, so the hint is gone and the maximum is not met. Tested at .1: food +30.5 ± 11.0, poison −9.4 ± 4.8, aim +.054 ± .033. Lived last 20k: food +.4 ± 15.0, poison +1.11 ± 1.13. The owner, from the video: at .1 the poison checkers are still visible, at .25 almost not
- [x] Stage 3: task 00088, experiment vm-00077
- [x] Write-up of stage 2 in `00076-white-noise/README.md` (R6)

## Verification

Unit tests of the world, the old draw-for-draw check, and a short life at q = .5.
