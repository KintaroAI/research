# Task 00138: A Dream That Rolls Dice (experiment vm-00125)

**Date:** 2026-11-27
**Status:** Complete (2026-11-27), pending tag `exp/vm-00125` (the owner's word: "yes - lets do that. can temperature help?"): a dream that rolls dice serves, and a temperature below 1 keeps most of that at a small cost for the lives that lived with the mix. No new life, one throw of the dice: a look.

## Context

- No dream of the scripted court ever serves a new ball (vm-00122, vm-00123, vm-00124). vm-00124's scratch test says why: shown the real frames up to a serve, every life's expected next frame holds a band of brightness in the court's top rows (0.6 to 1.2 ball pixels in all, the brightest pixel 1% to 5% of a ball's brightness) in the last two ticks before the serve and none four ticks before. The model expects the serve when it is due and does not know where: the world serves at a random column (`breakout_world._serve`).
- A dreamed frame is drawn with `pair`: each pixel's most probable bin and its more probable neighbour. At every column of the top rows that is the empty court. The dreamed frame is empty, the next tick's window is as empty, and so on to the end.
- The owner, asked how to solve it, took my first step: let the dream roll dice where the model is split, on the checkpoints we have; and asked whether a temperature can help.

## The decoding: `draw`

A value (a pixel) is predicted as chances over 32 grey bins. A grey level between two bins is written on both (two-hot), so a sure value is a hump of two neighbouring bins.

1. **The humps.** Every bin belongs to the hump it climbs to: from a bin go to its more probable neighbour while there is one. A hump's chance is the sum of its bins' chances.
2. **The dice.** One random number a value picks a hump by the humps' chances. With a temperature T each hump's chance is raised to 1/T first (as `online.instinct_temp` does for moves): T = 1 the model's own chances; below 1 the likely hump more often; toward 0 the heaviest hump always.
3. **The value** is the sharp value at the chosen hump's peak: the peak and its more probable neighbour, weighted by their chances, as `pair` decodes.

So a value with one hump is drawn exactly as `pair` draws it, whatever the dice: where the model is sure nothing changes. The dice of a dreamed tick are thrown once and kept through its decoding passes, so the decoding settles as before and is the same as drawing token by token. The body's values stay the expected ones, as before.

What the dice cannot do: the 64 pixels of a patch are drawn apart. A served ball is 4 pixels that would have to come up together, so the dice give a pixel of a ball now and then, not a ball. The question of the experiment is what the model makes of that pixel at the next tick.

## Changed on the way (2026-11-27, after the first dreams of one life and before the runs)

- **Two ways to weigh a hump.** In the first dreams (M7B, the free dream) every pixel's likeliest value without dice (`draw0`) was not `pair`'s frame: 0.7% of the pixels differ, and the ball is lost far sooner (the last ball's median tick 35 against 116). `pair` ranks values by their most probable bin. A grey level that sits on a bin has all its chance in that bin; one between two bins has it split. In the court the ball's white (255) sits on bin 31 and the background (20) lies between bins 2 and 3 (.57 and .43), so `pair` draws a ball's pixel from a chance of .36 on, not .50. Today's sharp decoding leans toward the ball, and that is part of what holds the ball in today's dreams. So the dice come in two families: `draw` weighs a hump by its chance (the sum of its bins); `drawp` by its tallest bin, as `pair` does, so that `drawp` without dice (`drawp0`) is `pair` exactly and the temperature leads from today's decoding to the dice.
- **A ball in a long dream is a blob:** a window of 3 x 3 with 3 bright pixels or more. With dice a ball is often drawn a pixel short or long; counted as exact blocks of 2 x 2, such balls read as gone and back.
- **The decodings run:** `pair`; `draw0` (the likeliest value, no dice: what `pair`'s lean is worth); `draw` at temperatures 1, .7 and .5; `drawp` at 1, .7 and .5.
- **A table tool,** `wm_dice.py` (the README's tables and picture from the scored files; `tests/test_wm_dice.py`).

## To build

- `src/vm_cuda.cu`, `src/vm_cuda.h`: the rule as a device function, a kernel for a step's values (`k_uni_draw_step`), `vmc_forward_uni_draw` (as `vmc_forward_uni_step` with the dice, the temperature and the way to weigh given) and `vmc_draw_values` (the rule on given chances, for the tests). Nothing existing is changed.
- `cuda_lib.py`: `CudaModel.forward_uni_draw`, `draw_values`.
- `wm_measures.py`: `draw_values` in numpy (the rule's reference); `Pieces(run, step, "draw", "draw0.7" or "drawp0.5", dice=0)`; `score --decode draw0.7 --dice 0 --out DIR` (the summary names the temperature, the dice and the way to weigh); the free part's count of the court's balls (below).
- `wm_video.py`: a decoding a run (`RUN@draw0.7:LABEL`), so that one life's dreams with and without dice stand side by side.
- `tests/test_draw.py`.
- `00125-dream-dice/`: README, Makefile.

**The court's balls in the free dream** (the `free` part, the court only; vm-00124 counted them by a scratch script): a whole ball is a block of 2 x 2 bright pixels in the ball's zone. By dreamed tick the share of dreams with a whole ball; over all dreams the balls served (a whole ball in the top rows after 3 ticks or more without one, there for 3 ticks or more), the balls that come back elsewhere after such a gap, the tick of the last whole ball, and the specks (bright pixels of the ball's zone that are in no whole ball, a frame). The same for the world's own continuation.

## The runs

No new life. The 60k checkpoints of the five court lives, seed 1: OB (neither) and SB3 (self-forcing) of vm-00122, MB (.5 its own prediction) of vm-00123, M7B (.7) and MNB (.5 with .2 random grey) of vm-00124.

Each with four decodings: `pair` (today's), `draw` (T = 1), `draw0.7`, `draw0.5`; one throw of the dice (`--dice 0`). The exam shown whole real frames. Parts: the free dream of 200 ticks (32 stretches) and the regular part (64 real ticks, a dream of 32). Everything is written to `exp_00125/`; nothing is written into the earlier experiments' folders. `pair` scored again must give the earlier numbers exactly (frozen tests repeat).

## Readings, set before the results

One throw of the dice and 32 long dreams a cell: a look.

1. **Does a dream with dice serve?** The balls served in the 32 free dreams (the world itself: about 11), and the dreams with a whole ball at ticks 100 and 200.
2. **What do the dice cost where today's way works?** The regular part against `pair` on the same checkpoint: the ball's horizon, the ball held at ticks 8 and 16, next-frame skill; and the specks a frame.
3. **Temperature:** which of 1, .7 and .5 serves at the least cost?
4. **Do the lives that lived with the mix make a ball of a drawn pixel more often than the plain ones?** In their lives a new ball always came as a fragment (some of its 4 pixels real, the others from their own prediction, which has none), with the whole ball as the target.

**What I expect** (written before any dream with dice was made):

- T = 1: after the ball is lost a bright pixel comes up in the top rows about once a tick. MB and M7B make balls of some of them and serve several times in the 32 dreams; OB and SB3 seldom or never; MNB I cannot call (it has learnt to look through grey specks, but a ball's pixel has a ball's exact brightness).
- The cost at T = 1 is plain: the ball's horizon in the regular part a few ticks shorter, next-frame skill a little lower, a few specks a frame.
- T = .5: nearly no cost and fewer serves (a drawn pixel every 40 to 100 ticks), still some for the lives with the mix. T = .7 between.
- A served ball is lost again as the others are, so the dreams do not hold a ball to tick 200 much more often; but a court that serves is not empty for good.

## Results (2026-11-27)

Forty scorings of five checkpoints (no new life), one throw of the dice a cell and a second throw of the main cells of the long dream (a look). The README has the tables, the picture and the readings one by one.

- **1. Does a dream with dice serve?** Yes. With `pair` no life serves a ball in its 32 long dreams; with dice by the tallest bin at T 1 3, 11, 20, 14 and 18 balls (the world 12), and a ball is in the court at tick 200 in .719 to .969 of the dreams against .000 to .500.
- **2. The cost in the short dreams:** against the world's own continuation the dice lose, the more the higher the temperature: at T 1 the ball held 6 to 11 dreamed ticks against 9 to 15, next-frame skill −.056 to −.303; at .5 the ball's horizon within 2 ticks of `pair`'s in every life. Stray pixels: under one a frame.
- **3. Temperature:** it helps the lives that lived with a mix. At .7 they serve 23, 7 and 18 balls and have a ball at tick 200 in .844 of their dreams each; the lives without a mix serve few or none below 1 (OB 0, SB3 4 at .7).
- **4. The lives with the mix and a drawn pixel:** yes. By a scratch test (the serve's frame shown with one of the ball's 4 pixels, `pair`, no dice) MB and M7B dream a ball in .84 and .86 of the 73 cases, OB and SB3 in none, MNB in .19 (from two pixels .67).
- **Found on the way:** `pair` is not the model's likeliest value. It leans toward grey levels that sit on a bin (the ball's white, 255, on bin 31; the background, 20, between bins 2 and 3), and with every pixel's likeliest value (`draw0`) the long dream's last ball goes sooner in every life (median tick 94, 78, 98, 35, 17 against 185, 138, 116, 116, 60) while the next frame is predicted a little better.
- **What I expected:** right that the lives with the mix serve and OB seldom, that T .5 costs little and still serves for them; wrong that the dreams would not hold a ball to tick 200 much more often (they do, by far), that SB3 would seldom serve at T 1 (it serves as often as the mix lives), and on the size of the next frame's cost at T 1.

## Verification

- `tests/test_draw.py`: a value with one hump is `pair`'s whatever the dice; two humps are drawn by their chances, and by their chances raised to 1/T; the kernel against the numpy rule on random chances; the same dice give the same dream, other dice another; the decoding settles and another first guess gives the same frame. `tests/test_cuda_model.py` and the whole dev suite after the change to the CUDA code.
- `pair` scored into `exp_00125/` equals the lives' earlier summaries number for number.
- Every scored file names its life, its decoding, temperature and dice, 64 windows a pass, nothing unsettled.
- Done: `tests/test_draw.py` (7), `tests/test_wm_dice.py` (2), the whole dev suite 2,170 passed, 34 skipped; 495 checks over the 95 scored files (15 of them the second throw of the dice), 0 failed, `pair` equal to the earlier summaries number for number (the free dream, the regular and the course parts of all five lives).
