# Task 00136: The Model Sees Its Own Prediction Mixed into Every Frame (experiment vm-00123)

**Date:** 2026-11-27
**Status:** Complete, tagged `exp/vm-00123` (2026-11-27; the owner's word: "lets do the next experiment"): the mix gives the court's dream what self-forcing gives it, harms the disc world's, and does not mend a long dream. One seed an arm: a look.

## Context

- vm-00121 and vm-00122: training on its own dreamed frames (self-forcing: 2 of 7 replayed windows get their last 1 to 3 steps whole from the model's dream) holds the court's ball for 15 to 17 dreamed ticks at 60k against 9, and 26 against 16 at 200k.
- vm-00122's dream of 1,000 ticks: the court's dream loses its ball within 57 to 585 ticks and never serves a new one.
- The owner, 2026-11-27 (the idea): "we need to add some sort of noise into the mix - so model learn to recover lost ball for example. And the way I'm thinking about that is - each time lets say 50% of pixels in current image comes from dream (prediction of the next) - so for current frame we take and swap 50% of random pixels with what was predicted on previous step and then we train that against real image - what is important - model sees only these 50% mixed images and newer full frame (it only used for training and full frames goes to memory but when replayed - it's also used 50% mix for frames - not originals)".

## The idea, as built

The owner, while it was being built: "replay - mix should be re-made each replay" and "describe how you do the mix". So:

`online.mix = q` (0: off; .5 here).

**In the life, at every tick:**

1. **The prediction.** Before the model is shown the new frame it predicts it from what it has seen so far: its last 7 frames as it saw them, the bodies and the moves. It is the model's sharp dream of one step: `online.mix_jacobi` = 2 decoding passes that start from the frame before as seen, `pair`, as self-forcing makes its dreamed frames. The body the frame is predicted with is the model's own prediction of it, so nothing of the tick being predicted enters.
2. **The frame as seen.** A fresh draw for every pixel: with probability q it is taken from the prediction, otherwise from the real frame. The model watches this frame, and the ring holds it for the ticks to come.
3. **Memory** keeps the real frame only.

**At every update (the newest window and the 7 replayed ones):**

1. The windows of 8 real frames come out of memory.
2. The mix is made again, step by step, with the model as it is now: frame 2 is predicted from frame 1 as seen and mixed with a fresh mask, frame 3 from frames 1 and 2 as seen, and so on.
3. The model trains with the frames as seen as its input and the real frames as its targets.

So the model is never given a whole real frame, in the life or in an update.

Where the owner's words leave a choice (my readings, said to the owner):

- **The prediction is the dream of one step,** not the prediction of each patch from the patches before it in the same frame, which has seen part of that frame.
- **A frame with nothing before it** (the first frame of a window, the first tick of a life, a step that starts an episode): the passes start from an empty frame with the real body, so the prediction is what the model expects of a frame it knows nothing about. It is mixed like every other.
- **The body's values are not mixed:** the model sees the real ones, as the owner's words speak of pixels.
- **The masks** are drawn pixel by pixel, fresh at every tick and at every update, from generators of their own.

One function makes the frames as seen for the life, for the update and for the exam (`online.mix_inputs`).

**The cost:** the mix made again for 8 windows of 8 frames is 16 decoding passes an update, 66 ms a tick on the RTX 4090: a life alone runs 12.7 ticks per second there (54 without the mix).

Refused with: several transformers (`online.pieces`), several bodies, no ring, the CPU, `noise_in`, `clog`, `blind`, `env.delays`, panels, `model.state`, `off`, `fast`, surprise gating, a refreshed value or label (their bootstrap windows are real frames), a lookahead (`instinct_h` above 1: it reads real windows), `env.thought` and `pred_record`. Self-forcing may be combined with it: the dreamed steps then start from the frames as seen.

## To build

- `online.py`: `mix_inputs` (the predictions and the frames as seen of windows, made whole or on top of frames already seen), the defaults `mix`, `mix_jacobi`, `mix_record`.
- `lifetime.py`: the option, its refusals, the frame as seen to the ring, the update's inputs made again, the metrics `mix_share`, `mix_wrong` (the share of the pixels taken from the prediction that are not the real ones) and `mix_err` (their mean distance in grey levels), `t_mix`; `online.mix_record = N` keeps the first and last N ticks as real, predicted, mask and seen (`mix_frames.npz`).
- `wm_measures.py`: `Pieces(..., lived=True)` and `score --lived`: the exam's real ticks shown as the life saw frames (the same function, the masks from a generator fixed by the number of windows), in a file of its own (`--name lived`). Without it the exam shows whole real frames, as to every other model. `score --home-from N`: the home set from a later tick than the life's end, so that a life of 60k is scored on the stretch vm-00122's lives were. `wm_video.py --lived`.
- `wm_curves.py`: arms of another output folder beside this one's, and a third arm in the picture.
- `tests/test_mix.py`.
- `00123-mixed-frames/`: README, Makefile.

## The lives

One seed (1), 60k ticks, `configs/W1_discs.json` and `configs/W1_breakout.json`:

| Arm | World | What it sees |
|---|---|---|
| MD | discs | `online.mix = .5` |
| MB | the scripted court | `online.mix = .5` |

Against seed 1 of the lives that exist, each run twice: OD and OB (vm-00120, and vm-00122 at 60k), SD3 and SB3 (vm-00121, and vm-00122 at 60k).

**Not run:** the mix together with self-forcing, which I had planned as two more arms. With the mix made again at every update a life costs about four times a plain one, so the two lives asked for run alone, one a card. It stays the follow-up.

## The numbers

`dev/wm_measures.py` with `pair`, the exam of seed 1001:

- at 20k, 40k and 60k the regular part and the course part, shown whole real frames (as vm-00122's lives, so `wm_curves.py` sets them side by side); the disc life also on its own world, on the stretch vm-00122's lives were scored on (64 starts from tick 200,000 on: the same world, seed 1; `--home-from 200000`, `home64`);
- at 60k the same shown frames as lived (`--lived`);
- at 60k the free dream of 200 ticks (`--parts free`, in a file of its own), for these lives and for vm-00122's four at 60k: is a ball still in the court, does the dream stay a possible world;
- a video of the long dream (20 watched ticks, 1,000 dreamed) of MB beside OB and SB3.

## Readings, set before the results

One seed an arm: a look. Two runs of one seed differ at 60k by up to 2 ticks of the ball's horizon, .06 of the things held at a tick and .025 of next-frame skill (vm-00122). A difference from a baseline counts only if it is outside the range of that baseline's two runs by more than that.

1. **Does the mix hold the things longer in the dream than no self-forcing?** The court: the ball's horizon (OB 9 and 9, SB3 17 and 15) and the ball held at tick 16 (OB .230 and .270, SB3 .524 and .468). The discs: the food held at ticks 4 and 8, on the life's own world first, then on the exam.
2. **Does the next frame pay?** Next-frame skill shown whole frames (OB .775 and .750, OD .355 and .367).
3. **Shown frames as lived or whole:** does the model need the whole frame it never saw, or the mix it lived with?
4. **The long dream:** is the ball in the court longer (the free dream's ball count at ticks 50, 100, 200), and is a new ball ever served?
5. **How wrong is the half it is shown?** `mix_wrong` and `mix_err` over the life: the noise the idea adds is the model's own error, and it shrinks as the model learns.

**What I expect** (written before the lives ran):

- The court's ball is held longer than without self-forcing and less long than with it (a horizon of 11 to 14). Half of every frame is real, so a thing is always half in sight and the model is seldom made to carry it on its own prediction.
- Next-frame skill a little lower than OB's and OD's (by .01 to .03): half the input is the model's guess.
- Frames as lived and whole frames score about the same.
- The life's own discs are dreamed a little better than OD's, new discs no better.
- The long dream still ends in an empty court: nothing here makes a serve.
- The half it is shown is soon nearly right (in a court life of 600 ticks, a trial, 0.75% of the pixels taken from the prediction were not the real ones), so late in life the mix adds little noise.

## Results (2026-11-27)

Two lives of 60k ticks, one seed an arm (a look). The README has the tables, the pictures and the readings one by one.

- **Does the mix hold the things longer in the dream?** The court: yes, as much as self-forcing: the ball held for 15 dreamed ticks at 60k against 9 and 9 without self-forcing and 15 and 17 with it; held at tick 16 in .492 of the dreams (OB .270 and .230, SB3 .468 and .524). The discs: no, less: the food held at tick 4 on own discs .555 against .781, on new discs .564 against .711.
- **Does the next frame pay?** A little in the court (.742 against .750 and .775), much in the disc world (own discs .639 against .689, new discs .166 against .367).
- **Frames as lived or whole:** whole frames score better for both (the court's next-frame skill .742 against .688 as lived; new discs .166 against −.107).
- **The long dream:** not mended. The free dream of 200 ticks: 0.47 ball pixels at tick 200 against 2.03 (OB) and 1.78 (SB3); no serve.
- **How wrong is the half it is shown?** In the court 0.48% of the lent pixels at 2k ticks and 0.05% at 60k (1 a frame); in the disc world 4.53% at 5k and 2.42% at 60k (50 a frame), 97% of the prediction's wrong pixels at a rim or in a checker.
- **What I expected:** half right for the court (held longer than without self-forcing, and as long as with it, where I had it below), right for the court's next frame and for the empty court of the long dream, wrong for the disc world (I had its own discs a little better; both are worse) and for frames as lived (whole frames are better).
- **Changed on the way, by the owner's word:** the mix is made again at every update ("replay - mix should be re-made each replay"); my first plan replayed it as lived. With that a life costs about four times a plain one, and the two arms of mine (the mix with self-forcing) were not run.

## Verification

- `tests/test_mix.py`; the whole dev suite. **Done:** `tests/test_mix.py` 24 passed (with the random grey of task 00137), the whole dev suite 2,160 passed, 34 skipped, on the code as it stands.
- Every life: 7 checkpoints and the final one, the last log line at tick 60,000, no error; `mix_wrong` in every row; the mixed share of pixels .5. **Done** (191 checks in all, 0 failed), with the record of what each life was shown: every recorded pixel is the prediction's or the real one, as its mask says.
- Every scored file names its life, one transformer, 64 windows a pass, `pair`, nothing unsettled; the files scored as lived say so. **Done** (the same checks): the exam's floors in every file are the anchors', the home set's those of vm-00122's lives.
- With the option off the life is the code as before (the guard is the option itself; lives on a card do not repeat bit for bit, so this is read from the code, and from the suite). **As said:** read from the code; the suite passes.
