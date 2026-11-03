# Task 00085: Clogged Channels, Learning to Restore What Does Not Arrive (experiment vm-00074)

**Date:** 2026-11-02
**Status:** Implemented; running, seeds 1–3 on the 4090 (2026-11-02), seeds 4–6 after vm-00072 frees the 3090

## Context

- The owner (2026-11-02): add noise to random pixels in random transformers, and empty patches over time, "the idea is that we add noise on transformers input image but backprop make against full image - thus we would train model to restore image even if some channels are clogged".
- Refined by the owner the same day: closed eyes are not corruption, the real frame is blank and so is the target (vm-00070 already does that); a blind spot or a dead area is permanent damage to the original image, so target and input both lack it; only temporary damage is on the input alone; "iimg should be different for each transformer so system as a whole learns to restore full from their own specific imperfections"; and black counts as missing for now.
- Self-forcing already gives a model inputs that differ from its targets (`iimg`, `update_rollout(inputs=...)`, task 00039). Summaries are built from `iimg`, so input summaries are damaged with the patches, and their targets stay clean.

## The three cases

| case | input | target | lasts | here? |
|---|---|---|---|---|
| Temporary damage | damaged, different per transformer | clean | one tick, or a whole window | **yes** |
| Closed eyes | blank | blank | while closed | no: vm-00070, a real event in the world |
| Permanent damage, a blind spot | no signal | no signal | the whole life | no: a later experiment. There is nothing to restore; it would test tracking food through the gap |

## Design

**`online.clog = p`:** in every training batch, for each transformer apart, about p of the patches of each step are damaged in its input, with the target clean. Memory stays clean, and damage is drawn fresh each time a window is used, so a replayed window is never damaged the same way twice. Acting stays clean, so that any loss in foraging comes from training alone.

- **Long and brief,** half of p each (`online.clog_long = .5`). Long: the patch is damaged in all 8 steps of the window, so nothing in the context shows it. Brief: one step. A model can often copy a brief hole from the frame before, so brief damage alone may teach little. A mask longer than the window looks the same as one of the window's length, so long means the whole window.
- **Two kinds, half each:** black, the whole patch at 0, which counts as missing (the owner), and speckle, half of the patch's pixels replaced by random values.
- **The whole image, each transformer its own.** A transformer's far summaries come from its own damaged copy. Averaging thins speckle out and keeps holes, fainter.
- **What is damaged:** patches, and summaries through them. Body and move stay clean for now, since the instinct reads hunger; a universal model should damage them too, later.

A token never sees its own target, so what damage removes is the context a prediction leans on: the same patch a step back, and its neighbours.

**Arms:** K4 with log summaries, the default world, `configs/U_default.json`. **C10:** `online.clog = .1`. **C30:** `online.clog = .3`. The control is vm-00073's K4, the same settings, not rerun. Six seeds, 60k ticks, seeds 1–3 on the 4090 and 4–6 on the 3090, as the control.

## Measures

`eval_clog.py`, frozen, final weights, 16 worlds, 1,500 ticks, on C10, C30 and the control. Each transformer's input is damaged in 30% of its patches, half black and half speckle, and a patch's prediction at the last step of the window is scored against the clean frame, on its own patches:

- **clean:** nothing damaged, the floor.
- **brief:** the patch damaged in the step before. It has to be rebuilt from older steps and neighbours.
- **long:** the patch damaged in all 8 steps. Only neighbours and summaries can give it.
- **copy,** the baseline: the patch of the damaged frame before, shifted by the last move.

Scored on all pixels and on the pixels that hold something (not background). `eval_hunger.py chain --clog p`: foraging with damage while acting, each transformer its own, 30%.

## The readings, set before the results

Each is C30 and C10 against the control, seed by seed, 95% interval on six seeds.

1. **Plumbing:** clean frames given as inputs change nothing in the loss, damaged ones raise it, and targets stay clean. **Met,** on a new model on the GPU: the loss is 3.466119 with no inputs and with clean frames as inputs, equal in every digit, about 3.476 with 30% damaged and 3.49 with 90%, and back to 3.466119 on the next batch without inputs.
2. **Restoration (the main one):** under `long` damage, error on things is below the control's by more than its interval, and below `copy`.
3. **Clean cost:** on clean input, food over the last 20k is not below by more than its interval, and poison not above by more than its interval.
4. **Robustness:** frozen food with damage while acting is above the control's by more than its interval.

A short check on one control life, 150 ticks: on things `clean` .04, `long` .11, `copy` .22. So a model trained on clean frames already beats `copy`, and the comparison that matters in reading 2 is against the control.

## Not in this one

Food with the eyes closed needs the two-senses world, and dream length reads a body that K4 does not predict, so both wait for a later step.

## Implementation

- [x] `online.py`: `clog_frames`, `ClogStream`, `clog`, `clog_long` in `DEFAULT_ONLINE`
- [x] `lifetime.py`: each transformer its own damaged inputs; refuse without the GPU or with self-forcing
- [x] `tests/test_clog.py`
- [x] `eval_hunger.py`: `--clog`; `eval_clog.py`
- [x] `00074-clogged-channels/`: README, Makefile
- [x] Reading 1 on the GPU
- [ ] Runs, frozen tests, write-up

## Verification

Unit tests of the damage: its share, whole-window and one-step masks, clean frames untouched, a transformer's damage its own. Reading 1 on the GPU.
