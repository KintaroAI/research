# Task 00088: Noise on the Input Only, Clean Targets (experiment vm-00077, stage 3 of vm-00076)

**Date:** 2026-11-02
**Status:** Complete (vm-00077)

## Context

- vm-00076 stage 1 (task 00087): white noise in the world, so input, targets and memory carry it. There was no maximum. Every model foraged best near the level it was raised on. At .5 the model raised there kept half of the clean lead, and none did better.
- vm-00074 found that damage on the input against clean targets cut poison on clean frames to about a third. The guess in vm-00076's analysis: with noise in the targets nothing can be restored, so the benefit is lost. This stage tests that guess.
- The owner (2026-11-02): "start stage 2 and once it's done - stage 3".

## Design

**`online.noise_in = q`:** the world stays clean, and the model gets the noise:
- **Acting:** each tick the frame is noised as `env.noise` does: each pixel replaced by a random gray with probability q, fresh each tick, one draw shared by all four transformers. The ring holds the noisy view.
- **Memory:** stores the clean frame.
- **Training:** each update draws fresh noise on the input copy of the replayed windows, one draw shared by all transformers, through `update_rollout(inputs=...)`. The targets are the clean frames.

It needs the universal model on the GPU, the ring, no self-forcing and no `online.clog`. `noise_in = 0` draws nothing, so a life is as before.

So the only difference from stage 1 at the same q is clean targets and clean memory. The model's view while acting is the same.

**Arms:** I10 and I50, K4 with log summaries, `online.noise_in` .1 and .5, seeds 1–3, 60k ticks, seeds 1 and 2 on the RTX 4090 and 3 on the RTX 3090, all six at once.

**Comparison:** N10 and N50 of stage 1 (the same noise in the world), and K4 of vm-00073, seeds 1–3.

**Frozen tests:** as stage 1, `eval_noise.py chain` and `error` at test levels 0, .1, .5 and 1. World noise at test time is what the model saw while acting, so the tools need no change.

## The readings, set before the results

1. **Plumbing:** `noise_in = 0` leaves a life unchanged. The input copy has about q of its pixels replaced, the targets and memory are clean, and the view is the same for all transformers. **Met:** 7 noise tests; a probe of a 120-tick life drew one acting view per tick (120) and one input per update (110), shared by the four transformers; refused with `online.clog`.
2. **Clean targets restore:** at its own level, I50's error against the clean frame is below N50's (.068) by more than its interval. **Met:** .050 against .068, −.018 ± .007.
3. **Clean targets help foraging at .5:** I50's food at .5 is above N50's by more than its interval. **Met:** 119.1 against 47.0, +72.0 ± 67.6; poison 3.14 against 17.7, −14.6 ± 4.6; aim +.21 ± .18.
4. **The maximum with clean targets:** I10 tested clean is above K4 on poison (lower) by more than its interval, as with vm-00074's damage. If so, the guess holds: what removed the benefit in stage 1 was the noise in the targets. **Not met on poison:** −1.15 ± 2.13 (aim +.04 ± .03, food +3.7 ± 35.0). But the lived curve rises: food over the last 20k ticks 73.5 (K4), 91.6 (I10), 116.6 (I50); every I50 seed (99.7, 104.2, 145.8) is above every K4 seed (88.2, 72.2, 60.1), with seeds not paired across experiments and wide intervals (+43.1 ± 95.1 paired). Where it turns down is untested: no level above .5.

## Implementation

- [x] `online.py`: `noise_in` in `DEFAULT_ONLINE`, `noise_frames`
- [x] `lifetime.py`: checks, acting view, training input
- [x] `tests/test_noise.py`: the share and the copy
- [x] Short life on the GPU
- [x] `00076-white-noise/Makefile`: `stage3`, `frozen3`
- [x] Runs, frozen tests
- [x] Write-up: `00077-input-noise/` (outputs moved to `exp_00077/`)

## Verification

Unit tests; a short life with `noise_in = .5`: the stored frames are clean and the acting view is noisy.
