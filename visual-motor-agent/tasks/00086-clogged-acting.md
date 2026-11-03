# Task 00086: Clogged Acting, Living With the Damage (experiment vm-00075)

**Date:** 2026-11-02
**Status:** Complete

## Context

- vm-00074: each transformer trained on its own damaged input against clean targets. Acting and memory stayed clean, so the damage was only ever practice. It restored what did not arrive, at no visible cost, and poison on clean input fell to about a third.
- The owner (2026-11-02): "I think acting during life should be on damaged image." Memory stays clean, and training on memory draws fresh damage, as it does now.

## Design

| | vm-00074 | here |
|---|---|---|
| Acting during a life | clean | **damaged, each transformer its own** |
| Memory | clean | clean |
| Training input | fresh damage | fresh damage |
| Training targets | clean | clean |

**`online.clog_act = 1`:** each transformer gets its own `ClogStream`, the one the frozen tests use, at the same p: half of the damage held for 16 ticks, half for one tick, black or speckle. Each step's damaged frame goes into that transformer's ring, so its context is damaged too. The world, the food and the frame put in memory do not change. Off by default, and with its own generators, so C10 and C30 stay repeatable. It needs `online.clog` above 0 and the ring.

**Memory stays clean,** because the targets come from it, and storing a damaged copy per transformer would cost five copies of each frame and repeat the same damage at each replay. A side effect: the damage seen while choosing a move does not match the damage drawn when that moment is replayed.

**Arms:** A10 and A30, `online.clog` .1 and .3 with `online.clog_act = 1`, K4 with log summaries, 64 x 64, six seeds, 60k ticks. C10 and C30 of vm-00074 are the comparison, not rerun, and vm-00073's K4 the control. All 12 lives at once, seeds 1–3 on the 4090, 4–6 on the 3090.

## Measures

The same frozen tests as vm-00074: `eval_hunger.py chain --clog .3`, `chain` on clean input, `eval_clog.py score`. Final weights, 16 worlds, 1,500 ticks. Each arm against the arm of vm-00074 with its p, seed by seed, 95% interval on six seeds. Note that C30's food split by batch in vm-00074, 55–62 in seeds 1–3 and 85–102 in 4–6, so differences in food against C30 need that caution.

## The readings, set before the results

1. **Plumbing:** with `clog_act = 0` a life draws what it drew; with it on, memory holds clean frames and the transformers act on different damaged views. **Met:** the switch off draws nothing, the stream leaves the frame it is given, the views differ on the GPU, and a life refuses `clog_act` without `clog`.
2. **The cost of growing up damaged:** food over the first 10k ticks is below the C arm's by more than its interval. Expected: a young model cannot restore yet. **Not met:** −.8 ± 11.8 and +4.5 ± 12.9.
3. **It catches up:** over the last 20k ticks, while acting on damage, food is not below the same arm's frozen food under damage in vm-00074, A10 against C10 and A30 against C30, by more than its interval. **Met:** +.4 ± 31.8 and +19.2 ± 24.0.
4. **The final model is no worse:** frozen food under damage not below, and poison not above, the C arm's by more than the interval, and restoration error under `long` not above. **Met.**
5. **Clean input:** frozen poison on clean input stays below the control's by more than the interval. **Met:** −1.9 ± .6 and −2.1 ± .4.

## Not in this one

The other two cases: closed eyes, and permanent damage. Damage to the body and the move.

## Implementation

- [x] `online.py`: `clog_act` in `DEFAULT_ONLINE`
- [x] `lifetime.py`: a stream per transformer, each acting on its own view; refuse without `clog` or the ring
- [x] `tests/test_clog.py`: the stream leaves its frame untouched
- [x] `00075-clogged-acting/`: README, Makefile
- [x] Reading 1 on the GPU
- [x] Runs, frozen tests, write-up

## Verification

Unit tests of the stream, and a short life with and without `clog_act`.
