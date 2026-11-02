# Task 00081: Two Senses, One Body (experiment vm-00070)

**Date:** 2026-11-01
**Status:** Complete

## Context

- The owner (2026-11-01): one sense should be able to recreate another, an image from words and words from an image, which should help the road from hearing speech to speaking. A caption drawn in the picture will not do: it is vision, and it goes when the eyes close. Hearing is not built, and text is implemented and basically the same, so use visual + text.
- Our prediction runs in slot order within a frame, and the scene barely changes from tick to tick. So a sense teaches another only while the other has been away for longer than the context.
- It is also the first rung of the baby trainer in §13 of `../ARCHITECTURE.md`: a scripted teacher naming what is near.

## Design

**The world.** `world/sense_world.py`, `env.sense`, wraps the circle world as `thought_world.py` does. One frame 128 wide, 64 high: left the egocentric view, right a text sense, two pieces split by column (as `U_thought.json`), joined by the summaries. `configs/U_sense.json`, channels on, `gain`, the hunger instinct as usual.

- **The teacher** writes what is nearest the cursor, three words on three rows: `food` or `poison`; `left`, `right`, `up` or `down` (the larger axis, in the ego view); `near` (within 16 px) or `far`. 16 sentences.
- **Letters** are grey levels, as in the text world, a whole 8 by 8 patch each. 16 letters and a blank, so 32 bins hold them. A patch keeps its single value through every thumbnail, so all 64 cells survive to the top summary, the only way text reaches the view's piece.
- **Each sense goes off by itself:** every 60–140 ticks the eyes close or the teacher falls silent, at even odds, for 10–16 ticks: longer than the context of 8, so what was there leaves the window. Never both at once.
- The world is a torus the size of the view, so nothing can leave the view. The eyes closing replaces that case.

**Prediction's target is what comes next,** so while the eyes are closed the model is taught black. The picture is taught from the words at the tick the eyes open, and scored there.

| arm | teacher |
|---|---|
| **ST** | true |
| **SS** | shuffle: a random sentence each time the true one changes, the same rhythm with no meaning |

Seeds 1–8, 60k ticks: 39 ticks/s alone beside other runs, about 2 hours for 8 lives on one GPU. Both on the 4090, one after the other: the frozen test of vm-00065 holds the 3090 for hours.

## Measures

Online: food and poison per 1k ticks with the eyes closed and open (`food_blind`, `poison_blind`, `food_seen`, `poison_seen`).

`eval_sense.py`: final weights frozen, 16 worlds, 1,500 ticks, twice: as lived, and with `sense.cut`, where the other sense goes off too (the control; the same model with nothing to restore from). Scored at a sense's first tick back, from a window that holds nothing of it:

- **Words from the picture:** at the teacher's return, its letters after the first, read from the text piece's prediction, and whole sentences. Within a frame each token sees the ones before it, so the first letter only says that the text is back. The direction and range words must come from the picture.
- **The picture from words:** at the eyes' return, the view piece's error per pixel, over the view and over the thing the teacher named.
- **Following words:** food and poison with the eyes closed.

## The readings, set before the results

Seed by seed, ST as lived minus ST cut:

1. **Words from the picture:** the sentence letters are above zero beyond the interval, and SS's are not.
2. **The picture from words:** the error over the named thing is below zero beyond the interval, and SS's is not.
3. **Words guide it:** food with the eyes closed is above zero beyond the interval, and SS's is not.

If 2 fails while 1 holds, words reach the view's piece too weakly, and the summaries are the first suspect.

## Verification

`tests/test_sense_world.py`, 12 tests: the view half is the world's, the text half the sentence; drops last longer than the context, never both; `cut` turns both off; shuffled words match by chance; each piece is read from its own slots; the first patch of a thing; a lifetime and its frozen test run. No model code changes.

## Results

Eight seeds each. **Readings 1 and 3 are met, reading 2 is not.** As lived minus cut, ST against SS: letters +0.059 ± 0.020 against +0.010 ± 0.014; error over the named thing −0.0008 ± 0.0016 against −0.0008 ± 0.0010; food with the eyes closed +7.7 ± 4.9 against +1.1 ± 3.4. Whole sentences 0.67 against 0.07, and poison with the eyes closed 3.5 against 14.1 per 1k ticks.

The fallback above is wrong: added after the results, `eval_steer.py` shows that the view piece proposes moves from the words as well as the text piece does, so words do reach it. The first-patch error, also added, does not clear zero. Words reach the view piece's move and not, as far as these scores can see, its picture. The scorer had a bug (a piece read from another's slots), fixed before SS was scored. Full results in `../00070-two-senses/README.md`.
