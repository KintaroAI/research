# Task 00056: Text That Appears One Letter per Step (experiment vm-00048)

**Date:** 2026-10-29
**Status:** Stopped (reveal kept as an option, default off)

## Context

- The owner's view (2026-10-29): an image is an instant. In the text world so far, each step shows a whole new page,
  and the order within it comes only from the reading order inside one frame. In the image world, the next frame is
  the current scene with a small change. Make text behave the same way: the page fills one letter per step, and when
  it is full the next step starts again from a page with one letter.
- Expected effects:
  - "next" means the same in both worlds (what the next frame changes): text and pixels become more swappable;
  - vm-00047's split loses its cause: when a letter appears, every earlier letter of its line is in the previous
    frame, with its B1 complete and readable by both halves;
  - one new letter per step (64× fewer letters per tick), so no overfitting in 100k–200k ticks;
  - the history is the page on screen (0–64 letters, plus the previous page for a few steps after a wipe) instead of
    8 pages (512 letters): the next-letter cross-entropy should be higher than today's 1.72;
  - 63 of 64 letter targets per step are copies of the previous frame.

## Design

- **`env.reveal = True`** (`world/text_world.py`; default False, unchanged): state per reader: the page's start `pos`
  and the letters shown `k` (1..page). A frame shows the page's first k letters in reading order, the rest empty. A
  step adds the next letter (k + 1); after a full page, the next page starts with its first letter (pos + page, k = 1).
  Readers start at k = 1 (with `tile`, on page boundaries, so at step s every reader's new letter is letter s % page).
  `get_state` / `set_state` carry k.
- **Empty is its own grey level:** with `reveal`, letter k (0..n − 1) is drawn at round((k + 1) · 255 / n) and empty
  at 0 (today letter 0 is at 0). With `onehot`, empty is all zeros (already distinct). `letters` / `read` decode an
  empty cell as code n ("" in `read`).
- **No model change.** The universal model sees frames as before.
- **`letter_ce.py`** with a reveal run: the target at each step is the new letter's first patch (top-left; with
  `sum_only`, its B1), predicted by the token before it in the same step; checked against the frames (the cell is
  filled at this step and empty at the step before).

## Verification

- `tests/test_text_world.py` (reveal): frame k shows exactly the first k letters, the rest at level 0; the levels of
  letters are 1..255 and distinct from empty; after a full page the next page starts with one letter at pos + page;
  `tile` readers on page boundaries; state round trip; `read` drops empty cells; the default mode unchanged (existing
  tests).

## Experiment vm-00048

- seed 1, 200k ticks (200k letters, a fifth of the training split), checkpoints every 20k:
  - **RB:** `configs/T_text_B.json` + reveal (the normal layout: 66 tokens per step).
  - **RB3:** `configs/T_text_B2.json` with `sum_only = 0`, `uni_pyr = 4` + reveal (vm-00046's NB3 in reveal mode).
  - **RS:** RB3 with `attn_split = 2`, `attn_cols = 1` (vm-00047's VS in reveal mode).
- **Measure:** next-letter cross-entropy on the val text (`letter_ce.py`), overall and by letter column. Hypothesis:
  in RS the right half predicts as well as the left, and RS as well as RB3.

## Results

- `tests/test_text_world.py` (reveal, 5 new): pages fill one letter per step and continue the text across pages (normal,
  2× and block layouts), empty cells at level 0 and the letters at 1..255, `tile` on page boundaries, state round trip,
  onehot empty = zeros, the default mode unchanged. Full suite: 1,493 passed, 30 skipped.
- `letter_ce.py`: reveal runs scored at each step's new letter (checked against the frames); `--device cuda` (the same
  computation on the GPU, `~/venvs/vm-gpu`, which needed Pillow) reproduces the CPU scores to 4 decimals (NB 1.7206,
  VS 1.9038 at 30k) in 37 s for both instead of minutes.
- Smoke run (RB, 2k ticks): 231 ticks/s alone, next-letter CE 5.35 (untrained: ln 256 = 5.55).
- vm-00048 runs: RB 59 ticks/s and RS 23 on the 4090 (together), RB3 14 on the 3090. RB finished 200k; RS stopped
  after 140k and RB3 after 80k (a reboot that left one GPU).

### Next-letter cross-entropy (val, 6,400 steps × 4 readers = 25,600 new letters; untrained ln 256 = 5.55)

| run | 20k | 80k | 100k | 200k |
|---|---|---|---|---|
| RB | 7.39 | | 6.29 | 6.09 |
| RB3 | 7.57 | 5.86 | | |
| RS | 7.18 | 5.15 | | |

- **Every run is worse than an untrained model: it memorized.** The cells that copy the previous frame score ~0; at
  the new cell RB (200k) puts e.g. 0.92 on a real but wrong letter, its top-1 accuracy is 21%, and its errors are not
  a shift of the text. The training text scores the same (6.05). Its training loss bounds the new letter's CE at
  ≤ 2.5: it fits the windows in its memory and nothing else.
- **The cause is the data rate.** Reveal adds one letter per tick, but the model still takes a gradient step per tick
  on its memory: at 20k ticks it has seen 20k letters, replayed many times, where NB had seen 1.28M. The expectation
  in the context ("no overfitting") was backwards.
- RS < RB3 here fits vm-00047's hint that the split overfits less; in this regime it means only "memorizes less".
- The owner chose to return to whole pages; reveal stays as `env.reveal` (default off). A fair reveal run would need
  the data rate of a whole page per tick, e.g. `online.bodies = 64` held for the life (one model in 64 worlds).
