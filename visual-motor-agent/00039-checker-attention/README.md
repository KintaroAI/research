# vm-00039: Attention Split as a Checkerboard

**Date:** 2026-10-28
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00039`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00039/`
**Task plan:** [`../tasks/00046-checker-attention.md`](../tasks/00046-checker-attention.md)

## Goal

vm-00037 split attention into halves of patch rows. Prediction survived, but behaviour collapsed: the move is
predicted by the last patch, and that patch saw only the bottom half. The owner's variant splits the halves as a
checkerboard instead. One half holds the odd patches of the first row, the even patches of the second, and so on; the
other half holds the rest. Movement is shared. Each half then sees the whole view at half resolution instead of half
the view at full resolution. Does that keep behaviour? And what does it do to text?

## Method

- **The mask** (`model.attn_checker = 1` with `model.attn_split = 2`; tests in `tests/test_attn_regions.py`): patch
  (row r, column c) is in region (r + c) % 2, and two patches attend to each other only if they share a region. The
  body and the move are in every region, so the move is shared. `model.attn_pred_global = 1` also puts the last patch
  in both regions; that patch predicts the move. There is no overlap. It is the same allow table as vm-00037
  (`vm_attn_allow`: C++ and both CUDA attention kernels; `torch_reference.attn_allow`: the reference and
  `text_dream.py`), checkpoint header slot 48.
- **Image arms**, in the egocentric touch world (`configs/U_touch_ego.json`), with no self-forcing and the 1-step
  instinct, 100k ticks, seeds 1–3:
  - **EC:** the checkerboard, move shared.
  - **ECg:** EC with `attn_pred_global`.
  - Controls: vm-00036's EB (full attention) and vm-00037's ES (row halves) and ESg (ES with `attn_pred_global`).
- **Text arm**, `configs/T_text_B.json` (a page of 8 lines of 8 letters, one letter per token, read line by line),
  100k ticks, seeds 1–3:
  - **TC:** the checkerboard.
  - Controls: vm-00030's full attention and vm-00037's TS (split by lines) and TV (split by columns).
- **Measures**, as in vm-00037:
  - Image: the frozen forage tests, the dream curve, held-out prediction by row and column.
  - Text: val nats per letter every 10k, by line and column, and generation from seed 1 at its best checkpoint (90k;
    32 val windows, 4 pages, T = 0.8), scored by the classical GPT (vm-00033).
- **Speeds**, all nine at once: EC 60–100 ticks/s (3 image + 3 text runs on the 4090), ECg 62 (three on the 3090),
  text 74–82.

## Results

### R1. Image: frozen at 100k, food / poison per 1k (food : poison)

| run | instinct off | 1-step | 4-step mean | 4-step pair |
|---|---|---|---|---|
| EC s1 | 22.1 / 18.4 (1.20) | 26.2 / 9.2 (2.83) | 30.9 / 7.8 (3.96) | 32.4 / 5.2 (6.17) |
| EC s2 | 14.4 / 15.3 (0.94) | 16.5 / 5.8 (2.84) | 20.4 / 3.4 (6.04) | 22.5 / 4.8 (4.74) |
| EC s3 | 22.4 / 18.5 (1.21) | 25.5 / 8.6 (2.98) | 22.3 / 4.9 (4.52) | 27.3 / 5.6 (4.86) |
| ECg s1 | 15.1 / 13.2 (1.14) | 14.8 / 7.4 (1.98) | 19.7 / 3.4 (5.83) | 21.8 / 3.6 (6.12) |
| ECg s2 | 16.3 / 18.7 (0.87) | 20.8 / 8.8 (2.37) | 26.2 / 7.6 (3.44) | 27.0 / 6.8 (4.00) |
| ECg s3 | 24.2 / 18.4 (1.32) | 27.5 / 9.9 (2.77) | 36.3 / 6.1 (5.99) | 39.6 / 6.8 (5.87) |

Means (food / poison per 1k):

| arm | instinct off | 1-step | 4-step mean | 4-step pair |
|---|---|---|---|---|
| EB (full attention, vm-00036) | 33.5 / 19.4 | 36.6 / 8.8 | 39.1 / 5.9 | 44.4 / 4.1 |
| ES (row halves, vm-00037) | 14.8 / 21.8 | 18.6 / 14.0 | 23.5 / 12.9 | 24.5 / 12.7 |
| ESg (row halves, move predictor shared) | 21.4 / 15.6 | 24.2 / 8.0 | 29.8 / 4.8 | 31.7 / 5.1 |
| **EC (checkerboard, move shared)** | 19.6 / 17.4 | 22.7 / 7.9 | 24.5 / 5.4 | 27.4 / 5.2 |
| **ECg (checkerboard, move predictor shared)** | 18.5 / 16.8 | 21.0 / 8.7 | 27.4 / 5.7 | 29.5 / 5.7 |

In life, over the last quarter: EB 29–50 food / 5.9–9.6 poison; ES 17–20 / 9.6–15.9; ESg 18–31 / 4.0–12.7;
EC 14–23 / 6.0–8.3; ECg 14–25 / 5.1–9.4.

### R2. Image: prediction at 100k

| run | held-out patch CE (s1 / s2 / s3) | move CE | body CE | dream: exact h16 | dream: hunger err h16 |
|---|---|---|---|---|---|
| EB | 0.779 / 0.815 / 0.778 | 0.68 / 0.72 / 0.41 | 0.55 / 0.74 / 0.68 | 0.882–0.892 | 0.062–0.103 |
| ES | 0.776 / 0.807 / 0.772 | 0.62 / 0.75 / 0.63 | 0.63 / 0.54 / 0.51 | 0.903–0.908 | 0.097–0.119 |
| ESg | 0.775 / 0.808 / 0.773 | 0.69 / 0.68 / 0.62 | 0.58 / 0.59 / 0.58 | 0.887–0.911 | 0.059–0.117 |
| EC | 0.772 / 0.790 / 0.765 | 0.81 / 0.87 / 0.78 | 0.62 / 0.66 / 0.59 | 0.900–0.913 | 0.036–0.067 |
| ECg | 0.773 / 0.790 / 0.768 | 0.79 / 0.61 / 0.70 | 0.81 / 0.67 / 0.58 | 0.888–0.901 | 0.089–0.128 |

By row (seed 1): EB 0.801 0.789 0.789 0.757 0.742 0.783 0.783 0.786; EC 0.794 0.781 0.780 0.752 0.739 0.780 0.778
0.777. By column, EC: 0.794 0.780 0.778 0.749 0.737 0.779 0.782 0.780. No row or column is worse than EB's.

### R3. Text: val nats per letter

| arm | best, s1 / s2 / s3 (mean) | at 100k (mean) |
|---|---|---|
| full attention (vm-00030) | 1.717 / 1.736 / 1.745 (1.733) | 1.977 |
| TS: by lines (vm-00037) | 1.713 / 1.734 / 1.724 (1.724) | 1.818 |
| TV: by columns (vm-00037) | 1.842 / 1.849 / 1.850 (1.847) | 1.851 |
| **TC: checkerboard** | 2.074 (90k) / 2.083 (80k) / 2.062 (100k) (**2.073**) | 2.073 |

At 10k: full 1.88–1.97, TV 2.22–2.28, TC 2.61–2.67. TC is still slowly improving at 100k and does not overfit.

By the predicted letter's line and column (TC, mean of the seeds at their best; the controls are in vm-00037's R3):

```
line:          0     1     2     3     4     5     6     7
TC           2.13  2.08  2.05  2.05  2.08  2.12  2.05  2.03

column:        0     1     2     3     4     5     6     7
full         1.79  1.74  1.73  1.72  1.73  1.72  1.73  1.70
TV (columns) 1.75  2.41  2.04  1.79  1.68  1.66  1.77  1.67
TC (checker) 2.08  1.90  2.27  2.03  2.15  2.03  2.08  2.04
```

Which of the most recent letters each column's predictor can see (0 = the predictor itself, the letter just before
the predicted one; 1 = the one before that; and so on):

```
column 0: 0 2 4 6 9     column 4: 0 2 5 7 9
column 1: 0 1 3 5 7     column 5: 0 2 4 5 7
column 2: 0 3 5 7 9     column 6: 0 2 4 7 9
column 3: 0 2 3 5 7     column 7: 0 2 4 6 7
```

### R4. Text: generation (seed 1 at its best, 32 val windows, 4 pages, T = 0.8)

| model | samples' nats per letter under GPT | mean line length | lines over 60 letters |
|---|---|---|---|
| real text | 1.756 | 28.8 | 0% |
| full attention (30k) | 1.451 | 30.7 | 0% |
| TS (40k) | 1.615 | 33.3 | 14% |
| TV (60k) | 2.255 | 32.6 | 10% |
| **TC (90k)** | **3.538** | 35.2 | 11% |

From the same context as vm-00037 ("Hearest thou, Biondello?"):

```
TC:    DUCHIFS mayor se is made methath an w, sirsed.

       Aw Af cou, st youre, and me t, on my davice do ce
       And bet st my touer ceavinte
```

All samples: `exp_00039/text_dream_TC_t0.8.json`.

### R5. Videos (in `exp_00039/`, not in git)

- **`video_EC_s3_vs_EB_s3/compare.mp4`** and **`video_ECg_s3_vs_EB_s3/compare.mp4`**: 1,000 steps at 15 fps. The
  checkerboard agent (left) and full attention (EB seed 3, right), frozen at 100k with the 4-step pair lookahead, in
  the same world (the forage test's world 0). A blue dot marks each patch of region 0 on the split agent's view.
  In world 0: **EC 22 food, 11 poison; ECg 36 food, 7 poison; EB 66 food, 3 poison**. Over the 16 worlds the frozen
  test's numbers are 27.3 / 5.6, 39.6 / 6.8 and 60.4 / 3.0 (seed 3 is the best checkerboard seed in both arms).
  `forage.mp4` and `forage_sheet.png`: the split agent alone.

## Analysis

- **Image prediction survives the checkerboard, as it survived the row split.** Held-out patch cross-entropy is equal
  or slightly better in every seed (0.765–0.790 against 0.778–0.815). No row or column is worse. The dream stays as
  exact (0.888–0.913 after 16 steps), and EC's dreamed hunger is the most accurate of all arms (0.036–0.067). A patch
  that sees only its diagonal neighbours (its own parity) still predicts its next content.
- **Behaviour is better than the row split but far from full attention.** With the move token shared (EC), the
  policy's own moves get 19.6 food per 1k (ES 14.8, EB 33.5). Poison with the instinct is close to EB's (5.2–5.4
  against 4.1–5.9), as in ESg, but food stays low (27.4 against 44.4 with the pair lookahead).
- **Putting the move's predictor in both halves no longer helps.** ECg is no better than EC (29.5 / 5.7 against
  27.4 / 5.2), while with row halves the same switch lifted food from 24.5 to 31.7. On a checkerboard the last patch
  already sees the whole view at half resolution, and a circle (8–11 px radius against 8 px patches) always covers
  patches of both parities. So what the policy lacks is not sight of the food. Both arms predict the move worse on
  average (mean move CE: EC 0.82, ECg 0.70, against EB 0.60 and ES / ESg 0.67 / 0.66).
- **A likely cause: the cursor's patches are split.** In the egocentric world the cursor sits at the centre, on the
  four patches (3, 3), (3, 4), (4, 3) and (4, 4). The checkerboard puts two of them in each half, so no single token
  in a half sees the whole cursor neighbourhood, and the relation between the cursor and nearby food is what steering
  needs. The row split kept those four patches together in its overlap rows (3–4). This is a hypothesis; it is not
  tested here.
- **In text the checkerboard is the worst split.** At 2.073 nats per letter, against 1.733 full, 1.724 by lines and
  1.847 by columns, the cost is 0.34 nats, three times TV's. The predictor always sees itself (the letter just before
  the predicted one), but at every column except one it cannot see the letter before that. The cost per column
  follows what is left of the last few letters. Column 1 is predicted by column 0, which sees the last letter of the
  previous line (the same region), so it keeps two in a row: 1.90, the best. Column 2 sees letters 0, 3 and 5 back:
  2.27, the worst. The others see 0 and 2, plus 3 or 4 at best: 2.03–2.15. As in vm-00037, the model leans on the
  last 3–4 letters, and the checkerboard removes one of them almost everywhere.
- **Generation shows it plainly.** Words are lost entirely ("Aw Af cou, st youre"). GPT scores the samples at 3.54
  nats per letter, against 2.26 for TV and 1.45 for full attention. Line lengths are as bad as TS's and TV's (11% over
  60 letters).
- **Regularization again.** TC does not overfit (best at 80k–100k, at 100k 2.073 against full's 1.977 at 100k from a
  best of 1.733). It also learns slowest of all (2.61–2.67 at 10k).

## Conclusions

1. A checkerboard split of attention leaves next-frame prediction intact in the circle world, as halves by rows did.
2. Behaviour lands between the row split and full attention: poison avoidance near full attention's, food at about 60%
   of it. Sharing the move's predictor adds nothing here. Seeing the whole view at half resolution is not enough for
   the policy; adjacent patches, probably those around the cursor, need to attend to each other.
3. In text the checkerboard is the costliest split tried (0.34 nats per letter, words lost in generation), because it
   hides the second-to-last letter almost everywhere. Text needs contiguous recent context; the image world needs
   local neighbourhoods.

## Next Steps

- Test the cursor hypothesis: a checkerboard that keeps the centre 2 × 2 patches in both regions, or a checkerboard
  of 2 × 2 patch blocks instead of single patches.
- Both splits in one model: row halves in one layer, a checkerboard in the other (a cheap form of Swin's shifted
  windows).
- Text: a checkerboard of 2-letter blocks, which keeps the last two letters together.
