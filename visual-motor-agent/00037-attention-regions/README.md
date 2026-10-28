# vm-00037: One Model, Attention Split Into Regions

**Date:** 2026-10-27
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00037`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00037/`
**Task plan:** [`../tasks/00044-attention-regions.md`](../tasks/00044-attention-regions.md)

## Goal

The owner's idea: split the work as if two transformers each handled half the tokens, with an overlap, while the
model stays one model with one loss. First step: an attention mask in the existing model. In the circle world physics
is local (a circle drifts 1–2 px per step, the cursor at most 4, a patch is 8), so a region may need only its own
patches, a border band and the global tokens. What does the model lose, in prediction and in behaviour? And in text,
where the classical comparison exists?

## Method

- **The mask** (`model.attn_split` S, `model.attn_overlap` O; tests in `tests/test_attn_regions.py`): the patch grid's
  rows (or columns, `model.attn_cols`) are cut into S bands, each reaching O rows past its edges; the body and move
  tokens are in every region. A token attends to another only if their regions meet (as well as causally, within its
  context). `model.attn_pred_global`: the last patch, which predicts the move, is in every region too. One allow table
  over the step's slots, applied where attention weights are formed (C++, both CUDA attention kernels, the PyTorch
  reference, `text_dream.py`'s forward); a disallowed key has weight exactly 0, so the backward passes are unchanged.
  It masks, it does not save compute.
- **Here:** halves (S = 2) with one row of overlap: rows 0–4 and 3–7 of the 8 × 8 grid.
- **Image arms**, the egocentric touch world (`configs/U_touch_ego.json`), no self-forcing, the 1-step instinct,
  100k ticks, seeds 1–3, against vm-00036's EB (the same, full attention):
  - **ES:** the split.
  - **ESg:** the split with `attn_pred_global` (added after ES's diagnosis, below).
- **Text arms**, `configs/T_text_B.json` (a page of 8 lines of 8 letters, one letter per token), 100k ticks, seeds 1–3,
  against vm-00030's B with bin embeddings (the same, full attention):
  - **TS:** the split by lines (lines 0–4 and 3–7), reading line by line.
  - **TV:** the split by columns (`attn_cols`: columns 0–4 and 3–7), reading still line by line.
- **Measures:** image: the frozen forage tests, the dream curve, held-out prediction (`eval_predict.py`, 16 bodies ×
  1,000 steps with the instinct; new: cross-entropy by patch row and column); text: val nats per letter every 10k, by
  line and column at each run's best, and generation (`reference/text_dream.py`, 32 val windows, 4 pages, T = 0.8,
  seed 1 at its best) judged by the classical GPT (vm-00033).
- Speeds: ES 91–100 ticks/s (three on the 4090), ESg 62 (three on the 3090), text 135–140.

## Log

### 2026-10-27
- ES halved food from the start. The frozen tests at 75k (seed 3) showed both the model's own moves and the instinct
  worse; the instinct overrode as often as in EB with similar predicted gains. Diagnosis: in the universal model the
  move is predicted by the token before it, the last patch (row 7), so ES's policy saw only rows 3–7 and the global
  tokens. ESg added: that token in both halves.
- `text_dream.py` has its own torch forward; the mask added there (it is checked against the reference on every run:
  0.0003).
- The baseline text runs rescored with `--tag rows` / `lines` for the by-line and by-column numbers; the totals equal
  the old ones (1.7175).

## Results

### R1. Image: frozen at 100k, food / poison per 1k (food : poison)

| run | instinct off | 1-step | 4-step mean | 4-step pair |
|---|---|---|---|---|
| ES s1 | 10.1 / 25.9 (0.39) | 14.9 / 18.4 (0.81) | 19.8 / 19.4 (1.02) | 16.5 / 18.4 (0.89) |
| ES s2 | 16.7 / 19.4 (0.86) | 19.4 / 8.8 (2.20) | 24.6 / 6.5 (3.78) | 27.4 / 6.2 (4.42) |
| ES s3 | 17.6 / 20.1 (0.87) | 21.5 / 14.7 (1.46) | 26.1 / 12.8 (2.04) | 29.5 / 13.4 (2.21) |
| ESg s1 | 16.3 / 19.8 (0.83) | 17.3 / 13.9 (1.25) | 20.6 / 7.0 (2.95) | 20.4 / 7.6 (2.69) |
| ESg s2 | 20.8 / 13.4 (1.54) | 24.7 / 4.8 (5.13) | 34.4 / 3.5 (9.82) | 33.3 / 4.7 (7.11) |
| ESg s3 | 27.2 / 13.7 (1.99) | 30.6 / 5.4 (5.70) | 34.4 / 4.0 (8.61) | 41.4 / 2.9 (14.11) |

Means (food / poison per 1k):

| arm | instinct off | 1-step | 4-step mean | 4-step pair |
|---|---|---|---|---|
| EB (full attention, vm-00036) | 33.5 / 19.4 | 36.6 / 8.8 | 39.1 / 5.9 | 44.4 / 4.1 |
| ES (split) | 14.8 / 21.8 | 18.6 / 14.0 | 23.5 / 12.9 | 24.5 / 12.7 |
| ESg (split, move predictor shared) | 21.4 / 15.6 | 24.2 / 8.0 | 29.8 / 4.8 | 31.7 / 5.1 |

In life, last quarter: EB 29–50 food / 5.9–9.6 poison; ES 17–20 / 9.6–15.9; ESg 18–31 / 4.0–12.7. No 1k window
without food in any life.

### R2. Image: prediction at 100k

| run | held-out patch CE (s1 / s2 / s3) | body CE | dream: exact h16 | dream: hunger err h16 |
|---|---|---|---|---|
| EB | 0.779 / 0.815 / 0.778 | 0.55 / 0.74 / 0.68 | 0.882–0.892 | 0.062–0.103 |
| ES | 0.776 / 0.807 / 0.772 | 0.63 / 0.54 / 0.51 | 0.903–0.908 | 0.097–0.119 |
| ESg | 0.775 / 0.808 / 0.773 | 0.58 / 0.59 / 0.58 | 0.887–0.911 | 0.059–0.117 |

By row (seed 1; the halves meet at rows 3–4): EB 0.801 0.789 0.789 0.757 0.742 0.783 0.783 0.786; ES 0.795 0.783 0.784
0.755 0.740 0.784 0.780 0.784. No row is worse.

### R3. Text: val nats per letter

| arm | best, s1 / s2 / s3 (mean) | at 100k (mean) |
|---|---|---|
| full attention (vm-00030) | 1.717 (30k) / 1.736 (20k) / 1.745 (30k) (1.733) | 1.972 / 2.019 / 1.940 (1.977) |
| **TS: by lines** | 1.713 (40k) / 1.734 (40k) / 1.724 (50k) (**1.724**) | 1.804 / 1.824 / 1.826 (1.818) |
| TV: by columns | 1.842 (60k) / 1.849 (50k) / 1.850 (50k) (1.847) | 1.843 / 1.853 / 1.856 (1.851) |

At 10k: full 1.88–1.97, TS 1.95–1.98, TV 2.22–2.28.

By the predicted letter's line and column (mean of the seeds at their best):

```
line:          0     1     2     3     4     5     6     7
full         1.88  1.75  1.69  1.69  1.72  1.76  1.69  1.68
TS (lines)   2.00  1.73  1.66  1.67  1.69  1.73  1.67  1.65

column:        0     1     2     3     4     5     6     7
full         1.79  1.74  1.73  1.72  1.73  1.72  1.73  1.70
TS (lines)   1.75  1.78  1.76  1.72  1.72  1.68  1.70  1.68
TV (columns) 1.75  2.41  2.04  1.79  1.68  1.66  1.77  1.67
```

### R4. Text: generation (seed 1 at its best, 32 val windows, 4 pages, T = 0.8)

| model | samples' nats per letter under GPT | mean line length | lines over 60 letters |
|---|---|---|---|
| real text | 1.756 | 28.8 | 0% |
| full attention (30k) | 1.451 | 30.7 | 0% |
| TS (40k) | 1.615 | 33.3 | 14% |
| TV (60k) | 2.255 | 32.6 | 10% |

From the same context ("Hearest thou, Biondello?"):

```
full:  LUCIO:
       I know, you wilt consort you fair promise
       That we him chance of son of not to the duke.
TS:    LUCIO:
       Ay, but stands pleased for the business:
       Thou hard-men after than a renowneds:
TV:    DUKE VINCENTIO:
       They are sharpoor beat it, whim go be to do;
       Ingrack to the dupon the bankard thing; atric per watch,
```

All samples: `exp_00037/text_dream_{base,TS,TV}_t0.8.json`.

### R5. Videos (in `exp_00037/`, not in git)

- **`video_ES_s3_vs_EB_s3/compare.mp4`** and **`video_ESg_s3_vs_EB_s3/compare.mp4`** (1,000 steps, 15 fps): the split
  agent (ES or ESg, seed 3, left) and full attention (EB seed 3, right), frozen at 100k with the 4-step pair
  lookahead, in the same world (the forage test's world 0); the split agents' views show the overlap band (rows 3–4,
  dashed). In world 0: **ES 23 food, 11 poison; ESg 40 food, 2 poison; EB 66 food, 3 poison**; over the 16 worlds the
  frozen test's numbers (29.5 / 13.4, 41.4 / 2.9, 60.4 / 3.0). `forage.mp4` and `forage_sheet.png`: the split agent
  alone.
- ES's hunger climbs through long stretches between meals and jumps at poison touches throughout; EB's stays near 0,
  a meal every 15 steps or so; ESg is in between, with few poison touches.

## Analysis

- **In the image world the split costs prediction nothing.** Held-out patch cross-entropy is equal or slightly better
  in every seed (0.772–0.808 against 0.778–0.815), no row is worse (not even at the border), the dream's pixels stay
  as exact, and held-out hunger is predicted as well or better (body CE 0.51–0.63 against 0.55–0.74; ES's dreamed
  hunger after 16 steps is a little worse, 0.097–0.119 against 0.062–0.103). A region plus one row of overlap and the
  global tokens is enough to predict the next frame: what moves into a patch comes from within a few pixels.
- **Behaviour collapsed, because of who predicts the move.** In the universal model every token predicts the next
  one, so the move is predicted by the last patch, which the split put in the bottom half: ES's policy saw only rows
  3–7. Its own moves fell from 33.5 to 14.8 food per 1k (ratio below 1), and the instinct, which chooses among the
  policy's candidates, recovered little (24.5 / 12.7 with the pair lookahead, against 44.4 / 4.1).
- **Sharing that one token recovers avoidance, not seeking.** ESg's poison with the lookahead (4.8–5.1 per 1k) is
  near EB's (4.1–5.9); its food is 30–32 against 39–44, and its own moves stay below EB's (21.4 against 33.5). One
  token that sees everything, at the end of the step, is weaker than full attention; likely because the other patch
  tokens it reads now carry only regional context, so food far up in the view reaches the move through it alone. Seed 3 is close
  to EB (41.4 / 2.9), seed 1 far (20.4 / 7.6), so the spread is large.
- **In text, a split by lines costs nothing and regularizes; a split by columns costs much.** TS matches or beats
  full attention (best 1.724 against 1.733) and overfits far less (1.818 against 1.977 at 100k). Its only loss is line
  0 (+0.12 nats), which can no longer see the end of the previous page. TV is 0.11 nats worse. Per column, the cost
  follows how many immediately preceding letters the predicting token can see: column 1 (its predictor sees one
  letter before it) 2.41, column 2 (two) 2.04, columns 3 and 6 (three) 1.77–1.79, columns 4, 5 and 7 (four or more, or
  the overlap) 1.66–1.68, better than full attention; column 0, after the line break, 1.75 (full 1.79). The model
  leans on the last 3–4 letters; a split that keeps them is nearly free, one that cuts them on every line is not.
- **Generation shows what prediction hides.** TS's samples keep their words but lose line structure
  (14% of lines over 60 letters, none in real text): ending a verse line needs the previous line break, often 30–40
  letters back, in the other half. TV's lose words too ("sharpoor", "dupon", "voicontent dges"): every page line's
  start is generated from one or two visible letters. GPT judges them 1.615 and 2.255 nats per letter against 1.451.
- **The split as regularizer.** In both worlds the split model is a little better on held-out prediction once it sees
  enough of the recent past (image rows, text lines 1–7, TV columns 4–7). Limiting attention removes some freedom to
  memorize; it slows early learning in text (10k: TS 1.95–1.98, TV 2.22–2.28 against 1.88–1.97).

## Conclusions

1. Attention in two halves with one row of overlap leaves next-frame prediction intact in the circle world and
   next-letter prediction intact in text (split by lines). Locality holds for prediction.
2. It breaks behaviour, through a property of the universal model: the token before the move predicts it. With that
   token regional, the policy sees half the world. Sharing it recovers poison avoidance but not all of the food.
3. In text the model depends mostly on the last 3–4 letters: a split that cuts them (by columns) costs 0.11 nats per
   letter and mangles words; one that keeps them (by lines) is free, overfits less, but loses long-range form (line
   length).

## Next Steps

- A move predictor that sees both halves fully: the owner's option of a move prediction from each half combined
  (two move tokens per step), or a global summary token before the move.
- No overlap (O = 0) and quarters (S = 4), image and text, now that prediction is known to survive halves.
- Separate weights per region (two transformers in fact), and a kernel that skips masked scores (the compute saving).
- In text, the regularizing effect: a split by lines with full attention in a later layer.
