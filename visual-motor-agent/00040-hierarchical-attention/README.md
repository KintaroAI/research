# vm-00040: Local Attention With Summary Tokens

**Date:** 2026-10-28
**Status:** Complete
**Source:** `exp/vm-00040` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00040/`
**Task plan:** [`../tasks/00047-hierarchical-attention.md`](../tasks/00047-hierarchical-attention.md)

## Goal

The owner's goal is high locality and a break from attention's squared cost, at a tolerable loss of quality. vm-00037
and vm-00039 showed that prediction is local, while behaviour and text need information from beyond a region. A
hierarchy fits that: each patch sees its neighbourhood in full detail and the rest only through summary tokens. This
is the first step, two levels: a 3 × 3 window of patches, and one summary token per row of the patch grid. Does it
keep prediction, behaviour and text? And what do the summaries alone, with full attention, change?

## Method

- **Summary tokens** (`model.uni_sum` = R; tests in `tests/test_hier_attention.py`): one token after every R rows of
  patches, in the step's sequence. With R = 1 a step is [body, row 0, S0, row 1, S1, …, row 7, S7, move], 74 tokens.
  - A summary has no values. Its input is b_in plus its slot and step embeddings, so its content comes only from
    attention.
  - The one rule stays: every token predicts the next token's values. A token whose next token is a summary has no
    target: that is the last patch of each row. S_r predicts the first patch of row r + 1, and S7 predicts the move.
    The loss is the mean over tokens with a target.
- **Local attention** (`model.attn_win` = W): two patches attend to each other only within W rows and W columns,
  in every step of the context. With summaries, a patch attends to every summary (in its own step, the rows above it,
  by causality), and a summary attends to its row's patches and every summary. The body and the move are global.
  Without the window, summaries see and are seen by everything: register tokens.
- **Implementation:** one slot layout (`vm_uni_slot` in `vm_agent.h`, `agent_lib.uni_layout`, `ref.uni_slots`)
  replaces every "patches are slots 1..nt − 2" assumption. It is used by the C++ and CUDA tokens, targets and loss, by
  the reference, and by the tools: `cuda_lib`, `eval_predict`, `lifetime`, `online`'s lookahead and self-forcing,
  `text_dream`, `eval_dream`, `dream_video` and `analyze_hazards`. Checkpoint header slots 49 and 50.
  - C++ and GPU are checked against the reference (loss and every gradient), and so are the instinct's candidates,
    the lookahead, self-forcing and the dream step.
  - Mutants that ignore the window, or that average the loss over every token, fail.
  - Full suite: 1,318 passed.
- **Arms**, 100k ticks, seeds 1–3, the same settings in both worlds:
  - Image, the egocentric touch world (`configs/U_touch_ego.json`, the 1-step instinct):
    - **EW:** `attn_win = 1` (a 3 × 3 window).
    - **EH:** `attn_win = 1` with `uni_sum = 1`.
    - **ER:** `uni_sum = 1` with full attention.
    - Control: vm-00036's EB.
  - Text, `configs/T_text_B.json`:
    - **TW:** `attn_win = 1`.
    - **TH:** `attn_win = 1` with `uni_sum = 1` (a summary after every line).
    - Controls: vm-00030's full attention, vm-00037's TS (by lines) and TV (by columns), vm-00039's TC (checkerboard).
- **Measures** as in vm-00039:
  - Image: the frozen forage tests, held-out prediction by row and column, the dream.
  - Text: val nats per letter every 10k, by line and column, and generation (seed 1 at its best: TW 60k, TH 90k)
    scored by the classical GPT.
  - The attention scores each mask allows per 8-step window, against full attention.
- **Speeds:** all 15 at once (9 image runs on the 4090, 6 text runs on the 3090): image 37 ticks/s, text 40–42.

## Log

### 2026-10-28
- The analysis first ran 9 image jobs and the text scoring at once, under the CUDA multi-process service. Clients ran
  out of device memory and left the service unusable: any client through it then failed, even alone. The missing
  parts were re-run without it, two per GPU. The service was restarted afterwards: its server had two stuck worker
  threads, so it was terminated and started fresh.
- **A tooling bug, found and fixed:** `eval_dream.py` (and `dream_video.py`, `analyze_hazards.py click-real`) still
  read the patches' predictions from slots 0..P − 1 and the body from slot P + 1. With summary tokens this gave EH
  and ER a dream only 0.80 exact after one step. Fixed to use the layout. `tests/test_eval_dream.py` (new) checks that
  the dreamed step equals patch-by-patch decoding with and without summaries; the old code fails it. The dream
  numbers below are from the fixed tool. The lifetimes and the frozen tests use paths that were already fixed and
  tested (moves, the instinct's candidates, the lookahead).

## Results

### R1. Attention scores computed per 8-step window

| mask | tokens per step | scores | of full |
|---|---|---|---|
| full attention | 66 | 139,656 | 1.00 |
| row halves, one row of overlap (vm-00037) | 66 | 102,792 | 0.74 |
| checkerboard (vm-00039) | 66 | 74,120 | 0.53 |
| **window W = 1 (EW, TW)** | 66 | 24,072 | **0.17** |
| **window + row summaries (EH, TH)** | 74 | 45,608 | **0.33** |
| summaries with full attention (ER) | 74 | 175,528 | 1.26 |

### R2. Text: val nats per letter

| arm | best, s1 / s2 / s3 (mean) | at 100k (mean) | at 10k |
|---|---|---|---|
| full attention (vm-00030) | 1.717 / 1.736 / 1.745 (1.733) | 1.977 | 1.88–1.97 |
| TS: split by lines (vm-00037) | 1.713 / 1.734 / 1.724 (1.724) | 1.818 | 1.95–1.98 |
| TV: split by columns (vm-00037) | 1.842 / 1.849 / 1.850 (1.847) | 1.851 | 2.22–2.28 |
| TC: checkerboard (vm-00039) | 2.074 / 2.083 / 2.062 (2.073) | 2.073 | 2.61–2.67 |
| **TW: window** | 1.981 (60k) / 1.969 (50k) / 1.942 (50k) (**1.964**) | 2.011 | 2.25–2.28 |
| **TH: window + line summaries** | 1.773 (90k) / 1.759 (80k) / 1.763 (90k) (**1.765**) | 1.775 | 2.16–2.18 |

By the predicted letter's column and line (mean of the seeds at their best):

```
column:        0     1     2     3     4     5     6     7
full         1.79  1.74  1.73  1.72  1.73  1.72  1.73  1.70
TW (window)  1.91  2.50  2.11  1.82  1.85  1.83  1.84  1.84
TH (+ sums)  1.75  1.76  1.74  1.71  1.78  1.78  1.80  1.80

line:          0     1     2     3     4     5     6     7
full         1.88  1.75  1.69  1.69  1.72  1.76  1.69  1.68
TW (window)  2.00  1.98  1.94  1.95  1.96  1.99  1.95  1.93
TH (+ sums)  1.85  1.77  1.73  1.75  1.77  1.80  1.74  1.72
```

### R3. Text: generation (seed 1 at its best, 32 val windows, 4 pages, T = 0.8)

| model | samples' nats per letter under GPT | mean line length | lines over 60 letters |
|---|---|---|---|
| real text | 1.756 | 28.8 | 0% |
| full attention | 1.451 | 30.7 | 0% |
| TS (lines) | 1.615 | 33.3 | 14% |
| TV (columns) | 2.255 | 32.6 | 10% |
| TC (checkerboard) | 3.538 | 35.2 | 11% |
| **TW (window)** | 2.574 | 35.4 | 18% |
| **TH (window + line summaries)** | **1.775** | 35.1 | 7% |

From the same context as vm-00037 and vm-00039 ("Hearest thou, Biondello?"):

```
TW:  KING HENRY VI:
     No mined heave howith you his woule conce,
     'Tis your Lords my liond, bewismock by to do non: no,
TH:  ISABELLA:
     Mether is not him; not for you.

     ISABELLA:
     Who, fell, and, and not be hone, which your doubtent states pute their he earms.
```

### R4. Image: frozen at 100k, food / poison per 1k (food : poison)

| run | instinct off | 1-step | 4-step mean | 4-step pair |
|---|---|---|---|---|
| EW s1 | 7.5 / 11.8 (0.63) | 8.6 / 7.9 (1.09) | 8.9 / 6.5 (1.38) | 10.2 / 5.5 (1.86) |
| EW s2 | 11.9 / 14.2 (0.84) | 12.1 / 8.6 (1.42) | 12.9 / 7.3 (1.76) | 14.9 / 7.3 (2.04) |
| EW s3 | 20.9 / 23.7 (0.88) | 22.4 / 14.1 (1.60) | 24.9 / 10.4 (2.40) | 28.1 / 9.4 (2.98) |
| EH s1 | 16.6 / 24.9 (0.67) | 15.5 / 19.1 (0.81) | 17.5 / 10.9 (1.61) | 19.5 / 10.3 (1.89) |
| EH s2 | 16.0 / 25.3 (0.63) | 18.9 / 16.1 (1.18) | 19.9 / 10.6 (1.87) | 25.9 / 9.7 (2.68) |
| EH s3 | 14.9 / 32.9 (0.45) | 20.7 / 27.9 (0.74) | 18.5 / 25.9 (0.71) | 28.3 / 22.7 (1.25) |
| ER s1 | 15.0 / 11.9 (1.26) | 18.9 / 7.3 (2.59) | 21.4 / 5.2 (4.12) | 22.9 / 5.9 (3.90) |
| ER s2 | 18.2 / 18.0 (1.01) | 25.0 / 10.2 (2.44) | 32.2 / 6.1 (5.31) | 37.2 / 5.4 (6.85) |
| ER s3 | 19.6 / 16.9 (1.16) | 16.4 / 6.1 (2.67) | 18.6 / 4.4 (4.20) | 19.2 / 4.3 (4.45) |

Means (food / poison per 1k):

| arm | instinct off | 1-step | 4-step mean | 4-step pair |
|---|---|---|---|---|
| EB (full attention, vm-00036) | 33.5 / 19.4 | 36.6 / 8.8 | 39.1 / 5.9 | 44.4 / 4.1 |
| **EW (window)** | 13.4 / 16.6 | 14.4 / 10.2 | 15.6 / 8.1 | 17.8 / 7.4 |
| **EH (window + row summaries)** | 15.9 / 27.7 | 18.4 / 21.0 | 18.6 / 15.8 | 24.6 / 14.2 |
| **ER (row summaries, full attention)** | 17.6 / 15.6 | 20.1 / 7.9 | 24.1 / 5.2 | 26.5 / 5.2 |

In life, over the last quarter: EB 29–50 food / 5.9–9.6 poison; EW 7–22 / 9.2–15.0; EH 16–19 / 16.4–17.2; ER 16–27 /
7.4–11.5.

### R5. Image: prediction at 100k

| run | held-out patch CE (s1 / s2 / s3) | move CE | body CE | dream: exact h16 | dream: hunger err h16 |
|---|---|---|---|---|---|
| EB | 0.779 / 0.815 / 0.778 | 0.68 / 0.72 / 0.41 | 0.55 / 0.74 / 0.68 | 0.882–0.892 | 0.062–0.103 |
| EW | 0.778 / 0.807 / 0.775 | 0.52 / 0.59 / 0.63 | 0.73 / 1.17 / 0.67 | 0.898–0.908 | 0.083–0.133 |
| EH | 0.773 / 0.808 / 0.776 | 0.65 / 0.69 / 0.46 | 0.47 / 0.56 / 0.60 | 0.902–0.914 | 0.090–0.136 |
| ER | 0.778 / 0.809 / 0.773 | 0.49 / 0.60 / 0.47 | 0.60 / 0.76 / 0.52 | 0.900–0.905 | 0.019–0.105 |

By column (seed 1): EW 0.820 0.781 0.777 0.750 0.739 0.785 0.784 0.786; EH 0.793 0.780 0.782 0.750 0.738 0.777 0.778
0.782. Only EW's column 0 is worse (+0.03).

### R6. Video (in `exp_00040/`, not in git)

- **`video_EH_s3_vs_EB_s3/compare.mp4`**: 1,000 steps at 15 fps. EH seed 3 (left) and full attention (EB seed 3,
  right), frozen at 100k with the 4-step pair lookahead, in the same world (the forage test's world 0).
  In world 0: **EH 27 food, 25 poison; EB 66 food, 3 poison**. Over the 16 worlds the frozen test's numbers are
  28.3 / 22.7 and 60.4 / 3.0. EH s3 is the seed whose poison avoidance broke most. `forage.mp4` and
  `forage_sheet.png`: EH alone.

## Analysis

- **In text the hierarchy nearly closes the gap at a third of the cost.**
  - The window alone costs 0.23 nats per letter (1.964 against 1.733).
  - Adding one summary per line brings it to 1.765, within 0.03 of full attention, with 33% of its attention scores.
    It is better than every earlier split except TS, which keeps 74%.
  - The per-column numbers show what the summaries supply. With the window alone, column 1 is the worst at 2.50: it
    is predicted by column 0, whose window cannot reach the end of the line above, so it lacks the letter before
    itself. With summaries, column 1 is at 1.76, because the line above's summary carries its end.
  - Like the splits, the hierarchy hardly overfits: 1.775 at 100k, against full attention's 1.977.
- **Generation shows the same, with a gap left.** TH's samples score 1.775 under GPT (TW 2.574, TV 2.255, TC 3.538),
  against full attention's 1.451. Words hold together, and only 7% of lines run over 60 letters (TW 18%, TS 14%).
- **Image prediction is local, even at a 3 × 3 window.** Held-out patch cross-entropy is unchanged in every arm
  (0.773–0.809 against 0.778–0.815), and the dream is as exact or better (0.898–0.914 after 16 steps). The one cost is
  EW's column 0 (+0.03). The world wraps around but the window does not, so column 0 cannot see what arrives from
  column 7. The row summaries remove it.
- **Image behaviour is lost in every arm, and the summaries themselves are part of the reason.**
  - EW (window alone) is the worst, at 17.8 food and 7.4 poison with the pair lookahead, against EB's 44.4 and 4.1.
    Its move predictor, the last patch, sees only the bottom-right corner, as in vm-00037's ES.
  - EH gets more food (24.6), but poison avoidance breaks (14.2, and 27.7 without the instinct).
  - ER has full attention and only adds the summaries. Its poison stays near EB's (5.2) but its food is 60% (26.5).
    So adding summaries costs food even with nothing masked.
  - What changes in ER is mainly who predicts the move. It is now S7, a token with no content of its own, instead of
    a patch. The move is predicted as well as EB's (move CE 0.47–0.60 against 0.41–0.72), yet the agent acts worse.
    The move's prediction loss does not measure how good the moves are.
  - A hypothesis, untested: the policy is a learned imitation of the instinct's past choices. A content-free token as
    its reader may imitate them as accurately but generalize them less well.
  - EH's poison suggests something more. One 128-number vector per row may not carry poison's checker texture apart
    from food's solid fill.

## Conclusions

1. A two-level hierarchy (a 3 × 3 window plus one summary per row) keeps text prediction within 0.03 nats per letter
   of full attention, with a third of the attention scores. It is the best split tried in text besides the 74% row
   split, and generation keeps its words.
2. Next-frame prediction in the image world is local, down to a 3 × 3 window of patches.
3. Behaviour is not kept. Even full attention with summaries added loses 40% of the food, so the layout itself
   (summaries, with the move predicted by one of them) costs behaviour, apart from any locality. With the window
   added, poison avoidance is lost as well.

## Next Steps

- **Find what the summaries do to behaviour:** ER with the move predicted by a patch again (summaries placed before
  the step's patches, or the last row unsummarized), against ER and EB.
- **The pyramid** (the owner's choice): summaries of 2 × 2 blocks, then of those, up to one global token. Each patch
  reads fine detail nearby and coarser summaries further out, for N·log N attention. Run it in text first, where the
  two-level version already works.
- A kernel that skips the masked attention scores, so the saving becomes real speed.
