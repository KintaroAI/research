# vm-00041: A Pyramid of Summary Tokens

**Date:** 2026-10-28
**Status:** Complete (pending tag)
**Source:** *tagged on completion as `exp/vm-00041`* (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00041/`
**Task plan:** [`../tasks/00048-pyramid-attention.md`](../tasks/00048-pyramid-attention.md)

> **Caveat (found 2026-10-29, task 00052):** in every model with summary tokens (here EP and TP; also vm-00040's
> `uni_sum` arms), the position codes added to the slot vectors at birth were misplaced. Cell k's code went to slot
> k + 1, so the summaries carried patch codes, 40 of 64 patches carried another cell's, and the last 16 slots none.
> Training did not repair it. The controls without summaries (EB, full-attention text) are unaffected. The behaviour
> results below may owe part of their loss to it; vm-00045 reruns the key arms with the codes placed by the layout.

## Goal

vm-00040's two levels (a 3 × 3 window plus one summary per row) came within 0.03 nats per letter of full attention in
text, but not in behaviour. The owner chose the pyramid: fine detail nearby, coarser summaries further out, up to one
token for the whole grid. With log N levels and a bounded number of keys per level, attention becomes N·log N instead
of N². Does it keep prediction, text and behaviour?

## Method

- **The pyramid** (`model.uni_pyr` = L; tests in `tests/test_pyramid_attention.py`): level ℓ = 1..L has one summary
  token per 2^ℓ × 2^ℓ block of patches. For the 8 × 8 grid and L = 3 that is 16 + 4 + 1 = 21 summaries and 87 tokens
  per step.
  - **Placement:** each summary comes right after its block's last patch in reading order, finer levels first. The
    global summary sits just before the move and predicts it.
  - **Values:** as in vm-00040, summaries have none, and the token before one has no target.
- **Attention** (`attn_win` = 1):
  - A patch reads its 3 × 3 window, and at every level the 3 × 3 nearest summaries (its position projected to that
    level).
  - A summary reads every patch of its block directly, so that any patch reaches any other in two hops with two
    layers. It also reads the summaries of its own and coarser levels within the window. No summary reads a finer
    one.
  - The body and the move are global.
- **Implementation:** the slot layout became a table built once per model (`vm_uni_layout` in `vm_agent.h`), shared by
  C++, CUDA (a device array) and the mask, and mirrored in `agent_lib.uni_layout` and `ref.uni_slot_table`. vm-00040's
  row summaries come from the same function. Checkpoint header slot 51.
  - Tests: the layout, the mask, the cost's growth, what a patch can see, C++ and GPU against the reference. Pyramid
    models also run in the candidates, lookahead, self-forcing and dream tests. A mutant whose summaries read the whole
    grid fails. Full suite: 1,372 passed.
- **Arms**, 100k ticks, seeds 1–3, the same settings in both worlds:
  - **EP:** the egocentric touch world (the 1-step instinct). Controls: vm-00036's EB (full attention), vm-00040's EH
    (window + row summaries) and ER (row summaries, full attention).
  - **TP:** the text world. Controls: vm-00040's TH (window + line summaries) and TW (window alone), and full
    attention.
- **Measures** as in vm-00040. Speeds: EP 74 ticks/s (three on the 4090), TP 55 (three on the 3090).

## Results

### R1. Cost: attention scores per 8-step window

| grid | patches | full | pyramid | 4 quarters (for comparison) |
|---|---|---|---|---|
| 8 × 8 (here) | 64 | 139,656 | 61,900 (0.44) | 41,352 (0.30) |
| 16 × 16 | 256 | 2,131,080 | 340,428 (0.16) | 558,216 (0.26) |
| 32 × 32 | 1,024 | 33,689,736 | 1,754,188 (0.05) | 8,523,912 (0.25) |

With 4× more patches, the pyramid grows 5.2–5.5× (N·log N), full attention 15–16× and fixed quarters 13–15× (N²/4). At
the 8 × 8 grid the pyramid saves less than the row summaries (0.33) or quarters. At 8 × 8, attention is also only 26%
of a layer's compute, against 57% at 16 × 16 and 84% at 32 × 32.

### R2. Text: val nats per letter

| arm | best, s1 / s2 / s3 (mean) | at 100k (mean) | at 10k |
|---|---|---|---|
| full attention (vm-00030) | 1.717 / 1.736 / 1.745 (1.733) | 1.977 | 1.88–1.97 |
| TH: window + line summaries (vm-00040) | 1.773 / 1.759 / 1.763 (1.765) | 1.775 | 2.16–2.18 |
| TW: window (vm-00040) | 1.981 / 1.969 / 1.942 (1.964) | 2.011 | 2.25–2.28 |
| **TP: window + pyramid** | 1.922 (90k) / 1.925 (80k) / 1.916 (50k) (**1.921**) | 1.933 | 2.24–2.28 |

By the predicted letter's column and line (mean of the seeds at their best):

```
column:        0     1     2     3     4     5     6     7
TW (window)  1.91  2.50  2.11  1.82  1.85  1.83  1.84  1.84
TP (pyramid) 1.85  2.37  2.06  1.81  1.82  1.83  1.81  1.82
TH (lines)   1.75  1.76  1.74  1.71  1.78  1.78  1.80  1.80

line:          0     1     2     3     4     5     6     7
TP (pyramid) 1.91  1.97  1.92  1.93  1.84  1.97  1.91  1.91
```

### R3. Text: generation (seed 1 at 90k, 32 val windows, 4 pages, T = 0.8)

| model | samples' nats per letter under GPT | mean line length | lines over 60 letters |
|---|---|---|---|
| real text | 1.756 | 28.8 | 0% |
| full attention | 1.451 | 30.7 | 0% |
| TH (window + line summaries, vm-00040) | 1.775 | 35.1 | 7% |
| TW (window, vm-00040) | 2.574 | 35.4 | 18% |
| **TP (window + pyramid)** | **2.365** | 34.9 | 19% |

From the same context as before ("Hearest thou, Biondello?"):

```
TP:  ISABELLA:
     Yes, to your de wised with you his woule companow
     I weare remem my deat you wity gent. Anges;
```

### R4. Image: frozen at 100k, food / poison per 1k (food : poison)

| run | instinct off | 1-step | 4-step mean | 4-step pair |
|---|---|---|---|---|
| EP s1 | 21.4 / 20.2 (1.06) | 22.6 / 11.1 (2.04) | 26.8 / 6.6 (4.08) | 32.6 / 7.1 (4.58) |
| EP s2 | 19.1 / 19.8 (0.97) | 16.4 / 10.1 (1.62) | 23.9 / 8.9 (2.69) | 23.8 / 6.1 (3.92) |
| EP s3 | 13.3 / 14.2 (0.93) | 14.6 / 9.4 (1.55) | 15.3 / 8.6 (1.79) | 17.3 / 7.1 (2.43) |

Means (food / poison per 1k):

| arm | instinct off | 1-step | 4-step mean | 4-step pair |
|---|---|---|---|---|
| EB (full attention, vm-00036) | 33.5 / 19.4 | 36.6 / 8.8 | 39.1 / 5.9 | 44.4 / 4.1 |
| EH (window + row summaries, vm-00040) | 15.9 / 27.7 | 18.4 / 21.0 | 18.6 / 15.8 | 24.6 / 14.2 |
| ER (row summaries, full attention, vm-00040) | 17.6 / 15.6 | 20.1 / 7.9 | 24.1 / 5.2 | 26.5 / 5.2 |
| **EP (window + pyramid)** | 17.9 / 18.1 | 17.9 / 10.2 | 22.0 / 8.0 | 24.6 / 6.8 |

In life, over the last quarter: EB 29–50 food / 5.9–9.6 poison; EH 16–19 / 16.4–17.2; EP 14–22 / 11.4–12.2.

### R5. Image: prediction at 100k

| run | held-out patch CE (s1 / s2 / s3) | move CE | body CE | dream: exact h16 | dream: hunger err h16 |
|---|---|---|---|---|---|
| EB | 0.779 / 0.815 / 0.778 | 0.68 / 0.72 / 0.41 | 0.55 / 0.74 / 0.68 | 0.882–0.892 | 0.062–0.103 |
| EH | 0.773 / 0.808 / 0.776 | 0.65 / 0.69 / 0.46 | 0.47 / 0.56 / 0.60 | 0.902–0.914 | 0.090–0.136 |
| EP | 0.778 / 0.837 / 0.773 | 0.56 / 0.68 / 0.57 | 0.71 / 0.70 / 0.90 | 0.904–0.913 | 0.089–0.116 |

EP seed 2 predicts worse than every other seed-2 run (0.837 against 0.808–0.815, in every row). By column, column 0 is
the worst (0.80–0.84). The window does not wrap around the torus, as in vm-00040's EW, and the pyramid's blocks do not
cover column 7 for column 0.

### R6. Video (in `exp_00041/`, not in git)

- **`video_EP_s3_vs_EB_s3/compare.mp4`**: 1,000 steps at 15 fps. EP seed 3 (left) and full attention (EB seed 3,
  right), frozen at 100k with the 4-step pair lookahead, in the same world (the forage test's world 0). In world 0:
  **EP 24 food, 11 poison; EB 66 food, 3 poison**. Over the 16 worlds the frozen test's numbers are 17.3 / 7.1 and
  60.4 / 3.0; seed 3 is EP's weakest seed. `forage.mp4` and `forage_sheet.png`: EP alone.

## Analysis

- **The pyramid scales as intended but does not pay off at this size.** Its share of the attention scores falls from
  44% at 8 × 8 to 5% at 32 × 32, as N·log N should. At our 8 × 8 grid it computes more scores than the row summaries
  (44% against 33%), and attention is only a quarter of the compute anyway.
- **In text the block shape matters more than the levels.** The pyramid gains 0.04 over the window alone (1.921
  against 1.964) and stays 0.16 behind the line summaries (1.765). Generation agrees: 2.365 nats per letter under GPT,
  against TW's 2.574 and TH's 1.775, and words fall apart ("I weare remem my deat you wity gent").
  - The cost sits where the window leaves gaps. Column 1, predicted by the line's first letter, is still at 2.37,
    because nothing carries the end of the previous line to the start of the next. Square blocks end mid-line, and
    the ones that contain the previous line's end are complete only every second or fourth line.
  - Line 4 is the best (1.84). It starts right after the 4 × 4 blocks of lines 0–3 complete, so its letters can read a
    summary of everything above.
  - Line summaries match how text is read; square blocks match how images are laid out.
- **In the image world the pyramid behaves like the row summaries,** with better poison avoidance than EH (6.8 against
  14.2 with the pair lookahead) and the same food (24.6). Both stay far from full attention (44.4 / 4.1). ER (vm-00040)
  showed that summaries with full attention already cost this much food; vm-00042 tests whether content in the
  summaries changes that.
- **Prediction holds,** except seed 2's 0.837, and the dream is as exact as before (0.904–0.913 after 16 steps).

## Conclusions

1. The pyramid gives N·log N attention (5% of full attention's scores at 32 × 32), with prediction intact in the
   image world.
2. In text, square blocks barely beat a plain window. The line-shaped summaries of vm-00040 are 0.16 nats per letter
   better, at a lower cost on this grid. The shape of the summaries must follow the data's order.
3. Behaviour is as in vm-00040: poison avoidance is near full attention's, food is about half.

## Next Steps

- vm-00042 (running): summaries with content, each a thumbnail of its block (the owner's idea), with the window and
  with full attention, to test whether a move predictor with content brings behaviour back.
- Shapes that follow the reading order in the same mechanism, for example blocks of 1 row × 2^ℓ columns, then rows,
  then bands. They would be universal and would suit text.
- A block-sparse kernel, worth it only at larger grids or longer contexts (attention is 26% of the compute here).
