# vm-00051: Separate Transformers That Share One Body

**Date:** 2026-10-30
**Status:** Complete (pending tag)
**Source:** `exp/vm-00051` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00051/`
**Task plan:** [`../tasks/00059-pieces-one-body.md`](../tasks/00059-pieces-one-body.md)

## Goal

vm-00049 and vm-00050 split the view among separate transformers in the text world, where moves do nothing. A visual
world needs them to act: every piece predicts a move, and the body makes one. The owner's rule: every piece keeps its
own move token, nothing fixes which piece drives which direction, and any division of labour must be learned. Which
rule for the one move lets pieces forage, and how do they compare with one model?

## Method

- **`online.pieces = 1`** (`lifetime.py`, with `model.band_n`): all pieces live one lifetime on one world and one body.
  Each has its own weights (seed + 7919 k), move prediction and instinct. They share one memory, and each takes its own
  gradient step on the same rows. Every piece's move token holds the move made. Tests: `tests/test_pieces_lifetime.py`;
  a run without pieces is bit for bit unchanged on the CPU.
- **`online.piece_rule`:** `mean` (the proposals' mean); `max` (per axis, the largest magnitude; ties at random: the
  first version gave ties to piece 0, and was restarted at 1k ticks); `gain` (the whole move of the piece whose instinct
  predicts the lowest next hunger); `random` (the whole move of a random piece).
- **Runs:** the egocentric touch world with checker poison (`configs/U_touch_ego_c2.json`), 8 × 8 patch grid,
  `uni_pyr = 3`, `sum_thumb = 1`; seeds 1–3, 60k ticks, the 3090. F (one model), F64 (one model, 64 candidates, as many
  as four pieces together), H (vertical halves, mean), Q (quadrants, mean), QM (quadrants, max), QG (quadrants, gain),
  QR (quadrants, random). QW (quadrants, `band_win = 1`, mean) was stopped at 6k ticks to make room.
- **`eval_pieces.py`:** frozen final checkpoints in 16 worlds for 1,500 ticks: food, poison, move size, aim (the cosine
  of the move made and the direction to food within 16 pixels), wins by the food's quadrant, and `--touch` (each poison
  touch taken apart). Rules can be swapped on frozen weights.

## Results

### R1. Foraging over the last 20k ticks, per 1k ticks

| arm | food | seeds | poison | seeds | food share | proposal size |
|---|---|---|---|---|---|---|
| F | 19.0 | 16.1, 10.2, 30.8 | 10.5 | 12.4, 5.2, 13.8 | 0.64 | not logged |
| F64 | 19.6 | 23.2, 16.1, 19.3 | 6.1 | 4.1, 6.8, 7.5 | 0.76 | not logged |
| H, mean | 8.0 | 8.2, 9.1, 6.7 | 9.6 | 6.4, 8.6, 13.7 | 0.46 | 0.19–0.28 |
| Q, mean | 8.8 | 5.9, 17.2, 3.2 | 7.1 | 6.0, 11.1, 4.3 | 0.55 | 0.11–0.32 |
| QM, max | 27.5 | 39.0, 13.2, 30.2 | 21.9 | 19.6, 27.2, 18.9 | 0.56 | 1.00 |
| QR, random | 13.4 | 10.3, 17.8, 12.2 | 6.8 | 6.2, 6.5, 7.7 | 0.66 | 0.49–0.65 |
| **QG, gain** | **42.8** | 50.1, 40.8, 37.5 | **4.8** | 5.8, 4.8, 3.8 | **0.90** | 0.63–0.75 |

QG starts at 25.2 food and 19.7 poison in its first 10k ticks, and learns over the life.

### R2. Frozen final checkpoints, means of three seeds

| arm, rule when acting | food | poison | move size | aim | smoothness |
|---|---|---|---|---|---|
| F | 19.9 | 10.2 | 0.53 | 0.28 | 0.19 |
| F64 | 19.4 | 5.3 | 0.53 | 0.34 | |
| QR, random | 14.8 | 6.6 | 0.55 | 0.21 | 0.10 |
| QG, gain | 43.0 | 4.7 | 0.69 | 0.63 | 0.43 |
| QG, random | 38.3 | 6.3 | 0.69 | 0.53 | |

Smoothness: the cosine of successive moves (800 ticks). Win shares are about 25% per piece in every arm. The piece whose
quadrant holds food within 16 pixels wins 21–26% of those ticks: chance.

### R3. QG's poison touches (339, frozen)

| | share |
|---|---|
| the piece that moved foresaw the rise (> 0.1) | ~1% (F: 30–69%) |
| only another piece foresaw it | 60% |
| nobody foresaw it | 40% |
| another piece's move would have missed it | 78% |
| food on the same tick | 26% (F: 2–7%) |
| hunger above 0.5 before it | 14% (F: 26–41%) |

Touches by ticks since food follow all ticks (1–2 ticks after food: 6–10% of touches, 8–9% of ticks).

## Analysis

- **The rule feeds back into the proposals.** A piece samples from its prediction of the move made. The mean makes a
  move smaller than the proposals, and they shrink (H, Q; the hesitant seeds never recovered). Max makes the largest,
  and they saturate at 1.00, a body that runs into everything. A whole move made as proposed (gain, random) keeps them
  at a working size.
- **Competition during learning is what makes QG.** QG with a random winner on its frozen weights keeps most of its food
  (38.3), but a lifetime with a random winner never gets there (13.4, aim 0.21). The selection at acting time adds
  little; the selection during learning builds pieces that aim at 0.53 by themselves, against F's 0.28. Every piece
  learns to predict the move made, and under gain that move is the best of four models' choices. A plausible
  mechanism, not shown directly.
- **More candidates are not the cause.** F64 halves F's poison and adds no food. Moving faster is not enough: QM moves
  fastest, with poor food share.
- **No division of labour by place or direction:** pieces propose alike, and the piece that sees the food does not win
  more. Near food, some pieces win more than others in a way that differs by seed; it is mild.
- **Poison gets past gain by the winner's curse.** Gain picks the lowest prediction, so it favours the piece that does
  not see the poison: the mover almost never foresaw its touch, and most of the time another piece did or had a safe
  move. Food on the same tick is the second hole. Poison shortly after food, the owner's worry, is not one.
- **What is lost:** each piece predicts the frame slightly worse than one model (patch CE 0.765 against 0.716), and a
  body needs four models' compute. Gain compares all the pieces in one place, so this split is not fully local.

## Conclusions

1. Pieces can act as one body with no fixed roles, and forage better than one model: 42.8 food and 4.8 poison per 1k
   ticks against 19.0 and 10.5, all three seeds well clear. The rule for the one move decides it.
2. The rule matters as a teacher: it decides what every piece learns to predict.
3. Whole proposals, chosen by their predicted outcome, work; averaging does not.
4. The winner's curse is a measured cost of choosing by each piece's own prediction.

## Next Steps

- vm-00052 (task 00060): the council rule and a world where circles never overlap, against QG.
- Whether the rule's lesson holds for one model: a piece's advantage may be its diversity, testable by four models over
  the whole view.
- The window (`band_win`) with a rule that works: QW stopped early.
- Smoother paths: the winner on 74% of the ticks is a tie, decided by noise.
