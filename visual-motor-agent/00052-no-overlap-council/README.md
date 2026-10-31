# vm-00052: Circles That Never Overlap, and a Council of Pieces

**Date:** 2026-10-31
**Status:** Complete
**Source:** `exp/vm-00052` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00052/`
**Task plan:** [`../tasks/00060-no-overlap-council.md`](../tasks/00060-no-overlap-council.md)

## Goal

vm-00051's QG (four quadrants, the winner by gain) makes 42.8 food and 4.8 poison per 1k ticks. Its 339 frozen poison touches
named two holes: food on the same tick (26%), and the winner's curse (the mover foresaw ~1%, only another piece 60%).
Close each, alone and together: a world where circles never overlap, and a council of pieces.

## Method

- **`env.no_overlap`:** a new circle is placed clear of the others; moving circles bounce off each other and off food.
  Tests: `tests/test_no_overlap.py`.
- **`online.piece_rule = council`:** every piece scores every piece's move; the lowest mean predicted hunger is made
  whole. S² scoring steps per tick.
- **Runs:** as QG (`configs/U_touch_ego_c2.json`, quadrants, `uni_pyr = 3`, `sum_thumb = 1`), seeds 1–3, 60k ticks:
  QN (gain, no overlap), QC (council), QCN (council, no overlap). Control: QG.
- **Frozen:** `eval_pieces.py --touch`, 16 worlds, 1,000 ticks, each arm with its own rule.

## Results

### R1. Foraging over the last 20k ticks, per 1k ticks

| arm | food | seeds | poison | seeds | food share | proposal size |
|---|---|---|---|---|---|---|
| QG (gain, old world) | 42.8 | 50.1, 40.8, 37.5 | 4.8 | 5.8, 4.8, 3.8 | 0.90 | 0.68 |
| **QN (gain, no overlap)** | **43.5** | 52.2, 31.9, 46.5 | **3.5** | 3.9, 3.9, 2.8 | **0.93** | 0.62 |
| QC (council, old world) | 29.3 | 41.1, 16.6, 30.1 | 1.9 | 2.6, 1.1, 2.1 | 0.94 | 0.52 |
| QCN (council, no overlap) | 34.9 | 41.1, 25.6, 37.8 | 2.1 | 1.1, 3.6, 1.6 | 0.94 | 0.52 |

### R2. By 10k ticks, food / poison

| arm | 0–10k | 10–20k | 20–30k | 30–40k | 40–50k | 50–60k |
|---|---|---|---|---|---|---|
| QG | 25.2 / 19.7 | 35.9 / 12.0 | 38.3 / 7.0 | 42.4 / 5.4 | 41.2 / 5.6 | 44.4 / 3.9 |
| QN | 22.0 / 18.6 | 32.0 / 12.9 | 40.2 / 7.4 | 41.0 / 5.7 | 44.2 / 3.6 | 42.8 / 3.4 |
| QC | 22.3 / 15.0 | 27.3 / 5.6 | 28.0 / 3.7 | 30.2 / 3.1 | 28.4 / 2.3 | 30.2 / 1.5 |
| QCN | 23.5 / 14.9 | 26.4 / 6.8 | 29.1 / 4.7 | 31.7 / 3.2 | 33.7 / 2.2 | 36.1 / 2.0 |

### R3. Frozen, 60k checkpoints, means of three seeds

| arm | food | poison | smoothness | food on the poison tick | the mover foresaw it |
|---|---|---|---|---|---|
| QN | 46.1 | 3.0 | 0.25 | 0% | ~1% |
| QC | 31.9 | 2.1 | 0.70 | 16–57% | 10–16% |
| QCN | 34.5 | 1.6 | 0.81 | 0% | 17–27% |

## Analysis

- **No overlap removes what it should:** same-tick touches go to zero, and poison falls 27% (4.8 to 3.5), matching the 26%
  of touches they were. Food is unchanged, and no learning cost shows.
- **Council closes much of the winner's curse.** Poison is the lowest of any arm, and arrives early (5.6 by 10k–20k against
  gain's 12.0). The mover now foresees more of its touches.
- **Council costs food during learning: more than frozen suggested.** Frozen on QG's weights it cost 14% of the food; as
  a lifetime, 32% in the old world and 20% with no overlap. Proposals settle at 0.52 against gain's 0.68, and one seed
  of QC stayed at 16.6. Choosing the best-looking move by everyone's vote teaches a more cautious body.
- **QCN is still rising at 60k** (36.1 food, 2.0 poison in the last 10k), and its worst seed is far above QC's worst.
  No overlap helps council's food by more than gain's, 19% against 2%, though that rests on QC's one stuck seed. Overlaps
  may make a cautious body avoid food with poison near it.
- **Council's paths are smooth:** 0.70–0.81 against gain's 0.25–0.43, with no turn cost. A move all pieces accept
  changes less from tick to tick than the one that looks best to one piece.

## Conclusions

1. `env.no_overlap` is free: keep it. The prediction from the touch analysis held to the percent.
2. Gain teaches food, council teaches caution: the best foraging is still gain with no overlap (QN), at 43.5 and 3.5.
3. With no overlap, council gets about 60% of QN's poison at about 80% of its food, and much smoother paths.

## Next Steps

- A mix: gain while learning, council when acting (frozen: 36.8 food and 1.7 poison on QG's weights), or council only
  when pieces disagree. `ARCHITECTURE.md` section 9 lists the other rules.
- QCN longer: it had not levelled off at 60k.
