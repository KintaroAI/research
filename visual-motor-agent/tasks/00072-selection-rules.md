# Task 00072: Ways to Choose the Move (experiment vm-00063)

**Date:** 2026-11-01
**Status:** Complete

## Context

- The list of algorithms is in `../ARCHITECTURE.md`, "How many pieces decide one move". Known: the pieces' predicted hungers differ by more than 0.005 on only ~26% of ticks, so gain chooses by noise
  on the rest; the piece that moved foresaw ~1% of its poison touches (the winner's curse); and a rule teaches, so only whole lifetimes count.
- The owner asked for different ways of choosing the move in the circle world (2026-11-01). Five rules that need no model change, chosen for those two faults.

## Design

`pieces_move` in `lifetime.py`, `online.piece_rule`:

| rule | how it chooses | fault it answers | status |
|---|---|---|---|
| `gain_sharp` | gain; among hungers within 0.005 of the lowest, the sharpest move prediction | ties | was frozen only (43.5 food, 5.8 poison) |
| `soft` | proposals' mean, weighted by softmin(predicted hunger / `piece_temp`, 0.01) | ties, with no arbitrary pick | new |
| `pessimism` | every piece scores every move: lowest mean + `piece_lambda` (1) × their spread | the curse | new |
| `rank` | every piece ranks every move, the lowest sum of ranks; ties at random | scores of different scale | new |
| `veto` | every piece scores every move: the lowest worst score | the curse, strictly | was frozen only (33.4, 1.4) |

`turn_cost` stays 0.

## Experiment vm-00063

- `configs/U_default.json` with each rule, seeds 1–8, 60k ticks: 40 lifetimes, seeds 1–3 on the 3090 and 4–8 on the 4090, with MPS.
- **Control:** vm-00055's eight seeds of QNA (gain, 34.7 ± 10.2 food, 3.5 ± 0.6 poison) and QCA (council, 28.8 ± 9.1, 1.5 ± 0.4): the same settings, an older build, not rerun.
- **Measure:** food and poison per 1k over the last 20k ticks, mean and 95% interval over seeds; food share; proposal size (a mean can shrink it); share of ticks per winning piece; ticks/s.
- **Reading:** seeds of one rule span 14–59 food, so only differences beyond the intervals count. A rule that keeps QNA's food with QCA's poison would beat both.

## An addition: true ties and a same-build control (2026-11-01)

The first 40 lifetimes showed wins in the order p0 > p1 > p2 > p3 in all eight seeds of every scoring rule, so ties were measured on frozen weights (RA s1, RA s5, QCA s4, 12,800 decisions each):

- `council`, `veto`, `pessimism`: scores tie exactly on 36–80% of decisions, but among **identical** moves (0.000–0.007 among different ones). The lowest piece gets the win, the move is the same: only the win share is an artefact.
- `rank`: the noise broke ties inside each scorer's ranks, not between the sums, which are small integers: 4–14% of decisions tied among **different** moves and went to the lowest piece. The table's "ties at random" was false for RA.

So: `rank` now breaks ties between the sums at random, and the first behaviour is kept as `rank_first`. **New arms** (seeds 1–8, 60k ticks, as above, 24 lifetimes): **RT** (`rank`, true ties), **NA** (`gain`) and **CO** (`council`), the last two the same-build control, since QNA and QCA came from an older build. RA stays reported as what it was: rank with ties to the lowest piece.

## Verification

`tests/test_pieces_lifetime.py`: each rule chooses as computed by hand (pessimism with and without the spread; rank ignoring how far a score is; soft from the mean to gain by temperature); short lifetimes on the
CPU and the GPU; invalid settings.

## Results

The last 20k ticks, per 1k, eight seeds, on one build:

| rule | food | poison |
|---|---|---|
| `gain` (NA) | 28.7 ± 5.7 | 3.90 ± 0.47 |
| **`council` (CO)** | **32.8 ± 10.4** | **1.30 ± 0.45** |
| `rank_first` (RA) | 33.0 ± 5.7 | 1.88 ± 0.61 |
| `rank` (RT) | 24.9 ± 6.8 | 2.90 ± 1.41 |
| `gain_sharp` (GS) | 32.1 ± 7.1 | 7.77 ± 1.93 |
| `veto` (VE) | 19.3 ± 5.5 | 1.66 ± 0.66 |
| `pessimism` (PE) | 17.6 ± 6.9 | 1.70 ± 0.69 |
| `soft` (SO) | 8.7 ± 5.1 | 7.49 ± 1.46 |

Council beats gain: poison −2.60 ± 0.75, food +4.0 ± 10.9. Fair ties made rank worse (−8.2 ± 4.3 food), so RA stays as the rule `rank_first`. The default stays `gain` until the owner decides. Full results in `../00063-selection-rules/README.md`.
