# vm-00069: Long Lives for Many Pieces

**Date:** 2026-11-01
**Status:** Complete
**Source:** `exp/vm-00069` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00069/`
**Task plan:** [`../tasks/00080-long-life-pieces.md`](../tasks/00080-long-life-pieces.md)

## Goal

60k ticks may be about 100 minutes of lived time. What does a life ten times longer change: does it get trapped, does it pay, and do the pieces keep their lead?

## Method

- **LT:** 16 pieces, `online.tie_tables=inout`, seeds 1–4 on the 4090, under MPS. **LD:** the default, 4 pieces, seeds 1–4 on the 3090.
- Both: `gain`, `configs/U_default.json` (channels on), `online.steps=500000`, `online.replay=500000`, so the memory holds the whole life and nothing is forgotten.
- **Frozen check:** `eval_pieces.py --ring` at 500k, 16 worlds, 1,500 ticks, every seed, `--rule random` beside gain, on the 3090 without MPS.
- **Set before the results (task 00080):**
  1. A life is trapped if its mean hunger over the last 20k ticks is above 0.9 or its food is under 1 per 1k.
  2. Long lives pay for an arm if its food over the last 20k minus its own 40–60k is above zero beyond its interval, or its poison is below zero beyond its interval with food not below.
  3. Pieces hold their lead if LT − LD in food over the last 20k is above zero beyond its interval.

With 4 seeds, df 3: t = 3.182.

## Results

### R1. Food / poison per 1k ticks, by 50k ticks, mean of four seeds

| arm | 0–50k | 50–100k | 100–150k | 150–200k | 200–250k | 250–300k | 300–350k | 350–400k | 400–450k | 450–500k |
|---|---|---|---|---|---|---|---|---|---|---|
| LD | 35.5 / 7.85 | 41.8 / 2.20 | 42.1 / 1.46 | 42.9 / 1.06 | 43.6 / 0.78 | 41.9 / 0.76 | 41.8 / 0.47 | 42.4 / 0.43 | 43.1 / 0.41 | 42.7 / 0.35 |
| LT | 51.4 / 9.42 | 61.4 / 3.69 | 55.6 / 2.55 | 55.1 / 2.02 | 54.7 / 1.56 | 54.9 / 1.29 | 55.4 / 0.97 | 57.4 / 0.93 | 58.2 / 0.85 | 58.4 / 0.60 |

### R2. The last 20k ticks, per 1k

| arm | food | poison | share | hunger | patch CE | body CE | move CE |
|---|---|---|---|---|---|---|---|
| LD | 42.8 ± 10.5 | 0.34 ± 0.04 | 0.99 | 0.041 | 0.739 | 0.239 | 0.908 |
| LT | 58.5 ± 16.5 | 0.57 ± 0.55 | 0.99 | 0.028 | 0.764 | 0.231 | 1.000 |

Food by seed: LD 45.2, 50.7, 39.9, 35.5; LT 74.0, 52.0, 54.0, 54.1.

Differences seed by seed:

| difference | food | poison |
|---|---|---|
| LD, last 20k − its 40–60k | +1.6 ± 5.2 | **−2.40 ± 0.60** |
| LT, last 20k − its 40–60k | −4.2 ± 7.5 | **−3.80 ± 1.87** |
| LT − LD, last 20k | +15.7 ± 18.1 (seeds +28.8, +1.3, +14.1, +18.6) | +0.24 ± 0.53 |

### R3. Frozen at 500k

| arm | rule | food | poison | share | move size | smooth | aim |
|---|---|---|---|---|---|---|---|
| LT | `gain` | 57.9 ± 7.9 | 1.10 ± 0.62 | 0.98 | 0.77 | 0.29 | 0.85 |
| LT | a piece at random | 55.1 ± 11.2 | 0.83 ± 0.20 | 0.99 | 0.75 | 0.24 | 0.80 |
| LD | `gain` | 42.3 ± 10.8 | 0.58 ± 0.51 | 0.98 | 0.65 | 0.24 | 0.72 |
| LD | a piece at random | 42.8 ± 11.1 | 0.43 ± 0.35 | 0.99 | 0.64 | 0.24 | 0.70 |

`gain` − random, seed by seed: LT food +2.8 ± 8.5, poison +0.28 ± 0.73; LD food −0.6 ± 1.9, poison +0.15 ± 0.54.

Not a reading, and made after seeing R2: LT − LD in frozen food, seed by seed, is +15.6 ± 11.0 (+12.8, +7.5, +23.3, +18.8).

### R4. Move CE climbs and the moves do not change

Mean of seeds over the 20k ticks from each mark:

| arm | measure | 0k | 100k | 200k | 300k | 400k | 480k |
|---|---|---|---|---|---|---|---|
| LD | move CE | 0.160 | 0.469 | 0.692 | 0.809 | 0.874 | 0.908 |
| LD | proposal size | 0.88 | 0.89 | 0.90 | 0.89 | 0.90 | |
| LD | smooth | 0.30 | 0.26 | 0.25 | 0.24 | 0.23 | 0.23 |
| LD | agreement of p0 with the move | 0.60 | 0.56 | 0.55 | 0.53 | 0.54 | |
| LT | move CE | 0.112 | 0.524 | 0.797 | 0.916 | 0.973 | 1.000 |
| LT | proposal size | 0.87 | 1.07 | 1.06 | 1.05 | 1.08 | |

Patch CE falls over the same lives, 0.792 to 0.739 in LD and 0.818 to 0.764 in LT, and body CE stays flat. Mean memory age rises in step with the life, as a memory of the whole life must: half the life at every point.

### R5. Speed

LD 22–27 ticks/s per life at the end, four lives on the 3090, 8.3 hours. LT 13.8 ticks/s per life, four on the 4090, 13.3 hours, of which over 3 hours at about 2 ticks/s while vm-00070 shared the GPU.

<!-- figures -->
## Figures

![Frames of the world](images/world.png)

*The world of this experiment as its model sees it, every 12 ticks. Moves are random: this shows the world, not the trained agent.*

![Food touched per 1k ticks; Poison touched per 1k ticks](images/curves.png)

*From each run's `metrics.jsonl`, the mean of its seeds by arm, smoothed.*
<!-- /figures -->

## Videos

Not in the repository: `~/data/research/visual-motor-agent/exp_00069/videos/`, 1,000 ticks at 500k. `ld_seeds/compare.mp4` and `lt_seeds/compare.mp4` show each arm's four seeds from the same start; `lt_vs_ld/compare.mp4` shows LT seed 4 against LD seed 1, the seeds closest to their arms' mean food; `ld_s1/forage.mp4` shows LD seed 1 alone.

## Readings

1. **Trapped: no.** No life of either arm is near the limits: the worst mean hunger of the last 20k is 0.076, the least food 35.5. Hunger touched 1.0 in at least one log row after 20k ticks in every life, and came back. The ring memory of vm-00031 trapped two lives of three; these are 4 seeds of each of two different models, so the whole-life memory is not proved the cause.
2. **Long lives pay: met for both, in poison and not in food.** LD −2.40 ± 0.60, LT −3.80 ± 1.87. "Food not below" is read in the reading's own terms, as not below zero beyond its interval. LT's food is lower than at 40–60k by 4.2 on its mean, and that interval includes zero.
3. **Pieces hold their lead: not met.** LT − LD in food is +15.7 ± 18.1, and the interval includes zero. All four seeds are above zero, and the frozen check gives the same size of lead, but that check was not the reading.

## Analysis

- **A long life cleans; it does not feed.** Each arm reaches its food in the first 100k ticks and then stays there, while poison falls for the whole life: LD to about an eighth of its level at 40–60k, LT to about an eighth too. Both arms end at a share of 0.99. Whatever the extra 440k ticks teach, it is what to avoid.
- **LT's food has a shape LD's does not.** It peaks at 61.4 at 50–100k, drops to 55 and climbs back to 58.4 by the end. The peak is what makes its long-life food difference negative. Seed 3 accounts for much of the dip and the return, 46.8 to 39.4 to 54.3 by 100k blocks. With 4 seeds no more can be said.
- **The lead may well be real, but this experiment did not show it.** Going from 4 to 16 pieces at 60k gave +20.6 ± 12.7 with shared tables on 8 seeds (vm-00067), a similar size, without channels. Here four seeds, one far from the others (74.0), leave too wide an interval. The reading as set is what stands.
- **Gain does nothing by the end, in either arm.** A piece drawn at random does as well as `gain`, though `gain` was what the lives were lived by. Gain's aim at near food is +0.051 ± 0.034 above a random piece in LT, +0.014 ± 0.029 in LD. The random draw is of pieces raised under gain, so this says nothing about what the rule taught them. A likely reason: `gain` chooses by predicted hunger, and the pieces' predictions now differ by more than 0.005 on only 12% of ticks in LT and 10% in LD, against 44% for G16 and 21% for NA at 60k, the same numbers of pieces without channels. On most ticks there is little for the rule to choose by. This was not tested.
- **Frozen poison is about twice the lifetime's.** LD 0.58 against 0.34, LT 1.10 against 0.57, with food unchanged. Either the frozen check's 16 worlds are harder than one life at the end, or a learner that keeps learning keeps its poison lower. This was not separated, and the intervals overlap.
- **Move CE is not explained.** A model that gets better at the frames and worse at its own moves, with the moves about the same, is a real oddity, and it is still rising at 500k. It is scored on the training batch, 7 of whose 8 windows come from the memory, so one guess is that a memory of the whole life keeps showing it moves from every age; another is that the model's move distribution widens over time, since it is trained on samples of itself. Neither was tested. The instinct draws its candidates from that distribution, so if the second is right, a longer life may slowly change what it can choose from. The readings above do not depend on this.
- **Caveats:** LT and LD have channels on, and their controls at 60k did not, so nothing here is read against those. Seeds of LT and LD are paired by number only, since they are different models. LT was slowed for over 3 hours by vm-00070, which changes wall time and not ticks. About 15 tests are read here; with df 3 only the poison differences are firm.

## Conclusions

1. **No life was trapped** in 500k ticks with a memory of the whole life, in either arm.
2. **Long lives pay in poison only.** Food is settled by 100k ticks; poison keeps falling to the end, by about seven eighths from 40–60k.
3. **The reading for pieces is not met:** LT − LD +15.7 ± 18.1 in food. The lead's size is like that of vm-00067, but four seeds did not show it.
4. By 500k `gain` does no better than a piece drawn at random, on food or on poison, likely because the pieces' predicted hungers rarely differ.
5. Move CE rises from about 0.15 to 0.91 and 1.00 while the moves stay about the same. That is open, and it is the first thing to look at before longer lives.

## After the write-up: what hunger does to the move (2026-11-02)

Three frozen tests of the final checkpoints, asked after the write-up by the owner: is there a gap in resolution around food, and is there a chain of moves? Hunger's bins are 1/31 = .032 apart and a food takes hunger to 0, so its relief is only as large as the hunger at that moment; poison always adds .5. `dev/eval_hunger.py` is newer than this experiment's tag. Rows: `hunger_reach.jsonl`, `hunger_told.jsonl`, `hunger_chain.jsonl` in the outputs; `make hunger` repeats them.

**The bins hide nothing.** With food one move away, LD / LT, 1,500 ticks in 16 worlds, all four seeds of each arm:

| Real hunger | Share of ticks | Predicted drop from food | Drop above the .005 tolerance | Lowest predicted hunger is a food move |
|---|---|---|---|---|
| 0–.005 | 21 / 28% | .0044 / .0035 | 12 / 10% | 95 / 93% |
| .005–.01 | 13 / 15% | .0081 / .0085 | 61 / 56% | 95 / 95% |
| .01–.02 | 20 / 20% | .0132 / .0130 | 81 / 76% | 96 / 96% |
| .02–.04 | 22 / 20% | .0219 / .0208 | 88 / 87% | 97 / 97% |
| .04–.1 | 19 / 13% | .0445 / .0407 | 90 / 89% | 96 / 97% |
| .1+ | 5 / 4% | .21 / .27 | 90 / 88% | 94 / 91% |

The model reads a drop of an eighth of a bin and ranks the food first. Near full the tolerance hides it: a piece with food in reach keeps a first sample that misses it on 27% (LD) and 19% (LT) of such ticks. The body still eats on 91% and 98% of the ticks with food in reach at that hunger, and on 96–99% above it. Of all food, 31% (LD) and 40% (LT) is eaten under .005, where it is worth almost nothing. A poison in reach is predicted as a rise of .32 and .34, above the tolerance on 95–96% of ticks, and the lowest predicted hunger is a poison move on about 1%.

**Hunger does not steer.** The same models told a false hunger, the world's own untouched, 750 ticks in 16 worlds, four seeds each. Aim: the cosine of the move to food within 16 pixels.

| Arm | LD food / poison | LT food / poison | Aim, LD / LT |
|---|---|---|---|
| Real hunger | 42.4 / 0.69 | 57.4 / 1.27 | .73 / .84 |
| Told 0 | 45.8 / 0.15 | 69.8 / 0.42 | .68 / .81 |
| Told .05 | 40.0 / 0.90 | 53.4 / 2.27 | .74 / .85 |
| Told .5 | 38.5 / 1.69 | 53.5 / 7.60 | .70 / .82 |
| Real hunger, tolerance 0 | 42.0 / 0.94 | 49.3 / 2.77 | .76 / .84 |

Aim barely moves with the hunger told. Told 0, food is up in all 8 seeds (LT +12.4 ± 11.9, LD +3.4 ± 4.4), though at hunger 0 a food cannot lower it further. Told .5, poison rises (LT +6.3 ± 4.9, LD +1.0 ± 0.8). A guess, untested: above .1 a state the model rarely sees spreads its predictions (.09 against .01–.05) and the instinct overrides on 27–29% of a piece's ticks against 2–5%, so noise replaces good moves. The tolerance earns its place: without it LT loses 8.1 ± 3.7 food and gains 1.5 ± 1.1 poison.

**A short reach, no chain.** Nothing in training links a move to a food several moves later: the instinct compares next-tick hunger and it overrides on 2–5% of ticks, and the model predicts the next step. Aim by the gap to the food's edge, 1,500 ticks in 16 worlds; blind is the same moves replayed in another world, the baseline:

| Gap (pixels) | Moves at top speed | Share of ticks | Aim LD | Aim LT | Blind |
|---|---|---|---|---|---|
| 0–4 | 1 | 4–5% | .89 | .91 | .16–.18 |
| 4–8 | 1–2 | 6–7% | .50 | .70 | .10 |
| 8–12 | 2–3 | 9–11% | .26 | .36 | .06 |
| 12–16 | 3 | 14–15% | .12 | .15 | .04 |
| 16–24 | 3–5 | 40–41% | .07 | .05 | .02 |
| 24+ | 5+ | 23–24% | .02 | .02 | .01 |

The whole torus is in view, yet on about 64% of ticks the food is hardly aimed at. Chaos and a grab, as a test: blind moves, except straight at the food whenever one move can touch it.

| Arm | LD food / poison | LT food / poison |
|---|---|---|
| Blind | 14.0 / 18.6 | 17.7 / 22.7 |
| Blind and a perfect grab | 26.6 / 18.6 | 34.6 / 22.5 |
| The model | 42.3 / 0.58 | 57.9 / 1.09 |

That explains about 60% of the food. The model leads the perfect grab by +15.6 ± 10.1 (LD) and +23.3 ± 8.9 (LT), in all 8 seeds, and the lead fits its aim at 2–3 moves. It is a wider reach and not a plan; the likely source, untested, is "toward food" learned up close and carried a little further. Sight does most for poison: 20 to 30 times less than blind. Successive moves are little alike: cosine .24 and .29.

**Memory** is blind to food's size: these lives store every step (`store_base` 1), replay is uniform, the relief gate when on only asks whether a food was touched, and surprise gating is refused with pieces.

**Caveats:** frozen and short, chosen after seeing the results, with about 30 comparisons. By the correction of a mean at least 1.9 times its interval, only LT's food without the tolerance stands alone; told 0 has its 8 of 8 seeds. Seeds are few, and LD against LT is not paired.

**Reading:** the fall of poison in R1 and the level food fit this asymmetry: poison is a fixed, large signal and food a small one, but no test here shows that this is the cause.

## Next steps

- Separate the two guesses about move CE: score it on the newest window alone, and on a memory limited to recent ticks.
- A longer life alone is not worth repeating for food. If lives go on, a changing world is the test: the adaptation experiment asks whether a settled model can still move.
- A chain of moves: score moves on what happens several moves later, without imagined frames (vm-00023's conclusion 3): predict hunger several moves ahead from the real window, trained on real futures. It needs the ring to take a horizon above 1.
- Why being told hunger 0 does better: if it holds on longer frozen runs, hunger in the input is costing food and poison.
