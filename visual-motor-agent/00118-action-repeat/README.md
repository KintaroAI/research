# vm-00118: Action Repeat — a Move Decided Every 4 Ticks and Held

**Date:** 2026-11-26
**Status:** Complete (negative)
**Source:** `exp/vm-00118` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00118/`
**Task plan:** [`../tasks/00131-action-repeat.md`](../tasks/00131-action-repeat.md)

## Goal

vm-00116 found that in Breakout the default agent's label knows where the paddle should be but orders two candidate moves correctly only about six times in ten, and that its misses are a paddle that travels enough but not to the ball. A move held for 4 ticks shifts the paddle up to 18 px, so what a decision does to the label is four times larger to learn, and a held move cannot dither. Does deciding every 4 ticks and holding the move make the agent play better, and does the label then order the moves?

## Method

- **The change** (`online.act_every = 4`, the owner's idea of 2026-11-17): a move is decided on ticks 1, 5, 9, ... as now and made again on the 3 ticks between. The model still sees, predicts and learns every tick; the move token's target is the move made. The instinct still runs on held ticks, its pick unused.
- **BA:** the default agent in Breakout stage 1 (`configs/U_breakout.json`) with that one change, seeds 1–12, 60k ticks, against BD (vm-00103), paired by seed. All 12 on the RTX 4090, six at a time.
- **Learning off** at 60k (`make frozen`): 6,000 ticks in new courts (seed 100 + s), acting as lived, against BD's test in vm-00116 (its ring arm: the same courts and ticks).
- **The miss autopsy** (`make autopsy`, `make read`; vm-00116's `dev/analyze_misses.py`): BA's frozen 60k weights play 16 new courts for 1,500 ticks a seed, about 3,800 balls.
  - **Deciding ticks:** what belongs to a choice (the move toward the ball, the label's order of the candidates, the label of the move made) is read on BA's deciding ticks only.
  - **Two scorings of the label's order.** vm-00116 scores a pair of candidates by where one tick of each leaves the paddle. A deciding move of BA is made 4 ticks running, so the order is also scored by where the hold leaves the paddle (4 ticks, or up to the landing if sooner). BD decides every tick, so for BD the two are the same number.
  - **The candidates:** the recorder now also keeps, of each transformer's 16 candidate dx, how many repeat the move made on the tick before and how many different values there are.
- **BD recorded again** by the same recorder (`make autopsy-bd`), for its candidates. Its play repeats vm-00116's recordings exactly: all 360 arrays of the 12 seeds are equal, and the tables agree to the last printed digit.
- **A test added after the results** (`make autopsy-grid`): both agents' frozen 60k weights in the same courts, each transformer picking the best dx of an even grid of 16 (no dy, no click) by its own label, in place of the best of its own candidates. The choice among the transformers and the hold are as lived.
- **By seed:** `make table` (`dev/analyze_lives.py repeat`) pairs the lives, the learning-off tests and the recordings by seed. ± is the 95% interval.

## Readings

Set in the task plan before the results (12 seeds, paired by seed):

1. **Lived:** BA − BD hit share at 40–60k, and at 20–40k. Above 0 with its 95% interval at 40–60k: action repeat helps, and the food world and k = 2 follow. The interval includes 0, or below: it does not, at k = 4.
2. **The start:** the first 5k window at .5 (BD: 32–55k, median 40.5k).
3. **Learning off:** BA − BD hit share over 6,000 ticks in the same courts.
4. **The autopsy,** the reason given for the arm: on deciding ticks in the last 8 ticks before a landing, the label's order of the candidates (BD: .60–.62) and the share of moves toward the ball (BD: .58–.60); the misses by kind (BD: 59% enough travel but not to the ball). Three outcomes were named: the order rises with the play (the reason holds); the play rises without the order (action repeat helps another way); neither rises (a larger effect per decision does not mend the order).
5. **What it may cost,** said beforehand: a decision only every 4 ticks cannot correct late, and a full held move overshoots the catching zone (18 px against 14).
6. **Cost:** ticks per second against BD's.

How they held up:

- **The second scoring of reading 4** was added on 2026-11-26 before any recording of BA existed (the lives of seeds 1–6 had ended).
- **The outcome of reading 4 was not among the three named:** the order rose and the play fell.
- **Reading 5 named the costs but no measure.** The measures in R5 and R6 (what 4 ticks do to the paddle, the paddle 4 ticks before the landing) were defined after the autopsy's first tables were seen. So were the candidates' table (R7) and the grid test (R8). They describe; they were not predicted.
- **The pairing of the recordings by seed** (R3's second table) was also built afterwards. Its quantities are those of reading 4.

## Results

The lives ran 2026-11-26 04:23–07:46, the learning-off tests to 08:07, the recordings 08:08–08:41, all on the RTX 4090. The lives had first been started twice at 04:20 by a queueing mistake (two queues writing the same folders); both were stopped after two minutes, the partial folders deleted and the lives started again. No run failed. The tables are `make table` and `make read`.

### R1. Lived: BA plays a little worse than BD

| Ticks | BD (12) | BA (12) | BA − BD (95%) | Seeds |
|---|---|---|---|---|
| 0–10k | .226 | .201 | −.025 ± .036 | 4 of 12 above |
| 10–20k | .251 | .229 | −.021 ± .042 | 5 of 12 above |
| 20–30k | .335 | .286 | −.049 ± .036 | 3 of 12 above |
| 30–40k | .442 | .405 | −.037 ± .037 | 3 of 12 above |
| 40–50k | .602 | .533 | −.069 ± .063 | 2 of 12 above |
| 50–60k | .678 | .617 | −.061 ± .050 | 2 of 12 above |

| Comparison | Ticks | BD | BA | Mean difference (95%) | Seeds |
|---|---|---|---|---|---|
| BA − BD | 20–40k | .387 | .344 | −.044 ± .030 | 1 of 12 above |
| BA − BD | 40–60k | .639 | .574 | −.065 ± .053 | 2 of 12 above |

| First 5k window at or above | BA | BD | BA − BD, seeds reaching it in both |
|---|---|---|---|
| .500 | 40–52k (median 43k) | 32–55k (median 40.5k) | +2.9k ± 4.3k (2 earlier, 10 later of 12) |
| .700 | 48–58k (median 54k), 6 of 12 | 44–58k (median 47.5k), 8 of 12 | +8.2k ± 4.2k (0 earlier, 5 later of 5) |

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BD, 40–60k | .611 | .571 | .696 | .566 | .644 | .581 | .484 | .723 | .669 | .696 | .777 | .652 |
| BA, 40–60k | .587 | .480 | .519 | .491 | .539 | .678 | .569 | .649 | .556 | .645 | .637 | .539 |

- **Reading 1: BA − BD = −.065 ± .053 at 40–60k** (2 of 12 seeds above) and −.044 ± .030 at 20–40k (1 of 12). Both intervals lie below 0: action repeat does not help at k = 4, and costs a little.
- **Reading 2:** BA reaches .5 at 40–52k (median 43k), in 10 of 12 seeds later than BD. The mean delay, 2.9k ± 4.3k ticks, is not clear of 0.
- BA is below BD in every 10k window of the life. The best 5k window of a life averages .685 for BA and .737 for BD.

### R2. Learning off

| | BD (12) | BA (12) | BA − BD (95%) | Seeds |
|---|---|---|---|---|
| Learning off, 60k weights, 6,000 ticks | .701 | .609 | −.092 ± .100 | 4 of 12 above |
| Lived, 50–60k | .678 | .617 | −.061 ± .050 | 2 of 12 above |
| The autopsy's recordings, 24,000 ticks a seed | .712 | .614 | −.098 ± .077 | 2 of 12 above |

- **Reading 3:** the frozen weights show the same difference as the life, in the test (71 to 88 balls a seed) and in the autopsy's recordings (about 310 balls a seed).

### R3. The label orders the moves better

By ticks to landing; BA on its deciding ticks:

| Ticks to landing | 1–2 | 3–4 | 5–8 | 9–16 | 17–32 | 33+ |
|---|---|---|---|---|---|---|
| BD: move made goes toward the ball | .58 | .60 | .59 | .53 | .50 | .46 |
| BA: move made goes toward the ball | .66 | .67 | .65 | .56 | .47 | .31 |
| BD: label's order of the candidates | .60 | .62 | .62 | .58 | .54 | .53 |
| BA: the order by one tick of each candidate | .65 | .68 | .67 | .59 | .52 | .49 |
| BA: the order by where the hold leaves the paddle | .66 | .70 | .69 | .63 | .55 | .49 |
| BD: the order of an even grid of dx | .61 | .64 | .63 | .59 | .55 | .54 |
| BA: the order of an even grid of dx, after the hold | .65 | .68 | .68 | .63 | .56 | .50 |
| BD: label's spread over the candidates (median) | .044 | .038 | .030 | .020 | .011 | .002 |
| BA: label's spread over the candidates (median) | .056 | .069 | .065 | .047 | .024 | .004 |

The last 8 ticks before a landing, by seed:

| Measure | BD (12) | BA (12) | BA − BD (95%) | Seeds |
|---|---|---|---|---|
| Move made goes toward the ball | .595 | .657 | +.061 ± .060 | 8 of 12 above |
| Label's order of the candidates, by one tick of each | .618 | .665 | +.047 ± .023 | 11 of 12 above |
| The same, by where the hold leaves the paddle | .618 | .688 | +.070 ± .021 | 12 of 12 above |
| Label's order of pairs ending 4+ px apart, by one tick | .690 | .757 | +.067 ± .033 | 10 of 12 above |
| The same, by where the hold leaves the paddle | .690 | .726 | +.037 ± .031 | 9 of 12 above |
| The grid's best dx goes toward the ball | .671 | .730 | +.058 ± .040 | 9 of 12 above |
| Label's correlation with the px left across the candidates, after the hold | .226 | .355 | +.129 ± .040 | 12 of 12 above |
| Label's spread over the candidates (median) | .0341 | .0652 | +.0311 ± .0108 | 12 of 12 above |

- **Reading 4, the order:** .688 against BD's .618 by where the hold leaves the paddle (+.070 ± .021, 12 of 12 seeds), .665 by one tick (+.047 ± .023).
- **Reading 4, the move:** BA's deciding move goes toward the ball .657 of the time against .595 (+.061 ± .060, 8 of 12).
- **The label varies about twice as much across the candidates** (.065 against .034) and follows where the move puts the paddle more closely (.355 against .226).
- **It is not the candidates:** on the same even grid of dx for both agents the order is .65–.68 against .61–.64.
- **The label knows the court as a state as BD's does:** in the last 4 ticks the label of the move made is .102 with the paddle under the ball and .384 with it 16–24 px away (BD: .085 and .407).
- **Across BA's 12 seeds the play does not go with the order** (over the last 16 ticks, r = −.04 after the hold; in BD r = +.66). It goes with the share of moves toward the ball (r = +.62).

### R4. Where the balls are lost

Each ball over its last 12 ticks, by how far the paddle had to travel 12 ticks before the landing (px; 0 means it was already under the ball):

| | 0 | 0–8 | 8–16 | 16–24 | 24–32 | 32+ |
|---|---|---|---|---|---|---|
| BD: balls | 1,670 | 1,196 | 545 | 169 | 34 | 21 |
| BA: balls | 1,502 | 1,101 | 712 | 305 | 110 | 48 |
| BD: returned | .84 | .67 | .57 | .33 | .15 | .00 |
| BA: returned | .80 | .53 | .51 | .38 | .32 | .17 |

| The misses | BD | BA |
|---|---|---|
| Misses | 1,058 | 1,463 |
| Wrong way: under half of the moves toward the ball | .39 | .55 |
| Too little travel: toward, but the moves add up to less than the gap | .02 | .08 |
| Enough travel, not to the ball | .59 | .38 |
| Missed by under 2 px | .31 | .23 |
| Px the paddle had to travel, before a return and before a miss | 2.9, 7.5 | 4.3, 9.1 |
| Px its moves asked for, before a return and before a miss | 30.7, 30.9 | 31.3, 30.7 |

- **BA loses the near balls and gains the far ones.** From within 8 px it returns .53 against BD's .67; from 24–32 px .32 against .15.
- **It starts further off.** Twelve ticks before the landing BA's paddle is 16 px or more from the catching zone for 463 of 3,778 balls, BD's for 224 of 3,635.
- **Reading 4, the misses:** most of BA's are wrong-way (.55 against .39). One wrong decision is four wrong ticks.

### R5. What 4 ticks do to the paddle

Spans of 4 ticks from a deciding tick (BD: every 4th tick), starting 5 to 16 ticks before a landing; 870 to 984 spans a recording. The catching zone is 14 px wide; a full move held for 4 ticks is 18 px.

| Spans of 4 ticks | BD (12) | BA (12) | BA − BD (95%) | Seeds |
|---|---|---|---|---|
| The paddle's net travel over the span, px | 3.7 | 5.4 | +1.7 ± 0.6 | 12 of 12 above |
| Start under the landing point: end outside the catching zone | .145 | .224 | +.079 ± .036 | 11 of 12 above |
| Start not under it: end under it | .296 | .270 | −.026 ± .057 | 4 of 12 above |
| Start not under it: end past it, on the other side | .001 | .012 | +.011 ± .010 | 11 of 12 above |
| Start not under it: end nearer, still short | .386 | .307 | −.079 ± .055 | 2 of 12 above |
| Start not under it: end no nearer | .317 | .411 | +.094 ± .057 | 11 of 12 above |
| Start within 8 px of the zone: end under it | .414 | .383 | −.031 ± .058 | 4 of 12 above |
| Start within 8 px: end past it | .001 | .025 | +.024 ± .022 | 11 of 12 above |
| Start within 8 px: end no nearer | .363 | .508 | +.145 ± .062 | 11 of 12 above |
| Start further off: end under it | .090 | .153 | +.063 ± .047 | 9 of 12 above |
| Start further off: end no nearer | .237 | .314 | +.077 ± .065 | 9 of 12 above |

- **Reading 5, the overshoot: rare.** A hold carries the paddle past the ball in .012 of the spans that start away from it (.025 from within 8 px).
- **The paddle cannot stay.** From under the ball it is out of the zone 4 ticks later in .224 of the spans, against .145.
- **One decision does less than four.** BD's four moves leave the paddle no nearer in .317 of the spans, BA's one held move in .411, although each of BD's moves goes toward the ball less often than BA's decision (R3).
- **Near the ball it is worst:** from within 8 px, half of BA's holds (.508) end no nearer.

### R6. The last 4 ticks

| Per ball | BD (12) | BA (12) | BA − BD (95%) | Seeds |
|---|---|---|---|---|
| Balls with the paddle under the landing point 4 ticks before | .637 | .539 | −.098 ± .085 | 2 of 12 above |
| Returned, of the balls it was under | .862 | .798 | −.064 ± .045 | 2 of 12 above |
| Returned, of the balls within 8 px of the zone | .561 | .523 | −.039 ± .096 | 4 of 12 above |
| Returned, of the balls further off | .097 | .166 | +.069 ± .040 | 10 of 12 above |

- **Reading 5, the late correction:** of the balls the paddle is under 4 ticks before the landing BA returns .798, BD .862 (−.064 ± .045).
- **Most of the loss is earlier:** BA's paddle is under the ball 4 ticks before the landing for .539 of the balls, BD's for .637.

### R7. The candidates repeat the move being held

Deciding ticks with a ball in play:

| Measure | BD (12) | BA (12) | BA − BD (95%) | Seeds |
|---|---|---|---|---|
| Of a transformer's 16 candidates: the share that repeat the move made on the tick before | .094 | .567 | +.472 ± .026 | 12 of 12 above |
| Of its 16 candidates: different values of dx | 9.1 | 5.6 | −3.5 ± 0.3 | 0 of 12 above |
| A decision goes the same way as the one before (BD: as the move 4 ticks before) | .577 | .774 | +.198 ± .012 | 12 of 12 above |
| Ticks on which the move is cut by a wall (all ticks) | .219 | .469 | +.250 ± .042 | 12 of 12 above |
| Size of the move, abs dx (1 is full speed) | .591 | .592 | +.001 ± .057 | 7 of 12 above |
| Move goes toward the ball, 5 to 16 ticks before a landing, paddle within 8 px of the zone | .518 | .493 | −.025 ± .059 | 5 of 12 above |
| The same, paddle further off | .608 | .686 | +.078 ± .063 | 8 of 12 above |

- **More than half of the candidates at a deciding tick are the old move:** .567 of a transformer's 16 repeat the dx just held, against .094 for BD, and they hold 5.6 different values of dx against 9.1.
- **The model barely tells a deciding tick from a held one:** on held ticks .638 of the candidates repeat the move, with 5.0 different values.
- **The cause is in the arm as built.** The candidates are sampled from the model's own move prediction, and its target on three ticks of four is the move just made.
- **So the paddle runs on:** a new decision goes the same way as the last .774 of the time, and its move is cut by a wall on .469 of ticks (BD: .219).
- **Neither agent slows down under the ball:** the move's size is .59 of full speed in both, about 10.6 px for a hold, against a catching zone of 14 px.
- **Near the ball neither agent knows which way to go:** from within 8 px of the zone the move goes toward the ball .518 (BD) and .493 (BA) of the time. BA's better decisions are the far ones (.686 against .608).

### R8. A free choice of dx does not bring the play back

The frozen 60k weights in the same courts, each transformer picking the best dx of an even grid of 16 by its own label:

| Balls returned | Its own candidates | The even grid | Grid − own (95%) | Seeds |
|---|---|---|---|---|
| BD | .712 | .694 | −.018 ± .020 | 5 of 12 above |
| BA | .614 | .608 | −.006 ± .031 | 7 of 12 above |
| BA − BD (95%) | −.098 ± .077 | −.086 ± .094 | +.012 ± .041 | 7 of 12 above |

| Measure | BD, own | BD, grid | BD, grid − own (95%) | BA, own | BA, grid | BA, grid − own (95%) |
|---|---|---|---|---|---|---|
| Move made goes toward the ball, the last 8 ticks | .595 | .599 | +.004 ± .019 | .657 | .680 | +.023 ± .035 |
| A decision goes the same way as the one before (BD: as the move 4 ticks before) | .577 | .599 | +.022 ± .011 | .774 | .635 | −.140 ± .022 |
| Ticks on which the move is cut by a wall | .219 | .214 | −.005 ± .022 | .469 | .356 | −.113 ± .037 |
| Spans of 4 ticks that start under the landing point and end outside the zone | .145 | .153 | +.008 ± .016 | .224 | .268 | +.044 ± .031 |
| Returned, of the balls the paddle was under 4 ticks before | .862 | .825 | −.037 ± .018 | .798 | .776 | −.022 ± .024 |

- **With no copies among its candidates BA returns as many balls as before:** .608 against .614 (−.006 ± .031).
- The grid does what it should to the moves: a decision follows the last one less (.635 from .774) and the walls cut fewer moves (.356 from .469).
- **BD gains nothing from the grid either** (−.018 ± .020), as vm-00116 expected from the label's order on it.
- **What the test cannot say:** these weights were learned in a life with the copies. A life with a free choice at each decision could have learned another label.

### R9. How the lives move, and the speed

| Arm, measure | 0–10k | 10–20k | 20–30k | 30–40k | 40–50k | 50–60k |
|---|---|---|---|---|---|---|
| BD, cosine between one move and the next | 0.23 | 0.21 | 0.23 | 0.23 | 0.23 | 0.20 |
| BA, cosine between one move and the next | 0.95 | 0.96 | 0.95 | 0.93 | 0.91 | 0.91 |
| BD, ticks on which the instinct overrules the model's own move | 0.41 | 0.48 | 0.48 | 0.48 | 0.48 | 0.47 |
| BA, ticks on which the instinct overrules the model's own move | 0.27 | 0.33 | 0.36 | 0.41 | 0.43 | 0.41 |
| BD, the move token's cross-entropy | 0.71 | 0.55 | 0.52 | 0.53 | 0.56 | 0.60 |
| BA, the move token's cross-entropy | 0.23 | 0.20 | 0.20 | 0.20 | 0.19 | 0.17 |

| Seeds | BD, ticks per second | BA, ticks per second | Cards (six lives at a time) |
|---|---|---|---|
| 1–6 | 5.58 | 9.81 | BD on the RTX 3090, BA on the RTX 4090 |
| 7–12 | 9.84 | 9.83 | both on the RTX 4090 |

- **Reading 6:** no cost and no saving, 9.83 against 9.84 ticks per second on the same card. The instinct still runs on held ticks.
- **Held moves** are .750 of body-ticks in every seed, as built.
- The move token's cross-entropy is a third of BD's (.17–.23 against .52–.71): on three ticks of four the move is a repeat.

## Analysis

- **Action repeat at k = 4 does not help in Breakout.** BA is below BD in every window of the life, by .065 ± .053 at 40–60k, with learning off and in the recordings.
- **The reason given for the arm holds, in the small.** A decision that moves the paddle four times as far is learned better: the label spreads twice as wide over the candidates, orders them correctly .688 of the time against .618, and the deciding move goes toward the ball more often. That is still far from an order that would track the ball (.5 is chance, 1 a perfect order).
- **The commitment costs more than the better decision gains.**
  - BD's four moves in 4 ticks are four chances: though each is right less often, the paddle ends no nearer in .317 of the spans, against .411 for BA's one decision.
  - A paddle that moves at .59 of full speed whatever the court cannot stay under a ball once its move is held: it leaves the zone in .224 of the holds that start there.
  - Near the ball the direction is a coin's toss in both agents (.49 and .52 from within 8 px), and BA's wrong toss lasts four ticks. So BA wins far balls and loses near ones, and there are many more near ones.
- **The overshoot named beforehand is not the cost:** a hold seldom carries the paddle past the ball (.012).
- **The arm as built also narrowed its own choice,** which the plan did not foresee: more than half of the candidates at a deciding tick are the move being held, so decisions follow one another and the paddle runs to the walls. With the frozen weights a free choice of dx removes that and returns no more balls (R8). So the copies are not what costs the play at 60k. Whether they shaped what was learned is not tested.
- **For the label, the finding of vm-00116 is now sharper:** where one move matters most, within 8 px of the catching zone, the move goes toward the ball half the time. Held moves did not change that.

## Conclusions

1. **Action repeat (k = 4) lowers the play in Breakout a little:** BA − BD = −.065 ± .053 hit share at 40–60k (2 of 12 seeds above), −.092 ± .100 with learning off, −.098 ± .077 in the recordings. It reaches .5 at 40–52k (median 43k), later than BD in 10 of 12 seeds.
2. **The label orders held moves better than one-tick moves:** .688 against .618 (+.070 ± .021, 12 of 12 seeds), and the deciding move goes toward the ball .657 of the time against .595. The plan named no outcome in which the order rises and the play falls.
3. **The loss goes with the commitment:** one decision in 4 ticks leaves the paddle no nearer to the ball more often than four (.411 against .317 of the spans), and the paddle leaves the catching zone more often when it was under the ball (.224 against .145). Passing the ball is rare (.012).
4. **A model that learns to predict held moves fills its candidates with them:** .567 of the 16 candidates at a deciding tick repeat the move being held (BD: .094). Picking from an even grid of dx instead returns no more balls (.608 against .614), for BD as well (.694 against .712).
5. **It costs no time and saves none:** 9.83 against 9.84 ticks per second.

## Next steps

To decide with the owner.

- **By the plan's rule the food world and k = 2 do not follow:** they were to follow a gain.
- **The arms left are those that teach the label what a move does** (reading 4 of the plan): an advantage-style target that widens the gap between moves, and labels for moves not made (task 00114, part A). The place to read them is the near range: the share of moves toward the ball from within 8 px of the catching zone, .52 in BD.
- **Not recommended now: a second action-repeat life with a free choice at each decision** (the move token trained on deciding ticks only). The grid test gives no sign that it would play better, and the cost of the commitment would stay.
- **For any arm in which moves repeat** (the food world's inertia, a later k): candidates sampled from the model's move prediction will repeat too. The recorder now measures it.

## Files

- `dev/lifetime.py` (`hold_move`), `dev/online.py` (`act_every`), `dev/eval_hunger.py` (`Life.act` holds as the life did), `dev/tests/test_act_every.py`: the change.
- `dev/analyze_misses.py`: `record` (the hold's scoring, the candidates, `--grid`), `read`, and one recording's measures `last_ticks`, `spans`, `moves`, `endgame`; `dev/tests/test_misses.py`.
- `dev/analyze_lives.py repeat`: the tables by seed; `dev/tests/test_lives.py`.
- `Makefile`: `runs`, `frozen`, `autopsy`, `autopsy-bd`, `autopsy-grid`, `read`, `table`.
- The tables as printed: `exp_00118/tables.md`, `exp_00118/autopsy/read.txt`, `exp_00118/autopsy_BD/read.txt`.
