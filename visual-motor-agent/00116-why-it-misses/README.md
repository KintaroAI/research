# vm-00116: Why the Trained Agent Misses — the Ring Control and a Miss Autopsy

**Date:** 2026-11-25
**Status:** Complete
**Source:** `dev/` (pending tag `exp/vm-00116`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00116/`
**Task plan:** [`../tasks/00129-why-it-misses.md`](../tasks/00129-why-it-misses.md)

## Goal

In Breakout stage 1 the default agent returns .65–.73 of the balls after 60k ticks, DreamerV3 .99 (vm-00108, vm-00110). A paddle that follows the ball's x returns every ball, so the game needs no foresight. Three new acting arms (vm-00113, vm-00114, vm-00115) have not said where the misses come from. This experiment measures it on checkpoints that exist, with no new learning code and no life.

## Method

- **Part A, the ring control** (`make check`, `make table`): BD's (vm-00103) 60k checkpoints, seeds 1–12, learning off (`ppo.lr = 0`), 6,000 ticks in new courts (seed 100 + s), acting on the ring as they lived and on full windows (`online.ring = 0`), both on one build. vm-00114 had read .765 on full windows against .644 on the ring, with its ring arm from older code and about 25 balls a seed.
- **Part B, the miss autopsy** (`make autopsy`, `make read`; `dev/analyze_misses.py`): an agent's frozen transformers play 16 new courts for 1,500 ticks a seed (seed 100 + s), 12 seeds, about 3,600 balls an agent. Every tick is recorded:
  - **from the court** (for scoring only; nothing of it reaches the model): where the ball will land and in how many ticks, by the court's own rules, and how far the paddle would still have to travel to be under it;
  - **from the agent:** each transformer's 16 candidate moves and their labels, each transformer's pick, the move made and whose it was;
  - **two more scorings of the label** each tick, before the ring takes the move: the same candidates with their dy and click set to nothing (in this world only dx moves the paddle), and an even grid of 16 dx from full left to full right.
- **The agents:** BD at 60k and at 30k, and RB at 60k (vm-00110: BD with seven times the replay), each as it lived; BD at 60k also on full windows.
- **The measures:**
  - **toward:** the share of ticks on which the move made goes toward the landing point (ticks with the paddle already under it left out);
  - **the label's order:** of the pairs of candidates that would leave the paddle different distances to travel, the share in which the label prefers the nearer one. One half is chance, 1 a perfect order;
  - **each ball over its last 12 ticks** (the paddle crosses its whole court in 11): how far the paddle had to travel, how far it did, and for a miss its kind.
- **The scoring is tested** (`tests/test_misses.py`): the landing point, the ticks to it and the hit test agree with what the world then does, under a random paddle and under a wobbling tracker.

## Readings

Set in the task plan before the results, with two notes on how they held up.

1. **The ring:** full − ring hit share, paired by seed; at least +.05 with its 95% interval above 0 would make a life with the ring off the next arm.
   - **The rule was badly made.** It counted all 6,000 ticks, but the first 2,000 are the very courts and ticks of the earlier tests, in which the difference was first noticed. The reading below is taken on the new ticks.
2. **Which arm follows the autopsy:** late starts or dithering point to action repeat, a transformer that cannot see the ball choosing the move to the council rule, the ring to the ring off.
   - **A fourth row was added after a first look at one seed** (88 balls), before the 12-seed tables: the label not ordering the candidates though it knows the court as a state points to an arm that teaches the label what a move does.

## Results

Part A ran 2026-11-25 21:19–22:12 on the RTX 4090; part B's recordings beside it, from about 21:30 to 22:25. No run failed.

### R1. The ring: no difference on new ticks

Hit share with learning off, 12 seeds, paired by seed (95% intervals):

| Ticks | Ring | Full windows | Full − ring | Seeds up, down |
|---|---|---|---|---|
| 1–2,000, seen before | .644 (310 balls) | .765 (296) | +.121 ± .067 | 10, 2 |
| 2,001–6,000, new | .732 (602 balls) | .763 (600) | +.032 ± .053 | 6, 6 |
| All 6,000 | .701 (912 balls) | .762 (896) | +.061 ± .041 | 9, 2 |
| The autopsy's recordings, 24,000 ticks a seed | .712 (3,630 balls) | .703 (3,627) | −.008 ± .015 | 5, 7 |

- **The build did not change the play:** each arm's first 2,000 ticks repeat the earlier test of the same weights in the same courts exactly, hit for hit, in 12 of 12 seeds (the ring: vm-00103's `frozen60k`; full windows: vm-00114's check).
- **So those 2,000 ticks are the old data,** and the difference in them is the one already seen. On the 4,000 new ticks it is +.032 ± .053, six seeds up and six down.
- **The autopsy's recordings,** four times the balls through another acting path (`eval_hunger.Life`), show none: −.008 ± .015.
- **The ring's .644 was low by chance:** .732 on the new ticks, .712 in the recordings. With about 25 balls a seed, a share is good to about ±.09 per seed.

### R2. The play

| | BD at 30k | BD at 60k | RB at 60k | BD at 60k, full windows |
|---|---|---|---|---|
| Balls | 4,213 | 3,635 | 3,576 | 3,634 |
| Returned | .38 | .71 | .75 | .70 |

Returned, by how far the paddle had to travel 12 ticks before the landing (px; 0 means it was already under the ball):

| | 0 | 0–8 | 8–16 | 16–24 | 24–32 | 32+ |
|---|---|---|---|---|---|---|
| BD at 30k | .72 | .41 | .23 | .09 | .02 | .00 |
| BD at 60k | .84 | .67 | .57 | .33 | .15 | .00 |
| RB at 60k | .85 | .75 | .65 | .52 | .19 | .00 |
| BD at 60k: balls | 1,670 | 1,196 | 545 | 169 | 34 | 21 |

- At 12 ticks every ball can still be reached (54 px of travel against a court of 48), yet the return falls steeply with the distance.
- BD at 60k loses 16% of the balls it was already under.
- By 12 ticks before the landing BD at 60k is under the ball for 46% of the balls and within 8 px for another 33%: most of what it returns it has drifted under during the flight.

### R3. The move and the label's order, by ticks to landing

BD at 60k, as lived:

| Ticks to landing | 1–2 | 3–4 | 5–8 | 9–16 | 17–32 | 33+ |
|---|---|---|---|---|---|---|
| Ticks recorded | 7,272 | 7,278 | 14,583 | 29,287 | 59,069 | 156,939 |
| Paddle not yet under the landing point | .32 | .36 | .44 | .54 | .70 | .77 |
| Move made goes toward it | .58 | .60 | .59 | .53 | .50 | .46 |
| Px nearer per tick (a full move is 4.5) | .36 | .63 | .71 | .63 | .51 | .14 |
| Label's order of the candidates | .60 | .62 | .62 | .58 | .54 | .53 |
| The same, pairs ending 4 px or more apart | .63 | .69 | .70 | .66 | .59 | .55 |
| Label's spread over the candidates (median) | .044 | .038 | .030 | .020 | .011 | .002 |

The three agents in the last 8 ticks (the bins 1–2, 3–4 and 5–8):

| | BD at 30k | BD at 60k | RB at 60k | BD at 60k, full windows |
|---|---|---|---|---|
| Move made goes toward the ball | .43–.46 | .58–.60 | .62–.66 | .59–.60 |
| Label's order of the candidates | .52 | .60–.62 | .63–.65 | .60–.63 |
| The same, pairs 4 px or more apart | .54–.55 | .63–.70 | .67–.72 | .63–.71 |

- **The move made goes toward the ball about six times in ten,** at best two in three (RB), even two ticks before the landing.
- **The label orders two candidates correctly about six times in ten,** seven in ten when they end 4 px or more apart.
- **Both rise with the play:** chance at 30k, .60 at 60k, a little more with seven times the replay. Within BD at 60k, a seed's share returned goes with its share of moves toward the ball (r = +.75) and with its label's order (r = +.66; 12 seeds).

### R4. The label knows the court as a state

The label of the move made (lower is better), BD at 60k, by how far the paddle still has to travel (px):

| Ticks to landing | 0 | 0–8 | 8–16 | 16–24 | 24–32 |
|---|---|---|---|---|---|
| 1–4 | .085 | .195 | .349 | .407 | .434 |
| 5–16 | .261 | .290 | .332 | .374 | .426 |
| 17+ | .450 | .452 | .455 | .461 | .469 |

- Within 16 ticks of the landing the label is clearly better with the paddle under the ball, and falls off with the distance. Further out it does not depend on it.
- So the model holds where the paddle is against where the ball will land. What it lacks is in R3: what a move does to that.

### R5. Better candidates would not help

In the last 8 ticks:

| | BD at 30k | BD at 60k | RB at 60k |
|---|---|---|---|
| Label's order, the sampled candidates | .52 | .60–.62 | .63–.65 |
| The same dx with no dy and no click | .53 | .61–.64 | .65–.66 |
| An even grid of 16 dx, no dy, no click | .52–.53 | .61–.64 | .63–.64 |
| The grid's best dx goes toward the ball | .53–.54 | .63–.68 | .70–.73 |
| Across the candidates, the label's correlation with the px left to travel | .05 | .18–.24 | .23–.27 |
| Its correlation with dy (absolute) | .27–.28 | .25–.26 | .36 |
| Its correlation with the click (absolute) | .33–.34 | .31–.33 | .42–.43 |

- **An ideal set of candidates gains little:** the best of an even grid of dx goes toward the ball .63–.68 of the time for BD, against .58–.60 for the move made.
- **Taking the do-nothing parts out of the move changes nothing:** the order is the same with dy and click set to nothing.
- **Across its candidates the label varies more with dy and with the click, which do nothing in this world, than with where dx puts the paddle.** More replay raises all three.

### R6. No transformer is to blame

- Each of the four makes about a quarter of the moves wherever the ball is (BD at 60k: .20–.33 in every quarter).
- Each orders its candidates as well with the ball in its own quarter as elsewhere (.61–.62 against .57–.61, the last 16 ticks).

### R7. The misses

Each miss over its last 12 ticks:

| | BD at 30k | BD at 60k | RB at 60k |
|---|---|---|---|
| Misses | 2,615 | 1,058 | 891 |
| Wrong way: under half of the moves toward the ball | .57 | .39 | .32 |
| Too little travel: toward, but the moves add up to less than the gap | .08 | .02 | .02 |
| Enough travel, not to the ball | .35 | .59 | .66 |
| Missed by under 2 px | .13 | .31 | .34 |
| Px the paddle had to travel, before a return and before a miss | 2.8, 14.4 | 2.9, 7.5 | 3.8, 8.3 |
| Px it travelled, before a return and before a miss | 29.9, 29.6 | 30.7, 30.9 | 31.7, 31.8 |

- **The misses are not late and not slow.** The paddle travels about 31 px in its last 12 ticks, before a return as before a miss, to close a gap of a few px.
- **It moves at more than half speed all the time,** a little more often toward the ball than away.

## Analysis

- **The ring is not it.** The difference vm-00114 read was the luck of 25 balls a seed. It looked confirmed here only while the ticks it was first seen on were counted again.
- **The horizon is not it, and neither is the choice among the transformers.** Hardly a miss comes from starting too late or moving too little, and no transformer orders the candidates better or worse than the others.
- **The candidates are not it** (as vm-00113 found by another road): a perfect spread of dx would move the paddle toward the ball two times in three.
- **What is left is the label itself: a good value of the state and a poor value of the move.** It knows the paddle should be under the ball (R4), and does not know which move takes it there (R3, R5).
- **A reason, not tested here:** one move changes the next state's value very little. From R4, 5 to 16 ticks before the landing the label changes by about .005 per px, so two moves ending 4 px apart differ by about .02. That is the size of the label's whole spread over its 16 candidates there (.020–.030), which also carries its variation with dy and the click. Only in the last 4 ticks is the slope larger (about .02 per px), and by then a wrong move costs the ball.
- **It is learned, slowly:** at chance at 30k, .60 at 60k, a little higher with seven times the replay (RB), and vm-00109's life returned .95 of the balls after a million ticks. vm-00115's policy-gradient actor learned from these same differences between candidates, which fits its staying at a random paddle's level.

## Conclusions

1. **Acting on the ring costs no play that can be measured:** +.032 ± .053 on 4,000 new ticks, −.008 ± .015 in the autopsy's recordings. vm-00114's side reading was noise.
2. **The trained agent's moves go toward the ball about six times in ten** (BD at 60k .58–.60 in the last 8 ticks, RB .62–.66), and its misses are a paddle that travels enough (about 31 px in 12 ticks) but not to the ball. Not lateness: 2% of the misses moved too little.
3. **The label knows the court as a state** (.085 with the paddle under the ball against .407 at 16–24 px, the last 4 ticks) **and barely orders the moves** (.60–.62; .63–.70 for moves ending 4 px or more apart).
4. **Neither better candidates nor another choice among the transformers would mend it:** an even grid of dx gets .63–.68 toward the ball, and the four transformers are alike.
5. **A re-test must leave out the ticks on which an effect was first seen.** Frozen tests repeat exactly, so the same weights, courts and seeds give the same first ticks.

## Next steps

- **The arm, to decide with the owner.** What has to grow is the label's dependence on the move:
  - **action repeat** (the owner, 2026-11-17; `ARCHITECTURE.md` section 9; to build): a move held for 4 ticks shifts the paddle up to 18 px, so what it does to the label is four times larger to learn, and a held move cannot dither;
  - **a label target that widens the gap between moves** (advantage learning);
  - **labels for moves not made** (task 00114, part A).
- **The same autopsy in the food world,** where the default agent beats DreamerV3: does the label order the candidates there? `analyze_misses.py` reads Breakout stage 1 only.
- **vm-00109's agent at a million ticks** as the high end of the scale (its court has bricks; the last ticks of a fall are below them).

## Files

- `dev/analyze_misses.py`: `ring` (R1), `record` and `read` (R2–R7).
- `dev/tests/test_misses.py`: the scoring against the world.
- `Makefile`: `check`, `table`, `autopsy`, `read`.
- The tables as printed: `exp_00116/ring.txt` and `exp_00116/autopsy/*.txt`.
