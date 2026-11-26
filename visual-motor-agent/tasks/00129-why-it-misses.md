# Task 00129: Why the Trained Agent Misses — the Ring Control and a Miss Autopsy (experiment vm-00116)

**Date:** 2026-11-25
**Status:** Complete, pending tag (both parts ran 2026-11-25; the write-up is `00116-why-it-misses/README.md`). The choice of the lived arm is open, with the owner. The owner, 2026-11-25, to the plan (the ring check and two worlds tonight, the autopsy next, then one lived arm chosen by what they show): "yes lets do this".

## Context

- **The acting rule so far** (task 00123): in Breakout stage 1 DreamerV3 returns .99 of the balls at 50–60k against the default agent's .65 (vm-00108). More replay speeds the start and levels off at .73 (vm-00110), not because the candidates narrow (vm-00113). Looking ahead in the model's dream hurts (vm-00114). A policy-gradient actor stays at a random paddle's level (vm-00115, six seeds, write-up pending).
- **Three new arms have not said where the misses come from.** Before a fourth, measure it on checkpoints that exist.
- **The game needs no foresight.** The tracker walker (`breakout_walkers.py`: the paddle moves toward the ball's x) returns every ball (hit share 1.0, vm-00103): the paddle moves up to 4.5 px a tick, the ball 1.5. So the question is not the horizon of the game but why the label and the instinct do not arrive at tracking in 60k ticks.
- **A side reading that was not a clean control.** vm-00114's check read BD's 60k weights at .765 on full windows (`online.ring = 0`) against .644 on the ring (+.121 ± .067, 10 of 12 seeds). But its ring arm was vm-00103's `frozen60k`, run on older code, and each seed's share rests on about 25 balls (2,000 ticks). In the food world, where both arms ran on one build, ring and full windows are equal (69.8 against 70.0 food).
- **Why the ring could matter:** acting on the ring, a cached token keeps what it saw when it was new, so the newest step reads keys and values computed with more context than any training window gives them (`ARCHITECTURE.md` section 9: "the reuse is not the same computation"). Breakout needs the ball's motion from consecutive frames; the food world's scene barely moves.

## Part A: the ring control

- **Arms:** BD's (vm-00103) 60k checkpoints, seeds 1–12, learning off (`ppo.lr = 0`), new courts (seed 100 + s), 6,000 ticks (about 75 balls a seed):
  - **ring:** as it lived (`online.ring = 1`);
  - **full:** full windows (`online.ring = 0`).
- Both on today's build, on the RTX 4090, 6 at a time.

### Readings, set before the results

1. **The build:** the ring arm's first 2,000 ticks against vm-00103's `frozen60k` (the same weights, courts and ticks). Equal counts mean neither the build nor the card changed the play. Unequal counts can come from either, and say that vm-00114's side reading compared two different plays.
2. **Full − ring hit share** over 6,000 ticks, paired by seed, 12 seeds:
   - at least +.05 with its 95% interval above 0: acting on the ring costs play in Breakout. A life with the ring off (BD with `online.ring = 0`) becomes an arm of the next lived experiment; training that matches the ring (a warm-up without loss at the start of each window) a later one;
   - the interval includes 0: vm-00114's side reading was the noise of 25 balls a seed, or the build. The ring stays.
3. **Descriptive:** the ring's share with three times the balls, against vm-00103's .644.

## Part B: the miss autopsy

A frozen life in new courts, recording every tick. The court's state is read for scoring only; nothing reaches the model.

- **From the world:** the ball's position and velocity, the paddle's position. From them, by the court's physics (walls, top): the landing point and the ticks to landing.
- **From the agent:** each transformer's 16 candidate moves and their predicted labels (`m.cand`, `m.pred`, as `eval_candidates.py` reads them), the move made and the transformer whose move it was.

### Measures, by ticks to landing (1–2, 3–4, 5–8, 9–16, 17–32, 33+)

1. **Toward:** the share of ticks on which the move made goes toward the landing point (ticks with the paddle already under it left out), and the px it comes nearer per tick (a full move is 4.5).
2. **Order:** per transformer, of the pairs among its 16 candidates that would leave the paddle different distances to travel, the share in which the label prefers the nearer one (.5 is chance, 1 a perfect order); also for pairs that end 4 px or more apart, and for the transformer whose move was made. Does the label order the moves correctly, and from how far out?
3. **Regret:** the distance left by the move made minus that of the best of all the candidates.
4. **Who:** which transformer's move was made, by the quarter the ball is in, and each transformer's order with the ball in its own quarter and elsewhere.
5. **What the label knows of the court:** the label of the move made by how far the paddle still has to travel, near landing and far from it. A label that is better with the paddle under the ball knows the court as a state; the order (measure 2) says whether it knows what a move does to it.

### Per ball

Each ball is read over its last 12 ticks (the paddle crosses its whole court in 11, so 12 ticks out every ball can still be reached).

- **Returned, by how far the paddle had to travel** 12 ticks before the landing (0, 0–8, 8–16, 16–24, 24–32, 32+ px).
- **Each miss classed:** wrong way (under half of its moves toward the ball), too little travel (toward, but the moves add up to less than the gap), enough travel but not to the ball (dithering).

### Which lived arm follows, set before the results

| The autopsy shows | The arm |
|---|---|
| Mostly too little travel, the label ordering right only a few ticks before landing | Action repeat (the owner, 2026-11-17; `online.act_every`, to build): a held move makes the candidates differ sooner |
| Mostly dithering | The same arm: commitment |
| The label does not order the candidates even near landing, though it knows the court as a state | Not a choice among existing arms: the label's dependence on the move is the limit, and the arm is one that teaches it (task 00114 part A, labels for moves not made; or the value read on the imagined next frame). To design with the owner |
| Missed moves made by a transformer that does not hold the ball in its quarter, another transformer's label right | The choice among transformers: `online.piece_rule = council` (exists; vm-00063 found a third of gain's poison) |
| Part A positive | The ring off |

One arm, 12 seeds in Breakout stage 1 against BD; whatever wins is then run in the food world (vm-00112: test any acting rule in both worlds).

- **Checkpoints:** BD and RB (vm-00110) at 60k, seeds 1–12; BD at 30k for the change over the life.
- **Acting:** on the ring or on full windows, by part A's result.

## Results (2026-11-25; the write-up has the tables)

- **Part A: no ring effect.** Over all 6,000 ticks full − ring is +.061 ± .041, but each arm's first 2,000 ticks repeat the earlier tests exactly (12 of 12 seeds), so they are the data the difference was first seen in. On the 4,000 new ticks: +.032 ± .053, six seeds up and six down. In the autopsy's recordings (3,630 balls an arm): −.008 ± .015. Reading 2's rule should have left the seen ticks out; `analyze_misses.py ring` now prints them apart. A life with the ring off was queued on the first reading and withdrawn before it started.
- **Part B:** BD at 60k returns .71 of the balls (BD at 30k .38, RB at 60k .75). In the last 8 ticks its move goes toward the ball .58–.60 of the time and its label orders two candidates correctly .60–.62 of the time (.63–.70 when they end 4 px or more apart). An even grid of dx without dy or click orders no better; its best dx goes toward the ball .63–.68. The label does know the court as a state (.085 with the paddle under the ball, .407 at 16–24 px, the last 4 ticks). Of 1,058 misses 39% went the wrong way, 59% travelled enough but not to the ball, 2% too little; the paddle travels about 31 px in the last 12 ticks. No transformer stands out.
- **By the table above:** not late starts, not the choice among transformers, not the ring. Dithering, and under it the label's weak dependence on the move: action repeat, or an arm that teaches the label what a move does. To decide with the owner.

## Code

- **`dev/analyze_misses.py`:** `ring DIR` (part A's table: the share by mode, the paired difference, the build check; tested on vm-00114's numbers, which it reproduces); `record OUT.npz RUN` (part B: `eval_hunger.Life` plays 16 new courts, the landing point and the ticks to it from the court's state, the candidates and labels from each transformer) and `read` (the tables).
- **`dev/tests/test_misses.py`** (3 tests, CPU): the landing point, the ticks to it and the hit test computed from the court's state agree with what the Breakout world then does, under a random paddle and a wobbling tracker; the order measure on a hand-made case.
- **`00116-why-it-misses/Makefile`:** `check`, `table`, `autopsy`, `read`. The frozen runs' copies of the weights are removed after each batch (they are the checkpoints tested).
- No change to the model, the world or `lifetime.py`.

## Not here

- Any new learning rule, and any life: this experiment only measures.
- RB under full windows (cheap to add if part A is positive).

## Implementation

- [x] `analyze_misses.py ring`, tested on vm-00114's check
- [x] `00116-why-it-misses/` Makefile and README; part A queued
- [x] Part A's table and reading (no ring effect on new ticks)
- [x] `analyze_misses.py record`, `read`, `tests/test_misses.py`; the autopsy's recordings started (BD at 60k and 30k, RB at 60k, as lived)
- [x] The autopsy's tables and reading (also on full windows, and with the label scored on dx alone)
- [x] The write-up
- [ ] The choice of the lived arm with the owner; commit and tag on the owner's word
