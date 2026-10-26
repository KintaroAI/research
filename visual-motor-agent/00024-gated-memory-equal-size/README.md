# vm-00024: Gated Memory at Equal Size

**Date:** 2026-10-26
**Status:** Complete
**Source:** `exp/vm-00024` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00024/`
**Task plan:** [`../tasks/00030-gated-memory-equal-size.md`](../tasks/00030-gated-memory-equal-size.md)

## Goal

vm-00022 compared memories that differed in size as well as content: keeping everything (the whole life) still behaved
best. Here every memory has the same size, a quarter of vm-00021's (25k steps), and is full for the second half of a
200k-tick life. When the memory cannot hold the whole life, what should it keep: the most recent steps, the surprising
ones, or the ones that led to relief and the surprising rest?

## Hypothesis

1. A memory of only the recent steps narrows the model to its latest habits (vm-00022's small random memory collapsed
   the policy).
2. A memory chosen by surprise, or by relief and surprise, keeps the rare events and predicts and behaves better than
   the recent steps, at the same size.
3. Chosen well, a quarter of the memory comes close to a memory of the whole life (vm-00021).

## Method

`configs/U_universal.json` (the default universal model, the 1-step instinct) with `online.replay = 25,000` steps
(3,125 chunks), 200k ticks, seeds 1–3, three gates. The long-term memory is a ring: when full, the oldest stored chunk is
overwritten, whatever the gate.

| Arm | Gate | Memory full from |
|---|---|---|
| U | every chunk: the last 25k steps | 26k |
| S | `store_surprise = 0.25`: the most surprising quarter of chunks (task 00028), reaching back about 100k steps | 90–94k |
| RS | `store_relief = 1`, `store_surprise = 0.25`: every chunk that led to relief, and the surprising quarter of the rest | 57–79k |

The reference: vm-00021 seeds 1–3, the same model with a memory of 100k steps (4× larger, the whole life up to 100k).
Five lifetimes ran on the 3090 (47 ticks/s each), four on the 4090 (32, then 93 when its other jobs ended), 01:56–03:08.
Measures: online clicks and time at the hunger cap; the frozen tests at 100k and 200k (eval_self, the forage test with
the instinct on 32 worlds, the policy alone at 200k); the click test on real windows (task 00027) at 100k and 200k.

## Log

### 2026-10-26
- **Ran** the 9 lifetimes 01:56–03:08; analysis until 03:29.

## Results

### R1. Online

Food / poison clicks per 1,000 steps, and the share of time at the hunger cap:

| Run | 40–60k | 80–100k | 180–200k | at the cap: 40–60k, 80–100k, 140–160k, 180–200k | relief share of memory at 100k, 200k |
|---|---|---|---|---|---|
| U s1 | 13.2 / 43.5 | 10.4 / 24.1 | 13.7 / 49.4 | 0.33, 0.19, 0.26, 0.39 | 0.18, 0.23 |
| U s2 | 1.9 / 2.3 | 2.2 / 2.8 | 0.0 / 0.2 | 0.10, 0.06, 0.21, **1.00** | 0.04, 0.01 |
| U s3 | 17.6 / 17.0 | 2.4 / 75.2 | 0.0 / 84.4 | 0.07, **0.90, 1.00, 1.00** | 0.03, 0.00 |
| S s1 | 7.6 / 11.7 | 7.7 / 10.6 | 9.0 / 9.1 | 0.13, 0.08, 0.10, 0.07 | 0.32, 0.30 |
| S s2 | 10.5 / 9.7 | 11.8 / 14.2 | 12.9 / 13.3 | 0.12, 0.10, 0.05, 0.12 | 0.39, 0.37 |
| S s3 | 5.0 / 16.4 | 4.0 / 13.8 | 3.3 / 13.8 | 0.26, 0.24, 0.25, 0.52 | 0.19, 0.15 |
| RS s1 | 20.4 / 24.6 | 20.1 / 17.4 | 29.7 / 15.6 | 0.14, 0.09, 0.17, 0.06 | 0.72, 0.79 |
| RS s2 | 5.4 / 6.3 | 3.9 / 2.2 | 2.5 / 2.7 | 0.27, 0.21, 0.62, 0.44 | 0.27, 0.10 |
| RS s3 | 13.7 / 19.2 | 13.9 / 22.9 | 12.2 / 17.4 | 0.12, 0.22, 0.10, 0.15 | 0.56, 0.50 |
| whole life s1 (vm-00021) | 6.5 / 27.3 | 9.6 / 5.2 | 7.5 / 4.8 | 0.29, 0.04, 0.04, 0.03 | 0.14, 0.13 |
| whole life s2 | 2.9 / 5.8 | 3.1 / 3.3 | 1.8 / 6.6 | 0.09, 0.03, 0.05, 0.37 | 0.06, 0.05 |
| whole life s3 | 8.7 / 11.1 | 7.3 / 7.8 | 2.0 / 1.4 | 0.12, 0.12, 0.07, 0.66 | 0.16, 0.08 |

U s3's policy collapsed to one move (the move token's cross-entropy 0.00 at 180–200k, against 0.08–0.19 in the other
runs of this experiment).

### R2. Frozen at 200k

| Run | eval_self success | forage with the instinct: food / poison per 1k (at the cap) | policy alone | click test: food right | poison right | nothing flat |
|---|---|---|---|---|---|---|
| U s1 | 1.000 | 16.8 / 44.3 (0.27) | 14.9 / 51.4 (0.36) | 0.74 | 0.66 | 0.90 |
| U s2 | 0.188 | 1.2 / 0.6 (0.07) | 0.8 / 1.0 (0.11) | 0.46 | 0.26 | 0.92 |
| U s3 | 0.820 | 5.2 / 14.9 (0.31) | 5.4 / 17.0 (0.42) | 0.18 | 0.58 | 0.40 |
| S s1 | 0.938 | 14.3 / 10.3 (0.06) | 10.7 / 9.6 (0.07) | 0.65 | 0.84 | 0.69 |
| S s2 | 0.945 | 14.8 / 10.0 (0.05) | 13.7 / 15.2 (0.09) | 0.71 | 0.87 | 0.87 |
| S s3 | 0.852 | 8.8 / 9.2 (0.08) | 7.2 / 8.2 (0.10) | 0.78 | 0.65 | 0.86 |
| RS s1 | 0.992 | 36.2 / 17.3 (0.07) | 30.6 / 25.6 (0.13) | 0.84 | 0.76 | 0.92 |
| RS s2 | 0.719 | 5.7 / 4.2 (0.07) | 2.8 / 2.6 (0.13) | 0.77 | 0.67 | 0.63 |
| RS s3 | 0.984 | 17.1 / 7.7 (0.04) | 12.6 / 8.9 (0.07) | 0.69 | 0.80 | 0.82 |
| whole life s1 | 0.961 | 10.8 / 3.7 (0.02) | 8.1 / 6.0 (0.04) | 0.60 | 0.66 | 0.95 |
| whole life s2 | 0.820 | 3.4 / 5.4 (0.06) | 3.4 / 4.6 (0.05) | 0.45 | 0.78 | 0.72 |
| whole life s3 | 0.977 | 12.7 / 3.7 (0.02) | 8.7 / 8.2 (0.09) | 0.81 | 0.79 | 0.83 |

(The click test, task 00027: the share of felt food clicks predicted to relieve hunger, of felt poison clicks predicted
to raise it, and of clicks on nothing predicted flat, in the agent's real 8-step windows.) At 100k the click test
ranks the arms the same way except U, which is still close to the others (food right 0.38–0.70, poison 0.53–0.80).

## Analysis

- **The recent steps alone are the worst memory** (hypothesis 1, supported): two of three U lifetimes ended stuck at
  the hunger cap (s2 from about 180k, never clicking; s3 from about 80k, clicking poison 84 times per 1,000 steps with a
  single move), and the third clicked poison 44–49 times per 1,000 steps. Its memory held almost no relief (0.00–0.23 of
  chunks). Its prediction decays with it: at 200k U s2 and s3 predict poison right in 0.26 and 0.58 of the cases, food
  in 0.46 and 0.18; U s3 even predicts relief for clicks on nothing. Once the agent stops eating, a recent-steps memory
  holds nothing but the trap, and the model learns the trap.
- **A memory chosen by salience keeps the model working at a quarter of the size** (hypothesis 2, supported): no S or
  RS model collapsed frozen (eval_self success 0.72–0.99, at the cap 0.04–0.08 of the time with the instinct), and their
  memories kept more relief chunks than U's in every seed (0.10–0.79 of chunks at 200k against 0.00–0.23). Their click predictions are as good as the whole-life memory's or
  better: food right 0.65–0.84 (whole life 0.45–0.81), poison 0.65–0.87 (0.66–0.79).
- **S is the most even, RS eats the most:** none of S's three lifetimes got stuck (at the cap 0.05–0.26 of the time through 160k), and they clicked both kinds
  9–15 times per 1,000 steps frozen; S s3 drifted toward the cap online at the end (0.52) but not frozen (0.08). RS
  kept 0.50–0.79 relief chunks in seeds 1 and 3 and foraged best of all (food 17–36 per 1,000 frozen), but RS s2 lost its
  foraging after 140k: once it ate less, the relief gate found less relief to keep (its relief share fell from 0.27 to
  0.10), and the memory did not hold it on course. A relief gate amplifies whatever the agent is doing.
- **The whole life is still the cautious one** (hypothesis 3, partly): with 4× the memory the agent clicks poison
  least frozen (3.7–5.4 per 1,000 against 9–17 for S and RS) and eats less; online it too drifted to the cap late in
  two seeds (0.37 and 0.66 at 180–200k). The salience-chosen quarter predicts as well and forages more, but clicks
  poison more.
- **Online and frozen differ for a trapped agent:** U s3 is at the cap all the time online, 0.31 frozen in fresh worlds.
  The trap is partly the lifetime's own state (its hunger and the memory it keeps re-learning), not only the weights.

## Conclusions

1. **When the memory cannot hold the whole life, what it keeps decides whether the agent survives.** The most recent
   steps lead to collapse (two of three lifetimes trapped at the hunger cap); the model's own surprise, with or without
   relief, keeps every lifetime working at the same size.
2. **The model's surprise alone is the steadiest rule;** storing every relief chunk on top raises foraging when the agent
   eats but cannot pull it out once it stops.
3. **A salience-chosen quarter predicts as well as the whole life,** but the whole life is still more cautious about
   poison: more memory still helps behaviour, the right content helps prediction.

## Next Steps

- A long life (500k+ ticks) with a fixed memory, where even the whole-life arm must choose: surprise against the most
  recent steps at the size of vm-00021's memory.
- A guard against traps that applies to every token: an entropy floor on the predicted distributions, or keeping a
  share of the memory for the oldest chunks.
