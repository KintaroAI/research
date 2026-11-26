# Task 00130: Two Worlds, One Set of Weights (experiment vm-00117)

**Date:** 2026-11-25
**Status:** Complete (2026-11-26; pending tag `exp/vm-00117`): transfer is one-sided, the old world's play is forgotten within 10k ticks, and the follow-up by the rule below is rehearsal; the write-up is `00117-two-worlds/README.md`. The owner, 2026-11-25, to the plan (the ring check and two worlds tonight): "yes lets do this".

## Context

- **`ARCHITECTURE.md` section 13, challenge 1:** "universal" so far means the same code, not the same mind. "Can one life pass through circles, text and sound and come back without having forgotten?" Its first step, swapping worlds within a life, has not been run.
- **What exists:** the default agent's two configs, `configs/U_agent.json` (the food world) and `configs/U_breakout.json` (Breakout stage 1), have the same `model`, `online` and `ppo` blocks and differ only in `env`. So the weights a life ended with in one world load in the other (`lifetime.py --init`), and nothing in the model has to change. Both first lives exist with 12 seeds: TT (vm-00101) and BD (vm-00103).
- **The nearest earlier results:**
  - vm-00095: inertia switched on in mid-life. The model adapts with no change to itself, food returns to its earlier level in a median of 7k ticks, and it settles at half of its food without inertia.
  - `llm-fundamentals` 00014: a dense GPT forgets a first task completely when trained on a second. 00020: a forgotten fact is relearned in about half the showings of a new one.

## The design

Three stages, every one the default agent unchanged (`lifetime.py`, no new code).

| Stage | Arm | World | Starts from | Ticks | Against |
|---|---|---|---|---|---|
| 1 (done) | TT, BD | the food world; Breakout | a random model | 60k | |
| 2 | **TB** | Breakout | TT's 60k weights | 60k | BD, by seed |
| 2 | **BF** | the food world | BD's 60k weights | 60k | TT, by seed |
| 3 | **TBF** | the food world again | TB's 60k weights | 30k | TT's first 30k, by seed |
| 3 | **BFB** | Breakout again | BF's 60k weights | 30k | BD's first 30k, by seed |

- **The old world, learning off:** TB's checkpoints at 10k, 30k and 60k in the food world (2,000 ticks, seed 100 + s, as vm-00114's `check/TT_ring`, which is the "before"); BF's in Breakout (6,000 ticks, seed 100 + s, as vm-00116's ring arm, which is the "before").
- **The floor:** the untrained default model, learning off, in each world: what "nothing left" scores, with the instinct choosing on an untrained label.
- **Seeds 1–12.** The second and third lives use the first life's own seed, so a court or field begins as the from-scratch life's did.

### What a life started from weights is, and is not

- `--init` loads the weights only. Memory starts empty and the optimizer new.
- So the old world is never replayed: this is learning one world after another without rehearsal, the harsh case.
- A real single life would still hold the old world's chunks in its memory and replay them among the new. That needs a change of world inside one process (`online.env_sched` switches only the settings of one world) and is the follow-up if the forgetting is large.

## Readings, set before the results (12 seeds, paired by seed)

1. **Into Breakout:** TB − BD hit share at 10–30k and at 40–60k; the first 5k window at .5 (BD: 32–55k, median 40.5k). No direction predicted: what the food world taught may help (seeing, the label's machinery) or hinder (its habits of moving).
2. **Into the food world:** BF − TT food and poison per 1k at 10–30k and at 40–60k.
3. **Forgetting:** the old world's frozen score after 10k, 30k and 60k ticks of the new world, as the share kept: (after − floor) ÷ (before − floor). Food per 1k and poison per 1k for TB, hit share for BF.
   - below one half kept at 60k: the forgetting is large. The follow-up is rehearsal (one process, one memory across both worlds), then the flexible layer (`ARCHITECTURE.md` section 13: "adaptation that does not erase").
   - one half or more kept: the weights hold two worlds without help.
4. **Savings:** the third life's first 30k ticks against the first life's first 30k. TBF against TT: food per 1k at 0–10k, 10–20k and 20–30k. BFB against BD: hit share at 10–30k. Ticks to half of the first life's 50–60k level.
5. **The cost of a past:** readings 1 and 2 at 40–60k. Does a model that already holds one world end lower in the next?

## Runs

- **Cards:** seeds 1–6 on the RTX 3090 and 7–12 on the RTX 4090 for every arm, BD's split in vm-00103 (TT ran with all 12 on the 3090).
- **The 4090's chain** (from about 21:20, when vm-00115's Breakout lives end): vm-00116's ring control, then TB, BF, the old-world tests and the floor, then TBF and BFB, seeds 7–12.
- **The 3090's chain** (from about 22:40, when vm-00115's food lives end): the same for seeds 1–6, without the ring control.
- **Expected:** the 4090 done at about 04:10, the 3090 at about 09:00.
- **Checked before:** 300 ticks of TB seed 1 and BF seed 1 run and learn (the smoke test: the weights load in the other world, the ring and the label on).

## Code

- **`00117-two-worlds/Makefile`:** `runs`, `frozen`, `floor`, `back`. The frozen runs' copies of the weights are removed after each batch.
- **The tables** (to write with the results): the paired differences by 10k for readings 1, 2 and 4; the share kept for reading 3.
- No change to the model, the world or `lifetime.py`.

## Not here

- A change of world inside one life with one memory (rehearsal).
- The text world, the two-senses world and the car: their configs' models differ from the default agent's (the text world's bins, patches and frame; the two senses' frame; the car's older settings), so today's weights do not load there as they are.

## Implementation

- [x] Smoke test (TB and BF, seed 1, 300 ticks)
- [x] `00117-two-worlds/` Makefile and README; both chains queued
- [x] The tables (`dev/analyze_lives.py two-worlds`, `make table`)
- [x] The write-up

## Result (2026-11-26)

The RTX 4090's chain ran 22:12–04:20, the RTX 3090's 22:38–09:16. No run failed. Details and tables: `00117-two-worlds/README.md`.

1. **Into Breakout:** TB − BD = +.145 ± .031 hit share at 10–30k (12 of 12 seeds above) and +.129 ± .054 at 40–60k; .5 is reached 14.8k ± 4.5k ticks earlier (TB 21–33k, median 26k).
2. **Into the food world:** BF − TT = −29.1 ± 3.0 food per 1k at 10–30k and −21.5 ± 4.3 at 40–60k (0 of 12 above); poison +5.9 ± 1.0 and +3.7 ± 1.1. BF is about 30k ticks behind and still rising at 60k.
3. **Forgetting:** the share kept after 10k, 30k and 60k ticks of the new world is .12, .06, .02 of the food, −.03, −.05, .00 of the poison avoided and −.02, −.04, −.06 of the hit share. Below one half at 60k: by the rule the forgetting is large and **the follow-up is rehearsal,** then the flexible layer.
4. **Savings:**
   - TBF − TT food: +1.8 ± 1.3 at 0–10k, −2.7 ± 5.1 at 10–20k, −10.1 ± 5.1 at 20–30k; half of TT's final level at 16–29k (median 22k) against 17–23k (median 18.5k). No savings against a new model.
   - BFB − BD = +.140 ± .044 hit share at 10–30k; half of BD's final level 9.7k ± 4.1k ticks earlier.
   - Not set beforehand, the third life against the other order's second life: TBF − BF = +22.7 ± 3.4 food per 1k at 10–30k (12 of 12 above); BFB − TB = −.006 ± .029. The earlier food-world life counts; no trace of the earlier Breakout life shows.
5. **The cost of a past:** none in Breakout (TB ends above BD). In the food world BF is 21.5 ± 4.3 below TT at 40–60k and still rising, so 60k ticks do not settle it.

**Also seen, not set beforehand:** the way of moving comes along in both directions (TB enters Breakout with smoothness .49, BF the food world with .10 against a new model's .18), and the old world's frame is still predicted after its play is gone (Breakout's patches 1.15 nats after 60k ticks of the food world; .66 before, 3.49 untrained).
