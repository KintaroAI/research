# vm-00021: A Longer Lifetime for the Universal Model

**Date:** 2026-10-25
**Status:** Complete
**Source:** `exp/vm-00021` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00021/`
**Task plan:** [`../tasks/00025-longer-lifetime.md`](../tasks/00025-longer-lifetime.md)

## Goal

vm-00019/20's universal model learned what a click does in only one seed in three by 60k ticks, and the instinct helped
only there. Here the default universal model lives 200k ticks, six seeds.

## Hypothesis

1. Given time, every seed learns the three click outcomes (food −0.30, poison +0.50, nothing 0).
2. The instinct then lowers poison clicks in every seed.
3. The policy learns the instinct's choices: the policy alone improves too.
4. Exploration holds (the instinct keeps the model clicking).

## Method

`configs/U_universal.json` (gray, checkerboard poison +0.5, 8 moving dots, slow hunger, continual torus; the universal
model, 8×8 patches, 32 bins, gray; the instinct with 16 candidates; no babbling, no command; memory U), 200k ticks,
seeds 1–6 (1–3 on the RTX 4090, 4–6 on the RTX 3090). Measures in task 00025.

## Log

### 2026-10-25
- **Launched** the 6 lifetimes at 19:49: 77 ticks/s each on the RTX 4090, 47 on the RTX 3090.
- **All 6 finished** by 21:02 (seeds 1–3 at about 20:35); analysis 21:02–21:50 (the new library, task 00026).

## Results

### R1. The click test (placed one-step windows, hunger 0.3): click minus stay (world: food −0.30, poison +0.50, nothing 0)

| Seed | 50k | 100k | 200k |
|---|---|---|---|
| 1 | −0.21 / +0.09 / +0.12 | −0.25 / +0.01 / +0.01 | **−0.28** / −0.00 / −0.00 |
| 2 | −0.04 / −0.05 / −0.06 | 0.00 / 0.00 / 0.00 | −0.06 / +0.02 / 0.00 |
| 3 | −0.16 / 0.00 / 0.00 | −0.11 / +0.02 / +0.01 | **−0.27 / +0.16** / +0.13 |
| 4 | −0.27 / +0.03 / −0.20 | +0.01 / +0.20 / +0.02 | −0.02 / +0.02 / −0.02 |
| 5 | −0.50 / −0.54 / −0.61 | −0.43 / −0.45 / −0.62 | −0.15 / +0.02 / +0.02 |
| 6 | −0.07 / −0.02 / 0.00 | −0.15 / +0.01 / +0.02 | **−0.29** / −0.03 / −0.01 |

(food / poison / nothing)

### R2. Behaviour: frozen forage test at 200k (food clicks / poison clicks per 1,000 steps)

| Seed | with the instinct | its policy alone | episodic test (instinct) |
|---|---|---|---|
| 1 | 10.8 / **3.7** | 8.1 / 6.0 | 0.96 |
| 2 | 3.4 / 5.4 | 3.4 / 4.6 | 0.82 |
| 3 | 12.7 / **3.7** | 8.7 / 8.2 | 0.98 |
| 4 | 23.5 / **18.8** | 23.6 / 47.3 | 0.99 |
| 5 | 14.0 / **26.4** | 14.4 / 43.3 | 0.98 |
| 6 | 4.9 / **3.7** | 3.5 / 5.1 | 0.74 |

At 100k the same pattern (seed 4: 26.7 / 25.8 with the instinct, 31.0 / 51.9 without).

### R3. Online (food / poison clicks per 1k, hunger at the cap)

40–60k: seeds 1–6 6.5/27.3, 2.9/5.8, 8.7/11.1, 25.8/30.4, 2.7/28.1, 2.5/6.4. 190–200k: 7.5/5.9, 2.1/1.9,
**0.0/0.0 (at the cap 100% of the steps)**, 19.5/21.4, 6.2/35.8, 2.3/16.4. The instinct overrode the policy's first sample
on 1–10% of ticks.

### R4. Tokens at 200k

All patch tokens: cursor 0.95–0.97, direction 0.88–0.91; the last patch's output (the policy's input): direction 0.23–0.39.

## Analysis

- **Food is learned with time, poison is not, in the click test** (hypothesis 1, partly): by 200k three seeds of six
  predict the food click's relief (−0.27 to −0.29 of −0.30), one partly, two not at all. Poison stays near 0 in five;
  where it appeared (seed 4: +0.20 at 100k–150k; vm-00020 seed 3: +0.41 at 40k) it faded again. One seed (5) spent
  50k–150k predicting relief for every click, background included (−0.6), before unlearning it.
- **Yet the instinct does avoid poison** (hypothesis 2, supported in five seeds of six): at 200k it cuts the frozen
  agent's poison clicks against its own policy (seed 4: 47.3 → 18.8, seed 5: 43.3 → 26.4, seeds 1, 3, 6: 6.0 → 3.7,
  8.2 → 3.7, 5.1 → 3.7) and raises food clicks in three. So inside real windows, with the last steps' history, the model
  does predict that a click on the checkerboard costs hunger; the click test's placed one-step windows (no history,
  hunger fixed at 0.3, the cursor centred on a disc) under-measure it. The click test needs real context.
- **The policy does not take the instinct's choices over** (hypothesis 3, not supported): without the instinct the
  policy alone still clicks poison as often as before (seed 4: 47.3 per 1k at 200k). The instinct overrides only 1–10%
  of ticks, so its choices are a small share of the moves the policy imitates.
- **Exploration mostly holds, with a trap** (hypothesis 4, partly): five seeds keep clicking; seed 3 stopped entirely
  for its last 10k ticks with hunger at the cap 100% of the time, although its frozen checkpoint still clicks from lower
  hunger (forage test: 12.7 food clicks per 1k). Hunger 1 is rare in its experience, its policy does not click there,
  and without a click hunger never falls: an absorbing state.
- **The scene stays held** over 200k (cursor 0.95–0.97).

## Conclusions

1. **The instinct, acting on the model's own predictions, lowers poison clicks in five lifetimes of six**, by up to
   60% against the same model's policy alone, without reward: the model's in-context prediction of a click's
   consequence is good enough to act on.
2. **The placed-state click test under-measures that prediction**; it should use real windows with history.
3. **The policy does not absorb the instinct's choices**, and one lifetime fell into a trap at the hunger cap.

## Follow-up: the click test on real windows (task 00027, 2026-10-25)

The frozen agent lives as in the forage test; every chosen move is tried with and without a click on copies of the
world. At 200k, four seeds of six (1, 2, 3, 6) predict that a poison click raises hunger in 66–79% of the felt cases
(seed 4: 46%, seed 5: 18%), where the placed test showed −0.03 to +0.16. The sign is mostly right but the size is
5–30% of the true +0.47. The frames use two steps of history (the patch error halves from one step to two, and does
not fall further at four or eight). Details: [`../tasks/00027-click-test-real-windows.md`](../tasks/00027-click-test-real-windows.md).

## Next Steps

- ~~The click test on real windows~~ (done: the follow-up above).
- A general remedy for the cap trap (an entropy floor on the move distribution, or the instinct also comparing "click"
  against "no click" so that it can choose relief).
- A larger share of instinct choices in what the policy learns from (e.g. more candidates, or the instinct's moves
  replayed more).

