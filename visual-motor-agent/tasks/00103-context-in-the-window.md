# Task 00103: A Longer Context in a World Larger Than the View (experiment vm-00092)

**Date:** 2026-11-13
**Status:** Complete (2026-11-15; experiment vm-00092, `00092-context-in-the-window/`)

## Context

- **vm-00091 (task 00102):** in a 128 × 128 world with 2 foods, seen through a 64 × 64 window, the value (VW) eats 47.6 food per 1k against K4's 21.2.
  - Food is out of view on 42% of VW's frozen ticks.
  - **Memory lasts the context:** 1–8 ticks after food was last in view, VW turns away from it less than a blind baseline (+.062 ± .017, 12 of 12, without the ticks after a meal). At 9–32 ticks there is nothing (−.022 ± .016).
  - The pre-set next step: a memory that outlasts the context, tested in this world.
- **The owner (2026-11-13):** "yes go ahead with the longer context."
- **vm-00082 (task 00093):** one transformer over the small world, ctx 8 to 32, K4: no food gain at any context, poison rising with the context (+5.45 ± 1.70 at 32), and no memory of hidden food. But there food was always in view, and that agent foraged poorly. Here food is often out of view, and VW forages.
- **Walkers in this world:** chasing whatever is in the window 96.2 food per 1k; seeing the whole world 136.7. The gap is what memory and search can win.

## Design

| Arm | Label | Model | Context | Seeds |
|---|---|---|---|---|
| C8 = vm-00091's VW (not rerun) | value, γ 0.9, n 4, fresh tail | 4 quarters × 8 layers, tolerance 0.002 | 8 | 1–12 |
| **C32** | the same | the same | 32 | 1–12 |

- **C16 was dropped after the timing (the owner, 2026-11-14):** it was only to check that memory follows the context, and can follow if C32 shows a gain.

- **The world and everything else are vm-00091's VW:** `--config configs/U_view.json --set online.future_k=4 --set model.n_layer=8 --set online.instinct_tol=0.002 --set online.value_gamma=0.9 --set online.value_refresh=1`, plus `--set model.ctx=32`.
- **What a longer context changes:**
  - acting: the ring holds ctx steps, so a move reads the last 32 frames;
  - training: each replayed window is ctx steps, so an update trains on 4 times C8's tokens (batch 8 windows);
  - memory: chunks of ctx steps; replay stays 100k steps.
- **C8 is not rerun:** the code that trains is unchanged since vm-00091, and GPU lives are not reproducible run to run anyway; the arms are compared over seeds. Its frozen tests are rerun with the finer bins (below), linked as `C8_sN` in this experiment's folder.
- **Cards:** seeds 1–6 on the RTX 4090, 7–12 on the RTX 3090, under MPS.
- **Timing** (1,000-tick lives, 2026-11-13 23:40–23:59, scratchpad `t92/`; 6 lives at ctx 16 and 6 at ctx 32 per card were planned):
  - GPU memory at steady state: about 1.4 GB per life at ctx 16, 2.9 GB at ctx 32. Twelve lives per card do not fit in 24 GB: 7 of the ctx-32 lives died out of memory. Six ctx-32 lives per card take about 17 GB.
  - Speed, with 9 lives per card: 4090 2.4 ticks/s at 16, 2.1 at 32; 3090 1.0 and 0.9. vm-00091's timing at ctx 8 was 4.8 (4090) and 3.3 (3090) with 12 lives per card, and its runs took 4 h.
  - Estimate: 10–13 h, the 3090 last.
- **Options put to the owner** (2026-11-14): C16 and C32 at 12 seeds in two waves (22–26 h), both at 8 seeds (13 h), or C32 alone at 12 seeds (10–13 h). The owner chose C32 alone.

## Code

1. **`eval_hunger.py chain`:** the out-of-view bins become 1–8, 9–16, 17–32, 33–64, 65+ ticks since food was last in view, so the gain of each context shows in its own bin. Each row carries its bin names (`since_names`); vm-00091's rows, without them, still print under their old bins.
2. **`eval_hunger.py show --base`:** the pairing also compares the memory measure (seen − blind, without the ticks after a meal) by bin, when both arms have the same bins.
3. No change to the model or the lifetime.

## Measures

- **Lived, 40–60k:** food, poison, hunger, overrides, move CE, smoothness; each arm against C8, paired by seed.
- **Frozen `chain`** (1,500 ticks): food, the share with no food in view, aim at unseen food by ticks since the last sight, seen against blind, without the ticks after a meal, and paired against C8.
- **Frozen `value` and `reach`,** as in vm-00091.

## The readings, set before the results

1. **Plumbing:** all 12 lives run 60k ticks with `model.ctx` 32, `env.view` 64, without errors.
2. **Memory reaches further (the main one):** C32's aim at unseen food at 9–16 and 17–32 ticks (seen − blind, without the ticks after a meal) is above C8's in the same bins, paired by seed over 12 seeds, by more than its interval, in at least one of the two bins.
3. **Memory pays:** C32's lived food at 40–60k is above C8's, paired by seed, by more than its interval.
4. **Poison is still seen:** C32's lived poison is above C8's by more than its interval (vm-00082 found +5.45 at 32).
5. **The share out of view falls:** C32's frozen share of ticks with no food in view is below C8's (.423), paired by seed.

**What follows:**
- **If 2 and 3 hold:** a longer context is the memory. Next is its cost (tokens per update, 2–3.5 times slower per tick) and whether memory follows the context (C16), or the same reach at less cost.
- **If 2 holds and 3 fails:** the agent remembers but does not use it to eat. Look at what it does with it (return to food, or only turn away less), then the value's horizon.
- **If 2 fails:** the context is not the bottleneck: the model does not keep what is in its input as a place to return to. Next is a summary carried across windows, written up for the owner's decision.
- **If 4 fails,** whatever else holds: vm-00082's poison cost carries over, and its cause (each update spread over more ticks) is tested before a longer context is used.

## Implementation

- [x] `eval_hunger.py`: finer bins with their names in each row; the memory measure in `show --base`. vm-00091's rows still print.
- [x] Timing at ctx 16 and 32 (1,000-tick lives, 12 per card planned; out of memory at 32, see Design).
- [x] `00092-context-in-the-window/` (README, Makefile); the runs (started 2026-11-14; every info.json checked: ctx 32, view 64, foods 2; about 17–18 GB per card).
- [x] Frozen tests (C8 included): seeds 1–6 on the idle 4090 while seeds 7–12 still lived (2026-11-14 20:42–21:11, Makefile `frozen_a`), seeds 7–12 after the lives (23:43–00:02, `frozen_b`: chain and value on the 4090, reach on the 3090).
- [x] Write-up (README, `readme_images.py` figures), videos (`make videos`: `videos/vm92_s5`, `vm92_s12`).

## Results (summary; full report in the experiment README)

- **Readings:**
  1. Plumbing: met.
  2. Memory reaches further: **fails.** C32 − C8 in the aim at unseen food (seen − blind, without meals) is +.018 ± .025 at 9–16 ticks and −.008 ± .032 at 17–32. At 1–8 ticks C32 is weaker (−.031 ± .021).
  3. Memory pays: **fails.** Lived food at 40–60k is 23.4 against 47.6 (−24.2 ± 6.0, lower in 12 of 12); frozen −28.1 ± 5.7.
  4. Poison: +7.16 ± 0.72 (12 of 12), met as written. The follow-up for this outcome was filed under "if 4 fails" by mistake, and it applies.
  5. The share out of view falls: **fails.** It rises, .460 against .423 (+.037 ± .021, higher in 10 of 12).
- **What follows (set above):**
  - Reading 2 failed: the context is not the bottleneck. Next is a summary carried across windows, written up in the README's Next steps for the owner's decision.
  - The poison cost carries over: its cause is tested before a longer context is used again.
- **Also:**
  - C32 never learns to avoid poison (12.9 → 10.9 per 1k over its life, C8 11.6 → 3.5).
  - Its move CE is far lower (.116 against .268).
  - Its instinct overrides rise over life (.34 → .50), while C8's fall (.46 → .27).
  - Value as a share of the label matches C8's (−32% against −29%, correlation .86 against .84).

## Verification

- [x] Every run's `info.json` has `model.ctx` 32, `env.view` 64, `n_layer` 8, `instinct_tol` 0.002, `value_gamma` 0.9, `value_refresh` 1 (and `env.foods` 2): checked for all 12 after the runs.
- [x] C8's rerun chain repeats vm-00091's food, poison, aim, out-of-view share and smoothness exactly, seed by seed, under the new bins. Its value rows repeat vm-00091's too (pred .030, real .042, bias −29%).
- [x] vm-00091's rows still print under their old bins (`eval_hunger.py show` on `exp_00091`, numbers as in its README).
