# Task 00124: A Long Label Horizon in the Bricks-Only Court (experiment vm-00111)

**Date:** 2026-11-24
**Status:** Planned. The owner, 2026-11-24: "for now we need to do #1 (0.997) and #2 - make both to be 3 lives"; on where: the bricks-only court (vm-00104's), recommended because of the label's rule below.

## Context

- **No taste for better bricks** in vm-00104 (bricks only), vm-00107 (paddle food) and vm-00109 (1M ticks): the agent hits each look a third of the time.
- **The suspected cause is the label's horizon:** γ .95 halves the value every 14 ticks, and a brick is hit 15 or more ticks after the return that aims it.
- **Why not vm-00107's court:** a meal step's label is its own worth with no tail (`meal_parts`: "a meal: real 1, no tail"). With paddle food the paddle hit is a meal, so the label of every aiming tick stops there: a brick's worth cannot reach the aim at any γ.
- **In the bricks-only court** (`configs/U_breakout2.json`, no paddle food) the next meal after a return is the brick. A brick k ticks after the aim is worth γ^k of its size: at k = 20, .36 at γ .95 and .94 at γ .997. A longer horizon also credits the return itself more strongly, which vm-00104 lacked (BS returned .33 at 50–60k, random .23).

## The arm

- **BH:** vm-00104's BS (`configs/U_breakout2.json`, `online.meal_size` .5: a big brick worth 1, a small one .2) with `online.label_gamma = 0.997`. Nothing else changed.
- **Seeds 1–3, 60k ticks**, `env.brick_shift` = seed mod 3 as in vm-00104, on the RTX 3090 under MPS. Paired with vm-00104's BS seeds 1–3 (and BS's 12 seeds as the range).
- **Frozen** at 30k and 60k (2,000 ticks, seed 100 + s, as vm-00104).

## Measures

- Paddle hit share, bricks and food per 1k, by 10k lived, against BS.
- The preference (rank hit minus standing mean, per brick) and the big bricks' share of hits, lived by 10k and frozen, against the aimer's chance level (vm-00104's walkers).
- The meal monitor: predicted against real label by ticks to the next meal (does the label carry value 15–60 ticks before a meal).

## The readings, set before the results (3 seeds: descriptive)

1. **Does a taste appear?** BH's preference at 40–60k above the aimer's level and above BS's, in each seed; the big bricks' share above a third.
2. **Does the play come back?** BH's hit share at 40–60k against BS's (.33 at 50–60k over 12 seeds).
3. **Does the long horizon break the label?** The meal monitor against the real label; hunger.

## Implementation

- [ ] `00111-long-horizon-bricks/` (README, Makefile: vm-00104's `SET` and look shift, BS's setting plus `--set online.label_gamma=0.997`, arm BH, seeds 1–3 on GPU 0; frozen at 30k and 60k).
- [ ] The runs, analysis, write-up.
