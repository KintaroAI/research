# vm-00111: A Long Label Horizon in the Bricks-Only Court

**Date:** 2026-11-24
**Status:** Planned (queued after vm-00110's frozen tests)
**Source:** `exp/vm-00111` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00111/`
**Task plan:** [`../tasks/00124-long-horizon-bricks.md`](../tasks/00124-long-horizon-bricks.md)

## Goal

The owner, 2026-11-24: "for now we need to do #1 (0.997) and #2 - make both to be 3 lives"; the court chosen with the owner: bricks only.

- **No taste for better bricks** in vm-00104, vm-00107 and vm-00109 (1M ticks): each look gets a third of the hits.
- **The suspected cause is the label's horizon:** at γ .95 a brick 20 ticks after the aim is worth .36 of its size there; at γ .997, .94.
- **Why the bricks-only court:** a meal step's label is its own worth with no tail (`online.meal_parts`). With paddle food (vm-00107) the paddle hit is a meal, so the label of the aiming ticks stops there at any γ. Without it, the next meal after a return is the brick.

## Method

- **BH:** vm-00104's BS (`configs/U_breakout2.json`, `online.meal_size` .5) with `online.label_gamma = .997`. Nothing else changed.
- **Seeds 1–3, 60k ticks**, `env.brick_shift` = seed mod 3, on the RTX 3090 under MPS; paired with vm-00104's BS seeds 1–3.
- **Frozen** at 30k and 60k (2,000 ticks, seed 100 + s).

## The readings, set before the results (3 seeds: descriptive)

1. **Does a taste appear?** BH's preference at 40–60k above the aimer's level and above BS's, in each seed; the big bricks' share above a third.
2. **Does the play come back?** BH's hit share at 40–60k against BS's (.33 at 50–60k over 12 seeds).
3. **Does the long horizon break the label?** The meal monitor against the real label; hunger.

## Results

(pending)
