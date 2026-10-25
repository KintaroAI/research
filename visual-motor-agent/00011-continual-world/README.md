# vm-00011: A Continual World: No Resets, Moving Circles, Hazards, a Wrap-Around and an Egocentric View

**Date:** 2026-10-25
**Status:** Complete
**Source:** `exp/vm-00011` (code: `visual-motor-agent/dev/`)
**Outputs:** `~/data/research/visual-motor-agent/exp_00011/`
**Task plan:** [`../tasks/00014-continual-world.md`](../tasks/00014-continual-world.md)

## Goal

Every world so far was episodic: a hit or 200 steps ended the episode and reset the world. Here the world never
resets. A hit circle is replaced at once by a new one, and the lifetime is one continuous stream. On top of that, the
world can have moving circles, hazards (blue circles that raise hunger while the cursor is inside them), a wrap-around
screen with no walls, and an egocentric view (the cursor stays at the center and moving shifts the scene).

The agent is vm-00008's DMI, the basic agent that aims, running on the GPU (tasks 00011–00013).

## Hypothesis

1. The basic agent forages in a continual world: it hits more often than at birth.
2. Moving circles are harder to hit, but the agent still learns them.
3. It does not learn to avoid hazards: nothing in its learning signal (the hindsight hunger command marks the moves
   before relief) points away from a rise in hunger.
4. Open: whether the egocentric view, where every move shifts the whole scene, is easier (the target's offset from
   the center is the direction to move) or harder (the cursor no longer moves on the screen).

## Method

- **Agent:** vm-00008's DMI (`configs/AHR_gated.json`, `ppo.pi_detach=1`, `model.pi_hidden=128`, inverse dynamics,
  memory U), `online.device=cuda`, 60k steps, seeds 1–3.
- **World options** (`world/circle_world.py`; all off by default, so earlier worlds are byte-identical; tests in
  `tests/test_world_ext.py`): `env.respawn` with `env.end_on_success=false` and `env.max_steps=1e9` (never reset);
  `env.target_speed`; `env.hazards`; `env.wrap` (a torus); `env.egocentric` (needs `wrap`). Their random draws use
  separate generators, so the first circle, the cursor and the drives are the same with or without them.
- **Arms:**

  | Arm | World | Runs |
  |---|---|---|
  | C, CM, CS, CH | continual with walls; + moving circles (0.5 px/step); + small circles (radius 6–8); + 2 hazards | 001–012 (stopped at about 12k) |
  | T | continual torus | 013–015 |
  | TE | + egocentric view | 016–018 |
  | TM | + moving circles | 019–021 |
  | TH | + 2 hazards (radius 5–8, 0.75 px/step, 0.02 hunger per step inside) | 022–024 |

- **Evaluation:** online hits per 1,000 steps, the share of steps in a hazard; `online.py --eval`: the episodic test
  (first hit within 200 steps, in the same world) and `eval_forage` (32 worlds × 1,000 steps of the frozen agent: hits
  per 1,000 steps, the share of steps in a hazard).

## Log

### 2026-10-25
- **Implemented** the world options and their tests (respawn never on the cursor, the circle's speed kept and the
  circle inside the walls, hazard pixels and hunger, state round trips, the torus hit test and drawing across the
  edge, the egocentric view keeping the cursor at the center and moving the circle by exactly the move). `online.py`
  counts every hit (`step_hits`), the hazard share and hunger, and `--eval` adds the forage test in continual worlds.
  All CPU tests pass (448), including the pixel-exact JS replica for the default world.
- **Launched C, CM, CS, CH** (runs 001–012, 12 lifetimes on the GPU at about 20 steps/s each).
- **With walls, the agent starves.** At 11k steps all 12 lifetimes hit 0–2 times per 1,000 steps, and hunger sits at
  1.0 from the first thousand steps on. A frozen check at 10k (32 worlds × 1,000 steps):

  | | hits per 1,000 steps | share of steps at a wall |
  |---|---|---|
  | the agent at 10k, C seed 1 | 0.2 | 0.55 |
  | the agent at 10k, CH seed 1 | 0.2 | 0.87 |
  | random babbling | 3.6 | 0.17 |

  The habit is a drift, and a wall absorbs a drift: the cursor slides into an edge and stays. In the episodic world
  the reset put the cursor somewhere new every 200 steps or sooner; the resets did the exploring. The runs were
  stopped at about 12k.
- **A wrap-around world** (`env.wrap`) has no walls to absorb the drift. Random agents hit 4.4–5.0 times per 1,000 steps
  there, against 2.8–3.0 with walls (32 worlds × 1,000 steps; the egocentric view the same).
- **The CUDA multi-process service** (`nvidia-cuda-mps-control -d`) lets the small kernels of the 12 lifetimes run
  at once: 69 steps/s per lifetime instead of 20 (830 in all).
- **Launched T, TE, TM, TH** (runs 013–024). At 6–7k steps they hit 4–17 times per 1,000 steps.
- **All 12 torus lifetimes finished at 01:06** (60k steps, about 15 minutes with the multi-process service). The frozen
  tests ran with the GPU forward (`online.py --eval ... --eval-device cuda`, about 10 s per checkpoint).

## Results

### R1. With walls: starvation (runs 001–012, steps 0–11k)

Online hits per 1,000 steps: 0–2 in every lifetime, hunger at 1.0 from the first thousand steps; the agent at 10k
spends 55–87% of its steps at a wall (random babbling: 17%) and hits 0.2 times per 1,000 steps (babbling: 3.6).

### R2. The torus: online hits per 1,000 steps (seeds 1, 2, 3; 20% of the moves babbled)

| Arm | steps 0–10k | 20k–30k | 50k–60k |
|---|---|---|---|
| T | 5.8, 5.4, 4.7 | 8.4, 7.8, 7.6 | 8.7, 8.8, 9.8 |
| TE (egocentric) | 0.9, 5.6, 14.7 | 11.5, 11.8, 17.1 | **17.3, 17.4, 19.3** |
| TM (moving circles) | 5.5, 7.8, 9.6 | 8.0, 6.8, 8.4 | 7.3, 6.3, 8.3 |
| TH (hazards) | 7.4, 6.7, 7.8 | 10.3, 9.4, 7.1 | 9.5, 8.0, 6.8 |

### R3. Frozen tests (birth → 20k → 40k → 60k; seeds 1, 2, 3)

| Arm | episodic test: first hit within 200 steps, same world | forage: hits per 1,000 steps (32 × 1,000 steps) |
|---|---|---|
| T | 0.62 0.88 0.92 0.89; 0.63 0.91 0.95 0.90; 0.59 0.93 0.91 0.94 | 5.0 → 9.8; 4.9 → 7.6; 4.7 → 10.0 |
| TE | 0.62 0.94 **1.00 1.00**; 0.62 0.95 **1.00 1.00**; 0.62 **1.00** 0.99 **1.00** | 4.4 → 18.2; 5.0 → 18.7; 4.7 → 20.8 |
| TM | 0.76 0.86 0.83 0.88; 0.76 0.82 0.86 0.84; 0.77 0.76 0.87 0.86 | 6.4 → 7.9; 5.4 → 8.0; 5.9 → 9.8 |
| TH | 0.63 0.88 0.87 0.95; 0.64 0.66 0.82 0.91; 0.60 0.68 0.65 0.88 | 5.0 → 11.2; 4.9 → 8.8; 4.5 → 7.5 |

Share of steps inside a hazard (TH, forage test): 0.057, 0.062, 0.069 at birth; 0.066, 0.067, 0.064 at 60k.

## Analysis

### Walls absorb the habit (hypothesis 1, not with walls)

- **In a continual world with walls the basic agent starves.** Its moves imitate its own moves, which drift; a drift
  ends at a wall, where the cursor stays. Nothing brings it back: no reset, no reward, and no hit to label any move.
- **The episodic world hid this.** Its resets put the cursor somewhere new every 200 steps or sooner, so the resets,
  not the agent, did the exploring. Every earlier result of the basic agent depends on them.

### On a torus it forages, and the egocentric view helps most (hypotheses 1 and 4)

- **On the torus, every arm learns.** Online hits rise from about 5 to 9–10 per 1,000 steps (T), and the episodic test
  in the same world from 0.6 at birth to 0.89–0.95 at 60k, above the best basic agent in the walled episodic world
  (DMI, 0.63–0.82 at 50k).
- **The egocentric view doubles it and solves the test**: 18–21 hits per 1,000 steps frozen, and every test episode a
  hit from 40k on in all three lifetimes. In that view the circle's offset from the screen center is the direction
  to move, the same at every place; in the allocentric view the agent must find the cursor first and compare two
  positions. The view does the binding of self and world that the CNN otherwise has to learn.

### Moving circles are harder (hypothesis 2)

- **Moving circles stay near the plain torus' level or below** (forage 7.9–9.8, episodic 0.84–0.88). The circle drifts at
  0.5 px per step against the cursor's 4; the window of 8 steps holds its velocity, but the head aims at where it
  is.

### No hazard avoidance (hypothesis 3, confirmed)

- **The basic agent spends as much time in hazards at 60k as at birth** (0.064–0.067 against 0.057–0.069), about the
  hazards' share of the area. It forages as well as in the hazard-free torus.
- **Its learning signal cannot see a hazard.** The hindsight command labels the moves before relief. A rise in hunger
  labels nothing, so no move is marked as one to avoid.

### Caveats

- 3 seeds per arm, 60k steps; online hits include the 20% babbled moves.
- The forage test is short (1,000 steps) and frozen; the episodic test keeps the continual world's motion and hazards
  but resets at a hit.
- The hazard cost (0.02 hunger per step) is small against the relief of a hit (1.0); hazards drift at 0.75 px per step.

## Conclusions

1. **Episode resets did the basic agent's exploring.** In a continual world with walls its drifting habit sticks to a
   wall and it starves (0–2 hits per 1,000 steps).
2. **In a world without walls it forages and keeps improving**: 0.89–0.95 on the episodic test at 60k, the best basic
   agent so far.
3. **An egocentric view makes the task easy**: every test episode a hit by 40k, twice the foraging rate. The view
   supplies the self-to-target geometry.
4. **It does not avoid hazards.** A drive that rises without relief marks nothing in its learning signal. The planner
   agent, which minimizes imagined hunger, is the test of whether a learned hunger model can (vm-00012).

## Next Steps

- **The planner agent in the same worlds** (vm-00012, running).
- **A learning signal for rises in hunger**: a hindsight command that also labels the moves before a hazard's cost, or
  a two-sided command (relief versus harm).
- **Moving circles with velocity in the aim**: does the hidden-layer head use the window's motion?
