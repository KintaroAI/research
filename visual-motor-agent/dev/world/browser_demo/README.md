# CircleWorld browser demo

A dependency-free HTML/JS copy of `world/circle_world.py`. It has the same 64×64 renderer and the
same step rules, so you can play the task yourself or watch recorded episodes. It runs straight from
`file://`: open `index.html` in any modern browser (no server needed).

## Play mode

- Arrow keys or WASD set `dx, dy = ±1`; pressing two keys gives a diagonal. Space sends one click
  per key press, and holding Space does not repeat it.
- With **Mouse control** on, each step does `dx, dy = clip((mouse − cursor) / max_step_px, −1, 1)`
  and the mouse button clicks.
- The world advances at a fixed rate, 15 steps/s by default (5–30 can be selected), even when you
  do nothing. The step cost and the drives keep running.
- Toggles cover exploration bonuses, drives (exposes the drives and enables the hunger cost), the
  hunger cost, the radius range (applied at the next reset) and the seed.
- The panel shows the last reward, the episode return, the step counter, hunger and restlessness
  bars, the remaining bonus budgets and a log of finished episodes. The small canvas shows the
  agent's actual 64×64 observation.

## Replay mode

Record an episode in Python (from `dev/`):

```python
from world import CircleWorld, EnvConfig
from world.record import EpisodeRecorder

env = CircleWorld(1, EnvConfig(), seed=0)
obs = env.reset()
rec = EpisodeRecorder(env.config, meta={"policy": "random"})
rec.begin(env)
while not rec.finished:
    action = policy(obs)                        # float array (1, 3) = [dx, dy, click]
    obs, reward, done, info = env.step(action)
    rec.record(action, reward, done, info)
rec.save_json("episode.json")
```

Then open `index.html`, go to **Replay**, and choose the file (or drop it on the page). Every frame
is re-rendered from the recorded state with the JS renderer; the file contains no images. The page
has play/pause, a step slider, a speed control, and per-step action, reward, bonus, return and
drive readouts. Keys: Space toggles play/pause, ←/→ step, Home/End jump.

## Files

| File | Purpose |
|---|---|
| `index.html` | The page (play and replay). |
| `circle_world.js` | Renderer and simulator. Exposed as the `CircleWorldJS` global in the browser and as a CommonJS module in Node. |
| `render_cli.js` | `node render_cli.js fixtures.json results.json`, used by `../check_js_equivalence.py`. |

## Equivalence with the Python world

```bash
./venv/bin/python -m world.check_js_equivalence     # from dev/; also run by pytest
```

The check runs about 300 fixtures (edge cases plus random states, configs and action sequences)
through both implementations. It compares every rendered image pixel for pixel and the step outputs
within 1e-5. In practice they are bit-identical: both use float64 with the same operation order,
and rewards and drives go out as float32.

One difference is by design: JS resets draw from a seeded mulberry32 PRNG, not numpy's PCG64, so
random episodes differ between the two. Fixtures therefore pass explicit states.

If you change a rule in `circle_world.py`, make the same change in `circle_world.js` and re-run
the check.
