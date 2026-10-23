"""Check that browser_demo/circle_world.js matches world/circle_world.py.

Generates about 300 fixtures: hand-written edge cases plus random states,
configs and action sequences.  Each fixture runs as a one-env world in Python
and in Node (browser_demo/render_cli.js).  The script then compares:

  * every rendered image, pixel for pixel: the initial frame, plus the
    terminal frame and the returned obs frame of every step;
  * float step outputs within 1e-5: cursor, reward, exposed drives, bonus,
    internal drives, counters and episode stats;
  * boolean outputs exactly: done, success, timeout, click, empty_click,
    first, inside, target_alive.

Fixtures use auto_reset=False (no random resets), except a few auto-reset
fixtures.  Those are compared only on pre-reset quantities, up to their first
done step, because JS and Python reset with different RNGs.

Exit status: 0 = equivalent, 1 = mismatch, 2 = node missing or failed.

Usage (from dev/):
    python -m world.check_js_equivalence [--node PATH] [--seed S] [--keep DIR] [-v]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # allow `python world/check_js_equivalence.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from world.circle_world import CURSOR_EDGE_EPS, CircleWorld, EnvConfig

HERE = Path(__file__).resolve().parent
CLI = HERE / "browser_demo" / "render_cli.js"
TOL = 1e-5

FLOAT_STATE = ("hunger", "restlessness", "t", "cum_move", "empty_clicks", "clicks",
               "ep_return", "path_length", "bonus_total")
EP_KEYS = ("ep_length", "ep_return", "ep_success", "ep_clicks", "ep_empty_clicks",
           "ep_path_length", "ep_init_dist", "ep_bonus_total")
BOOL_KEYS = ("done", "success", "timeout", "click", "empty_click", "first", "inside")
#: Per-step keys that are still meaningful after an auto-reset.
PRE_RESET_SCALARS = ("reward", "bonus")


# =============================================================================
# Fixtures
# =============================================================================

def _json_num(v: float) -> Any:
    """JSON has no NaN/Infinity literals; encode them as strings."""
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v


def _state(cursor, target, radius, alive=True, **kw) -> Dict[str, Any]:
    st = {"cursor": [float(cursor[0]), float(cursor[1])],
          "target": [float(target[0]), float(target[1])],
          "radius": float(radius), "target_alive": bool(alive),
          "t": 0, "hunger": 0.2, "restlessness": 0.3, "cum_move": 0.0,
          "empty_clicks": 0, "clicks": 0, "ep_return": 0.0, "path_length": 0.0,
          "init_dist": 5.0, "bonus_total": 0.0}
    st.update(kw)
    return st


def _fx(fid: str, state: Dict[str, Any], actions=(), **config) -> Dict[str, Any]:
    cfg = {"auto_reset": False}
    cfg.update(config)
    return {"id": fid, "config": cfg, "state": state,
            "actions": [[_json_num(v) for v in a] for a in actions]}


def edge_case_fixtures() -> List[Dict[str, Any]]:
    """Hand-written cases at the boundaries of the rendering and step rules."""
    hi = 64 - CURSOR_EDGE_EPS
    inf, nan = float("inf"), float("nan")
    F: List[Dict[str, Any]] = []

    # Cursor at the four corners and at the clamp boundary; push outward, then in.
    for name, (x, y), push in [("tl", (0.0, 0.0), (-1, -1)), ("tr", (hi, 0.0), (1, -1)),
                               ("bl", (0.0, hi), (-1, 1)), ("br", (hi, hi), (1, 1))]:
        F.append(_fx(f"corner_{name}", _state((x, y), (32, 32), 10),
                     [(push[0], push[1], 0), (push[0], 0, 0), (-push[0], -push[1], 0)], bonuses=True))
    # Integer and just-below-integer cursor positions (hotspot = floor).
    for k, (x, y) in enumerate([(10.0, 20.0), (10.999999999, 20.999999999),
                                (11 - 1e-12, 21 - 1e-12), (0.5, 63.5), (63.0, 0.0)]):
        F.append(_fx(f"cursor_pos_{k}", _state((x, y), (40, 40), 12), [(0.25, -0.25, 0)]))
    # Circles whose boundary passes exactly through pixel centers.
    for k, (c, r) in enumerate([((32.5, 32.5), 0.0), ((32.5, 32.5), 0.5), ((32.5, 32.5), 1.0),
                                ((32.5, 32.5), 5.0), ((32.5, 32.5), 13.0), ((32.5, 32.5), 2.5),
                                ((32.5, 32.5), math.sqrt(2)), ((32.0, 32.0), math.sqrt(0.5)),
                                ((32.0, 32.0), math.sqrt(2.5)), ((20.0, 40.0), math.sqrt(12.5)),
                                ((10.25, 50.75), 7.0), ((31.5, 31.5), 32.0)]):
        F.append(_fx(f"boundary_circle_{k}", _state((2, 2), c, r)))
    # Circles partially or entirely outside the viewport, or covering it.
    for k, (c, r) in enumerate([((-5, 30), 10), ((70, 70), 12), ((32, -3), 8),
                                ((-20, -20), 5), ((32, 32), 50)]):
        F.append(_fx(f"offscreen_circle_{k}", _state((20, 20), c, r), [(1, 0, 1)]))
    # Hit test exactly on the boundary, dead targets, cursor drawn over the circle.
    F.append(_fx("click_on_boundary", _state((40, 30), (30, 30), 10), [(0, 0, 1)]))
    F.append(_fx("click_just_outside", _state((40 + 1e-9, 30), (30, 30), 10), [(0, 0, 1)]))
    F.append(_fx("dead_target_click", _state((30, 30), (30, 30), 10, alive=False),
                 [(0, 0, 1), (0.5, 0, 1)], bonuses=True))
    F.append(_fx("cursor_over_circle", _state((30.7, 29.2), (30, 30), 14), [(0, 0, 0)]))
    # Click threshold and non-finite actions.
    F.append(_fx("click_threshold", _state((5, 5), (40, 40), 12),
                 [(0, 0, 0.5), (0, 0, 0.5000001), (0, 0, inf), (0, 0, nan), (0, 0, -inf),
                  (0, 0, 1e300)], bonuses=True))
    F.append(_fx("nonfinite_moves", _state((30, 30), (50, 10), 6),
                 [(nan, inf, 0), (-inf, nan, 1), (2.5, -7, 0), (1e300, -1e300, 0),
                  (-0.0, 0.0, 0), (0.3, nan, nan)], bonuses=True, drives=True, hunger_cost=0.01))
    # Budget edges (factor exactly 0, just positive, and beyond the budget).
    for k, (cum, clicks) in enumerate([(200.0, 10), (199.0, 9), (250.0, 15), (0.0, 0)]):
        F.append(_fx(f"budget_{k}", _state((20, 20), (50, 50), 8, cum_move=cum,
                                           empty_clicks=clicks, clicks=clicks),
                     [(1, 0, 1), (0, 1, 1), (-1, -1, 0)], bonuses=True, drives=True))
    # Drive clamping.
    F.append(_fx("hunger_clamp_high", _state((20, 20), (50, 50), 8, hunger=0.999, restlessness=0.995),
                 [(0, 0, 0)] * 3, drives=True, hunger_cost=0.01))
    F.append(_fx("restless_clamp_low", _state((20, 20), (50, 50), 8, hunger=0.0, restlessness=0.01),
                 [(1, 1, 0)] * 3, drives=True))
    F.append(_fx("success_hunger_relief", _state((50, 50), (50, 50), 8, hunger=0.7),
                 [(0, 0, 1)], drives=True, hunger_cost=0.01, h_relief=0.4))
    # Episode limits.
    F.append(_fx("timeout_last_step", _state((5, 5), (40, 40), 10, t=4), [(1, 0, 0), (1, 0, 0)],
                 max_steps=5))
    F.append(_fx("success_on_last_step", _state((40, 40), (40, 40), 10, t=4), [(0, 0, 1)],
                 max_steps=5))
    F.append(_fx("continue_after_success", _state((40, 40), (40, 40), 10),
                 [(0, 0, 1), (0, 0, 1), (0.5, 0.5, 1), (0, 0, 0)], end_on_success=False,
                 max_steps=4, bonuses=True))
    # Auto-reset fixtures (compared on pre-reset quantities only).
    F.append(_fx("autoreset_success", _state((40, 40), (40, 40), 10), [(0, 0, 1), (1, 0, 0)],
                 auto_reset=True))
    F.append(_fx("autoreset_timeout", _state((5, 5), (40, 40), 10, t=1), [(0, 1, 0), (1, 1, 1)],
                 auto_reset=True, max_steps=2, drives=True))
    # Other viewport sizes and step sizes.
    F.append(_fx("size_1", _state((0.5, 0.5), (0.5, 0.5), 0.5), [(1, 1, 1)], size=1,
                 radius_min=0.25, radius_max=0.5))
    F.append(_fx("size_2", _state((1.2, 0.3), (1.0, 1.0), 0.75), [(-1, 1, 0)], size=2,
                 radius_min=0.5, radius_max=1.0))
    F.append(_fx("size_5", _state((4.9, 4.9), (2.5, 2.5), 2.0), [(-0.5, -0.5, 1)], size=5,
                 radius_min=1.0, radius_max=2.5))
    F.append(_fx("size_17", _state((8.5, 3.0), (8.5, 8.5), 6.0), [(0, 1, 0), (0, 1, 1)], size=17,
                 radius_min=1.0, radius_max=8.5))
    for k, step_px in enumerate([0.3, 2.7, 10.0]):
        F.append(_fx(f"step_px_{k}", _state((30, 30), (45, 45), 9), [(0.7, 0.9, 0), (-1, 0.2, 1)],
                     max_step_px=step_px, bonuses=True, drives=True))
    F.append(_fx("custom_colors", _state((12, 40), (30, 30), 14),
                 [(1, 0, 0)], bg_color=[255, 255, 255], target_color=[0, 0, 0],
                 cursor_color=[0, 255, 0], cursor_outline=[220, 40, 40]))
    return F


def _random_config(rng: np.random.Generator) -> Dict[str, Any]:
    size = int(rng.choice([64] * 6 + [32, 48, 17, 9, 100]))
    rmin = float(rng.uniform(0.5, size / 2))
    cfg: Dict[str, Any] = {
        "size": size, "max_steps": int(rng.integers(1, 40)),
        "max_step_px": float(4.0 if rng.random() < 0.5 else rng.uniform(0.2, 9.0)),
        "radius_min": rmin, "radius_max": float(rng.uniform(rmin, size / 2)),
        "end_on_success": bool(rng.random() < 0.7), "auto_reset": False,
        "bonuses": bool(rng.random() < 0.5), "drives": bool(rng.random() < 0.5),
    }
    if rng.random() < 0.3:
        for name in ("bg_color", "target_color", "cursor_color", "cursor_outline"):
            cfg[name] = [int(v) for v in rng.integers(0, 256, 3)]
    if rng.random() < 0.5:
        cfg.update(success_reward=float(rng.uniform(0, 20)), step_cost=float(rng.uniform(0, 0.05)),
                   move_bonus=float(rng.uniform(0, 0.1)), move_budget_px=float(rng.uniform(1, 300)),
                   click_bonus=float(rng.uniform(0, 0.1)), click_budget=int(rng.integers(1, 15)),
                   h_rate=float(rng.uniform(0, 0.05)), h_relief=float(rng.uniform(0, 1.5)),
                   hunger_cost=float(rng.uniform(0, 0.05)), b_rate=float(rng.uniform(0, 0.1)),
                   b_relief=float(rng.uniform(0, 0.5)))
    return cfg


def _random_state(rng: np.random.Generator, cfg: EnvConfig) -> Dict[str, Any]:
    size = cfg.size
    cursor = rng.uniform(0, size, 2)
    if rng.random() < 0.15:  # exact edges and integers
        cursor = rng.choice([0.0, float(int(rng.integers(0, size))), size - CURSOR_EDGE_EPS], 2)
    cursor = np.minimum(cursor, size - CURSOR_EDGE_EPS)
    target = rng.uniform(-0.2 * size, 1.2 * size, 2)
    radius = rng.uniform(0, 0.6 * size)
    if rng.random() < 0.2:  # exact half-pixel geometry
        target = np.round(target * 2) / 2
        radius = float(rng.choice([0.5, 1.0, 2.5, 5.0, 13.0, math.sqrt(2)]))
    empty = int(rng.integers(0, int(1.5 * cfg.click_budget) + 2))
    return _state(cursor, target, radius, alive=bool(rng.random() < 0.85),
                  t=int(rng.integers(0, cfg.max_steps)), hunger=float(rng.random()),
                  restlessness=float(rng.random()),
                  cum_move=float(rng.uniform(0, 1.3 * cfg.move_budget_px)),
                  empty_clicks=empty, clicks=empty + int(rng.integers(0, 3)),
                  ep_return=float(rng.uniform(-5, 5)), path_length=float(rng.uniform(0, 100)),
                  init_dist=float(rng.uniform(0, 50)), bonus_total=float(rng.uniform(0, 1)))


def _set_fixture_state(env: CircleWorld, st: Dict[str, Any]) -> None:
    env.set_state({k: [v] for k, v in st.items()})


def random_fixtures(rng: np.random.Generator, count: int) -> List[Dict[str, Any]]:
    """Random configs/states; actions mix noise, special values and target seeking.

    Seeking actions are generated by stepping a Python env forward, so that
    fixtures contain real successes, timeouts and budget exhaustion.  The
    comparison itself replays the fixed action list from scratch.
    """
    special = [0.0, 1.0, -1.0, 0.5, float("nan"), float("inf"), -float("inf"), 1.7, -3.0]
    F = []
    for k in range(count):
        cfg_d = _random_config(rng)
        cfg = _config({"config": cfg_d})
        st = _random_state(rng, cfg)
        env = CircleWorld(1, cfg, seed=0)
        _set_fixture_state(env, st)
        seek = rng.random() < 0.4
        actions = []
        for _ in range(int(rng.integers(0, 13))):
            if seek and rng.random() < 0.8:
                d = (env.target[0] - env.cursor[0]) / cfg.max_step_px
                a = [float(np.clip(d[0], -1, 1) + rng.normal(0, 0.1)),
                     float(np.clip(d[1], -1, 1) + rng.normal(0, 0.1)), float(rng.random() < 0.6)]
            else:
                a = [float(v) for v in rng.uniform(-1.3, 1.3, 2)] + [float(rng.random())]
                if rng.random() < 0.1:
                    a[int(rng.integers(0, 3))] = float(rng.choice(special))
            actions.append(a)
            env.step(np.array([a]))
        F.append(_fx(f"random_{k:03d}", st, actions, **cfg_d))
    return F


# =============================================================================
# Running
# =============================================================================

def _config(fx: Dict[str, Any]) -> EnvConfig:
    base = EnvConfig().to_dict()
    base.update(fx["config"])
    return EnvConfig.from_dict(base)


def run_python(fx: Dict[str, Any]) -> Dict[str, Any]:
    """Python counterpart of render_cli.js:runFixture (images as arrays)."""
    env = CircleWorld(1, _config(fx), seed=0)
    _set_fixture_state(env, fx["state"])
    out = {"id": fx["id"], "image0": env.observe()["image"][0], "steps": []}
    for action in fx["actions"]:
        obs, reward, done, info = env.step(np.array([[float(v) for v in action]]))
        s = env.get_state()
        out["steps"].append({
            "image": info["terminal_image"][0],
            "obs_image": obs["image"][0],
            "cursor": info["gt_cursor"][0].tolist(),
            "reward": float(reward[0]),
            "drives": info["terminal_drives"][0].astype(np.float64).tolist(),
            "obs_drives": obs["drives"][0].astype(np.float64).tolist(),
            "done": bool(done[0]), "success": bool(info["success"][0]),
            "timeout": bool(info["timeout"][0]), "click": bool(info["click"][0]),
            "empty_click": bool(info["empty_click"][0]), "first": bool(info["first"][0]),
            "inside": bool(info["gt_inside"][0]), "bonus": float(info["bonus"][0]),
            "state": {"target_alive": bool(info["gt_target_alive"][0]),
                      "hunger": float(info["gt_drives"][0, 0]),
                      "restlessness": float(info["gt_drives"][0, 1]),
                      **{k: float(s[k][0]) for k in FLOAT_STATE if k not in ("hunger", "restlessness")}},
            "ep": {k: float(info[k][0]) for k in EP_KEYS},
        })
    return out


def run_node(fixtures: List[Dict[str, Any]], node: str, workdir: Path) -> List[Dict[str, Any]]:
    fx_path, res_path = workdir / "fixtures.json", workdir / "results.json"
    fx_path.write_text(json.dumps({"fixtures": fixtures}, allow_nan=False))
    proc = subprocess.run([node, str(CLI), str(fx_path), str(res_path)],
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"node failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return json.loads(res_path.read_text())["results"]


def _decode(b64: str, size: int) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(b64)), dtype=np.uint8).reshape(size, size, 3)


# =============================================================================
# Comparison
# =============================================================================

class Report:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.images = 0
        self.bad_images = 0
        self.bad_pixels = 0
        self.steps = 0
        self.max_diff: Dict[str, float] = {}
        self.bool_checks = 0
        self.events = {"success": 0, "timeout": 0, "empty_click": 0, "done": 0, "clamped": 0}

    def image(self, where: str, py: np.ndarray, js: np.ndarray) -> None:
        self.images += 1
        diff = np.any(py != js, axis=-1)
        if diff.any():
            self.bad_images += 1
            self.bad_pixels += int(diff.sum())
            ys, xs = np.nonzero(diff)
            self.errors.append(f"{where}: {int(diff.sum())} pixels differ, first at "
                               f"(x={xs[0]}, y={ys[0]}) py={py[ys[0], xs[0]].tolist()} "
                               f"js={js[ys[0], xs[0]].tolist()}")

    def num(self, where: str, key: str, py: Optional[float], js: Optional[float]) -> None:
        py_nan = py is None or math.isnan(py)
        js_nan = js is None
        if py_nan or js_nan:
            if py_nan != js_nan:
                self.errors.append(f"{where}.{key}: py={py} js={js}")
            return
        d = abs(py - js)
        self.max_diff[key] = max(self.max_diff.get(key, 0.0), d)
        if not d <= TOL:
            self.errors.append(f"{where}.{key}: py={py!r} js={js!r} (|diff|={d:.3g})")

    def flag(self, where: str, key: str, py: bool, js: bool) -> None:
        self.bool_checks += 1
        if bool(py) != bool(js):
            self.errors.append(f"{where}.{key}: py={py} js={js}")


def compare(fixtures, py_results, js_results) -> Report:
    rep = Report()
    for fx, py, js in zip(fixtures, py_results, js_results):
        fid = fx["id"]
        size = _config(fx).size
        auto = bool(fx["config"].get("auto_reset", False))
        if js["id"] != fid:
            rep.errors.append(f"fixture order mismatch: {fid} vs {js['id']}")
            continue
        rep.image(f"{fid}/image0", py["image0"], _decode(js["image0"], size))
        if len(py["steps"]) != len(js["steps"]):
            rep.errors.append(f"{fid}: {len(py['steps'])} python steps vs {len(js['steps'])} js")
            continue
        for k, (ps, jst) in enumerate(zip(py["steps"], js["steps"])):
            where = f"{fid}/step{k}"
            rep.steps += 1
            reset_happened = auto and ps["done"]
            rep.image(f"{where}/terminal_image", ps["image"], _decode(jst["image"], size))
            if not reset_happened:
                rep.image(f"{where}/obs_image", ps["obs_image"], _decode(jst["obs_image"], size))
            for i in range(2):
                rep.num(where, "cursor", ps["cursor"][i], jst["cursor"][i])
                rep.num(where, "drives", ps["drives"][i], jst["drives"][i])
                if not reset_happened:
                    rep.num(where, "obs_drives", ps["obs_drives"][i], jst["obs_drives"][i])
            for key in PRE_RESET_SCALARS:
                rep.num(where, key, ps[key], jst[key])
            for key in BOOL_KEYS:
                rep.flag(where, key, ps[key], jst[key])
            rep.flag(where, "target_alive", ps["state"]["target_alive"], jst["state"]["target_alive"])
            for key in ("hunger", "restlessness"):
                rep.num(where, key, ps["state"][key], jst["state"][key])
            if not reset_happened:
                for key in FLOAT_STATE:
                    if key not in ("hunger", "restlessness"):
                        rep.num(where, key, ps["state"][key], jst["state"][key])
            for key in EP_KEYS:
                rep.num(where, key, ps["ep"][key], jst["ep"][key])
            for ev in ("success", "timeout", "empty_click", "done"):
                rep.events[ev] += int(ps[ev])
            edge = size - CURSOR_EDGE_EPS
            rep.events["clamped"] += int(any(c in (0.0, edge) for c in ps["cursor"]))
            if reset_happened:
                break  # later steps start from different random resets
    return rep


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--node", default=None, help="node executable (default: from PATH)")
    parser.add_argument("--seed", type=int, default=0, help="fixture RNG seed")
    parser.add_argument("--num-random", type=int, default=260, help="number of random fixtures")
    parser.add_argument("--keep", default=None, help="directory to keep fixtures/results JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="print all mismatches")
    args = parser.parse_args(argv)

    node = args.node or shutil.which("node")
    if not node:
        print("node not found; cannot check JS equivalence")
        return 2
    version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()

    rng = np.random.default_rng(args.seed)
    edge = edge_case_fixtures()
    fixtures = edge + random_fixtures(rng, args.num_random)
    py_results = [run_python(fx) for fx in fixtures]
    try:
        if args.keep:
            workdir = Path(args.keep)
            workdir.mkdir(parents=True, exist_ok=True)
            js_results = run_node(fixtures, node, workdir)
        else:
            with tempfile.TemporaryDirectory(prefix="cw_js_eq_") as tmp:
                js_results = run_node(fixtures, node, Path(tmp))
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
        print(f"node run failed: {e}")
        return 2

    rep = compare(fixtures, py_results, js_results)
    worst = max(rep.max_diff.items(), key=lambda kv: kv[1], default=("-", 0.0))
    ev = rep.events
    print(f"JS/Python equivalence (node {version}): {len(fixtures)} fixtures "
          f"({len(edge)} edge-case, {len(fixtures) - len(edge)} random), {rep.steps} steps")
    print(f"  images:   {rep.images} compared pixel-exact, {rep.bad_images} mismatched "
          f"({rep.bad_pixels} pixels)")
    print(f"  floats:   {len(rep.max_diff)} fields, max |py - js| = {worst[1]:.3g} "
          f"({worst[0]}), tolerance {TOL:g}")
    if args.verbose:
        for k, v in sorted(rep.max_diff.items()):
            print(f"            {k:16s} {v:.3g}")
    print(f"  booleans: {rep.bool_checks} compared")
    print(f"  coverage: {ev['success']} successes, {ev['timeout']} timeouts, {ev['done']} dones, "
          f"{ev['empty_click']} empty clicks, {ev['clamped']} clamped cursors")
    if rep.errors:
        shown = rep.errors if args.verbose else rep.errors[:15]
        for e in shown:
            print("  MISMATCH", e)
        if len(shown) < len(rep.errors):
            print(f"  ... {len(rep.errors) - len(shown)} more (use -v)")
        print(f"FAIL: {len(rep.errors)} mismatches")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
