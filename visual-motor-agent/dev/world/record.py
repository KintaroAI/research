"""Episode recording, replay export and image helpers for CircleWorld.

``EpisodeRecorder`` stores the *state* of one env per step, not images.  Every
frame can be re-rendered exactly with ``render_state`` in Python or with
``renderState`` in the browser replay viewer (browser_demo/index.html).

Recorded episode layout (T = number of steps):
    frames (T+1 entries; frame k is the state the agent observed before
    action k, and frame T is the terminal state):
        cursor (T+1, 2), target (T+1, 2), radius (T+1,), target_alive (T+1,),
        hunger (T+1,), restlessness (T+1,)    raw drive values, whether or not
                                              they were exposed to the agent
    steps (T entries):
        action (T, 3)   applied action [dx, dy, click] (sanitized: NaN -> 0,
                        dx, dy clipped to [-1, 1], click in {0, 1})
        reward (T,), bonus (T,), done, success, click, empty_click (T,)

Typical use:
    env = CircleWorld(N, cfg, seed)
    obs = env.reset()
    rec = EpisodeRecorder(env.config, env_index=0, meta={"policy": "random"})
    rec.begin(env)
    while not rec.finished:
        actions = policy(obs)
        obs, reward, done, info = env.step(actions)
        rec.record(actions, reward, done, info)
    rec.save_json("episode.json")      # load in browser_demo/index.html (Replay)
    save_gif(rec.render_frames(), "episode.gif")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from .circle_world import EnvConfig, render_state, sanitize_actions

FORMAT_NAME = "circle_world_episode"
FORMAT_VERSION = 1

_FRAME_KEYS = ("cursor", "target", "radius", "target_alive", "hunger", "restlessness")
_STEP_KEYS = ("action", "reward", "bonus", "done", "success", "click", "empty_click")
_BOOL_KEYS = ("target_alive", "done", "success", "click", "empty_click")

PathLike = Union[str, Path]


# =============================================================================
# Episode recorder
# =============================================================================

class EpisodeRecorder:
    """Records one episode of one env (``env_index``) of a CircleWorld.

    Post-step states come from the ``gt_*`` entries of ``info`` (evaluation-only
    ground truth), so terminal frames are correct even with auto-reset.
    """

    def __init__(self, config: EnvConfig, env_index: int = 0,
                 meta: Optional[Dict[str, Any]] = None):
        self.config = config
        self.env_index = int(env_index)
        self.meta: Dict[str, Any] = dict(meta or {})
        self._frames: Dict[str, List[Any]] = {k: [] for k in _FRAME_KEYS}
        self._steps: Dict[str, List[Any]] = {k: [] for k in _STEP_KEYS}
        self.finished = False
        self._started = False

    def __len__(self) -> int:
        """Number of recorded steps T."""
        return len(self._steps["reward"])

    # ----------------------------------------------------------------- input
    def begin(self, source: Any) -> None:
        """Start a new episode from the current state of ``source``.

        ``source`` is a CircleWorld or a ``get_state()`` dict; call this right
        after reset() (or after the done step that auto-reset this env).
        """
        state = source if isinstance(source, dict) else source.get_state()
        i = self.env_index
        for k in self._frames.values():
            k.clear()
        for k in self._steps.values():
            k.clear()
        self._append_frame(state["cursor"][i], state["target"][i], state["radius"][i],
                           state["target_alive"][i], state["hunger"][i], state["restlessness"][i])
        self.finished = False
        self._started = True

    def record(self, actions: np.ndarray, reward: np.ndarray, done: np.ndarray,
               info: Dict[str, np.ndarray]) -> bool:
        """Record one step from the arguments and return values of ``env.step``.

        ``actions`` may be the full (N, 3) batch or this env's (3,) action;
        ``reward`` and ``done`` may be (N,) arrays or scalars.  Returns True
        once the episode has finished.
        """
        if not self._started:
            raise RuntimeError("call begin() before record()")
        if self.finished:
            raise RuntimeError("episode already finished; call begin() to record another")
        i = self.env_index
        a = np.asarray(actions, dtype=np.float64)
        a = a[i] if a.ndim == 2 else a
        move, click = sanitize_actions(a.reshape(1, 3))
        d = bool(np.asarray(done).reshape(-1)[i] if np.ndim(done) else done)
        r = float(np.asarray(reward).reshape(-1)[i] if np.ndim(reward) else reward)

        self._steps["action"].append([float(move[0, 0]), float(move[0, 1]), float(click[0])])
        self._steps["reward"].append(r)
        self._steps["bonus"].append(float(info["bonus"][i]))
        self._steps["done"].append(d)
        self._steps["success"].append(bool(info["success"][i]))
        self._steps["click"].append(bool(info["click"][i]))
        self._steps["empty_click"].append(bool(info["empty_click"][i]))
        self._append_frame(info["gt_cursor"][i], info["gt_target"][i], info["gt_radius"][i],
                           info["gt_target_alive"][i], info["gt_drives"][i, 0], info["gt_drives"][i, 1])
        self.finished = d
        return d

    def _append_frame(self, cursor, target, radius, alive, hunger, restlessness) -> None:
        f = self._frames
        f["cursor"].append([float(cursor[0]), float(cursor[1])])
        f["target"].append([float(target[0]), float(target[1])])
        f["radius"].append(float(radius))
        f["target_alive"].append(bool(alive))
        f["hunger"].append(float(hunger))
        f["restlessness"].append(float(restlessness))

    # ---------------------------------------------------------------- output
    def arrays(self) -> Dict[str, np.ndarray]:
        """Recorded data as numpy arrays (frame arrays have T+1 rows, step arrays T)."""
        out: Dict[str, np.ndarray] = {}
        for k, v in list(self._frames.items()) + list(self._steps.items()):
            dtype = bool if k in _BOOL_KEYS else np.float64
            arr = np.array(v, dtype=dtype)
            if k in ("cursor", "target"):
                arr = arr.reshape(-1, 2)
            elif k == "action":
                arr = arr.reshape(-1, 3)
            out[k] = arr
        return out

    def summary(self) -> Dict[str, Any]:
        """Episode statistics computed from the recording."""
        a = self.arrays()
        steps = np.diff(a["cursor"], axis=0)
        return {
            "length": int(len(self)),
            "return": float(a["reward"].sum()),
            "success": bool(a["success"].any()),
            "finished": bool(self.finished),
            "clicks": int(a["click"].sum()),
            "empty_clicks": int(a["empty_click"].sum()),
            "path_length": float(np.sqrt((steps ** 2).sum(axis=1)).sum()) if len(steps) else 0.0,
            "init_dist": float(max(0.0, np.hypot(*(a["cursor"][0] - a["target"][0])) - a["radius"][0])),
            "bonus_total": float(a["bonus"].sum()),
        }

    def render_frames(self) -> np.ndarray:
        """Re-render all T+1 frames: uint8 (T+1, size, size, 3)."""
        a = self.arrays()
        return render_state(a["cursor"], a["target"], a["radius"], a["target_alive"], self.config)

    def to_json_dict(self) -> Dict[str, Any]:
        """The browser replay document (see module docstring)."""
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "config": self.config.to_dict(),
            "meta": {"env_index": self.env_index, **self.meta},
            "num_steps": len(self),
            "frames": {k: list(v) for k, v in self._frames.items()},
            "steps": {k: list(v) for k, v in self._steps.items()},
            "summary": self.summary(),
        }

    def save_json(self, path: PathLike) -> None:
        """Write the browser replay format (strict JSON, full float precision)."""
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, allow_nan=False, separators=(",", ":"))

    def save_npz(self, path: PathLike) -> None:
        """Write the arrays plus JSON-encoded config/meta/summary (no pickling needed)."""
        np.savez_compressed(
            path, **self.arrays(),
            config_json=np.array(json.dumps(self.config.to_dict())),
            meta_json=np.array(json.dumps({"env_index": self.env_index, **self.meta})),
            summary_json=np.array(json.dumps(self.summary())))


def load_episode(path: PathLike) -> Dict[str, Any]:
    """Load a recording written by save_json or save_npz.

    Returns a dict of numpy arrays (frame and step keys) plus ``config``
    (EnvConfig), ``meta`` and ``summary`` (dicts).
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            out: Dict[str, Any] = {k: z[k] for k in _FRAME_KEYS + _STEP_KEYS}
            out["config"] = EnvConfig.from_dict(json.loads(str(z["config_json"])))
            out["meta"] = json.loads(str(z["meta_json"]))
            out["summary"] = json.loads(str(z["summary_json"]))
        return out
    with open(path) as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT_NAME:
        raise ValueError(f"{path}: not a {FORMAT_NAME} file")
    out = {}
    for group in ("frames", "steps"):
        for k, v in doc[group].items():
            arr = np.array(v, dtype=bool if k in _BOOL_KEYS else np.float64)
            if k in ("cursor", "target"):
                arr = arr.reshape(-1, 2)
            elif k == "action":
                arr = arr.reshape(-1, 3)
            out[k] = arr
    out["config"] = EnvConfig.from_dict(doc["config"])
    out["meta"] = doc.get("meta", {})
    out["summary"] = doc.get("summary", {})
    return out


# =============================================================================
# Image helpers
# =============================================================================

def _to_rgb_u8(img: Any) -> np.ndarray:
    """HxWx3 uint8 from uint8/float ([0, 1]) images, grayscale, or RGBA."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(np.round(np.nan_to_num(a) * 255.0), 0, 255)
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"expected an HxW or HxWxC image, got shape {a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] == 4:
        a = a[:, :, :3]
    elif a.shape[2] != 3:
        raise ValueError(f"expected 1, 3 or 4 channels, got {a.shape[2]}")
    return np.ascontiguousarray(a)


def _resize_nearest(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize (exact pixel replication for integer factors)."""
    h, w = img.shape[:2]
    rows = (np.arange(height) * h) // height
    cols = (np.arange(width) * w) // width
    return img[rows][:, cols]


def save_gif(frames: Iterable[Any], path: PathLike, scale: int = 4, fps: float = 15) -> None:
    """Save frames (HxWx3) as a looping GIF, upscaled ``scale``x with nearest neighbor."""
    imgs = []
    for f in frames:
        a = _to_rgb_u8(f)
        imgs.append(Image.fromarray(_resize_nearest(a, a.shape[0] * scale, a.shape[1] * scale)))
    if not imgs:
        raise ValueError("save_gif: no frames")
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=max(1, int(round(1000.0 / fps))), loop=0)


def save_grid(rows: Sequence[Sequence[Any]], path: PathLike, scale: int = 4, pad: int = 2,
              pad_color: Sequence[int] = (128, 128, 128)) -> None:
    """Save a grid of images as a PNG.

    Args:
        rows: list of rows, each a list of HxWx3 uint8 images (float [0, 1],
            grayscale and RGBA are converted).  Sizes may differ: every image
            is resized (nearest neighbor) to the largest cell, then ``scale``x.
            Short rows are padded with empty cells.
        pad: gap between and around cells, in output pixels.
    """
    grid = [[_to_rgb_u8(im) for im in row] for row in rows]
    cells = [im for row in grid for im in row]
    if not cells:
        raise ValueError("save_grid: no images")
    ch = max(im.shape[0] for im in cells) * scale
    cw = max(im.shape[1] for im in cells) * scale
    ncols = max(len(row) for row in grid)
    canvas = np.empty((len(grid) * (ch + pad) + pad, ncols * (cw + pad) + pad, 3), dtype=np.uint8)
    canvas[:] = np.asarray(pad_color, dtype=np.uint8)
    for r, row in enumerate(grid):
        for c, im in enumerate(row):
            y, x = pad + r * (ch + pad), pad + c * (cw + pad)
            canvas[y:y + ch, x:x + cw] = _resize_nearest(im, ch, cw)
    Image.fromarray(canvas).save(path)


def save_strip(frames: Iterable[Any], path: PathLike, scale: int = 4, pad: int = 2) -> None:
    """Save frames side by side as one PNG row (a one-row save_grid)."""
    save_grid([list(frames)], path, scale=scale, pad=pad)
