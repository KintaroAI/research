"""TextWorld (task 00032): a text drawn as grey frames, with CircleWorld's interface, so that the universal model
lives in it unchanged.

A letter is a brightness: the corpus's sorted alphabet gives letter k the grey level round(k * 255 / (n - 1)).
A frame is a page of (size / cell)^2 letters, each a cell x cell block of its level, written in block x block blocks of
consecutive letters (blocks in raster order, letters within a block row by row; block 1: an ordinary page, line by
line). Each step shows the next page; each body reads from its own seeded offset and wraps around at the end of its
split (train: the first train_frac of the corpus, val: the rest), or with tile from evenly spaced pages (scoring: N
readers for pages / N steps read every whole page of the split once). Moves are ignored and the drives stay 0.
"""

import dataclasses
import os
from typing import Any, Dict, Optional

import numpy as np

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "llm-fundamentals", "dev",
                              "data", "tinyshakespeare", "input.txt")


@dataclasses.dataclass
class TextConfig:
    kind: str = "text"
    corpus: str = DEFAULT_CORPUS
    split: str = "train"          # train | val
    train_frac: float = 0.9
    size: int = 16                # frame side in pixels
    cell: int = 2                 # letter side in pixels
    block: int = 1                # letters per side of a block of consecutive letters
    tile: bool = False            # readers start evenly spaced, whole pages apart (together they read the split once)
    # read by the agent loops, as in EnvConfig: drives are observed (always 0 here), no hunger, no episodes
    drives: bool = True
    h_rate: float = 0.0
    respawn: bool = True
    auto_reset: bool = True

    def validate(self) -> "TextConfig":
        if self.kind != "text":
            raise ValueError("TextConfig.kind must be 'text'")
        if self.split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if self.size % self.cell:
            raise ValueError("size must be a multiple of cell")
        if (self.size // self.cell) % self.block:
            raise ValueError("the page's letters per side (size / cell) must be a multiple of block")
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must be in (0, 1)")
        return self

    @property
    def side(self) -> int:
        """Letters per page side."""
        return self.size // self.cell

    @property
    def page(self) -> int:
        """Letters per page (one step)."""
        return self.side ** 2


def load_corpus(path: str) -> str:
    with open(os.path.expanduser(path)) as f:
        return f.read()


class TextWorld:
    """N readers of one text, in lockstep. State: pos (N,) int64, each reader's position in its split."""

    def __init__(self, num_envs: int, config: Optional[TextConfig] = None, seed: int = 0):
        if int(num_envs) < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.config = dataclasses.replace(config if config is not None else TextConfig()).validate()
        c = self.config
        text = load_corpus(c.corpus)
        self.alphabet = "".join(sorted(set(text)))
        n = len(self.alphabet)
        if n < 2 or n > 256:
            raise ValueError("the corpus needs 2 to 256 distinct characters, has %d" % n)
        self.levels = np.round(np.arange(n) * 255.0 / (n - 1)).astype(np.uint8)
        lut = np.zeros(0x110000 if any(ord(ch) > 255 for ch in self.alphabet) else 256, np.int64)
        for k, ch in enumerate(self.alphabet):
            lut[ord(ch)] = k
        codes = lut[np.frombuffer(text.encode("utf-32-le"), np.uint32)]
        cut = int(len(codes) * c.train_frac)
        self.codes = codes[:cut] if c.split == "train" else codes[cut:]
        if len(self.codes) < c.page:
            raise ValueError("the split is shorter than one page")
        # where the i-th letter of a page sits on the page grid (flat index y * side + x)
        G, b = c.side, c.block
        i = np.arange(c.page)
        blk, r = i // (b * b), i % (b * b)
        y = (blk // (G // b)) * b + r // b
        x = (blk % (G // b)) * b + r % b
        self.grid_index = y * G + x
        if c.tile:
            pages = len(self.codes) // c.page
            self.pos = (np.arange(self.num_envs) * pages // self.num_envs * c.page).astype(np.int64)
        else:
            self.pos = np.random.default_rng(seed).integers(0, len(self.codes), self.num_envs).astype(np.int64)

    # ------------------------------------------------------------------ text <-> frames
    def page_codes(self, pos: np.ndarray) -> np.ndarray:
        """The letter codes of the pages starting at pos (N,): (N, page), wrapping around the split."""
        return self.codes[(np.asarray(pos)[:, None] + np.arange(self.config.page)[None]) % len(self.codes)]

    def render(self, codes: np.ndarray) -> np.ndarray:
        """Pages of letter codes (N, page) -> frames (N, size, size, 3) uint8."""
        c, G = self.config, self.config.side
        grid = np.zeros((len(codes), G * G), np.uint8)
        grid[:, self.grid_index] = self.levels[codes]
        img = grid.reshape(-1, G, G).repeat(c.cell, 1).repeat(c.cell, 2)
        return np.repeat(img[..., None], 3, axis=-1)

    def decode_levels(self, values: np.ndarray) -> np.ndarray:
        """Grey levels (any shape, 0..255) -> the nearest letter's code."""
        return np.abs(np.asarray(values, np.float64)[..., None] - self.levels.astype(np.float64)).argmin(-1)

    def read(self, images: np.ndarray) -> list:
        """Frames (N, size, size, 3) -> the pages' text (N strings), read at each letter's top-left pixel."""
        c, G = self.config, self.config.side
        grid = images[:, ::c.cell, ::c.cell, 0].reshape(len(images), G * G)
        codes = self.decode_levels(grid[:, self.grid_index])
        return ["".join(self.alphabet[k] for k in row) for row in codes]

    # ------------------------------------------------------------------ CircleWorld's interface
    def observe(self) -> Dict[str, np.ndarray]:
        return {"image": self.render(self.page_codes(self.pos)),
                "drives": np.zeros((self.num_envs, 2), np.float32)}

    def reset(self, env_ids=None) -> Dict[str, np.ndarray]:
        return self.observe()

    def step(self, actions: np.ndarray):
        if np.shape(actions) != (self.num_envs, 3):
            raise ValueError(f"expected actions of shape ({self.num_envs}, 3), got {np.shape(actions)}")
        self.pos = (self.pos + self.config.page) % len(self.codes)
        obs = self.observe()
        n, f = self.num_envs, np.zeros(self.num_envs, bool)
        nan = np.full(n, np.nan)
        info = dict(terminal_image=obs["image"], terminal_drives=obs["drives"], success=f, timeout=f.copy(),
                    click=np.asarray(actions)[:, 2] > 0.5, ep_length=nan, ep_return=nan.copy(),
                    ep_success=nan.copy(), ep_clicks=nan.copy(), gt_pos=self.pos.copy())
        return obs, np.zeros(n, np.float32), f.copy(), info

    def get_state(self) -> Dict[str, Any]:
        return {"pos": self.pos.copy()}

    def set_state(self, state: Dict[str, Any]) -> None:
        self.pos = np.asarray(state["pos"], np.int64).copy()
