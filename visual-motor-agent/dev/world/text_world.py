"""TextWorld (task 00032): a text drawn as grey frames, with CircleWorld's interface, so that the universal model
lives in it unchanged.

A letter is a glyph, a cell x cell block from a table of one glyph per letter of the corpus's sorted alphabet
(encoding): level (letter k's every pixel at the grey level round(k * 255 / (n - 1)): a letter is a brightness) or
onehot (task 00033: pixel k of the cell, row by row, at 255, the rest 0; needs cell^2 >= n).
A frame is a page of (size / cell)^2 letters, written in block x block blocks of
consecutive letters (blocks in raster order, letters within a block row by row; block 1: an ordinary page, line by
line). Each step shows the next page; each body reads from its own seeded offset and wraps around at the end of its
split (train: the first train_frac of the corpus, val: the rest), or with tile from evenly spaced pages (scoring: N
readers for pages / N steps read every whole page of the split once). Moves are ignored and the drives stay 0.
With reveal (task 00056), a page fills one letter per step instead: a frame shows the page's first k letters in
reading order and the rest empty; each step adds the next letter, and after a full page the next page starts with
its first. Empty is its own grey level (0; the letters at 1..255) or, with onehot, all zeros.
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
    encoding: str = "level"       # the glyphs: level | onehot
    reveal: bool = False          # task 00056: one letter per step, the page filling up (then the next page)
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
        if self.encoding not in ("level", "onehot"):
            raise ValueError("encoding must be 'level' or 'onehot'")
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
        k0, den = (1, n) if c.reveal else (0, n - 1)                   # with reveal, level 0 is the empty cell
        self.levels = np.round((np.arange(n) + k0) * 255.0 / den).astype(np.uint8)
        if c.encoding == "level":
            self.glyphs = np.repeat(self.levels[:, None, None], c.cell, 1).repeat(c.cell, 2)   # [n, cell, cell]
        else:
            if c.cell ** 2 < n:
                raise ValueError("onehot needs cell^2 >= %d letters" % n)
            self.glyphs = np.zeros((n, c.cell * c.cell), np.uint8)
            self.glyphs[np.arange(n), np.arange(n)] = 255
            self.glyphs = self.glyphs.reshape(n, c.cell, c.cell)
        self.empty = n                                                  # the code of an empty cell (reveal)
        self.cells = np.concatenate([self.glyphs, np.zeros((1, c.cell, c.cell), np.uint8)])   # glyphs + empty
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
        self.shown = np.full(self.num_envs, c.page if not c.reveal else 1, np.int64)   # letters of the page on screen

    # ------------------------------------------------------------------ text <-> frames
    def page_codes(self, pos: np.ndarray) -> np.ndarray:
        """The letter codes of the pages starting at pos (N,): (N, page), wrapping around the split."""
        return self.codes[(np.asarray(pos)[:, None] + np.arange(self.config.page)[None]) % len(self.codes)]

    def render(self, codes: np.ndarray, shown: Optional[np.ndarray] = None) -> np.ndarray:
        """Pages of letter codes (N, page) -> frames (N, size, size, 3) uint8; with shown (N,), only each page's first
        shown letters, the rest empty."""
        c, G = self.config, self.config.side
        if shown is not None:
            codes = np.where(np.arange(c.page)[None] < np.asarray(shown)[:, None], codes, self.empty)
        grid = np.zeros((len(codes), G * G), np.int64)
        grid[:, self.grid_index] = codes
        img = self.cells[grid.reshape(-1, G, G)].transpose(0, 1, 3, 2, 4).reshape(-1, c.size, c.size)
        return np.repeat(img[..., None], 3, axis=-1)

    def decode_levels(self, values: np.ndarray) -> np.ndarray:
        """Grey levels (any shape, 0..255) -> the nearest letter's code (with reveal, level 0 is empty: code n)."""
        lev = np.append(self.levels, 0) if self.config.reveal else self.levels
        return np.abs(np.asarray(values, np.float64)[..., None] - lev.astype(np.float64)).argmin(-1)

    def decode_glyphs(self, cells: np.ndarray) -> np.ndarray:
        """Cells (..., cell, cell) of grey values -> the nearest glyph's letter code (with reveal, empty: code n)."""
        g = (self.cells if self.config.reveal else self.glyphs).reshape(-1, self.config.cell ** 2).astype(np.float64)
        x = np.asarray(cells, np.float64).reshape(cells.shape[:-2] + (1, -1))
        return ((x - g) ** 2).sum(-1).argmin(-1)

    def letters(self, images: np.ndarray) -> np.ndarray:
        """Frames (N, size, size, 3) -> the pages' letter codes in reading order (N, page): the nearest glyphs."""
        c, G = self.config, self.config.side
        cells = images[..., 0].reshape(len(images), G, c.cell, G, c.cell).transpose(0, 1, 3, 2, 4)
        return self.decode_glyphs(cells.reshape(len(images), G * G, c.cell, c.cell)[:, self.grid_index])

    def read(self, images: np.ndarray) -> list:
        """Frames (N, size, size, 3) -> the pages' text (N strings; empty cells left out)."""
        return ["".join(self.alphabet[k] for k in row if k < self.empty) for row in self.letters(images)]

    # ------------------------------------------------------------------ CircleWorld's interface
    def observe(self) -> Dict[str, np.ndarray]:
        return {"image": self.render(self.page_codes(self.pos), self.shown if self.config.reveal else None),
                "drives": np.zeros((self.num_envs, 2), np.float32)}

    def reset(self, env_ids=None) -> Dict[str, np.ndarray]:
        return self.observe()

    def step(self, actions: np.ndarray):
        if np.shape(actions) != (self.num_envs, 3):
            raise ValueError(f"expected actions of shape ({self.num_envs}, 3), got {np.shape(actions)}")
        c = self.config
        if c.reveal:   # the next letter; after a full page, the next page's first
            self.shown += 1
            full = self.shown > c.page
            self.pos = np.where(full, (self.pos + c.page) % len(self.codes), self.pos)
            self.shown[full] = 1
        else:
            self.pos = (self.pos + c.page) % len(self.codes)
        obs = self.observe()
        n, f = self.num_envs, np.zeros(self.num_envs, bool)
        nan = np.full(n, np.nan)
        info = dict(terminal_image=obs["image"], terminal_drives=obs["drives"], success=f, timeout=f.copy(),
                    click=np.asarray(actions)[:, 2] > 0.5, ep_length=nan, ep_return=nan.copy(),
                    ep_success=nan.copy(), ep_clicks=nan.copy(), gt_pos=self.pos.copy(),
                    gt_shown=self.shown.copy())
        return obs, np.zeros(n, np.float32), f.copy(), info

    def get_state(self) -> Dict[str, Any]:
        return {"pos": self.pos.copy(), "shown": self.shown.copy()}

    def set_state(self, state: Dict[str, Any]) -> None:
        self.pos = np.asarray(state["pos"], np.int64).copy()
        if "shown" in state:
            self.shown = np.asarray(state["shown"], np.int64).copy()
