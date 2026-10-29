"""Position codes at birth by the slot layout (task 00052): every patch slot's vector starts with the 2-D sine-cosine code
of its cell, every summary's with the code of its block's centre, the body and the move with none. Before the fix, slot
k + 1 got cell k's code, which interleaved summaries shifted.

Requires libvmagent.so.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from reference import torch_reference as ref  # noqa: E402
from test_model_reference import SMALL  # noqa: E402

D = 64   # wide enough that the noise (0.02) projects onto a code by ~0.035


def code(y, x, d=D):
    """The init's code of position (row y, column x) in cell units: 0.1 * [sin x w, cos x w, sin y w, cos y w]."""
    q = d // 4
    w = 10000.0 ** (-np.arange(q) / q)
    return 0.1 * np.concatenate([np.sin(x * w), np.cos(x * w), np.sin(y * w), np.cos(y * w)])


def expected_centre(cfg, t, k):
    """Slot k's centre (row, column) from the reference's slot table, or None for the body and the move."""
    g = cfg["img"] // cfg["patch"]
    kind, lv, r, c = t["kind"][k], t["level"][k], t["r"][k], t["c"][k]
    if kind == 1:
        return float(r), float(c)
    if kind != 2:
        return None
    if cfg.get("uni_sum", 0):
        return (r + 0.5) * cfg["uni_sum"] - 0.5, 0.5 * (g - 1)
    b = 2 ** lv
    return (r + 0.5) * b - 0.5, (c + 0.5) * b - 0.5


def slots_at_birth(cfg, seed=3):
    a = agent_lib.Agent(cfg, seed=seed)
    off, size = [(o, s) for n, o, s in a.param_layout() if n == "uni_slot"][0]
    nt = len(ref.uni_slot_table(cfg)["kind"])
    assert size == nt * D
    return a.params()[off:off + size].reshape(nt, D).astype(np.float64)


IMG = dict(SMALL, d_model=D, img=64, patch=8, universal=1, bins=8, gray=1)    # an 8 x 8 grid
TXT = dict(SMALL, d_model=D, img=16, patch=2, universal=1, bins=8, gray=1)    # 8 x 8 letters, as T_text_B
G4 = dict(SMALL, d_model=D, img=32, patch=8, universal=1, bins=8)             # a 4 x 4 grid, colour


@pytest.mark.parametrize("extra", [dict(), dict(uni_pyr=1), dict(uni_pyr=2), dict(uni_pyr=3), dict(uni_sum=1),
                                   dict(uni_sum=2), dict(uni_pyr=3, sum_thumb=1, attn_win=1)],
                         ids=["none", "pyr1", "pyr2", "pyr3", "rows1", "rows2", "pyr3_thumb_win"])
@pytest.mark.parametrize("base", [IMG, TXT, G4], ids=["img8", "text8", "img4"])
def test_each_slot_starts_with_its_own_position_code(base, extra):
    cfg = dict(base, **extra)
    g = cfg["img"] // cfg["patch"]
    if cfg.get("uni_pyr", 0) and g % (2 ** cfg["uni_pyr"]):
        pytest.skip("grid not divisible")
    t = ref.uni_slot_table(cfg)
    S = slots_at_birth(cfg)
    nt = len(t["kind"])
    # every centre any slot could carry: all cells, all block centres
    cand = {expected_centre(cfg, t, k) for k in range(nt)} - {None}
    cand |= {(float(r), float(c)) for r in range(g) for c in range(g)}
    C = {p: code(*p) for p in cand}
    for k in range(nt):
        p = expected_centre(cfg, t, k)
        if p is not None:
            assert S[k] @ C[p] / (C[p] @ C[p]) == pytest.approx(1.0, abs=0.2), (k, p)
        rest = S[k] - (C[p] if p is not None else 0.0)   # what is left must be the 0.02 noise alone
        assert abs(rest.std() - 0.02) < 0.006, k
        worst = max(abs(rest @ v) / (v @ v) for v in C.values())
        assert worst < 0.2, (k, p, worst)


def test_the_body_and_the_move_carry_no_position():
    cfg = dict(IMG, uni_pyr=3)
    S = slots_at_birth(cfg)
    for k in (0, len(S) - 1):
        assert abs(S[k].std() - 0.02) < 0.006
        assert max(abs(S[k] @ code(y, x)) / (code(y, x) @ code(y, x)) for y in range(8) for x in range(8)) < 0.2


def test_a_summary_starts_where_its_block_is():
    """B1(0, 0) at (0.5, 0.5), B2(0, 0) at (1.5, 1.5), the global summary at (3.5, 3.5): not at a patch's cell."""
    cfg = dict(IMG, uni_pyr=3)
    t = ref.uni_slot_table(cfg)
    S = slots_at_birth(cfg)
    first = {lv: int(np.flatnonzero((t["kind"] == 2) & (t["level"] == lv))[0]) for lv in (1, 2, 3)}
    for lv, centre in [(1, 0.5), (2, 1.5), (3, 3.5)]:
        v = code(centre, centre)
        assert S[first[lv]] @ v / (v @ v) == pytest.approx(1.0, abs=0.2)
    # the slot right after patch (1, 1) is B1(0, 0): it must not carry cell 10's code (the old init's)
    k = first[1]
    assert k == 11
    assert abs((S[k] - code(0.5, 0.5)) @ code(1, 2)) / (code(1, 2) @ code(1, 2)) < 0.2
