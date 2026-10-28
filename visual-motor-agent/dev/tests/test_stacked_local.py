"""Two stacked boxes with backprop cut between them (task 00045, reference/stacked_local.py): no gradient crosses the
cut except end to end; the difference target with an exact inverse; the gradient at the cut; every arm learns a toy
sequence. Requires torch (CPU) in the venv."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reference import stacked_local as sl  # noqa: E402

V, CTX = 5, 16


def toy(S=4, seed=0):
    """Windows of the repeating sequence 0 1 2 3 4 0 1 ... from random phases: one-hot inputs, next-letter targets."""
    rng = np.random.default_rng(seed)
    w = (rng.integers(0, V, (S, 1)) + np.arange(CTX + 1)[None]) % V
    w = torch.tensor(w)
    return torch.nn.functional.one_hot(w[:, :-1], V).float(), w[:, 1:]


@pytest.mark.parametrize("arm", sl.ARMS)
def test_only_end_to_end_passes_gradient_through_the_cut(arm):
    stack = sl.Stack(arm, V, CTX, seed=1, device="cpu")
    x, y = toy()
    L, _ = stack.losses(x, y, diag=True)
    L["top"].backward()
    g = [p.grad for p in stack.mods["bottom"].values()]
    crossed = any(t is not None and t.abs().sum() > 0 for t in g)
    assert crossed == (arm == "e2e")
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stack.mods["top"].values())


def test_difference_target_with_an_exact_inverse_reaches_the_output_target():
    rng = np.random.default_rng(2)
    A = torch.tensor(rng.normal(size=(V, V)) + 3 * np.eye(V))
    b = torch.tensor(rng.normal(size=V))
    f = lambda h: h @ A.T + b                                          # noqa: E731
    g = lambda y: torch.linalg.solve(A, (y - b).T).T                   # noqa: E731  the exact inverse
    h = torch.tensor(rng.normal(size=(7, V)))
    y_hat = f(h)
    t = sl.output_target(y_hat, torch.tensor(rng.integers(0, V, 7)), beta=0.5)
    torch.testing.assert_close(f(sl.dtp_target(h, y_hat, t, g)), t)
    gb = lambda y: g(y) + 1.0                                          # noqa: E731  a biased inverse: the bias cancels
    torch.testing.assert_close(f(sl.dtp_target(h, y_hat, t, gb)), t)


def test_gradient_at_the_cut_matches_finite_differences():
    stack = sl.Stack("dni", V, CTX, seed=3, device="cpu")
    x, y = toy(seed=1)
    h = stack.bottom(x).detach()
    loss, g = sl.grad_at_cut(stack.top, h, y)
    d = torch.randn_like(h)
    d /= d.norm()
    eps = 1e-2
    with torch.no_grad():
        fd = (sl.ce(stack.top(h + eps * d), y) - sl.ce(stack.top(h - eps * d), y)) / (2 * eps)
    assert abs(fd.item() - (g * d).sum().item()) < 0.05 * abs(fd.item()) + 1e-6


@pytest.mark.parametrize("arm", [a for a in sl.ARMS if a != "dni"])
def test_every_arm_learns_a_toy_sequence(arm):
    torch.manual_seed(0)
    stack = sl.Stack(arm, V, CTX, seed=4, device="cpu", lr=2e-3)
    x, y = toy(S=8, seed=5)
    first = stack.step(x, y)["top"]
    for _ in range(60):
        last = stack.step(x, y)["top"]
    assert last < 0.7 * first, (arm, first, last)


def test_the_gradient_predictor_learns_the_true_gradient():
    """dni: with both boxes held fixed, M's prediction comes to point along the top loss's gradient at the cut (on the
    toy the top soon reaches zero loss, so the arm's own learning is not a test of M)."""
    torch.manual_seed(0)
    stack = sl.Stack("dni", V, CTX, seed=4, device="cpu", lr=2e-3)
    for k in ("bottom", "top"):
        for g in stack.opt[k].param_groups:
            g["lr"] = 0.0
    x, y = toy(S=8, seed=5)
    cos0 = stack.step(x, y, diag=True)["cos_sg"]
    for _ in range(150):
        cos = stack.step(x, y, diag=True)["cos_sg"]
    assert cos > 0.8 and cos > cos0 + 0.3, (cos0, cos)


@pytest.mark.parametrize("arm", sl.ARMS)
def test_tie_shares_one_matrix_and_the_top_shapes_it(arm):
    """--tie: the bottom's output map is the top's input map transposed (one tensor); in the cut arms the top's loss
    reaches that matrix but none of the bottom's own parameters."""
    stack = sl.Stack(arm, V, CTX, seed=1, device="cpu", tie=True)
    assert "out_w" not in stack.mods["bottom"] and "in_w" not in stack.mods["top"]
    assert "cut" in stack.opt                                         # it learns in every arm, also frozen
    x, y = toy()
    h0 = stack.bottom(x).detach()
    L, _ = stack.losses(x, y)
    L["top"].backward()
    assert stack.mods["cut"]["w"].grad.abs().sum() > 0
    crossed = any(p.grad is not None and p.grad.abs().sum() > 0 for p in stack.mods["bottom"].values())
    assert crossed == (arm == "e2e")
    with torch.no_grad():   # (random: a constant shift cancels, the bottom's last layer reads a zero-mean LayerNorm)
        stack.mods["cut"]["w"].add_(0.1 * torch.randn_like(stack.mods["cut"]["w"]))
    assert (stack.bottom(x) - h0).abs().max() > 1e-3                 # the shared matrix is the bottom's output map


def test_cut_norm_bounds_h():
    stack = sl.Stack("dtp", V, CTX, seed=1, device="cpu", cut_norm=True)
    with torch.no_grad():
        for p in stack.mods["bottom"].values():
            p.mul_(3.0)                                               # a bottom grown large
        h = stack.bottom(toy()[0])
    torch.testing.assert_close(h.mean(-1), torch.zeros(h.shape[:2]), atol=1e-5, rtol=0)
    torch.testing.assert_close(h.var(-1, unbiased=False), torch.ones(h.shape[:2]), atol=1e-3, rtol=0)

