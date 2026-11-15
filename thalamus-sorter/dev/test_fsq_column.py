"""Tests for the FSQ column and the correlation sign option (ts-00028).

Run: ./venv/bin/python test_fsq_column.py   (or with pytest, if installed)
"""

import numpy as np
import torch

from column_manager import FSQColumn
from solvers.drift_torch import DriftSolver


def make_column(m=3, inputs=6, levels=(8, 5, 5, 5), lr=1e-3, **kw):
    col = FSQColumn(m, n_outputs=len(levels), max_inputs=inputs, window=4, lr=lr,
                    levels=levels, predict_k=2, hidden=16, n_samples=3,
                    report_every=0, **kw)
    for c in range(m):
        for s in range(inputs):
            col.wire(c, c * inputs + s)
    return col


def test_fsq_values_lie_on_the_level_grid():
    col = make_column()
    z = torch.randn(2, 500, 4, device=col._device) * 5
    codes, idx = col.fsq(z)
    for j, L in enumerate(col.levels):
        half = L // 2
        expect = (np.arange(L) - half) / half
        got = np.unique(codes[..., j].cpu().numpy().round(6))
        assert set(got) <= set(expect.round(6)), (L, got)
        assert len(got) == L, f"level count {L}: only {got} reached"
        i = idx[..., j].cpu().numpy()
        assert i.min() >= 0 and i.max() <= L - 1
        np.testing.assert_allclose(codes[..., j].cpu().numpy(),
                                   (i - half) / half, atol=1e-6)
    assert codes.abs().max() <= 1.0 + 1e-6


def test_fsq_gradient_passes_straight_through():
    col = make_column()
    z = torch.zeros(1, 1, 4, device=col._device, requires_grad=True)
    codes, _ = col.fsq(z)
    codes.sum().backward()
    assert torch.all(z.grad > 0)


def test_decoder_reads_only_the_codes():
    col = make_column()
    n = col.m * col.max_inputs
    x1 = torch.rand(col.m, 1, col.max_inputs * col.enc_window, device=col._device)
    x2 = torch.rand_like(x1)
    with torch.no_grad():
        c1, _ = col.encode(x1)
        p1 = col.decode(c1)
        p_same = col.decode(c1.clone())
    torch.testing.assert_close(p1, p_same)
    assert col.decode.__code__.co_argcount == 2      # (self, codes): no other input
    assert n > 0 and x2.shape == x1.shape


def test_tick_outputs_codes_and_learns():
    rng = np.random.default_rng(0)
    m, inputs = 4, 8
    col = make_column(m=m, inputs=inputs, lr=3e-3)
    n = m * inputs
    T = 3000
    # Each input is a slow sine with its own phase: predictable two ticks ahead.
    phase = rng.uniform(0, 2 * np.pi, n)
    t = np.arange(T)
    sig = (0.5 + 0.4 * np.sin(0.15 * t[None, :] + phase[:, None])).astype(np.float32)
    first = last = None
    for tick in range(col.window, T):
        window = sig[:, tick - np.arange(col.window)]        # col 0 = newest
        col.tick(window)
        if tick == col.window + 50:
            first = col.loss_ema['model']
    last = col.loss_ema['model']
    out = col.get_outputs()
    assert out.shape == (m, 4)
    assert np.all(np.abs(out) <= 1.0)
    assert last < first * 0.5, (first, last)
    assert last < col.loss_ema['mean'], col.loss_ema


def test_lr_zero_freezes_the_weights():
    col = make_column(lr=0.0)
    before = [p.detach().clone() for p in col._params]
    sig = np.random.rand(col.m * col.max_inputs, col.window).astype(np.float32)
    for _ in range(5):
        col.tick(sig)
    for b, p in zip(before, col._params):
        torch.testing.assert_close(b, p.detach())


def test_save_and_load_round_trip(tmp_path):
    col = make_column()
    sig = np.random.rand(col.m * col.max_inputs, col.window).astype(np.float32)
    col.tick(sig)
    col.save(str(tmp_path))
    state = torch.load(tmp_path / "column_states.pt", weights_only=True)
    col2 = make_column()
    col2.load_state(state, np.load(tmp_path / "column_slot_map.npy"))
    for a, b in zip(col._params, col2._params):
        torch.testing.assert_close(a.detach(), b.detach())
    col.tick(sig)
    col2.tick(sig)
    # Same weights after one more identical update from the same optimizer state
    np.testing.assert_allclose(col.get_outputs(), col2.get_outputs(), atol=1e-5)


def test_corr_sign_rule():
    score = torch.tensor([[0.9, -0.9, 0.2, -0.2]])
    pos = DriftSolver._corr_mask(score, 0.5, 'pos', False)
    ab = DriftSolver._corr_mask(score, 0.5, 'abs', True)
    auto_deriv = DriftSolver._corr_mask(score, 0.5, 'auto', True)
    auto_plain = DriftSolver._corr_mask(score, 0.5, 'auto', False)
    assert pos.tolist() == [[True, False, False, False]]
    assert ab.tolist() == [[True, True, False, False]]
    assert auto_deriv.tolist() == pos.tolist()
    assert auto_plain.tolist() == ab.tolist()


if __name__ == "__main__":
    import pathlib
    import tempfile
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d))
        else:
            fn()
        print("ok", name)
    print(f"{len(tests)} passed")
