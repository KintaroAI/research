"""The instinct's horizon (task 00029, online.uni_lookahead): one step equals the instinct's reading, Jacobi decoding
with patches + 1 iterations equals sequential decoding, the frame helpers invert each other, and a short lifetime with
a 3-step horizon and its frozen tests run."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
import online  # noqa: E402

# 32x32 frames: 16 patches of 8x8, 18 tokens per step
SMALL_UNI = dict(img=32, c1=4, c2=8, c3=8, feat=16, d_model=16, n_layer=2, n_head=4, d_ff=32, ctx=8, pred_img=16,
                 pc1=4, pc2=8, pred_hidden=16, use_prev_reward=1, logstd_init=-0.3, patch=8, universal=1, bins=8)


@pytest.fixture(scope="module", params=[0, 1], ids=["rgb", "gray"])
def agent(request):
    agent_lib.set_threads(4)
    return agent_lib.Agent(dict(SMALL_UNI, gray=request.param), seed=3)


def windows(cfg, S, T, M, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (S, T, cfg["img"], cfg["img"], 3), dtype=np.uint8)
    if cfg.get("gray"):
        img[..., 1] = img[..., 2] = img[..., 0]
    drv = rng.random((S, T, 2)).astype(np.float32)
    act = np.concatenate([rng.uniform(-1, 1, (S, T, 2)), rng.random((S, T, 1)) < 0.5], -1).astype(np.float32)
    es = np.zeros((S, T), np.uint8)
    es[:, 0] = 1
    cand = np.concatenate([rng.uniform(-1, 1, (S, M, 2)), rng.random((S, M, 1)) < 0.5], -1).astype(np.float32)
    return img, drv, act, es, cand


def test_frame_helpers_invert(agent):
    c = agent.config
    img = windows(c, 3, 1, 1, 0)[0][:, 0]
    pv = online.uni_patches(c, img)
    assert pv.shape == (3, 16, 64 if c.get("gray") else 192)
    np.testing.assert_array_equal(online.uni_frame(c, pv), img)


def test_one_step_is_the_instincts_reading(agent):
    img, drv, act, es, cand = windows(agent.config, 2, 8, 3, 1)
    h = online.uni_lookahead(agent, img, drv, act, es, cand, 1)
    for j in range(3):
        a2 = act.copy()
        a2[:, -1] = cand[:, j]
        nb = agent.forward_uni(img, drv, a2, es, values=False, move_probs=False, next_body=True)["next_body"]
        np.testing.assert_allclose(h[:, j], nb[:, -1, 0], atol=1e-6)


def sequential(agent, img, drv, act, es, cand, H):
    """Sequential decoding by hand: each imagined step's body, then its patches one by one."""
    c = agent.config
    T, M, R, nt = img.shape[1], cand.shape[1], agent.config["ctx"] - (H - 1), 18
    out = np.empty((img.shape[0], M))
    for s in range(img.shape[0]):
        for m in range(M):
            im = list(img[s, T - R:])
            dv = list(drv[s, T - R:])
            ac = list(act[s, T - R:])
            ac[-1] = cand[s, m]
            for _ in range(H - 1):   # one more step: the current frame as the guess, then fixed token by token
                im.append(im[-1].copy())
                dv.append(dv[-1].copy())
                ac.append(cand[s, m])
                e = np.zeros((1, len(im)), np.uint8)
                e[0, 0] = 1
                f = lambda: agent.forward_uni(np.stack(im)[None], np.stack(dv)[None], np.stack(ac)[None], e,  # noqa
                                              next_body=True)
                dv[-1] = np.clip(f()["next_body"][0, -2], 0, 1)
                pv = online.uni_patches(c, im[-1][None])
                for p in range(nt - 2):
                    pv[0, p] = f()["values"][0, -1, p]
                    im[-1] = online.uni_frame(c, pv)[0]
            e = np.zeros((1, len(im)), np.uint8)
            e[0, 0] = 1
            out[s, m] = agent.forward_uni(np.stack(im)[None], np.stack(dv)[None], np.stack(ac)[None], e, values=False,
                                          move_probs=False, next_body=True)["next_body"][0, -1, 0]
    return out


@pytest.mark.parametrize("H", [2, 3])
def test_jacobi_with_enough_iterations_is_sequential(agent, H):
    img, drv, act, es, cand = windows(agent.config, 2, 8, 2, 2)
    want = sequential(agent, img, drv, act, es, cand, H)
    got = online.uni_lookahead(agent, img, drv, act, es, cand, H, J=17)   # 16 patches + the body
    np.testing.assert_allclose(got, want, atol=1e-6)
    few = online.uni_lookahead(agent, img, drv, act, es, cand, H, J=1)
    assert np.isfinite(few).all()


def test_short_windows_and_move_sequences(agent):
    """Windows shorter than ctx keep all their steps; a held move equals the same move given as a sequence."""
    img, drv, act, es, cand = windows(agent.config, 2, 3, 2, 3)
    held = online.uni_lookahead(agent, img, drv, act, es, cand, 4)
    seq = online.uni_lookahead(agent, img, drv, act, es, np.repeat(cand[:, :, None], 4, 2), 4)
    np.testing.assert_array_equal(held, seq)
    with pytest.raises(ValueError):
        online.uni_lookahead(agent, img, drv, act, es, cand, 9)


def test_uni_act_with_a_horizon(agent):
    img, drv, act, es, _ = windows(agent.config, 3, 8, 1, 4)
    a, over, gain = online.uni_act(agent, img, drv, act, es, agent.config["bins"], np.random.default_rng(0), 6,
                                   0.005, 4)
    assert a.shape == (3, 3) and 0 <= over <= 1 and gain >= 0
    a1 = online.uni_act(agent, img, drv, act, es, agent.config["bins"], np.random.default_rng(0), 6, 0.005, 1)[0]
    a0 = online.uni_act(agent, img, drv, act, es, agent.config["bins"], np.random.default_rng(0), 6, 0.005)[0]
    np.testing.assert_array_equal(a1, a0)                             # horizon 1: the instinct as before


def test_lifetime_with_a_horizon(tmp_path):
    out = str(tmp_path / "run")
    sets = ["model.img=32", "env.size=32", "model.d_model=16", "model.n_head=2", "model.d_ff=32", "model.bins=8",
            "online.steps=40", "online.log_every=20", "online.save_every=40", "online.threads=2", "online.stm=64",
            "online.replay=2000", "online.instinct_m=3", "online.instinct_h=3", "online.device=cpu"]
    cmd = [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "U_universal.json"),
           "--seed", "2", "-o", out] + sum([["--set", s] for s in sets], [])
    subprocess.run(cmd, check=True, cwd=DEV, capture_output=True)
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert len(rows) == 2 and "instinct_override" in rows[-1]
    ev = subprocess.run([sys.executable, os.path.join(DEV, "online.py"), "--eval", out, "--ckpts", "40",
                         "--eval-horizon", "2"], cwd=DEV, capture_output=True, text=True, check=True)
    assert "horizon 2" in ev.stdout
    r = json.load(open(os.path.join(out, "eval", "ckpt_00000040_h2.json")))
    assert r["instinct_h"] == 2 and "forage_poison_per_1k" in r
    ev = subprocess.run([sys.executable, os.path.join(DEV, "online.py"), "--eval", out, "--ckpts", "40",
                         "--eval-horizon", "3", "--eval-tests", "forage", "--eval-worlds", "4"], cwd=DEV,
                        capture_output=True, text=True, check=True)
    r = json.load(open(os.path.join(out, "eval", "ckpt_00000040_h3_forage4.json")))
    assert "success" not in r and r["instinct_h"] == 3 and "forage_hits_per_1k" in r


def test_gpu_step_values_and_lookahead():
    """The GPU's one-step values equal the full forward's; the lookahead on the GPU matches the C++ model's (the
    imagined frames are rounded to u8, so a few pixels may differ)."""
    cuda_lib = pytest.importorskip("cuda_lib")
    if not cuda_lib.available() or not cuda_lib.has("vmc_forward_uni_step"):
        pytest.skip("libvmcuda.so with vmc_forward_uni_step not available")
    agent = agent_lib.Agent(dict(SMALL_UNI, gray=1), seed=3)
    m = cuda_lib.CudaModel(agent)
    img, drv, act, es, cand = windows(agent.config, 3, 6, 4, 5)
    full = m.forward_uni(img, drv, act, es, next_body=True)
    for t in (0, 5):
        o = m.forward_uni_step(img, drv, act, es, t)
        np.testing.assert_array_equal(o["values_t"], full["values"][:, t])
        np.testing.assert_array_equal(o["next_body"], full["next_body"])
    g = online.uni_lookahead(m, img, drv, act, es, cand, 3)
    c = online.uni_lookahead(agent, img, drv, act, es, cand, 3)
    np.testing.assert_allclose(g, c, atol=2e-3)
    m.close()
    agent.close()


def test_gpu_mode_decoding_matches_the_reference():
    """mode=True: each value of step t as its most probable bin's centre, as the PyTorch reference's argmax; the
    lookahead's 'mode' decoding runs on the GPU and is refused on the CPU."""
    torch = pytest.importorskip("torch")
    cuda_lib = pytest.importorskip("cuda_lib")
    if not cuda_lib.available() or not cuda_lib.has("vmc_forward_uni_step"):
        pytest.skip("libvmcuda.so with vmc_forward_uni_step not available")
    from reference import torch_reference as ref
    agent = agent_lib.Agent(dict(SMALL_UNI, gray=1), seed=3)
    c = agent.config
    m = cuda_lib.CudaModel(agent)
    img, drv, act, es, cand = windows(c, 3, 5, 4, 6)
    P = ref.params_from_flat(agent.params(), agent.param_layout(), c, dtype=torch.float64)
    with torch.no_grad():
        lg = ref.forward_uni(P, c, img, drv, act, es)["logits"].numpy()      # [S, T*nt, V, K]
    nt, V, K = ref.uni_dims(c)
    want = lg.reshape(3, 5, nt, V, K).argmax(-1) / (K - 1)
    for t in (0, 4):
        got = m.forward_uni_step(img, drv, act, es, t, mode=True)["values_t"]
        assert (np.abs(got - want[:, t]) > 1e-6).mean() < 1e-3                # (near-ties aside)
    h = online.uni_lookahead(m, img, drv, act, es, cand, 3, decode="mode")
    assert h.shape == (3, 4) and np.isfinite(h).all()
    ev = online._GpuEval(agent, m)                                           # the frozen tests' model
    np.testing.assert_array_equal(online.uni_lookahead(ev, img, drv, act, es, cand, 3, decode="mode"), h)
    with pytest.raises(ValueError):
        online.uni_lookahead(agent, img, drv, act, es, cand, 3, decode="mode")
    m.close()
    agent.close()


def pair_decode(p):
    """The reference of 'pair' decoding (task 00038): probabilities [..., K] -> the most probable bin and its more
    probable neighbour, their probability-weighted centre."""
    K = p.shape[-1]
    b = p.argmax(-1)
    lo, hi = np.take_along_axis(p, np.maximum(b - 1, 0)[..., None], -1)[..., 0], \
        np.take_along_axis(p, np.minimum(b + 1, K - 1)[..., None], -1)[..., 0]
    nb = np.where(b == 0, 1, np.where(b == K - 1, K - 2, np.where(hi > lo, b + 1, b - 1)))
    pb, pn = np.take_along_axis(p, b[..., None], -1)[..., 0], np.take_along_axis(p, nb[..., None], -1)[..., 0]
    return (pb * b + pn * nb) / ((pb + pn) * (K - 1))


def test_pair_decoding_recovers_two_hot_values():
    """A two-hot distribution decodes to its own value (mode would give the nearest bin centre); a mixture of two
    distinct values chooses the more probable one."""
    K = 32
    v = np.array([0.0, 21 / 255, 94 / 255, 0.5, 1.0])
    x = v * (K - 1)
    lo = np.floor(x).astype(int)
    p = np.zeros((len(v), K))
    p[np.arange(len(v)), lo] = 1 - (x - lo)
    p[np.arange(len(v)), np.minimum(lo + 1, K - 1)] += x - lo
    np.testing.assert_allclose(pair_decode(p), v, atol=1e-12)
    q = 0.4 * p[1] + 0.6 * p[2]                     # background 21 or circle 94: the circle, exactly
    np.testing.assert_allclose(pair_decode(q), 94 / 255, atol=1e-12)


def test_gpu_pair_decoding_matches_the_reference():
    """mode=2: the GPU's 'pair' values equal the reference formula on the PyTorch reference's probabilities; the
    lookahead accepts decode='pair' (the GPU only); other modes are refused."""
    torch = pytest.importorskip("torch")
    cuda_lib = pytest.importorskip("cuda_lib")
    if not cuda_lib.available() or not cuda_lib.has("vmc_forward_uni_step"):
        pytest.skip("libvmcuda.so with vmc_forward_uni_step not available")
    from reference import torch_reference as ref
    for in_bins in (0, 1):
        agent = agent_lib.Agent(dict(SMALL_UNI, gray=1, in_bins=in_bins), seed=3)
        c = agent.config
        m = cuda_lib.CudaModel(agent)
        img, drv, act, es, cand = windows(c, 3, 5, 4, 6)
        P = ref.params_from_flat(agent.params(), agent.param_layout(), c, dtype=torch.float64)
        with torch.no_grad():
            lg = ref.forward_uni(P, c, img, drv, act, es)["logits"]      # [S, T*nt, V, K]
        nt, V, K = ref.uni_dims(c)
        want = pair_decode(torch.softmax(lg, -1).numpy().reshape(3, 5, nt, V, K))
        for t in (0, 4):
            got = m.forward_uni_step(img, drv, act, es, t, mode=2)["values_t"]
            assert (np.abs(got - want[:, t]) > 1e-5).mean() < 1e-3            # (near-ties aside)
            mean = m.forward_uni_step(img, drv, act, es, t, mode=0)["values_t"]
            assert not np.allclose(got, mean)
        h = online.uni_lookahead(m, img, drv, act, es, cand, 3, decode="pair")
        assert h.shape == (3, 4) and np.isfinite(h).all()
        with pytest.raises(RuntimeError):
            m.forward_uni_step(img, drv, act, es, 0, mode=3)
        with pytest.raises(ValueError):
            online.uni_lookahead(agent, img, drv, act, es, cand, 3, decode="pair")
        m.close()
        agent.close()
