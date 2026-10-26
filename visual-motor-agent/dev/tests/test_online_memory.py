"""online.py gated memory (vm-00007): hindsight labels, the salience gate, chunked long-term memory, surprise.

Requires libvmagent.so (make libvmagent.so); torch only for the reference downsampling.
"""
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

L, K = 8, 12


def make_stream(n, seed, p_hit=0.04, p_timeout=0.01):
    """Episodes end with a hit (hit = done = 1) or a timeout (done = 1). Returns hit, done, hunger [n]."""
    rng = np.random.default_rng(seed)
    hit = np.zeros(n, np.uint8)
    done = np.zeros(n, np.uint8)
    for j in range(n):
        u = rng.random()
        if u < p_hit:
            hit[j] = done[j] = 1
        elif u < p_hit + p_timeout:
            done[j] = 1
    return hit, done, rng.random(n).astype(np.float32)


def brute_labels(hit, done, value, k):
    """value[j] if j's episode ends with a hit within k steps of j (j itself counts), else 0."""
    out = np.zeros(len(hit), np.float32)
    for j in range(len(hit)):
        e = j
        while e < len(hit) - 1 and not done[e]:
            e += 1
        if done[e] and hit[e] and e - j < k:
            out[j] = value[j]
    return out


def test_hindsight_labels_stop_at_the_episode_end():
    hit, done, h = make_stream(3000, 0)
    lab = online.hindsight_labels(hit, done, h, K)
    assert np.array_equal(lab, brute_labels(hit, done, h, K))
    # the stream memory's look-ahead (Replay.fill) also counts hits of the next episode
    ahead = np.array([hit[j:j + K].any() for j in range(len(hit))])
    differ = (lab > 0) != ahead
    assert differ.any()
    for j in np.flatnonzero(differ):
        e = j + np.argmax(done[j:])         # j's episode ends with a timeout, and a hit follows within K steps
        assert not hit[e] and hit[e + 1:j + K].any() and lab[j] == 0


def gated(**kw):
    on = dict(online.DEFAULT_ONLINE, memory="gated", stm=64, replay=L * 400, relief_k=K, store_relief=0,
              store_surprise=0.0, store_base=1.0, store_warmup=0, surprise_window=200)
    on.update(kw)
    return online.GatedMemory(on, L, 8, K, seed=3)


def feed(gm, hit, done, h, parts=None):
    """Steps 1..n into the gated memory; step j's id is stored in drv[:, 1] and its first pixel."""
    s = gm.stm
    for j in range(len(hit)):
        img = np.full((8, 8, 3), j % 256, np.uint8)
        s.add(img=img, nimg=img, drv=np.array([h[j], j], np.float32), ndrv=np.array([h[j], j + 1], np.float32),
              pa=np.zeros(3), pr=0.0, es=0, act=np.zeros(3), rew=0.0, done=done[j], hit=hit[j], cmd=0.0)
        gm.note(parts[j] if parts is not None else (1.0, 1.0, 1.0))
        gm.decide(j + 1)


def test_store_all_keeps_every_chunk_in_order_with_relief_and_labels():
    n = 400 * L
    hit, done, h = make_stream(n, 1)
    gm = gated(store_relief=1)
    feed(gm, hit, done, h)
    D = K - 1
    n_dec = (n - D) // L
    assert gm.n == n_dec
    lab = brute_labels(hit, done, h, K)
    rel = brute_labels(hit, done, np.ones(n), K) > 0
    for m in range(n_dec):
        ids = gm.ltm["drv"][m, :, 1]
        assert np.array_equal(ids, np.arange(m * L, (m + 1) * L)), m       # chunk m = steps mL+1 .. (m+1)L
        assert gm.relief[m] == rel[m * L:(m + 1) * L].any()
        assert gm.reason[m] == (2 if gm.relief[m] else 4)
        assert np.allclose(gm.ltm["lbl"][m], lab[m * L:(m + 1) * L])
    # training rows: the newest window (ending K - 1 steps back), then whole chunks
    r = gm.rows(6, np.random.default_rng(0), D)
    ids = r["drv"][:, :, 1]
    assert np.array_equal(ids[0], np.arange(n - D - L, n - D))
    for row in ids[1:]:
        assert row[0] % L == 0 and np.array_equal(row, row[0] + np.arange(L))
    new_lab = online.hindsight_labels(hit, done, h, K)[n - D - L:n - D]
    assert np.allclose(r["pr"][0], new_lab)
    for b in range(1, 6):
        m = int(ids[b, 0]) // L
        assert np.allclose(r["pr"][b], lab[m * L:(m + 1) * L])


def test_gate_reasons_and_rates():
    n = 600 * L
    hit, done, h = make_stream(n, 2)
    # relief only: relief chunks always, others with probability store_base
    gm = gated(store_relief=1, store_base=0.25, store_warmup=40 * L)
    feed(gm, hit, done, h)
    reasons = gm.reason[:gm.n]
    assert (reasons[:40] == 1).all()                             # warm-up chunks
    later = reasons[40:]
    assert set(np.unique(later)) <= {2, 4}
    assert gm.relief[:gm.n][40:][later == 2].all()
    n_dec = (n - (K - 1)) // L
    rel = brute_labels(hit, done, np.ones(n), K) > 0
    n_rel = sum(rel[m * L:(m + 1) * L].any() for m in range(40, n_dec))
    assert (later == 2).sum() == n_rel
    frac_base = (later == 4).sum() / (n_dec - 40 - n_rel)
    assert 0.18 < frac_base < 0.32


def test_surprise_gate_stores_the_top_fraction():
    n = 3000 * L
    hit = done = np.zeros(n, np.uint8)
    rng = np.random.default_rng(4)
    parts = np.stack([rng.exponential(1.0, n), rng.exponential(1.0, n), np.full(n, np.nan)], 1)
    gm = gated(store_surprise=0.1, store_base=0.0, surprise_window=1000)
    feed(gm, hit, done, np.zeros(n), parts)
    frac = gm.n / ((n - (K - 1)) // L)
    assert 0.08 < frac < 0.12
    assert (gm.reason[:gm.n] == 3).all()


def test_combined_surprise_is_the_sum_of_normalized_parts():
    gm = gated(surprise_window=5)
    seq = [(1.0, 2.0, np.nan), (3.0, 2.0, np.nan), (2.0, 8.0, np.nan), (6.0, 1.0, np.nan)]
    hist = [[], []]
    for j, p in enumerate(seq):
        gm.stm.add(img=np.zeros((8, 8, 3), np.uint8), nimg=np.zeros((8, 8, 3), np.uint8), drv=np.zeros(2),
                   ndrv=np.zeros(2), pa=np.zeros(3), pr=0.0, es=0, act=np.zeros(3), rew=0.0, done=0, hit=0, cmd=0.0)
        gm.note(p)
        hist[0].append(p[0])
        hist[1].append(p[1])
        want = p[0] / np.mean(hist[0]) + p[1] / np.mean(hist[1])
        assert np.isclose(gm.sur[j], want)
        assert gm.top[j] == int(np.argmax([p[0] / np.mean(hist[0]), p[1] / np.mean(hist[1]), 0]))


def test_token_surprise_is_the_largest_slot_ratio():
    """The universal model's surprise (task 00028): the largest ratio of a token's cross-entropy to its slot's recent
    mean, its kind recorded; an unmeasured step counts 0; the surprise gate then stores by it."""
    gm = online.GatedMemory(dict(online.DEFAULT_ONLINE, memory="gated", stm=64, replay=L * 400, relief_k=K,
                                 store_relief=0, store_surprise=0.5, store_base=0.0, store_warmup=0,
                                 surprise_window=3), L, 8, K, seed=3, tokens=True)
    kinds = np.array([0, 0, 1, 2])
    seq = [np.array([1.0, 2.0, 1.0, 1.0]), np.array([1.0, 2.0, 1.0, 5.0]), None, np.array([3.0, 2.0, 1.0, 1.0]),
           np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 8.0, 1.0, 1.0])]
    rows = []
    for j, ce in enumerate(seq):
        gm.stm.add(img=np.zeros((8, 8, 3), np.uint8), nimg=np.zeros((8, 8, 3), np.uint8), drv=np.zeros(2),
                   ndrv=np.zeros(2), pa=np.zeros(3), pr=0.0, es=0, act=np.zeros(3), rew=0.0, done=0, hit=0, cmd=0.0)
        gm.note_tokens(ce, kinds)
        if ce is None:
            assert gm.sur[j] == 0.0
            continue
        rows = (rows + [ce])[-3:]                                    # the window: the last 3 measured steps
        z = ce / np.mean(rows, 0)
        assert np.isclose(gm.sur[j], z.max()) and gm.top[j] == kinds[np.argmax(z)]
    assert np.isclose(gm.sur[1], 5.0 / 3.0) and gm.top[1] == 2       # the body token (the drives after the move)
    assert gm.top[5] == 0                                             # a patch
    m = gm.metrics()
    assert set(k for k in m if k.startswith("surprise_mean_")) == {"surprise_mean_" + p for p in online.TOKEN_KINDS}
    assert np.isclose(m["surprise_mean_move"], 1.0)


def test_token_surprise_gate_stores_the_top_fraction():
    n = 3000 * L
    rng = np.random.default_rng(5)
    gm = online.GatedMemory(dict(online.DEFAULT_ONLINE, memory="gated", stm=64, replay=L * 4000, relief_k=K,
                                 store_relief=0, store_surprise=0.1, store_base=0.0, store_warmup=0,
                                 surprise_window=1000), L, 8, K, seed=3, tokens=True)
    kinds = np.array([0] * 6 + [1, 2])
    for j in range(n):
        gm.stm.add(img=np.zeros((8, 8, 3), np.uint8), nimg=np.zeros((8, 8, 3), np.uint8), drv=np.zeros(2),
                   ndrv=np.zeros(2), pa=np.zeros(3), pr=0.0, es=0, act=np.zeros(3), rew=0.0, done=0, hit=0, cmd=0.0)
        gm.note_tokens(rng.exponential(1.0, 8) * np.arange(1, 9), kinds)
        gm.decide(j + 1)
    frac = gm.n / ((n - (K - 1)) // L)
    assert 0.08 < frac < 0.12 and (gm.reason[:gm.n] == 3).all()
    c = gm.metrics()
    assert sum(c["surprise_top_" + p] for p in online.TOKEN_KINDS) == c["stored_surprise"]


SMALL = dict(img=32, c1=4, c2=8, c3=8, feat=16, d_model=16, n_layer=2, n_head=4, d_ff=32, ctx=8,
             pred_img=16, pc1=4, pc2=8, pred_hidden=16, use_prev_reward=1, logstd_init=-0.3, coord=1, pb=8,
             pred_residual=1, pred_skip=1)


def test_downsample_matches_the_reference():
    ref = pytest.importorskip("reference.torch_reference")
    img = np.random.default_rng(5).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    assert np.array_equal(online.downsample_u8(img, 2), ref.downsample_u8(img, 2))


def test_surprise_parts_match_a_direct_computation():
    agent = agent_lib.Agent(SMALL, seed=1)
    rng = np.random.default_rng(6)
    T = 5
    img = rng.integers(0, 256, (1, T, 32, 32, 3), dtype=np.uint8)
    out = agent.forward(img, rng.random((1, T, 2)), rng.uniform(-1, 1, (1, T, 3)), rng.random((1, T)),
                        np.eye(1, T, dtype=np.uint8), pred_actions=rng.uniform(-1, 1, (1, T, 3)))
    nimg = img[0, T - 1].copy()
    nimg[4:12, 6:14] = 200                     # some pixels change
    ndrv = rng.random(2).astype(np.float32)
    sd = np.array([0.5, 0.7])
    for click in (0.0, 1.0):
        act = np.array([0.3, -0.8, click], np.float32)
        frame, drives, move = online.surprise_parts(out, T - 1, img[0, T - 1], nimg, ndrv, act, sd, 2, 10.0)
        se = 0.0
        for y in range(16):
            for x in range(16):
                c = img[0, T - 1, 2 * y:2 * y + 2, 2 * x:2 * x + 2].astype(int).sum((0, 1))
                g = nimg[2 * y:2 * y + 2, 2 * x:2 * x + 2].astype(int).sum((0, 1))
                c, g = (c + 2) // 4, (g + 2) // 4
                wgt = 11.0 if (c != g).any() else 1.0
                se += wgt * (((out["pred_frames"][0, T - 1, y, x] - g / 255.0) ** 2).sum())
        assert np.isclose(frame, se / (16 * 16 * 3), rtol=1e-5)
        assert np.isclose(drives, ((out["pred_drives"][0, T - 1] - ndrv) ** 2).sum(), rtol=1e-5)
        p = out["click_p"][0, T - 1]
        mv = 0.5 * (((act[:2] - out["mu"][0, T - 1]) / sd) ** 2).sum() - np.log(p if click else 1 - p)
        assert np.isclose(move, mv, rtol=1e-5)
        f2, d2, m2 = online.surprise_parts(out, T - 1, img[0, T - 1], nimg, ndrv, act, None, 2, 10.0, move=False)
        assert (f2, d2) == (frame, drives) and np.isnan(m2)
    agent.close()


@pytest.mark.parametrize("config,extra", [
    ("A_basic.json", ["online.goal=1", "model.use_prev_reward=1", "online.batch=4", "online.explore_eps=0.2"]),
    ("L_online.json", ["online.batch=4"]),
])
def test_gated_run_smoke(tmp_path, config, extra):
    sets = ["online.memory=gated", "online.steps=60", "online.log_every=30", "online.save_every=1000000",
            "online.replay=800", "online.stm=64", "online.store_warmup=16", "online.store_relief=1",
            "online.store_surprise=0.2", "online.store_base=0.3", "online.threads=4"] + extra
    cmd = [sys.executable, "online.py", "--config", os.path.join("configs", config), "-o", str(tmp_path)]
    for s in sets:
        cmd += ["--set", s]
    subprocess.run(cmd, cwd=DEV, check=True, capture_output=True)
    rows = [json.loads(x) for x in open(tmp_path / "metrics.jsonl")]
    assert len(rows) == 2
    r = rows[-1]
    assert r["chunks"] > 0 and 0 < r["ltm_chunks"] <= 800 // L
    assert r["stored"] == r["stored_warmup"] + r["stored_relief"] + r["stored_surprise"] + r["stored_base"]
    assert all(np.isfinite(r[k]) for k in ("pred_mse", "drive_mse", "surprise_mean_frame", "surprise_mean_drives"))
    if config == "A_basic.json":
        assert np.isfinite(r["surprise_mean_move"]) and r["ltm_label_share"] is not None
    else:
        assert r["surprise_mean_move"] is None and np.isfinite(r["dyn_h1_mse"])
