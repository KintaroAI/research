"""The text world (task 00032): pages of letters as grey frames, read by the unchanged universal model.

Requires libvmagent.so; torch for the reference model.
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
from world import TextConfig, TextWorld, make_config, make_world  # noqa: E402

TEXT = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n"
        "First Citizen:\nYou are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n") * 40

LAYOUTS = [dict(size=16, cell=2, block=1), dict(size=8, cell=1, block=8), dict(size=16, cell=1, block=4),
           dict(size=64, cell=1, block=8)]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    p = tmp_path_factory.mktemp("text") / "corpus.txt"
    p.write_text(TEXT)
    return str(p)


def world(corpus, n=3, seed=0, **kw):
    return TextWorld(n, TextConfig(corpus=corpus, **kw), seed=seed)


def test_alphabet_and_levels(corpus):
    w = world(corpus)
    assert w.alphabet == "".join(sorted(set(TEXT)))
    n = len(w.alphabet)
    assert w.levels[0] == 0 and w.levels[-1] == 255 and len(set(w.levels.tolist())) == n
    assert np.allclose(w.levels, np.round(np.arange(n) * 255 / (n - 1)))
    assert (w.decode_levels(w.levels) == np.arange(n)).all()


@pytest.mark.parametrize("kw", LAYOUTS, ids=lambda k: "s%d_c%d_b%d" % (k["size"], k["cell"], k["block"]))
def test_pages_read_back_as_the_text_and_continue_it(corpus, kw):
    w = world(corpus, **kw)
    cut = int(len(TEXT) * 0.9)
    train = TEXT[:cut]
    pages = [w.read(w.observe()["image"])]
    for _ in range(3):
        obs, rew, done, inf = w.step(np.zeros((3, 3), np.float32))
        pages.append(w.read(obs["image"]))
        assert (inf["terminal_image"] == obs["image"]).all() and not done.any() and (obs["drives"] == 0).all()
    start = w.pos - 3 * w.config.page
    for b in range(3):
        got = "".join(p[b] for p in pages)
        want = "".join(train[(start[b] + i) % len(train)] for i in range(4 * w.config.page))
        assert got == want
    img = w.observe()["image"]
    assert img.shape == (3, kw["size"], kw["size"], 3) and (img[..., 0] == img[..., 2]).all()
    c = kw["cell"]
    if c > 1:   # each letter a cell x cell block of one level
        blocks = img[..., 0].reshape(3, kw["size"] // c, c, kw["size"] // c, c)
        assert (blocks == blocks[:, :, :1, :, :1]).all()


def test_the_model_sees_consecutive_letters_in_its_token_order(corpus):
    """Arm A: an 8x8 page in one 8x8 patch: value i of the token is letter i. Arm B: a 16x16 page of 2x2 letters,
    2x2 patches: token j (row by row) is letter j, its 4 values the same."""
    from reference import torch_reference as ref
    for kw, patch in ((dict(size=8, cell=1, block=8), 8), (dict(size=16, cell=2, block=1), 2)):
        w = world(corpus, n=2, **kw)
        codes = w.page_codes(w.pos)                                       # [2, 64] letters in reading order
        cfg = dict(img=kw["size"], patch=patch, gray=1, bins=256)
        v, used = ref.uni_values(cfg, w.observe()["image"][:, None], np.zeros((2, 1, 2)), np.zeros((2, 1, 3)))
        lv = w.levels[codes] / 255.0
        pv = v[:, 0, 1:-1]                                                # the patch tokens' values
        if patch == 8:
            assert np.allclose(pv[:, 0], lv)
        else:
            assert np.allclose(pv, np.repeat(lv[..., None], 4, -1))


def test_splits_tiles_and_state(corpus):
    tr, va = world(corpus, split="train"), world(corpus, split="val")
    assert len(tr.codes) + len(va.codes) == len(TEXT) and len(tr.codes) == int(len(TEXT) * 0.9)
    assert "".join(va.alphabet[k] for k in va.codes) == TEXT[len(tr.codes):]
    t = world(corpus, n=4, split="val", tile=True)
    pages = len(t.codes) // t.config.page
    assert (t.pos == np.arange(4) * pages // 4 * t.config.page).all()
    s = t.get_state()
    a = t.step(np.zeros((4, 3), np.float32))[0]["image"]
    t.set_state(s)
    assert (t.step(np.zeros((4, 3), np.float32))[0]["image"] == a).all()
    assert (world(corpus, seed=5).pos == world(corpus, seed=5).pos).all()
    assert (world(corpus, seed=5).pos != world(corpus, seed=6).pos).any()


def test_factory_and_validation(corpus):
    assert isinstance(make_world(2, make_config(dict(kind="text", corpus=corpus)), seed=0), TextWorld)
    c = make_config(dict(kind="text", corpus=corpus, size=16), auto_reset=True)
    assert c.auto_reset and c.page == 64
    assert type(make_world(1, make_config({}), seed=0)).__name__ == "CircleWorld"
    for bad in (dict(size=10, cell=4), dict(size=16, cell=2, block=3), dict(split="test")):
        with pytest.raises(ValueError):
            make_config(dict(kind="text", corpus=corpus, **bad))


SMALL_TEXT = dict(img=16, patch=2, gray=1, bins=256, universal=1, ctx=4, pred_img=8, d_model=32, n_layer=1, n_head=2,
                  d_ff=64)


def small_run(tmp_path, corpus, steps=40):
    cfg = json.load(open(os.path.join(DEV, "configs", "T_text_B.json")))
    cfg["env"]["corpus"] = corpus
    cfg["model"].update(SMALL_TEXT)
    cfg["online"].update(steps=steps, save_every=steps, log_every=steps // 2, device="cpu", replay=400, stm=64,
                         store_warmup=0)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(cfg))
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, "lifetime.py", "--config", str(p), "--seed", "1", "-o", str(out)], cwd=DEV,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return out


def test_lifetime_in_the_text_world_and_eval_predict(tmp_path, corpus):
    """A short lifetime runs unchanged in the text world; eval_predict.py's patch cross-entropy is the reference
    loss's patch-target tokens on the same windows; char_gpt's val windows are the same letters."""
    torch = pytest.importorskip("torch")
    import eval_predict
    from reference import char_gpt, torch_reference as ref
    out = small_run(tmp_path, corpus)
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert rows[-1]["step"] == 40 and np.isfinite(rows[-1]["uni_ce_patch"])
    r = subprocess.run([sys.executable, "eval_predict.py", str(out), "--ckpts", "40", "final", "--env-set", "split=val",
                        "--env-set", "tile=true", "--bodies", "2"], cwd=DEV, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.load(open(out / "eval" / "ckpt_00000040_predict.json"))
    agent = agent_lib.Agent.load(str(out / "ckpt_00000040.bin"))
    cfg = agent.config
    info = json.load(open(out / "info.json"))
    w = make_world(2, make_config(dict(info["config"]["env"], split="val", tile=True)), seed=12345)
    steps = res["steps"]
    ro = eval_predict.rollout(agent, w, steps, np.random.default_rng(12345))
    win = eval_predict.windows(ro, cfg["ctx"])
    P = ref.params_from_flat(agent.params(), agent.param_layout(), cfg, dtype=torch.float64)
    _, t = ref.uni_loss(P, cfg, dict(images=win["img"], drives=win["drv"], action=win["act"], ep_start=win["es"],
                                     next_drives=win["ndrv"]))
    nt = (cfg["img"] // cfg["patch"]) ** 2 + 2
    assert abs(res["ce_patch"] - t["per_token"][:, :, :nt - 2].mean().item()) < 1e-4
    assert abs(res["ce_body"] - t["per_token"][:, :, nt - 1].mean().item()) < 1e-4
    assert len(res["ce_patch_by_value"]) == 4 and len(res["ce_patch_by_step"]) == cfg["ctx"]
    # the reference's val windows: the same letters, in the same windows
    letters = np.stack([w.decode_levels(win["img"][:, t, ::2, ::2, 0].reshape(len(win["img"]), -1)[:, w.grid_index])
                        for t in range(cfg["ctx"])], 1).reshape(len(win["img"]), -1)
    cg = char_gpt.val_windows(dict(info["config"]["env"]), 2, cfg["ctx"])
    assert (cg == letters).all()


def test_char_gpt_starts_at_ln_vocab(corpus):
    torch = pytest.importorskip("torch")
    from reference import char_gpt
    m = dict(d_model=32, n_layer=1, n_head=2, d_ff=64)
    P = char_gpt.init_params(40, 64, 32, 1, 64, torch.Generator().manual_seed(0), "cpu")
    win = np.random.default_rng(0).integers(0, 40, (8, 64))
    v, by = char_gpt.evaluate(P, m, win, "cpu")
    assert abs(v - np.log(40)) < 0.05 and by.shape == (63,)
