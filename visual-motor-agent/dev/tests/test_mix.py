"""Task 00136 (online.mix): the model's own prediction of every frame mixed into what it sees of it. The frames as
seen (online.mix_inputs): what is taken from where, that no real frame enters its own prediction, a window made in
parts against one made whole, a frame with nothing before it. A short life with the mix: the model is never shown a
whole real frame, the metrics, what it is refused with. The exam shown as the life saw frames (wm_measures.Pieces)."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import agent_lib  # noqa: E402
import online  # noqa: E402
from test_lookahead import SMALL_UNI, windows  # noqa: E402

cuda_lib = pytest.importorskip("cuda_lib")
if not cuda_lib.available() or not cuda_lib.has("vmc_forward_uni_step") or not cuda_lib.has("vmc_set_uni_inputs"):
    pytest.skip("libvmcuda.so with vmc_forward_uni_step and vmc_set_uni_inputs not available", allow_module_level=True)


@pytest.fixture(scope="module", params=[dict(in_bins=1), dict(in_bins=1, age_key=1)], ids=["steps", "ages"])
def model(request):
    agent = agent_lib.Agent(dict(SMALL_UNI, gray=1, **request.param), seed=5)
    m = cuda_lib.CudaModel(agent)
    yield m
    m.close()
    agent.close()


def test_a_pixel_is_the_predictions_or_the_real_one(model):
    c = model.config
    T = c["ctx"]
    img, drv, act, es, _ = windows(c, 3, T, 1, 2)
    img0, drv0, act0 = img.copy(), drv.copy(), act.copy()
    seen, hit, pred = online.mix_inputs(model, img, drv, act, es, np.random.default_rng(7), 0.5)
    assert seen.dtype == np.uint8 and seen.shape == img.shape and hit.shape == img.shape[:4] and pred.shape == img.shape
    np.testing.assert_array_equal(seen, np.where(hit[..., None], pred, img))
    assert 0.47 < hit.mean() < 0.53 and all(0.4 < hit[:, s].mean() < 0.6 for s in range(T))    # the first frame too
    assert (pred != img).mean() > 0.9                              # an untrained model's prediction is no real frame
    for a, b in ((img, img0), (drv, drv0), (act, act0)):           # what it was given stays as it was
        np.testing.assert_array_equal(a, b)
    whole, hit1, pred1 = online.mix_inputs(model, img, drv, act, es, np.random.default_rng(7), 1.0)
    assert hit1.all()
    np.testing.assert_array_equal(whole, pred1)                    # every pixel its own prediction: no real pixel at all
    with pytest.raises(ValueError):
        online.mix_inputs(model, img, drv, act, es, np.random.default_rng(7), 0.5, J=0)


def test_no_frame_enters_its_own_prediction(model):
    """The prediction of step s is made from the steps before it: another real frame, another body, another move at
    step s or later leaves it as it was."""
    c = model.config
    T, s = c["ctx"], 5
    img, drv, act, es, _ = windows(c, 2, T, 1, 3)
    _, _, pred = online.mix_inputs(model, img, drv, act, es, np.random.default_rng(1), 0.5)
    img2, drv2, act2 = img.copy(), drv.copy(), act.copy()
    img2[:, s:] = 255 - img2[:, s:]
    drv2[:, s:] = 1.0 - drv2[:, s:]
    act2[:, s:] = -act2[:, s:]
    _, _, pred2 = online.mix_inputs(model, img2, drv2, act2, es, np.random.default_rng(1), 0.5)
    np.testing.assert_array_equal(pred[:, :s + 1], pred2[:, :s + 1])
    assert (pred[:, s + 1:] != pred2[:, s + 1:]).any()             # and what was seen of step s does reach the next


def test_the_passes_of_one_step(model):
    """What the docstring says of a step with steps before it: the passes start from the frame before as seen; the
    first is made with the body of the step before, the others with the body the model predicted (the move token's of
    the step before, as read in the first pass)."""
    c = model.config
    T, s = c["ctx"], 3
    pp = agent_lib.uni_layout(c)["patch"] - 1
    img, drv, act, es, _ = windows(c, 2, T, 1, 4)
    seen, hit, pred = online.mix_inputs(model, img, drv, act, es, np.random.default_rng(2), 0.5, J=3)
    wi, wd = seen[:, :s + 1].copy(), drv[:, :s + 1].astype(np.float32)
    wi[:, s], wd[:, s] = seen[:, s - 1], drv[:, s - 1]
    for j in range(3):
        o = model.forward_uni_step(wi, wd, act[:, :s + 1], es[:, :s + 1], s, mode=2)
        wi[:, s] = online.uni_frame(c, o["values_t"][:, pp])
        if j == 0:
            wd[:, s] = np.clip(o["next_body"][:, s - 1, :2], 0.0, 1.0)
    np.testing.assert_array_equal(pred[:, s], wi[:, s])


def test_a_window_made_in_parts_is_one_made_whole(model):
    """The life makes one step a tick on top of the frames it has seen; an update makes a whole window."""
    c = model.config
    T = c["ctx"]
    img, drv, act, es, _ = windows(c, 2, T, 1, 5)
    seen, hit, pred = online.mix_inputs(model, img, drv, act, es, np.random.default_rng(3), 0.5)
    for given in (T - 1, 4):
        part = img.copy()
        part[:, :given] = seen[:, :given]
        seen2, hit2, pred2 = online.mix_inputs(model, part, drv, act, es, np.random.default_rng(3), 0.5, given=given)
        np.testing.assert_array_equal(seen2, seen)
        assert not hit2[:, :given].any()
        np.testing.assert_array_equal(hit2[:, given:], hit[:, given:])
    np.testing.assert_array_equal(online.mix_inputs(model, seen, drv, act, es, np.random.default_rng(3), 0.5, given=T)[0], seen)


def test_a_frame_with_nothing_before_it(model):
    """The first step of a window and a step that starts an episode: the passes start from an empty frame with the real
    body, and see nothing of the steps before."""
    c = model.config
    T = c["ctx"]
    pp = agent_lib.uni_layout(c)["patch"] - 1
    img, drv, act, es, _ = windows(c, 2, T, 1, 6)
    es[1, 3] = 1                                                  # the second window: a new episode at step 3
    _, _, pred = online.mix_inputs(model, img, drv, act, es, np.random.default_rng(4), 0.5)
    es_cut = np.zeros((1, T - 3), np.uint8)
    es_cut[:, 0] = 1
    _, _, cut = online.mix_inputs(model, img[1:, 3:], drv[1:, 3:], act[1:, 3:], es_cut, np.random.default_rng(4), 0.5)
    np.testing.assert_array_equal(pred[1, 3], cut[0, 0])
    wi, wd = np.zeros_like(img[:, :1]), drv[:, :1].astype(np.float32)   # step 0 by hand
    for _ in range(2):
        wi[:, 0] = online.uni_frame(c, model.forward_uni_step(wi, wd, act[:, :1], es[:, :1], 0, mode=2)["values_t"][:, pp])
    np.testing.assert_array_equal(pred[:, 0], wi[:, 0])
    drv2 = drv.copy()
    drv2[:, 0] = 1.0 - drv2[:, 0]                                  # its own real body is what such a frame is made with
    assert (online.mix_inputs(model, img, drv2, act, es, np.random.default_rng(4), 0.5)[2][:, 0] != pred[:, 0]).any()


def test_a_share_of_random_grey(model):
    """Task 00137 (the owner: "30% real, 50% from prediction and 20% just grey noise"): a pixel is the prediction's, a
    random grey or the real one; without the share nothing is drawn for it and the frames are as they were."""
    c = model.config
    T = c["ctx"]
    img, drv, act, es, _ = windows(c, 3, T, 1, 8)
    plain = online.mix_inputs(model, img, drv, act, es, np.random.default_rng(9), 0.5)
    same = online.mix_inputs(model, img, drv, act, es, np.random.default_rng(9), 0.5, noise=0.0)
    assert all((a == b).all() for a, b in zip(plain, same)) and plain[1].dtype == np.uint8 and set(np.unique(plain[1])) == {0, 1}
    seen, kind, pred = online.mix_inputs(model, img, drv, act, es, np.random.default_rng(9), 0.5, noise=0.2)
    shares = [(kind == k).mean() for k in (0, 1, 2)]
    assert 0.28 < shares[0] < 0.32 and 0.48 < shares[1] < 0.52 and 0.18 < shares[2] < 0.22
    np.testing.assert_array_equal(kind == 1, plain[1] == 1)        # the prediction's pixels are the ones they were
    np.testing.assert_array_equal(seen[kind == 0], img[kind == 0])
    np.testing.assert_array_equal(seen[kind == 1], pred[kind == 1])
    grey = seen[kind == 2]                                         # [pixels, 3]: one grey in every channel, any level
    assert (grey[:, 0] == grey[:, 1]).all() and (grey[:, 0] == grey[:, 2]).all()
    assert 115 < grey[:, 0].mean() < 140 and grey[:, 0].std() > 65 and grey[:, 0].min() < 5 and grey[:, 0].max() > 250
    np.testing.assert_array_equal(pred[:, 0], plain[2][:, 0])      # the first frame's prediction has seen nothing yet
    assert (pred[:, 1:] != plain[2][:, 1:]).any()                  # the later ones have seen the greys before them
    part = img.copy()                                              # a window made in parts has the greys of one made whole
    part[:, :T - 1] = seen[:, :T - 1]
    np.testing.assert_array_equal(online.mix_inputs(model, part, drv, act, es, np.random.default_rng(9), 0.5, given=T - 1, noise=0.2)[0], seen)
    with pytest.raises(ValueError):
        online.mix_inputs(model, img, drv, act, es, np.random.default_rng(9), 0.9, noise=0.2)


# ---------------------------------------------------------------------------------------------- a short life
SETS = ["online.steps=96", "online.log_every=48", "online.save_every=96", "online.stm=64", "online.replay=2000",
        "online.mix=0.5", "online.mix_record=8"]


def life_cmd(out, sets, config="W1_discs.json"):
    return [sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", config),
            "--seed", "4", "-o", out] + sum([["--set", s] for s in sets], [])


@pytest.fixture(scope="module")
def life(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("mix") / "run")
    subprocess.run(life_cmd(out, SETS), check=True, cwd=DEV, capture_output=True)
    return out


def test_the_life_never_shows_a_whole_real_frame(life):
    rows = [json.loads(line) for line in open(os.path.join(life, "metrics.jsonl"))]
    assert len(rows) == 2 and all(r["uni_ce_patch"] > 0 for r in rows)
    assert all(0.49 < r["mix_share"] < 0.51 and 0.0 < r["mix_wrong"] <= 1.0 and r["mix_err"] > 0 and r["t_mix"] > 0 for r in rows)
    z = np.load(os.path.join(life, "mix_frames.npz"))
    assert z["tick"].tolist() == list(range(1, 9)) + list(range(89, 97))
    real, pred, mask, seen = z["real"], z["pred"], z["mask"], z["seen"]
    np.testing.assert_array_equal(seen, np.where(mask, pred, real))
    assert all(0.45 < m.mean() < 0.55 for m in mask)              # half of every frame, the life's first included
    assert all((s != r).mean() > 0.2 for s, r in zip(seen[:8], real[:8]))   # a newborn's guess is not the real frame
    assert os.path.exists(os.path.join(life, "ckpt_00000096.bin"))


def test_a_life_with_random_grey_in_the_mix(tmp_path):
    out = str(tmp_path / "run")
    subprocess.run(life_cmd(out, SETS + ["online.mix_noise=0.2"]), check=True, cwd=DEV, capture_output=True)
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert all(0.49 < r["mix_share"] < 0.51 and 0.19 < r["noise_share"] < 0.21 and r["uni_ce_patch"] > 0 for r in rows)
    z = np.load(os.path.join(out, "mix_frames.npz"))
    real, pred, mask, seen = z["real"], z["pred"], z["mask"], z["seen"]
    np.testing.assert_array_equal(seen[mask == 0], real[mask == 0])
    np.testing.assert_array_equal(seen[mask == 1], pred[mask == 1])
    assert all(0.15 < (m == 2).mean() < 0.25 and 0.25 < (m == 0).mean() < 0.35 for m in mask)
    import wm_measures as W                                        # the exam as lived draws the greys too
    assert W.Pieces(out, 96, "pair", lived=True).mix_noise == 0.2 and W.Pieces(out, 96, "pair").mix_noise == 0.0


def test_the_mix_with_self_forcing(tmp_path):
    out = str(tmp_path / "run")
    subprocess.run(life_cmd(out, SETS + ["online.sf_windows=2", "online.sf_steps=3"]), check=True, cwd=DEV, capture_output=True)
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert rows[-1]["sf_dreamed"] > 0 and all(0.49 < r["mix_share"] < 0.51 for r in rows)


@pytest.mark.parametrize("more, config", [(["online.noise_in=0.1"], "W1_discs.json"), (["online.device=cpu", "online.ring=0"], "W1_discs.json"),
                                          (["online.ring=0"], "W1_discs.json"), ([], "W_discs.json"), (["online.mix=1.5"], "W1_discs.json"),
                                          (["online.store_surprise=0.5"], "W1_discs.json"), (["online.mix=0.9", "online.mix_noise=0.2"], "W1_discs.json"),
                                          (["online.mix=0", "online.mix_noise=0.2"], "W1_discs.json")],
                         ids=["noise_in", "cpu", "no ring", "four transformers", "a share above 1", "surprise gating", "shares above 1 together",
                              "grey without the mix"])
def test_what_the_mix_is_refused_with(tmp_path, more, config):
    bad = subprocess.run(life_cmd(str(tmp_path / "run"), SETS + more, config), cwd=DEV, capture_output=True, text=True)
    assert bad.returncode != 0 and "online.mix" in bad.stderr


# ---------------------------------------------------------------------------------------------- the exam as lived
def test_the_exam_shown_as_the_life_saw_frames(life, tmp_path):
    import wm_measures as W
    env = json.load(open(os.path.join(life, "info.json")))["config"]["env"]
    exam = W.Exam(env, seed=5, copies=2, ticks=160, starts=(100, 120))
    e, s = exam.regular()
    old = W.CHUNK
    W.CHUNK = 64
    try:
        plain, lived = W.Pieces(life, 96, "pair"), W.Pieces(life, 96, "pair", lived=True)
        assert plain.mix == 0.0 and lived.mix == 0.5
        for m in (plain, lived):
            m.reset(len(e))
            for j in range(11, -1, -1):
                m.observe(exam.frames[e, s - j], exam.body[e, s - j], np.zeros((len(e), 3), np.float32))
        real = [exam.frames[e, s - j] for j in range(plain.L - 1, -1, -1)]
        assert all((a == b).all() for a, b in zip(plain.fr, real))                 # whole frames, as to every model
        assert all((a != b).any() and (a == b).mean() > 0.4 for a, b in zip(lived.fr, real))   # half of each is the real frame
        assert all((a == b).all() for a, b in zip(lived.bd, plain.bd))             # the bodies are the real ones
        first = [f.copy() for f in lived.fr]
        d1 = W.dreams(lived, exam, e, s, 3, warm=12)[0]                            # a scoring repeats: the masks are fixed
        assert all((a == b).all() for a, b in zip(lived.fr, first))
        assert (d1 == W.dreams(lived, exam, e, s, 3, warm=12)[0]).all() and d1.shape == (len(e), 3, exam.S, exam.S)
        r = W.score(life, 96, parts=("regular",), H=3, out_dir=str(tmp_path), exam=exam, name="lived", lived=True)
        assert r["lived"] is True and json.load(open(str(tmp_path / "step_00000096_pair_lived.json")))["lived"] is True
        assert "lived" not in W.score(life, 96, parts=("regular",), H=3, out_dir=str(tmp_path), exam=exam)
        with pytest.raises(ValueError):
            W.score(life, 96, parts=("regular",), H=3, out_dir=str(tmp_path), exam=exam, lived=True)   # no name of its own
        plain.close(), lived.close()
        other = str(tmp_path / "nomix")                                            # a run that never had the mix
        shutil.copytree(life, other)
        info = json.load(open(os.path.join(other, "info.json")))
        info["config"]["online"]["mix"] = 0.0
        json.dump(info, open(os.path.join(other, "info.json"), "w"))
        with pytest.raises(ValueError):
            W.Pieces(other, 96, "pair", lived=True)
    finally:
        W.CHUNK = old
