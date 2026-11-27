"""Task 00132: world-model measures that do not depend on the model (wm_measures.py).

Predictors made by hand go through the same calls as a model, and each measure must say of them what is known:
the world itself is exact at every horizon, the copy has no skill, a frame shifted by d px has its things d px off, a
blank frame has lost them, a blurred frame beats a shifted one on pixels and loses the ball. Then the exam (the same
every time, its events where the starts say), the reader on real frames, and the universal model's adapter on a GPU:
its decoding is exact (another first guess gives the same frame) and dreaming leaves what it carries untouched.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, DEV)

import cuda_lib  # noqa: E402
import wm_measures as W  # noqa: E402

gpu = pytest.mark.skipif(not cuda_lib.available(), reason="needs libvmcuda.so and a GPU")
STARTS = (100, 180, 260, 340)


def env_of(name):
    return json.load(open(os.path.join(DEV, "configs", name)))["env"]


@pytest.fixture(scope="module")
def discs():
    return W.Exam(env_of("W_discs.json"), seed=5, copies=4, ticks=420, starts=STARTS)


@pytest.fixture(scope="module")
def court():
    return W.Exam(env_of("W_breakout.json"), seed=5, copies=6, ticks=900, starts=STARTS, keep=STARTS)


class Shifted:
    """The real continuation moved by (dx, dy) px (rolled: right for the torus; in the court the walls move too)."""
    name = "shifted"

    def __init__(self, dx, dy):
        self.d = (dy, dx)

    def dream_from(self, exam, e, s, H):
        f, b = W.TheWorld().dream_from(exam, e, s, H)
        return np.roll(f, self.d, (2, 3)), b


class Blurred:
    """The real continuation, every pixel the mean of the 5 x 5 around it."""
    name = "blurred"

    def dream_from(self, exam, e, s, H):
        f, b = W.TheWorld().dream_from(exam, e, s, H)
        g = sum(np.roll(f.astype(np.float32), (i, j), (2, 3)) for i in range(-2, 3) for j in range(-2, 3)) / 25.0
        return g, b


class Blank:
    """The world's background with nothing on it."""
    name = "blank"

    def dream_from(self, exam, e, s, H):
        f, b = W.TheWorld().dream_from(exam, e, s, H)
        bg = exam._court() if exam.kind == "breakout" else np.full((exam.S, exam.S), exam.cfg.bg, np.uint8)
        return np.broadcast_to(bg, f.shape).copy(), b


# ---------------------------------------------------------------------------------------------- small functions
def test_auc():
    assert W.auc([3, 4], [1, 2]) == 1.0 and W.auc([1, 2], [3, 4]) == 0.0
    assert W.auc([1, 1], [1, 1, 1]) == 0.5                              # all equal: chance
    assert W.auc([2, 0], [1]) == 0.5
    assert W.auc([1, 3], [1, 2]) == pytest.approx((0.5 + 0 + 1 + 1) / 4)
    assert np.isnan(W.auc([], [1.0]))


def test_horizon_counts_the_ticks_from_the_first_on():
    assert W.horizon([True, True, False, True]) == 2 and W.horizon([False, True]) == 0 and W.horizon([True] * 5) == 5


def test_grey_distance_is_in_grey_levels():
    a = np.random.default_rng(0).integers(0, 200, (3, 8, 8)).astype(np.uint8)
    assert W.grey_w1(a, a) == 0.0
    assert W.grey_w1(a + 3, a) == pytest.approx(3.0)
    assert W.grey_w1(np.full((2, 4, 4), 50.0), np.r_[np.zeros(16), np.full(16, 100)].reshape(2, 4, 4)) == pytest.approx(50.0)


def test_skill_is_one_minus_the_share_of_the_floors_error():
    assert W.skill(np.array([1.0, 1.0]), np.array([4.0, 4.0])) == pytest.approx(0.75)
    assert np.isnan(W.skill(np.array([1.0]), np.array([0.0])))          # the floor is exact: no skill to speak of


# ---------------------------------------------------------------------------------------------- the exam
@pytest.mark.parametrize("name", ["W_discs.json", "W_breakout.json"])
def test_an_exam_is_the_same_every_time(name):
    a = W.Exam(env_of(name), seed=3, copies=2, ticks=150, starts=(100,))
    b = W.Exam(env_of(name), seed=3, copies=2, ticks=150, starts=(100,))
    c = W.Exam(env_of(name), seed=4, copies=2, ticks=150, starts=(100,))
    assert (a.frames == b.frames).all() and all((a.state[k] == b.state[k]).all() for k in a.state)
    assert (a.frames != c.frames).any()


def test_an_exam_is_the_world_stepped_with_null_moves(discs):
    from world import make_config, make_world
    w = make_world(4, make_config(env_of("W_discs.json"), auto_reset=True), seed=5)
    for t in range(5):
        assert (w.observe()["image"][..., 0] == discs.frames[:, t]).all()
        w.step(np.zeros((4, 3)))


def test_a_skipped_exam_is_the_same_world_later():
    a = W.Exam(env_of("W_breakout.json"), seed=3, copies=1, ticks=60, starts=(40,))
    b = W.Exam(env_of("W_breakout.json"), seed=3, copies=1, ticks=20, skip=40, starts=(10,))
    assert (a.frames[:, 40:] == b.frames).all() and (a.body[:, 40:] == b.body).all()
    assert (a.state["ball"][:, 40:] == b.state["ball"]).all()


def test_the_starts(discs, court):
    e, s = discs.regular()
    assert len(e) == 16 and sorted(set(s)) == list(STARTS) and (np.bincount(e) == 4).all()
    e, s = court.every_tick()
    assert s.min() == 100 and s.max() == court.T - 2 and len(e) == court.E * (court.T - 101)
    for name in ("hit", "miss"):
        ev = None
        for d in W.LEADS:
            e, s = court.event_starts(name, d)
            assert len(e) and court.state[name][e, s + d].all() and s.min() >= 100       # the event is d ticks ahead
            assert ev is None or (ev == s + d).all()                                    # the same events at every lead
            ev = s + d
    e, s = discs.bounce_starts(8)
    assert len(e) == 8 and discs.state["turned"][e, s + 1].any(1).all()
    e, s = discs.free_starts(8, step=60)
    assert len(e) == 8 and not any(discs.state["turned"][a, b - 1:b + 3].any() for a, b in zip(e, s))


def test_the_real_frames_are_valid_and_a_copy_knows_its_own_range(discs, court):
    for ex in (discs, court):
        e, s = ex.regular()
        fr, _ = W.real(ex, e, s, 8)
        assert ex.valid(fr, e).all()
        assert not ex.valid(np.zeros_like(fr), e).any()                 # a black frame has no bright pixel
    assert court.zone_names == ["walls", "paddle rows", "court"]
    assert (court.count_lo[:, 0] == court.count_hi[:, 0]).all()        # the walls never change


# ---------------------------------------------------------------------------------------------- the reader
@pytest.mark.parametrize("which", ["discs", "court"])
def test_the_things_make_the_real_frame_and_are_read_where_they_are(which, discs, court):
    ex = discs if which == "discs" else court
    e, s = ex.every_tick()
    e, s = e[::7], s[::7]
    ths = ex.things(e, s)
    assert [t["kind"] for t in ths] == (["food", "food", "poison", "poison"] if which == "discs" else ["ball", "paddle"])
    for th in ths:
        assert (ex.compose(th) == ex.frames[e, s])[th["ok"]].all()      # the frame without it, with it painted back
        shift, gain = W.locate(ex.frames[e, s], th)
        assert (shift[th["ok"]] == 0).all() and (gain[th["ok"]] > 0).all()
    if which == "court":
        assert not ths[0]["ok"].all() and ths[0]["ok"].mean() > 0.9      # no ball in the pause after a miss


def test_a_shifted_frame_has_its_things_that_far_off(discs):
    e, s = discs.regular()
    for d in ((4, 0), (0, -4), (4, 4)):                                 # 4 px keeps the poison's 2 px checker in phase
        fd, _ = W.dreams(Shifted(*d), discs, e, s, 3)
        dist, found, kinds = W.read_things(discs, e, s, fd)
        assert found.all() and np.allclose(dist, np.hypot(*d))
    fd, _ = W.dreams(Shifted(1, 0), discs, e, s, 3)                      # the solid discs at any shift
    dist, found, kinds = W.read_things(discs, e, s, fd)
    food = [k == "food" for k in kinds]
    assert found[food].all() and (dist[food] == 1).all()


def test_the_paddle_is_read_along_its_row_and_nothing_outside_the_court(court):
    e, s = court.regular()
    fd, _ = W.dreams(Shifted(0, -3), court, e, s, 2)                     # everything 3 px up: no paddle in its rows
    dist, found, kinds = W.read_things(court, e, s, fd)
    assert not found[kinds.index("paddle")].any()
    th = court.things(e, s)[1]                                          # a paddle at the left wall is not looked for left of it
    shift, gain = W.locate(np.roll(court.frames[e, s], -8, 2), th)
    x0 = th["shape"].any(1).argmax(1)
    assert (x0 + shift[:, 0] >= 0).all()


def test_blank_frames_have_lost_everything(discs, court):
    for ex in (discs, court):
        e, s = ex.regular()
        r, _, _ = W.regular_part(Blank(), ex, 4)
        for k in ex.thing_kinds:
            assert r["hold"][k]["lost"] == [1.0] * 4 and r["hold"][k]["held"] == [0.0] * 4 and r["hold"][k]["horizon"] == 0
        assert r["real"]["valid"] == [0.0] * 4


def test_a_blurred_ball_beats_a_shifted_one_on_pixels_and_is_lost(court):
    e, s = court.regular()
    fr, _ = W.real(court, e, s, 4)
    blur, _ = W.dreams(Blurred(), court, e, s, 4)
    move, _ = W.dreams(Shifted(2, 0), court, e, s, 4)
    masks, _ = court.kind_masks(e, s)
    ball = masks["ball"]
    assert W.sq(blur[:, 0], fr[:, 0], ball).sum() < W.sq(move[:, 0], fr[:, 0], ball).sum()      # the pixel error prefers the blur
    d_b, f_b, kinds = W.read_things(court, e, s, blur)
    d_m, f_m, _ = W.read_things(court, e, s, move)
    i, there = kinds.index("ball"), ~np.isnan(d_m[kinds.index("ball")])
    assert not f_b[i][there].any()                                      # the reader does not: no ball in the blur
    assert f_m[i][there].all() and (d_m[i][there] == 2).all()           # and a whole one 2 px off


# ---------------------------------------------------------------------------------------------- the numbers on known predictors
@pytest.mark.parametrize("which", ["discs", "court"])
def test_the_world_itself_is_exact_and_the_copy_has_no_skill(which, discs, court):
    ex = discs if which == "discs" else court
    H = 12
    w, arr, _ = W.regular_part(W.TheWorld(), ex, H)
    c, _, _ = W.regular_part(W.Copy(), ex, H)
    assert w["next"]["skill"] == 1.0 and w["curve"]["skill_copy"] == [1.0] * H and w["curve"]["horizon"] == H
    assert w["real"]["w1"] == [0.0] * H and w["real"]["valid"] == [1.0] * H
    for z in w["real"]["zones"].values():                               # by zone: all valid, the dreamed counts the real ones
        assert z["valid"] == [1.0] * H and z["count"] == z["real"]
    for k in ex.thing_kinds:
        assert w["hold"][k]["held"] == [1.0] * H and w["hold"][k]["dist"] == [0.0] * H and w["hold"][k]["horizon"] == H
        assert np.isfinite(w["next"]["by"][k]["skill"]) and w["next"]["by"][k]["skill"] == 1.0
    assert np.isnan(w["next"]["by"]["rest"]["skill"]) and w["next"]["by"]["rest"]["rmse"] == 0.0   # nothing else changes
    assert c["next"]["skill"] == 0.0 and c["curve"]["skill_copy"] == [0.0] * H and c["curve"]["horizon"] == 0
    assert arr["err"].shape == (len(ex.starts) * ex.E, H) and not arr["err"].any()
    assert c["next"]["copy_rmse"] == c["next"]["rmse"] > 0


def test_linear_is_exact_in_free_flight_and_wrong_after_a_bounce(discs):
    e, s = discs.free_starts(8, step=60)
    lin, cop, kinds = W.floor_things(discs, e, s, 2)
    assert np.allclose(lin, 0.0, atol=1e-9)                              # no bounce from s - 1 to s + 2
    assert np.allclose(cop[..., 0][[k == "food" for k in kinds]], 1.5)   # a food disc moves 1.5 px a tick
    assert np.allclose(cop[..., 0][[k == "poison" for k in kinds]], 0.75)
    e, s = discs.bounce_starts(8)
    lin, _, _ = W.floor_things(discs, e, s, 1)
    turned = discs.state["turned"][e, s + 1].T                            # (a disc pushed by one that turned moves off its
    assert turned.any(0).all() and (lin[..., 0][turned] > 1e-6).all()    # line too, without turning: it is not scored)


def test_a_straight_line_through_the_frames_as_seen(discs, court):
    e, s = discs.free_starts(8, step=60)
    d, kinds = W.seen_things(discs, e, s, 2, 2)                          # a disc's centre is read off a frame to a
    assert d.shape == (4, 8, 2) and (d < 0.6).all()                      # fraction of a pixel: in free flight the line holds
    e, s = court.regular()
    d2, kinds = W.seen_things(court, e, s, 16, 2)
    d8, _ = W.seen_things(court, e, s, 16, 8)
    lin, _, _ = W.floor_things(court, e, s, 16)
    ball = kinds.index("ball")
    far = lambda x: np.nanmean(x[ball][:, 7] > W.HOLD)                   # noqa: E731  the ball lost by tick 8
    assert far(lin) < far(d8) < far(d2)                                  # whole pixels: 2 frames know the least
    r, _, _ = W.regular_part(W.Copy(), court, 16)
    assert set(r["hold_seen"]) == {"2", "4", "8"}
    assert r["hold_seen"]["2"]["ball"]["horizon"] < r["hold_seen"]["8"]["ball"]["horizon"] <= r["hold_linear"]["ball"]["horizon"]


class ExactAwayFromSeams:
    """The real continuation, but within the seam's band the last real frame (a predictor that fails only at the seams)."""
    name = "seamless"

    def dream_from(self, exam, e, s, H):
        f, b = W.TheWorld().dream_from(exam, e, s, H)
        near = W.seam_mask(exam.S)
        return np.where(near, exam.frames[e, s][:, None], f), b


def test_the_seams(discs, court):
    near = W.seam_mask(64)
    assert near.sum() == 960 and near[:, 28:36].all() and near[28:36].all() and not near[:28, :28].any() and not near[36:, 36:].any()
    for ex in (discs, court):
        w = W.seam_part(W.TheWorld(), ex, 6)
        c = W.seam_part(W.Copy(), ex, 6)
        x = W.seam_part(ExactAwayFromSeams(), ex, 6)
        assert w["near"]["skill"] == [1.0] * 6 and w["far"]["skill"] == [1.0] * 6 and w["pixels"] == dict(near=960, far=3136)
        assert c["near"]["skill"] == [0.0] * 6 and c["far"]["skill"] == [0.0] * 6
        assert x["far"]["skill"] == [1.0] * 6 and x["near"]["skill"] == [0.0] * 6      # it loses at the seams only
        for k in ex.thing_kinds:
            for part in ("across", "clear"):
                held, n = np.array(w["things"][k][part]["held"]), np.array(w["things"][k][part]["n"])
                assert (held[n > 0] == 1.0).all() and np.isnan(held[n == 0]).all()
        total = sum(np.array(w["things"][k][part]["n"]) for k in ex.thing_kinds for part in ("across", "clear"))
        assert (total > 0).all()
    x = W.seam_part(ExactAwayFromSeams(), discs, 6)["things"]["food"]   # a disc across a seam is lost by the predictor
    held, n = np.array(x["across"]["held"]), np.array(x["across"]["n"])  # that fails there, from the third tick on
    assert n.sum() > 0 and np.nanmean(held[2:]) < np.nanmean(np.array(x["clear"]["held"])[2:])


def test_the_things_by_where_they_have_been(discs, court):
    quarters = [(a, b) for a in (slice(0, 32), slice(32, 64)) for b in (slice(0, 32), slice(32, 64))]
    for ex in (discs, court):
        e, s = ex.regular()
        been, kinds = W.across_borders(ex, e, s, 6)
        assert been.shape == (len(kinds), len(e), 6) and kinds == [th["kind"] for th in ex.things(e, s)]
        assert (been[:, :, 1:] >= been[:, :, :-1]).all()                # once across a border, across from then on
        many = []                                                       # by hand: the true shape has pixels in two quarters
        for t in range(7):                                              # the last real frame, then 6 ticks
            many.append(np.stack([sum(th["shape"][:, a, b].any((1, 2)).astype(int) for a, b in quarters) > 1
                                  for th in ex.things(e, np.asarray(s) + t)]))
        many = np.logical_or.accumulate(np.stack(many, 2), 2)
        assert (been == many[:, :, 1:]).all() and been.any() and not been.all()
        d0, f0, _ = W.read_things(ex, e, s, W.TheWorld().dream_from(ex, e, s, 6)[0])
        there = ~np.isnan(d0)
        w = W.borders_part(ex, d0, f0)                                  # the world itself holds everything, wherever it was
        x = W.borders_part(ex, np.where(there, np.where(been, 5.0, 0.0), np.nan), np.ones(been.shape, bool))
        for k in ex.thing_kinds:                                        # a reader's distances made by hand: 5 px off once
            sel = np.array([kk == k for kk in kinds])                   # the thing has lain across a border, exact before
            n_in, n_ac = np.array(x[k]["inside"]["n"]), np.array(x[k]["across"]["n"])
            assert (n_in == (there & ~been)[sel].sum((0, 1))).all() and (n_ac == (there & been)[sel].sum((0, 1))).all()
            assert (np.array(x[k]["inside"]["held"])[n_in > 0] == 1.0).all() and (np.array(x[k]["across"]["held"])[n_ac > 0] == 0.0).all()
            assert np.isnan(np.array(x[k]["inside"]["held"])[n_in == 0]).all()
            for part, n in (("inside", n_in), ("across", n_ac)):
                assert w[k][part]["n"] == n.tolist() and (np.array(w[k][part]["held"])[n > 0] == 1.0).all()
    e, s = discs.regular()                                              # a torus: a disc over the frame's edge is in two quarters
    been, kinds = W.across_borders(discs, e, s, 1)
    edge = np.stack([th["shape"][:, :, 0].any(1) & th["shape"][:, :, 63].any(1) for th in discs.things(e, s)])
    assert edge.any() and been[:, :, 0][edge].all()


def test_the_ticks_after_which_a_curve_is_at_a_level():
    assert W.reach([10, 20, 30], [0.1, 0.3, 0.5], 0.2) == (15.0, True)
    assert W.reach([10, 20, 30], [0.1, 0.2, 0.3], 0.2) == (20.0, True)
    assert W.reach([10, 20], [0.5, 0.6], 0.2) == (10, False)            # at the first checkpoint already
    assert W.reach([10, 20], [0.1, 0.15], 0.2) is None                  # never
    assert W.reach([10, 20, 30], [0.1, 0.05, 0.3], 0.2) == (26.0, True)  # a dip does not count


def test_the_events_of_the_court_on_known_forecasts(court):
    e, s = court.every_tick()
    true = court.body[e, s + 1, 0]
    r = W.court_next_tick(court, e, s, true)
    assert r["hit"]["auc"] == 1.0 and r["miss"]["auc"] == 1.0 and r["hit"]["size"] == pytest.approx(1.0)
    assert r["hit"]["events"] == court.state["hit"][:, 101:].sum() > 5 and r["miss"]["events"] > 0
    r = W.court_next_tick(court, e, s, court.body[e, s, 0])             # no change forecast: chance
    assert r["hit"]["auc"] == 0.5 and r["miss"]["auc"] == 0.5 and r["miss"]["size"] == 0.0
    half = court.body[e, s, 0] + 0.5 * (true - court.body[e, s, 0])     # half of every change: the same order, half the size
    r = W.court_next_tick(court, e, s, half)
    assert r["hit"]["auc"] == 1.0 and r["miss"]["size"] == pytest.approx(0.5)
    kept = {}
    w = W.events_part(W.TheWorld(), court, keep=kept)                    # the forecasts by tick are kept for later readings
    assert (kept["next_hunger"] == true).all() and (kept["next_s"] == s).all() and (kept["next_e"] == e).all()
    c = W.events_part(W.Copy(), court)
    for name in ("hit", "miss"):
        for d in W.LEADS:
            assert w["within"][name][str(d)]["auc"] == 1.0 and c["within"][name][str(d)]["auc"] == 0.5


def test_the_events_of_the_disc_world_on_known_predictors(discs):
    w = W.events_part(W.TheWorld(), discs)
    c = W.events_part(W.Copy(), discs)
    for name in ("bounce", "free"):
        for k in ("food", "poison"):
            assert w[name][k]["skill"] == 1.0 and w[name][k]["dist"] == 0.0 and w[name][k]["held"] == 1.0
            assert c[name][k]["skill"] == 0.0
    assert w["free"]["food"]["linear_dist"] < 1e-9 < w["bounce"]["food"]["linear_dist"]
    assert w["bounce"]["food"]["things"] >= w["bounce"]["starts"] / 4   # only the discs that turn are scored


def test_other_dice_is_the_world_until_chance_enters(court):
    e, s = court.regular()
    fr, _ = W.real(court, e, s, 30)
    fd, bd = W.dreams(W.OtherDice(), court, e, s, 30)
    miss = np.stack([court.state["miss"][e, s + j] for j in range(1, 31)], 1)
    same = court.state["live"][e, s] & ~miss.any(1)                      # a ball in play and no miss: no serve, no chance
    assert same.any() and (fd[same] == fr[same]).all()
    assert (bd[same][:, -1] == court.body[e, s + 30][same]).all()
    assert (fd[~same] != fr[~same]).any() and court.valid(fd, e).all()   # a serve falls elsewhere; still frames of this world


def test_the_contract_shows_the_real_ticks_in_order_and_no_more(court):
    seen = []

    class Spy(W.Copy):
        def reset(self, n):
            seen.append([])

        def observe(self, frame, body, move):
            seen[-1].append((frame.copy(), body.copy()))
            assert not move.any()
            W.Copy.observe(self, frame, body, move)

    e, s = court.regular()
    fd, bd = W.dreams(Spy(), court, e, s, 3, warm=5)
    assert len(seen) == 1 and len(seen[0]) == 5
    for j, (f, b) in enumerate(seen[0]):
        assert (f == court.frames[e, s - 4 + j]).all() and (b == court.body[e, s - 4 + j]).all()
    assert (fd == court.frames[e, s][:, None]).all() and fd.shape == (len(e), 3, 64, 64)
    with pytest.raises(ValueError):
        W.dreams(Spy(), court, e, s, 3, warm=200)


# ---------------------------------------------------------------------------------------------- the universal model's adapter
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for name in ("W_discs.json", "W_breakout.json"):
        d = tmp_path_factory.mktemp("wm") / name[:-5]
        r = subprocess.run([sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", name),
                            "--set", "online.steps=8", "--set", "online.log_every=8", "--set", "online.save_every=100000",
                            "--seed", "1", "-o", str(d)], cwd=DEV, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        out[name] = str(d)
    return out


@gpu
def test_the_adapter_decodes_exactly_and_carries_nothing_away(runs, court):
    m = W.Pieces(runs["W_breakout.json"], 0, "pair")
    assert m.S == 4 and m.max_passes == 65 and m.params == 4 * 2159138
    e, s = court.regular()
    e, s = e[:6], s[:6]
    null = np.zeros((6, 3), np.float32)
    m.reset(6)
    for j in range(9, -1, -1):
        m.observe(court.frames[e, s - j], court.body[e, s - j], null)
    f1, b1 = m.dream(3)
    assert f1.shape == (6, 3, 64, 64) and f1.dtype == np.uint8 and b1.shape == (6, 3, 2) and m.unsettled == 0
    assert max(m.passes) <= m.max_passes and len(m.passes) == 3
    f2, b2 = m.dream(3)                                                  # dreaming changed nothing it carries
    assert (f1 == f2).all() and (b1 == b2).all()
    _, b3 = m.dream(1, frames=False)                                     # the body alone is the same body
    assert (b3[:, 0] == b1[:, 0]).all()
    m.guess = np.random.default_rng(0).integers(0, 256, (6, 64, 64)).astype(np.uint8)
    f4, b4 = m.dream(3)                                                  # another first guess, the same frames: the
    assert (f4 == f1).all() and np.allclose(b4, b1, atol=1e-6)           # decoding is token by token, not a guess kept
    m.guess = None
    m.reset(6)                                                           # a window of one tick
    m.observe(court.frames[e, s], court.body[e, s], null)
    f5, _ = m.dream(2)
    assert f5.shape == (6, 2, 64, 64)
    m.reset(6)                                                           # only the last ctx ticks count
    for j in range(7, -1, -1):
        m.observe(court.frames[e, s - j], court.body[e, s - j], null)
    f6, _ = m.dream(3)
    assert (f6 == f1).all()
    m.close()
    for dec in ("instinct", "instinct_pair"):                            # the instinct's way: 2 passes, each piece alone
        m = W.Pieces(runs["W_breakout.json"], 0, dec)
        m.reset(6)
        for j in range(7, -1, -1):
            m.observe(court.frames[e, s - j], court.body[e, s - j], null)
        f7, b7 = m.dream(3)
        f8, b8 = m.dream(3)
        assert f7.shape == (6, 3, 64, 64) and m.passes == [2] * 6 and (f7 == f8).all() and (b7 == b8).all()
        assert (f7 != f1).any()                                          # not the exact decoding
        m.close()


@gpu
def test_one_transformer_goes_through_the_same_calls(tmp_path_factory, court):
    """Task 00133: the adapter with one transformer on the whole frame (no pieces): the same three calls, exact decoding."""
    d = tmp_path_factory.mktemp("wm1") / "one"
    r = subprocess.run([sys.executable, os.path.join(DEV, "lifetime.py"), "--config", os.path.join(DEV, "configs", "W1_breakout.json"),
                        "--set", "online.steps=8", "--set", "online.log_every=8", "--set", "online.save_every=100000",
                        "--seed", "1", "-o", str(d)], cwd=DEV, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    m = W.Pieces(str(d), 0, "pair")
    assert m.S == 1 and m.max_passes == 65 and m.params == 2166178
    e, s = court.regular()
    e, s = e[:4], s[:4]
    null = np.zeros((4, 3), np.float32)
    m.reset(4)
    for j in range(7, -1, -1):
        m.observe(court.frames[e, s - j], court.body[e, s - j], null)
    f1, b1 = m.dream(2)
    assert f1.shape == (4, 2, 64, 64) and m.unsettled == 0 and max(m.passes) <= 65
    m.guess = np.random.default_rng(1).integers(0, 256, (4, 64, 64)).astype(np.uint8)
    f2, b2 = m.dream(2)                                                  # another first guess, the same frames
    assert (f2 == f1).all() and np.allclose(b2, b1, atol=1e-6)
    m.close()
    old = W.CHUNK
    try:                                                                 # a smaller batch gives the same dream (to the last
        W.CHUNK = 2                                                      # bits of a rounding edge: here the same frames)
        ex = W.Exam(env_of("W1_breakout.json"), seed=5, copies=2, ticks=330, starts=(100, 200))
        r = W.score(str(d), 0, "pair", "work", ("regular", "seam"), H=2, out_dir=str(tmp_path_factory.mktemp("wm1out")), exam=ex)
        assert r["pieces"] == 1 and r["chunk"] == 2 and r["regular"]["starts"] == 4 and r["unsettled"] == dict(regular=0, seam=0)
        assert len(r["seam"]["near"]["skill"]) == W.SEAM_TICKS
    finally:
        W.CHUNK = old


@gpu
@pytest.mark.parametrize("name", ["W_discs.json", "W_breakout.json"])
def test_a_checkpoint_is_scored(runs, name, tmp_path):
    ex = W.Exam(env_of(name), seed=5, copies=2, ticks=330, starts=(100, 200))
    r = W.score(runs[name], 0, "mean", "work", ("regular", "events", "warm", "warmdream", "seam"), H=2, out_dir=str(tmp_path), exam=ex)
    assert r["world"] == ex.kind and r["unsettled"] == dict(regular=0, events=0, warm=0, warmdream=0, seam=0) and r["regular"]["starts"] == 4
    assert r["pieces"] == 4 and len(r["seam"]["near"]["skill"]) == W.SEAM_TICKS and set(r["seam"]["things"]) == set(ex.thing_kinds)
    assert set(r["warm_dream"]) == {"2", "4", "8"} and len(r["warm_dream"]["8"]["skill"]) == 8
    assert set(r["warm_dream"]["2"]["held"]) == set(ex.thing_kinds)
    assert len(r["regular"]["curve"]["skill_copy"]) == 2 and set(r["regular"]["hold"]) == set(ex.thing_kinds)
    assert set(r["warm"]) == {str(k) for k in W.WARM_K}
    assert r["warm"]["8"]["skill"] == r["warm"]["16"]["skill"] == pytest.approx(r["regular"]["next"]["skill"])   # 8 ticks of context
    saved = json.load(open(tmp_path / "step_00000000_mean_work.json"))
    assert saved["regular"]["next"]["skill"] == r["regular"]["next"]["skill"]
    arr = np.load(tmp_path / "step_00000000_mean_work.npz")
    assert arr["err"].shape == (4, 2) and arr["dist"].shape[1:] == (4, 2)
    if ex.kind == "breakout":
        assert set(r["events"]) == {"next", "within"} and r["events"]["next"]["hit"]["events"] > 0
    else:
        assert set(r["events"]) == {"bounce", "free"}


@gpu
@pytest.mark.parametrize("name", ["W_discs.json", "W_breakout.json"])
def test_the_tables_print_from_scored_lives(runs, name, tmp_path, capsys):
    import shutil
    ex = W.Exam(env_of(name), seed=5, copies=2, ticks=330, starts=(100, 200), keep=(100, 200))
    a = W.anchors(env_of(name), "work", H=4, exam=ex)
    assert a["the world"]["next"]["skill"] == 1.0 and a["copy"]["next"]["skill"] == 0.0
    json.dump(a, open(tmp_path / ("anchors_%s_work.json" % ex.kind), "w"))
    for sd in (1, 2):
        run = tmp_path / ("XX_s%d" % sd)
        shutil.copytree(runs[name], run)
        for dec in ("pair", "mean"):
            W.score(str(run), 0, dec, "work", ("regular", "events", "warm", "seam") if dec == "pair" else ("regular",), H=4, exam=ex)
        r = W.score(str(run), 0, "pair", "work", ("regular", "next"), H=4, exam=ex, name="learn")
        assert "within" not in r["events"]                              # the learning curve leaves the long dreams out
        W.score(str(run), 0, "pair", "home", ("regular",), H=4, exam=ex)
    b = W.borders([str(tmp_path / ("XX_s%d" % sd)) for sd in (1, 2)], 0, "pair", exams={ex.kind: ex})   # from the kept distances
    saved = json.load(open(tmp_path / "XX_s2" / "wm" / "step_00000000_pair_work_borders.json"))
    assert saved["run"] == "XX_s2" and saved["pieces"] == 4 and set(saved["things"]) == set(ex.thing_kinds)
    assert saved["things"] == json.loads(json.dumps(b[1]["things"])) and len(saved["things"][ex.thing_kinds[0]]["inside"]["held"]) == 4
    W.figure(str(tmp_path), "XX", [1, 2, 3], str(tmp_path / "images" / "curves.png"), step=0)      # the README's pictures
    size = W.strip(str(tmp_path / "XX_s1"), 0, str(tmp_path / "images" / "dream.png"), start=150, ticks=(1, 2), exam=ex)
    assert size[0] <= 960 and all(os.path.getsize(tmp_path / "images" / f) < 200 * 1024 for f in ("curves.png", "dream.png"))
    W.tables(str(tmp_path), "XX", [1, 2, 3], step=0, steps=(0,), exam=ex)
    W.compare(str(tmp_path), "XX", str(tmp_path), "XX", [1, 2, 3], step=0)                         # an arm against itself: no difference
    out = capsys.readouterr().out
    for part in ("### The board", "### The next frame, by kind of thing", "### The dream, tick by tick", "### The things in the dream",
                 "### Is a dreamed frame a possible frame?", "### Events", "### Warm-up", "### Learning", "### Home and away",
                 "### By seed", "XX: 2 seeds at 0k (1, 2)", "did not settle: 0", "## XX against XX: 2 seeds",
                 "### At the seams between the quarters", "### The next frame and the dream by pixels", "+.000 ± .000",
                 "### The things by where they have been", "inside one quarter so far:"):
        assert part in out, part
    width, n_tables = None, 0
    for line in out.split("\n"):                                        # every row of a table has as many cells as its head:
        if not line.startswith("|"):                                    # no pipe inside a cell
            width = None
            continue
        if width is None:
            width, n_tables = line.count("|"), n_tables + 1
        assert line.count("|") == width, line
    assert n_tables >= 9
