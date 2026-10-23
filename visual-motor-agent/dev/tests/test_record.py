"""EpisodeRecorder round trips and the GIF / strip / grid image helpers."""

import json

import numpy as np
import pytest
from PIL import Image

from world import CircleWorld, EnvConfig, render_state
from world.record import EpisodeRecorder, load_episode, save_gif, save_grid, save_strip


def record_episode(cfg, n=3, index=1, seed=0):
    """Record env ``index`` while it seeks the target; other envs act randomly."""
    env = CircleWorld(n, cfg, seed=seed)
    obs = env.reset()
    rec = EpisodeRecorder(env.config, env_index=index, meta={"policy": "seek"})
    rec.begin(env)
    images = [obs["image"][index]]
    rng = np.random.default_rng(seed)
    while not rec.finished:
        a = rng.uniform(-1, 1, (n, 3))
        delta = env.target[index] - env.cursor[index]
        a[index, :2] = np.clip(delta / cfg.max_step_px, -1, 1) + rng.normal(0, 0.3, 2)
        a[index, 2] = float(np.hypot(*delta) < env.radius[index] - 1)
        obs, reward, done, info = env.step(a)
        rec.record(a, reward, done, info)
        images.append(info["terminal_image"][index])
    return rec, np.array(images)


def test_recorded_states_rerender_to_the_observations():
    rec, images = record_episode(EnvConfig(drives=True, bonuses=True))
    np.testing.assert_array_equal(rec.render_frames(), images)   # incl. the terminal frame
    s = rec.summary()
    assert s["success"] and s["finished"] and s["length"] == len(rec) == len(images) - 1
    arrays = rec.arrays()
    assert arrays["cursor"].shape == (len(rec) + 1, 2) and arrays["action"].shape == (len(rec), 3)
    assert not arrays["target_alive"][-1] and arrays["target_alive"][:-1].all()


def test_json_and_npz_round_trip(tmp_path):
    rec, _ = record_episode(EnvConfig(bonuses=True))
    rec.save_json(tmp_path / "ep.json")
    rec.save_npz(tmp_path / "ep.npz")
    doc = json.loads((tmp_path / "ep.json").read_text())
    assert doc["format"] == "circle_world_episode" and doc["version"] == 1
    assert doc["num_steps"] == len(rec) == len(doc["steps"]["reward"]) == len(doc["frames"]["radius"]) - 1
    for path in (tmp_path / "ep.json", tmp_path / "ep.npz"):
        ep = load_episode(path)
        assert ep["config"] == rec.config and ep["meta"]["policy"] == "seek"
        for key, value in rec.arrays().items():
            np.testing.assert_array_equal(ep[key], value, err_msg=f"{path.suffix}:{key}")


def test_recorded_actions_are_sanitized():
    env = CircleWorld(1, seed=0)
    rec = EpisodeRecorder(env.config)
    rec.begin(env.get_state())
    a = np.array([[np.nan, 5.0, 0.7]])
    rec.record(a, *env.step(a)[1:])
    assert rec.arrays()["action"].tolist() == [[0.0, 1.0, 1.0]]


def test_record_requires_begin_and_stops_at_done():
    env = CircleWorld(1, seed=0)
    rec = EpisodeRecorder(env.config)
    with pytest.raises(RuntimeError):
        rec.record(np.zeros((1, 3)), *env.step(np.zeros((1, 3)))[1:])
    rec.begin(env)
    env.set_state({"cursor": [[30.0, 30.0]], "target": [[30.0, 30.0]], "radius": [10.0]})
    rec.begin(env)
    assert rec.record(np.array([[0, 0, 1.0]]), *env.step(np.array([[0, 0, 1.0]]))[1:])
    with pytest.raises(RuntimeError):
        rec.record(np.zeros((1, 3)), *env.step(np.zeros((1, 3)))[1:])


def test_gif_strip_and_grid(tmp_path):
    cfg = EnvConfig()
    frames = render_state([[10.0, 10.0], [14.0, 10.0], [18.0, 12.0]], [[40.0, 40.0]] * 3,
                          [12.0] * 3, [True, True, False], cfg)
    up2 = lambda im: np.repeat(np.repeat(im, 2, axis=0), 2, axis=1)

    save_gif(frames, tmp_path / "ep.gif", scale=2, fps=15)
    with Image.open(tmp_path / "ep.gif") as im:
        assert im.size == (128, 128) and im.n_frames == 3
        for k in range(3):
            im.seek(k)
            np.testing.assert_array_equal(np.array(im.convert("RGB")), up2(frames[k]))

    save_strip(frames, tmp_path / "strip.png", scale=2, pad=2)
    strip = np.array(Image.open(tmp_path / "strip.png"))
    assert strip.shape == (128 + 4, 3 * 128 + 4 * 2, 3)
    np.testing.assert_array_equal(strip[2:130, 2 + 130:2 + 130 + 128], up2(frames[1]))

    small = frames[0][::2, ::2]                          # 32x32, upscaled to the 64x64 cell
    as_float = frames[2].astype(np.float64) / 255.0      # float images in [0, 1] are accepted
    save_grid([[frames[0], small], [as_float]], tmp_path / "grid.png", scale=1, pad=2)
    grid = np.array(Image.open(tmp_path / "grid.png"))
    assert grid.shape == (2 * 66 + 2, 2 * 66 + 2, 3)
    np.testing.assert_array_equal(grid[2:66, 68:132], up2(small))
    np.testing.assert_array_equal(grid[68:132, 2:66], frames[2])
    assert (grid[68:132, 68:132] == 128).all()          # missing cell = padding color
