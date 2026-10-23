"""Videos and screenshot sheets of the agent at successive training checkpoints.

Each view plays continuously, as in training: after every hit or timeout the world draws a new random
circle and cursor. Every checkpoint gets its own random sequence of starts (both runs of part 3 share
it); actions are sampled from the policy. The charts and the "% hit" numbers come from a fixed set of
evaluation episodes per checkpoint. Views are the agent's own 64x64 frames, upscaled; the cursor path
and click markers are drawn on top for the viewer only.

Writes into OUT_DIR:
  1_learning.mp4      PPO from scratch, checkpoint by checkpoint                          (--run)
  2_babbling.mp4      a reward-free babbler's predicted next frame for 9 actions vs the true
                      outcome, checkpoint by checkpoint                                 (--babbler)
  3_transfer.mp4      PPO from scratch vs PPO from the babbler, side by side              (--run2)
  all.mp4             the three parts with title cards
  learning_paths.png, transfer_paths.png
                      screenshot sheets: one picture per episode (start frame, whole cursor path,
                      every click): the first episode of each view, played to its end
  babbling_<steps>.png  the prediction card of each babbler checkpoint

Usage:
    ./venv/bin/python make_video.py --run RUN --run2 RUN_FROM_BABBLER --babbler BABBLER_RUN -o OUT_DIR
"""
import argparse
import dataclasses
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_lib  # noqa: E402
from analyze_dynamics import downsample, make_states, predict_actions, true_outcomes  # noqa: E402
from evaluate import executed_action  # noqa: E402
from record_episodes import env_config_from_info  # noqa: E402
from world import CircleWorld  # noqa: E402
from world.circle_world import render_state  # noqa: E402
from world.record import EpisodeRecorder  # noqa: E402

W, H, FPS = 1280, 720, 15
ITER = 16384                                   # env steps per PPO iteration (128 envs x 128 steps)
BG, PANEL = (12, 12, 18), (26, 26, 36)
TEXT, MUTED = (232, 232, 238), (150, 152, 168)
HIT, MISS, TRAIL = (90, 225, 125), (255, 95, 75), (255, 205, 60)
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_fonts = {}


def font(size, bold=False):
    if (size, bold) not in _fonts:
        path = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        _fonts[size, bold] = ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()
    return _fonts[size, bold]


def steps_label(s):
    return "untrained (random weights)" if s == 0 else "after {:,} env steps".format(s)


# ----------------------------------------------------------------------------- runs and episodes

def load_run(run_dir):
    with open(os.path.join(run_dir, "info.json")) as f:
        info = json.load(f)
    cfg = dataclasses.replace(env_config_from_info(info), bonuses=False, auto_reset=False)
    return cfg, info["ppo"].get("reward_scale", 0.1)


def checkpoints(run_dir, max_steps=None):
    """[(steps, path)] of the numbered checkpoints ckpt_NNNNNNNN.bin, sorted."""
    out = []
    for p in glob.glob(os.path.join(run_dir, "ckpt_*.bin")):
        m = re.fullmatch(r"ckpt_(\d{8})\.bin", os.path.basename(p))
        if m and (max_steps is None or int(m.group(1)) <= max_steps):
            out.append((int(m.group(1)), p))
    return sorted(out)


def play(agent, cfg, n, seed, reward_scale):
    """One episode in each of n envs from fixed start states, actions sampled as in training.

    Returns EpisodeRecorder.arrays() per env: cursor/target/radius/target_alive have T+1 rows,
    action/success/click/empty_click have T rows.
    """
    env = CircleWorld(n, cfg, seed=seed)
    obs = env.observe()
    recs = [EpisodeRecorder(cfg, i) for i in range(n)]
    for r in recs:
        r.begin(env)
    ctx = agent.new_context(n, seed=seed)
    ep_start = np.ones(n, np.uint8)
    prev_a = np.zeros((n, 3), np.float32)
    prev_r = np.zeros(n, np.float32)
    done_all = np.zeros(n, bool)
    for _ in range(cfg.max_steps):
        a = agent.act(ctx, obs["image"], obs["drives"], prev_a, prev_r, ep_start=ep_start)["action"]
        a = np.where(done_all[:, None], 0.0, a)          # finished envs idle (auto_reset off)
        obs, rew, done, inf = env.step(a)
        for i in np.flatnonzero(~done_all):
            recs[i].record(a, rew, done, inf)
        done_all |= done
        prev_a = executed_action(a)
        prev_r = (rew * reward_scale).astype(np.float32)
        ep_start = np.zeros(n, np.uint8)
        if done_all.all():
            break
    ctx.close()
    return [r.arrays() for r in recs]


def play_stream(agent, cfg, n, seed, reward_scale, steps):
    """n envs played for `steps` steps as in training: after every hit or timeout the world draws a new
    random circle and cursor. Returns, per env, its list of episodes (the last one may be unfinished)."""
    env = CircleWorld(n, dataclasses.replace(cfg, auto_reset=True), seed=seed)
    obs = env.observe()
    recs = [EpisodeRecorder(env.config, i) for i in range(n)]
    for r in recs:
        r.begin(env)
    eps = [[] for _ in range(n)]
    ctx = agent.new_context(n, seed=seed)
    ep_start = np.ones(n, np.uint8)
    prev_a = np.zeros((n, 3), np.float32)
    prev_r = np.zeros(n, np.float32)
    for _ in range(steps):
        a = agent.act(ctx, obs["image"], obs["drives"], prev_a, prev_r, ep_start=ep_start)["action"]
        obs, rew, done, inf = env.step(a)
        for i in range(n):
            recs[i].record(a, rew, done, inf)
            if done[i]:
                eps[i].append(recs[i].arrays())
                recs[i].begin(env)                        # the episode the world just drew
        first = inf["first"].astype(bool)                 # same resets of the agent's inputs as train.py
        prev_a = executed_action(a)
        prev_a[first] = 0.0
        prev_r = (rew * reward_scale).astype(np.float32)
        prev_r[first] = 0.0
        ep_start = first.astype(np.uint8)
    ctx.close()
    for i in range(n):
        if len(recs[i]):
            eps[i].append(recs[i].arrays())
    return eps


def evaluate_ckpts(ckpts, cfg, reward_scale, n, seed, frames, cache):
    """{steps: dict(stats, stream, first)} per checkpoint (cached by path).

    stats: success rate and median length of n episodes from a fixed set of start states.
    stream: 4 views played continuously for `frames` steps; first: the first episode of each view,
    played to its end. Their random starts depend only on the checkpoint's step count, so two runs
    evaluated at the same step see the same circles.
    """
    out = {}
    for s, path in ckpts:
        if path not in cache:
            agent = agent_lib.Agent.load(path)
            eps = play(agent, cfg, n, seed, reward_scale)
            vseed = 1000 * seed + s // ITER
            stream = play_stream(agent, cfg, 4, vseed, reward_scale, frames)
            first = play(agent, cfg, 4, vseed, reward_scale)
            agent.close()
            lens = np.array([len(e["reward"]) for e in eps])
            succ = np.array([e["success"].any() for e in eps])
            cache[path] = dict(stats=dict(success=float(succ.mean()), median_len=float(np.median(lens))),
                               stream=stream, first=first)
            hits = sum(int(e["success"].any()) for st in stream for e in st)
            print("  %8d  success %.2f  median length %5.1f  clip hits %d" % (s, succ.mean(), np.median(lens), hits),
                  flush=True)
        out[s] = cache[path]
    return out


# ----------------------------------------------------------------------------- drawing

def up(a, s):
    return a.repeat(s, 0).repeat(s, 1)


def hot(p, s):
    """Screen position of the cursor hotspot pixel centre (world position p, scale s)."""
    return (np.floor(p[0]) + 0.5) * s, (np.floor(p[1]) + 0.5) * s


def layers(cursor, target, radius, alive, cfg):
    """(frame without cursor, frame with cursor), uint8 64x64x3."""
    t, r, a = np.reshape(target, (1, 2)), np.reshape(radius, (1,)), np.reshape(alive, (1,))
    bare = render_state(np.full((1, 2), -100.0), t, r, a, cfg)[0]
    full = render_state(np.reshape(cursor, (1, 2)), t, r, a, cfg)[0]
    return bare, full


def paste_cursor(im, bare, full, s):
    mask = np.any(full != bare, axis=-1)
    im.paste(Image.fromarray(up(full, s)), mask=Image.fromarray(up(mask.astype(np.uint8) * 255, s)))


def ring(d, x, y, r, color, width):
    d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=width)


def cross(d, x, y, r, color, width):
    d.line([(x - r, y - r), (x + r, y + r)], fill=color, width=width)
    d.line([(x - r, y + r), (x + r, y - r)], fill=color, width=width)


def tag(d, xy, text, f, fg=TEXT, bg=(0, 0, 0, 150), pad=4):
    x, y = xy
    _, _, r, b = d.textbbox((x + pad, y + pad), text, font=f)
    d.rectangle([x, y, r + pad, b + pad], fill=bg)
    d.text((x + pad, y + pad), text, font=f, fill=fg)


def banner(d, w, h, text, color, f):
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    bh = b - t + 12
    d.rectangle([0, h - bh, w, h], fill=(0, 0, 0, 175))
    d.text(((w - (r - l)) / 2, h - bh + 6 - t), text, font=f, fill=color)


def draw_path(d, ep, k, s, fresh_k=None):
    """Cursor path up to frame k and the missed clicks so far (fresh_k: frame whose click is new)."""
    lw = s // 2 + 1
    pts = [hot(p, s) for p in ep["cursor"][:k + 1]]
    if len(pts) > 1:
        d.line(pts, fill=TRAIL + (170,), width=lw, joint="curve")
    ring(d, pts[0][0], pts[0][1], 1.5 * s, TRAIL + (210,), max(1, s // 3))
    for j in np.flatnonzero(ep["empty_click"][:k]):
        x, y = hot(ep["cursor"][j + 1], s)
        age = 0 if fresh_k is None else fresh_k - j - 1
        cross(d, x, y, 1.3 * s, MISS + (max(120, 255 - 30 * age),), lw)
        if fresh_k is not None and age == 0:
            ring(d, x, y, 2.6 * s, MISS + (210,), max(1, s // 3))


def draw_hit(d, ep, k, s, fresh_k=None):
    """Ring around the successful click, drawn over the cursor sprite."""
    for j in np.flatnonzero(ep["success"][:k]):
        x, y = hot(ep["cursor"][j + 1], s)
        ring(d, x, y, (7 if j + 1 == fresh_k else 5.5) * s, HIT + (255,), s // 2 + 2)


def view(ep, k, cfg, s, label, note=None):
    """Frame k (0..T) of an episode at scale s, with the cursor path and click markers."""
    bare, full = layers(ep["cursor"][k], ep["target"][k], ep["radius"][k], ep["target_alive"][k], cfg)
    im = Image.fromarray(up(bare, s))
    d = ImageDraw.Draw(im, "RGBA")
    for j in np.flatnonzero(ep["success"][:k]):          # the circle that was just clicked away
        t, r = ep["target"][j] * s, ep["radius"][j] * s
        ring(d, t[0], t[1], r, HIT + (150,), max(1, s // 3))
    draw_path(d, ep, k, s, fresh_k=k)
    paste_cursor(im, bare, full, s)
    draw_hit(d, ep, k, s, fresh_k=k)
    tag(d, (0, 0), label, font(max(12, 3 * s)))
    if note:
        banner(d, im.width, im.height, note[0], note[1], font(max(12, 3 * s), True))
    return im


def stream_seq(episodes, hold=4):
    """Video frames of one continuously playing view: (episode, frame) pairs. Each episode runs from its
    start to its last frame; a finished episode's last frame is held for `hold` frames."""
    seq = []
    for e, ep in enumerate(episodes):
        T = len(ep["reward"])
        seq += [(e, k) for k in range(T + 1)] + [(e, T)] * (hold if ep["done"][-1] else 0)
    return seq


def stream_view(eps, seq, f, cfg, s, letter):
    """Video frame f of one view and its number of hits so far."""
    e, k = seq[min(f, len(seq) - 1)]
    ep = eps[e]
    over = k == len(ep["reward"]) and ep["done"][-1]
    hits = sum(int(x["success"].any()) for x in eps[:e]) + int(over and ep["success"].any())
    note = outcome_note(ep, k) if over else None
    return view(ep, k, cfg, s, "%s   t = %d   hits: %d" % (letter, k, hits), note), hits


def path_summary(ep, cfg, s):
    """One picture of a whole episode: start frame, the full cursor path, every click, final cursor."""
    T = len(ep["reward"])
    bare, _ = layers(ep["cursor"][0], ep["target"][0], ep["radius"][0], ep["target_alive"][0], cfg)
    _, full = layers(ep["cursor"][T], ep["target"][0], ep["radius"][0], ep["target_alive"][0], cfg)
    im = Image.fromarray(up(bare, s))
    d = ImageDraw.Draw(im, "RGBA")
    draw_path(d, ep, T, s)
    paste_cursor(im, bare, full, s)
    draw_hit(d, ep, T, s)
    return im


def outcome_note(ep, shown):
    """Banner text once an episode is over (or was cut off at `shown` frames)."""
    T = len(ep["reward"])
    misses = int(ep["empty_click"].sum())
    miss_txt = "" if misses == 0 else " · %d miss%s" % (misses, "es" if misses > 1 else "")
    if ep["success"].any():
        return ("%s✓ hit at step %d%s" % ("… " if T > shown else "", T, miss_txt), HIT)
    return ("%s✗ no hit in %d steps" % ("… " if T > shown else "", T), MISS)


def short_outcome(ep):
    T = len(ep["reward"])
    return ("✓ %d steps" % T, HIT) if ep["success"].any() else ("✗ %d steps" % T, MISS)


def chart(series, size, xmax, ylim, title, log=False):
    """Pre-rendered line chart. series: [(xs, ys, rgb, label)]. Returns (image, x->px, y->px, (top, bottom))."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    w, h = size
    c = lambda rgb: np.array(rgb) / 255.0  # noqa: E731
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=c(PANEL))
    ax = fig.add_axes([0.1, 0.17, 0.86, 0.68], facecolor=c(PANEL))
    for xs, ys, rgb, label in series:
        ax.plot(xs, ys, color=c(rgb), lw=2, marker="o", ms=3, label=label)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([5, 10, 20, 50, 100, 200])
    ax.set_xlim(0, xmax)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%dk" % round(v / 1000) if v else "0"))
    ax.set_title(title, color=c(TEXT), fontsize=11, loc="left")
    ax.tick_params(colors=c(MUTED), labelsize=9)
    ax.grid(alpha=0.15)
    for sp in ax.spines.values():
        sp.set_color(c(MUTED))
        sp.set_alpha(0.5)
    if any(lab for *_, lab in series):
        ax.legend(fontsize=9, facecolor=c(PANEL), edgecolor=c(MUTED), labelcolor=c(TEXT), loc="center right")
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    tr = ax.transData.transform
    (x0, _), (x1, _) = tr([(0, ylim[0]), (xmax, ylim[0])])
    fy = lambda y: h - tr([(0, y)])[0][1]  # noqa: E731
    top, bottom = fy(ylim[1]), fy(ylim[0])
    plt.close(fig)
    return img, (lambda x: x0 + (x1 - x0) * x / xmax), fy, (top, bottom)


def put_chart(img, d, ch, xy, x, ys=(), colors=()):
    """Paste a chart and draw the 'now' marker at x (with dots at the given y values)."""
    cim, fx, fy, (top, bottom) = ch
    img.paste(cim, xy)
    px = xy[0] + fx(x)
    d.line([(px, xy[1] + top), (px, xy[1] + bottom)], fill=TEXT + (150,), width=2)
    for y, col in zip(ys, colors):
        py = xy[1] + fy(y)
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=col + (255,), outline=TEXT)


def title_bar(d, text, right=""):
    d.rectangle([0, 0, W, 52], fill=PANEL)
    d.text((24, 12), text, font=font(24, True), fill=TEXT)
    f = font(14)
    l, _, r, _ = d.textbbox((0, 0), right, font=f)
    d.text((W - 24 - (r - l), 20), right, font=f, fill=MUTED)


def legend(d, x, y):
    f = font(16)
    d.line([(x, y + 10), (x + 34, y + 10)], fill=TRAIL + (200,), width=3)
    d.text((x + 46, y), "cursor path (circle = start)", font=f, fill=TEXT)
    cross(d, x + 17, y + 40, 7, MISS + (255,), 3)
    d.text((x + 46, y + 30), "click that missed", font=f, fill=TEXT)
    ring(d, x + 17, y + 70, 10, HIT + (255,), 3)
    d.text((x + 46, y + 60), "click on the circle: +10 reward, episode ends", font=f, fill=TEXT)


def card(title, lines):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((100, 230), title, font=font(42, True), fill=TEXT)
    y = 310
    for line in lines:
        d.text((100, y), line, font=font(22), fill=MUTED)
        y += 38
    return np.asarray(img)


# ----------------------------------------------------------------------------- video output

class Video:
    def __init__(self, path):
        import imageio_ffmpeg
        self.path, self.frames = path, 0
        self.gen = imageio_ffmpeg.write_frames(path, (W, H), fps=FPS, codec="libx264", quality=None,
                                               pix_fmt_out="yuv420p", macro_block_size=8,
                                               output_params=["-crf", "18", "-preset", "medium"])
        self.gen.send(None)

    def write(self, frame, repeat=1):
        data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        for _ in range(repeat):
            self.gen.send(data)
        self.frames += repeat

    def close(self):
        self.gen.close()
        print("wrote %s (%.0f s)" % (self.path, self.frames / FPS))


class Tee:
    def __init__(self, *videos):
        self.videos = videos

    def write(self, frame, repeat=1):
        for v in self.videos:
            v.write(frame, repeat)


# ----------------------------------------------------------------------------- part 1: learning

def learning_frame(d1s, f, s, last, cfg, charts, run_name):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    title_bar(d, "1 · Learning from reward: PPO from scratch", run_name)
    total = 0
    for i, (eps, seq) in enumerate(zip(d1s["stream"], d1s["seqs"])):
        v, hits = stream_view(eps, seq, f, cfg, 5, "ABCD"[i])
        total += hits
        img.paste(v, (24 + (i % 2) * 332, 62 + (i // 2) * 332))
    x = 700
    d.text((x, 66), steps_label(s), font=font(30, True), fill=TEXT)
    sub = "of PPO training · iteration %d of %d" % (s // ITER, last // ITER) if s else "PPO training starts here"
    d.text((x, 106), sub + "   ·   hits in this clip: %d" % total, font=font(17), fill=MUTED)
    st = d1s["stats"]
    put_chart(img, d, charts[0], (x, 140), s, [st["success"]], [HIT])
    put_chart(img, d, charts[1], (x, 332), s, [st["median_len"]], [TRAIL])
    d.text((x, 524), "Each view keeps playing: after a hit or a timeout, a new random", font=font(15), fill=MUTED)
    d.text((x, 544), "circle and cursor appear. Charts: 128 test episodes per checkpoint.", font=font(15),
           fill=MUTED)
    legend(d, x, 576)
    d.text((x, 668), "The agent sees only these 64×64 pixels (+ its previous action)", font=font(15), fill=MUTED)
    d.text((x, 690), "and outputs dx, dy and click.", font=font(15), fill=MUTED)
    return np.asarray(img)


def part_learning(data, cfg, out, frames, hold, run_name):
    steps = sorted(data)
    last = steps[-1]
    succ = [data[s]["stats"]["success"] for s in steps]
    lens = [data[s]["stats"]["median_len"] for s in steps]
    charts = (chart([(steps, succ, HIT, None)], (556, 186), last, (0, 1.03), "test episodes that hit the circle (fraction)"),
              chart([(steps, lens, TRAIL, None)], (556, 186), last, (3, 230), "median steps per episode (200 = gave up)",
                    log=True))
    for s in steps:
        d1s = dict(data[s], seqs=[stream_seq(st) for st in data[s]["stream"]])
        for f in range(frames):
            frame = learning_frame(d1s, f, s, last, cfg, charts, run_name)
            out.write(frame)
        out.write(frame, hold)


# ----------------------------------------------------------------------------- part 2: babbling

GRID = [(c - 1, r - 1) for r in range(3) for c in range(3)]          # (dx, dy) of the 3x3 cells
ARROWS = ["↖", "↑", "↗", "←", "click", "→", "↙", "↓", "↘"]
ZOOM = 16                                                            # zoom window, prediction pixels


def pick_state(env, cfg, inside):
    """Index of a state with the whole circle in view, the cursor clear of the walls, and high hunger;
    the cursor well inside the circle, or just outside its edge."""
    st = env.get_state()
    cur, tgt, rad = st["cursor"], st["target"], st["radius"]
    dist = np.hypot(cur[:, 0] - tgt[:, 0], cur[:, 1] - tgt[:, 1])
    in_view = np.minimum.reduce([tgt[:, 0] - rad, cfg.size - tgt[:, 0] - rad, tgt[:, 1] - rad, cfg.size - tgt[:, 1] - rad])
    clear = np.minimum.reduce([cur[:, 0], cfg.size - cur[:, 0], cur[:, 1], cfg.size - cur[:, 1]])
    near = (dist > 3) & (dist < rad - 4) if inside else (dist - rad > 3) & (dist - rad < 8)
    ok = near & (in_view > 2) & (clear > 8)
    return int(np.argmax(np.where(ok, st["hunger"], -1.0)))


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def zoom_box(cursor, wsize, p):
    """Corner (ox, oy) of a ZOOM x ZOOM window of a p x p frame around the cursor, and the cursor centre."""
    c = (np.floor(cursor) + 0.5) * p / wsize
    ox, oy = (int(np.clip(round(v - ZOOM / 2), 0, p - ZOOM)) for v in c)
    return ox, oy, c


def zoomed(frame, box, s=8):
    """The zoom window of a p x p frame, with a small ring where the cursor is now."""
    ox, oy, c = box
    a = frame[oy:oy + ZOOM, ox:ox + ZOOM]
    im = Image.fromarray(up(a if a.dtype == np.uint8 else to_u8(a), s))
    ring(ImageDraw.Draw(im, "RGBA"), (c[0] - ox) * s, (c[1] - oy) * s, 0.8 * s, TRAIL + (220,), 2)
    return im


def overview(frame, box, wsize, p):
    """The full frame at 2x with the zoom window outlined."""
    im = Image.fromarray(up(frame, 2))
    f = 2 * wsize // p
    ImageDraw.Draw(im).rectangle([box[0] * f, box[1] * f, (box[0] + ZOOM) * f - 1, (box[1] + ZOOM) * f - 1],
                                 outline=MUTED)
    return im


def babble_card(s, grid, click, wsize, run_name):
    """grid: the cursor next to the circle, 9 actions (moves + click); click: the cursor inside the circle,
    click only. Each: dict(now: u8 frame, now_p: p x p frame, box, pred/true: p x p frames, hunger: dict)."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    title_bar(d, "2 · Without any reward: what the babbler predicts its actions will do", run_name)
    for i, line in enumerate((
            "Random moves and clicks, no reward. The model is trained only to predict its next frame (and hunger).",
            "Each cell: the frame after one action from the same start (centre = click, others = one full step), zoomed 2×",
            "around the cursor; the small ring marks where the cursor is now. (Over the circle, its movement predictions are still poor.)")):
        d.text((40, 60 + 22 * i), line, font=font(16), fill=MUTED)
    d.text((40, 128), steps_label(s).replace("env steps", "reward-free steps"), font=font(26, True), fill=TEXT)
    y0, hf, p = 178, font(17, True), grid["now_p"].shape[0]
    d.text((40, y0), "now", font=hf, fill=TEXT)
    img.paste(overview(grid["now"], grid["box"], wsize, p), (40, y0 + 28))
    img.paste(zoomed(grid["now_p"], grid["box"]), (40, y0 + 166))
    for i, line in enumerate(("cursor next to", "the circle", "hunger %.2f" % grid["hunger"]["now"])):
        d.text((40, y0 + 304 + 22 * i), line, font=font(15), fill=MUTED)
    for gx, title, cells in ((200, "model predicts", grid["pred"]), (626, "what actually happens", grid["true"])):
        d.text((gx, y0), title, font=hf, fill=TEXT)
        for c, cell in enumerate(cells):
            cx, cy = gx + (c % 3) * 134, y0 + 28 + (c // 3) * 134
            img.paste(zoomed(cell, grid["box"]), (cx, cy))
            tag(d, (cx, cy), ARROWS[c], font(14, c == 4), fg=TRAIL if c == 4 else TEXT)
    h = grid["hunger"]
    yh = y0 + 28 + 3 * 134 + 8
    d.text((200, yh), "click outside the circle → hunger %.2f predicted" % h["click_pred"], font=font(17), fill=TEXT)
    d.text((626, yh), "→ hunger %.2f in fact" % h["click_true"], font=font(17), fill=TEXT)
    gx = 1080
    d.text((gx, y0), "cursor inside", font=hf, fill=TEXT)
    d.text((gx, y0 + 20), "the circle", font=hf, fill=TEXT)
    for i, (lab, im) in enumerate((("now", overview(click["now"], click["box"], wsize, p)),
                                   ("click → model", zoomed(click["pred"][0], click["box"])),
                                   ("click → actual", zoomed(click["true"][0], click["box"])))):
        y = y0 + 50 + i * 150
        img.paste(im, (gx, y))
        tag(d, (gx, y), lab, font(13))
    h = click["hunger"]
    d.text((gx, y0 + 482), "hunger %.2f, after click:" % h["now"], font=font(14), fill=MUTED)
    d.text((gx, y0 + 502), "%.2f predicted, %.2f true" % (h["click_pred"], h["click_true"]), font=font(14), fill=TEXT)
    return np.asarray(img)


def part_babbling(babbler, out, out_dir, hold, run_name):
    cfg, _ = load_run(babbler)
    actions = np.array([[dx, dy, 0.0] for dx, dy in GRID], np.float32)
    actions[4] = [0.0, 0.0, 1.0]                                      # centre: click in place
    states = []
    for seed, inside, acts in ((202, False, actions), (101, True, actions[4:5])):
        env = make_states(cfg, 64, seed, inside=inside)
        states.append((env, pick_state(env, cfg, inside), acts))
    for s, path in checkpoints(babbler):
        agent = agent_lib.Agent.load(path)
        p = agent.config["pred_img"]
        f = cfg.size // p
        cards = []
        for env, i, acts in states:
            obs = env.observe()
            true, true_d = true_outcomes(env, acts)
            pred, pred_d, _ = predict_actions(agent, dict(image=obs["image"][i:i + 1], drives=obs["drives"][i:i + 1]),
                                              acts)
            c = 4 if len(acts) == 9 else 0                            # the click
            cards.append(dict(now=obs["image"][i], now_p=downsample(obs["image"][i], f), pred=pred[0],
                              true=downsample(true[i], f), box=zoom_box(env.get_state()["cursor"][i], cfg.size, p),
                              hunger=dict(now=float(obs["drives"][i, 0]), click_pred=float(pred_d[0, c, 0]),
                                          click_true=float(true_d[i, c, 0]))))
        agent.close()
        frame = babble_card(s, cards[0], cards[1], cfg.size, run_name)
        Image.fromarray(frame).save(os.path.join(out_dir, "babbling_%08d.png" % s))
        out.write(frame, hold)
        print("  babbler %8d  hunger after click: outside %.3f predicted (true %.3f), inside %.3f (true %.3f)"
              % (s, cards[0]["hunger"]["click_pred"], cards[0]["hunger"]["click_true"],
                 cards[1]["hunger"]["click_pred"], cards[1]["hunger"]["click_true"]), flush=True)


# ----------------------------------------------------------------------------- part 3: transfer

def transfer_frame(left, right, f, s, last, cfg, names):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")
    title_bar(d, "3 · Babbling first, then reward", names)
    fb = font(26, True)
    txt = steps_label(s) + ("" if s == 0 else " of PPO")
    l, _, r, _ = d.textbbox((0, 0), txt, font=fb)
    d.text(((W - (r - l)) / 2, 60), txt, font=fb, fill=TEXT)
    bx, by, bw = W / 2 - 200, 98, 400                                 # progress through the shown range
    d.rectangle([bx, by, bx + bw, by + 6], fill=PANEL)
    d.rectangle([bx, by, bx + bw * s / last, by + 6], fill=MUTED)
    heads = (("PPO from scratch", "random initial weights"),
             ("PPO from the babbler", "initial weights from 1M reward-free steps"))
    for p, (side, (h1, h2)) in enumerate(((left, heads[0]), (right, heads[1]))):
        x0 = 40 if p == 0 else W - 40 - 522
        d.text((x0, 114), h1, font=font(20, True), fill=TEXT)
        d.text((x0, 140), h2, font=font(14), fill=MUTED)
        total = 0
        for i, (eps, seq) in enumerate(zip(side["stream"], side["seqs"])):
            v, hits = stream_view(eps, seq, f, cfg, 4, "ABCD"[i])
            total += hits
            img.paste(v, (x0 + (i % 2) * 266, 164 + (i // 2) * 266))
        st = side["stats"]
        d.text((x0, 690), "hits in this clip: %d  ·  tests: %d%% hit, median %d steps"
               % (total, round(100 * st["success"]), st["median_len"]), font=font(16), fill=TEXT)
    return np.asarray(img)


def part_transfer(dl, dr, cfg, out, frames, hold, names):
    steps = sorted(set(dl) & set(dr))
    for s in steps:
        left, right = (dict(dd[s], seqs=[stream_seq(st) for st in dd[s]["stream"]]) for dd in (dl, dr))
        for f in range(frames):
            frame = transfer_frame(left, right, f, s, steps[-1], cfg, names)
            out.write(frame)
        out.write(frame, hold)


# ----------------------------------------------------------------------------- screenshot sheets

def sheet(rows, cfg, s, path, groups, n=4):
    """rows: [(label, [stats per group], [eps per group])]; one path picture per episode."""
    cell, label_w, gap, row_h = 64 * s, 190 if len(groups) == 1 else 260, 8, 64 * s + 30
    gw = n * (cell + gap)
    img = Image.new("RGB", (label_w + len(groups) * (gw + 24), 50 + len(rows) * row_h), BG)
    d = ImageDraw.Draw(img, "RGBA")
    for g, name in enumerate(groups):
        d.text((label_w + g * (gw + 24), 14), name, font=font(20, True), fill=TEXT)
    for r, (label, stats, eps_groups) in enumerate(rows):
        y = 50 + r * row_h
        d.text((12, y + 8), label, font=font(17, True), fill=TEXT)
        for g, st in enumerate(stats):
            d.text((12, y + 34 + 22 * g), ("%s: " % groups[g].split()[-1] if len(groups) > 1 else "") +
                   "%d%% hit, median %d" % (round(100 * st["success"]), st["median_len"]), font=font(14), fill=MUTED)
        for g, eps in enumerate(eps_groups):
            for i, ep in enumerate(eps[:n]):
                x = label_w + g * (gw + 24) + i * (cell + gap)
                img.paste(path_summary(ep, cfg, s), (x, y))
                txt, col = short_outcome(ep)
                d.text((x + 2, y + cell + 4), txt, font=font(14), fill=col)
    img.save(path)
    print("wrote", path)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="PPO run from scratch (dense checkpoints)")
    ap.add_argument("--run2", help="PPO run initialized from the babbler (same seed and config)")
    ap.add_argument("--babbler", help="reward-free babbling run")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=3, help="env seed of the replayed start states")
    ap.add_argument("--episodes", type=int, default=128, help="episodes per checkpoint for the statistics")
    ap.add_argument("--frames", type=int, default=48, help="video frames per checkpoint (part 1)")
    ap.add_argument("--frames-transfer", type=int, default=45, help="video frames per checkpoint (part 3)")
    ap.add_argument("--transfer-max", type=int, default=294912, help="last checkpoint shown in part 3")
    ap.add_argument("--threads", type=int, default=64)
    args = ap.parse_args()
    agent_lib.set_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    name = lambda r: os.path.basename(os.path.normpath(r))  # noqa: E731
    cfg, rs = load_run(args.run)
    cache = {}
    print("evaluating", args.run, flush=True)
    frames = max(args.frames, args.frames_transfer)
    d1 = evaluate_ckpts(checkpoints(args.run), cfg, rs, args.episodes, args.seed, frames, cache)

    full = Video(os.path.join(args.out, "all.mp4"))
    full.write(card("Pixels in, mouse out", [
        "A small network sees only 64×64 pixels and moves a cursor: (dx, dy, click).",
        "Task: click the red circle. No coordinates, no labels, no demonstrations.",
        "",
        "1  learning from reward (PPO)",
        "2  learning without any reward (motor babbling + prediction)",
        "3  babbling first, then reward"]), 4 * FPS)

    full.write(card("1 · Learning from reward", [
        "PPO from random weights. Reward: +10 for clicking the circle, −0.001 per step.",
        "Four views play on, as in training: after each hit or timeout, a new random circle",
        "and cursor appear. A few seconds per training checkpoint."]), 4 * FPS)
    v = Video(os.path.join(args.out, "1_learning.mp4"))
    part_learning(d1, cfg, Tee(v, full), args.frames, 10, name(args.run))
    v.close()

    if args.babbler:
        full.write(card("2 · Learning without any reward", [
            "A babbler moves and clicks at random and is trained only to predict its next frame",
            "and its hunger. Nothing rewards it for the circle, clicks or hunger.",
            "Does it learn what its own actions do?"]), 4 * FPS)
        v = Video(os.path.join(args.out, "2_babbling.mp4"))
        print("babbler", args.babbler, flush=True)
        part_babbling(args.babbler, Tee(v, full), args.out, 3 * FPS, name(args.babbler))
        v.close()

    if args.run2:
        print("evaluating", args.run2, flush=True)
        d2 = evaluate_ckpts(checkpoints(args.run2, args.transfer_max), cfg, rs, args.episodes, args.seed, frames,
                            cache)
        d1t = {s: d1[s] for s in d1 if s <= args.transfer_max}
        full.write(card("3 · Babbling first, then reward", [
            "Left: PPO from scratch. Right: the same PPO, starting from the babbler's weights",
            "(its policy head is untrained). Both sides get the same random sequence of circles."]), 4 * FPS)
        v = Video(os.path.join(args.out, "3_transfer.mp4"))
        part_transfer(d1t, d2, cfg, Tee(v, full), args.frames_transfer, 10,
                      "%s vs %s" % (name(args.run), name(args.run2)))
        v.close()
        rows = [("{:,} steps".format(s), [d1t[s]["stats"], d2[s]["stats"]], [d1t[s]["first"], d2[s]["first"]])
                for s in sorted(d1t) if s in d2]
        sheet(rows, cfg, 2, os.path.join(args.out, "transfer_paths.png"), ["PPO from scratch", "PPO from the babbler"])
    full.close()

    steps = sorted(d1)
    shown = steps[::2] + ([steps[-1]] if (len(steps) - 1) % 2 else [])   # every other checkpoint and the last
    rows = [("{:,} steps".format(s), [d1[s]["stats"]], [d1[s]["first"]]) for s in shown]
    sheet(rows, cfg, 3, os.path.join(args.out, "learning_paths.png"), ["PPO from scratch: one picture per episode"])


if __name__ == "__main__":
    main()
