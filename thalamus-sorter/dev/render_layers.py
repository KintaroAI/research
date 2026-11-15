"""Render a run's layers on the plane: what do V1, V2, V3 ... show (ts-00028).

Replays a finished FSQ-column run with learning off over fresh stretches of
smooth eye drift (restarting at a random place every 100 ticks), records every neuron's value (pixels and fed-back codes),
and draws:

  1. the whole plane: every neuron placed by a 2-D projection (UMAP) of its
     sorted embedding, each cell showing the neuron's current value;
     a second plane coloured by layer;
  2. layer by layer: V1 (the pixels), V2 (codes of columns that read
     pixels), V3 (codes of columns that read V2 codes), ... each projected on
     its own, rotated/flipped onto the receptive-field centres of its
     neurons (display only), as images beside the input crop;
  3. receptive fields: each code neuron's correlation with every pixel over
     the replay, a gallery per layer.

Layers: a column is depth 1 if any of its wired inputs is a pixel, depth d+1
if any input is a code of a depth-d column (BFS, as ts-00023). Pixels are V1;
a code of a depth-d column is V(d+1).

Measures go to <run>/layers/measures.json; images and videos beside it.

Usage:
    python render_layers.py RUN_DIR [--ticks 4000] [--video-ticks 300] [--seed 7]
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import torch

from column_manager import FSQColumn

RF_FLOOR = 0.05   # a code whose peak abs correlation with any pixel is below this has no receptive field (silent)
TEXT = (235, 235, 235)
BG = (24, 24, 24)
# Layer colours (BGR) for the layer map, V1 first
# grey, blue, green, orange, purple, yellow, red
LAYER_BGR = [(200, 200, 200), (240, 150, 60), (90, 200, 90), (40, 150, 240),
             (200, 90, 180), (60, 220, 230), (70, 70, 230)]
DIM_BGR = [(240, 150, 60), (90, 200, 90), (40, 150, 240), (200, 90, 180),
           (60, 220, 230), (70, 70, 230), (255, 200, 120), (120, 255, 200)]


# ---------------------------------------------------------------- loading

def load_run(run):
    info = json.load(open(os.path.join(run, "info.json")))
    a = info["args"]
    emb = np.load(os.path.join(run, "model.npy")).astype(np.float64)
    state = torch.load(os.path.join(run, "column_states.pt"), weights_only=True)
    slot_map = np.load(os.path.join(run, "column_slot_map.npy"))
    return a, emb, state, slot_map


def build_column(state, slot_map):
    col = FSQColumn(int(state["m"]), n_outputs=int(state["n_outputs"]),
                    max_inputs=int(state["max_inputs"]), window=int(state["enc_window"]),
                    lr=0.0, levels=tuple(int(v) for v in state["levels"]),
                    predict_k=int(state["predict_k"]), hidden=int(state["hidden"]),
                    n_samples=int(state["n_samples"]), report_every=0)
    col.load_state(state, slot_map)
    col.train_every = 10 ** 12          # no training-window forward passes in the replay
    col._tick_count = 1
    return col


def layers_from_wiring(slot_map, n_sensory, d, live=None):
    """Column depth by BFS on the wiring; neuron level (pixels 1, codes depth+1).

    live: optional bool (m*d,) of code neurons that carry signal. When given, only live
    codes pass depth on (a column fed only by silent codes is not deeper), and silent
    codes get level 0 like unreached ones, so dead codes cannot manufacture layers."""
    m = slot_map.shape[0]
    if live is None:
        live = np.ones(m * d, dtype=bool)
    depth = np.full(m, -1, dtype=np.int64)
    inputs = [row[row >= 0] for row in slot_map]
    depth[[c for c in range(m) if (inputs[c] < n_sensory).any()]] = 1
    frontier, dd = set(np.where(depth == 1)[0].tolist()), 1
    while frontier:
        new = []
        for c in np.where(depth < 0)[0]:
            codes = inputs[c][inputs[c] >= n_sensory] - n_sensory
            src = codes[live[codes]] // d
            if np.isin(src, list(frontier)).any():
                new.append(c)
        dd += 1
        depth[new] = dd
        frontier = set(new)
    n = n_sensory + m * d
    level = np.ones(n, dtype=np.int64)
    code_col = (np.arange(n_sensory, n) - n_sensory) // d
    level[n_sensory:] = np.where((depth[code_col] > 0) & live, depth[code_col] + 1, 0)  # 0 = unreached or silent
    pixel_share = np.array([(r < n_sensory).mean() if len(r) else 0.0 for r in inputs])
    return depth, level, pixel_share


# ---------------------------------------------------------------- replay

def drift_path(src_h, src_w, ch, cw, ticks, step, momentum, rng, restart_every=0):
    """The run's drift (same rule as main.py --saccade-momentum); with
    restart_every > 0 the eye restarts at a random place every so many ticks,
    so that over the replay distant pixels see unrelated content."""
    max_dy, max_dx = src_h - ch, src_w - cw
    y, x = int(rng.integers(0, max_dy + 1)), int(rng.integers(0, max_dx + 1))
    v = [0, 0]
    out = []
    for t in range(ticks):
        if restart_every and t and t % restart_every == 0:
            y, x = int(rng.integers(0, max_dy + 1)), int(rng.integers(0, max_dx + 1))
            v = [0, 0]
        if rng.random() >= momentum or v == [0, 0]:
            v = [int(rng.integers(-step, step + 1)), int(rng.integers(-step, step + 1))]
        if not 0 <= y + v[0] <= max_dy:
            v[0] = -v[0]
        if not 0 <= x + v[1] <= max_dx:
            v[1] = -v[1]
        y, x = int(np.clip(y + v[0], 0, max_dy)), int(np.clip(x + v[1], 0, max_dx))
        out.append((y, x))
    return out


def replay(a, col, n_sensory, ticks, seed, warmup=100, restart_every=100):
    """Run the frozen loop: pixels from a fresh drift, codes one tick late.
    Returns values (ticks, n), crops (ticks, h, w), path."""
    w, h = int(a["width"]), int(a["height"])
    src = cv2.imread(a["signal_source"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    use_raw = bool(a.get("use_mse")) or bool(a.get("use_deriv_corr"))
    rng = np.random.default_rng(seed)
    total = ticks + warmup
    path = drift_path(src.shape[0], src.shape[1], h, w, total,
                      int(a.get("saccade_step", 2)), float(a.get("saccade_momentum", 0.0)), rng,
                      restart_every)
    m, d = col.m, col.n_outputs
    n = n_sensory + m * d
    W = col.window
    buf = np.zeros((n, W), dtype=np.float32)            # col 0 = newest
    outputs = np.zeros((m, d), dtype=np.float32)
    vals = np.zeros((ticks, n), dtype=np.float32)
    crops = np.zeros((ticks, h, w), dtype=np.float32)
    for t, (y, x) in enumerate(path):
        crop = src[y:y + h, x:x + w]
        sens = crop.ravel() if use_raw else crop.ravel() - crop.mean()
        buf = np.roll(buf, 1, axis=1)
        buf[:n_sensory, 0] = sens
        buf[n_sensory:, 0] = outputs.ravel()
        col.tick(buf)
        outputs = col.get_outputs()
        if t >= warmup:
            vals[t - warmup] = buf[:, 0]
            crops[t - warmup] = crop
    return vals, crops, path[warmup:]


# ---------------------------------------------------------------- measures

def zscore(x, axis=0):
    s = x.std(axis=axis, keepdims=True)
    return (x - x.mean(axis=axis, keepdims=True)) / np.where(s < 1e-8, 1.0, s)


def receptive_fields(vals, n_sensory, w, h):
    """Correlation of each code neuron with each pixel: (K, h*w); with peak
    sign, centre (y, x) and size (pixels above half the peak)."""
    zp = zscore(vals[:, :n_sensory])
    zc = zscore(vals[:, n_sensory:])
    rf = (zc.T @ zp) / len(vals)
    absr = np.abs(rf)
    peak = absr.max(axis=1)
    above = absr >= 0.5 * peak[:, None]
    yy, xx = np.divmod(np.arange(n_sensory), w)
    wts = absr * above
    tot = wts.sum(axis=1).clip(min=1e-12)
    cy, cx = (wts * yy).sum(axis=1) / tot, (wts * xx).sum(axis=1) / tot
    sign = np.sign(rf[np.arange(len(rf)), absr.argmax(axis=1)])
    alive = vals[:, n_sensory:].std(axis=0) > 1e-6
    return rf, np.stack([cy, cx], 1), above.sum(axis=1), sign, peak, alive


def knn(emb, k):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    return nn.kneighbors(emb, return_distance=False)[:, 1:]


def spearman_topography(emb, centres, rng, pairs=20000):
    from scipy.stats import spearmanr
    n = len(emb)
    if n < 10:
        return float("nan"), float("nan")
    i, j = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    de = np.linalg.norm(emb[i] - emb[j], axis=1)
    dc = np.linalg.norm(centres[i] - centres[j], axis=1)
    r = spearmanr(de, dc).correlation
    sh = rng.permutation(n)
    r0 = spearmanr(de, np.linalg.norm(centres[sh][i] - centres[sh][j], axis=1)).correlation
    return float(r), float(r0)


# ---------------------------------------------------------------- drawing

def project(emb, seed):
    if len(emb) < 20:
        return emb[:, :2].copy()
    import umap
    return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                     random_state=seed).fit_transform(emb).astype(np.float64)


def rotate_onto(pos, target):
    """Rotation/reflection of pos (centred) that best matches target; no
    scaling (the panel is filled afterwards), so the layout is the sort's own."""
    P, Tg = pos - pos.mean(0), target - target.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Tg)
    return P @ (U @ Vt)


def voronoi_index(pos, side, lo=None, hi=None, gap=2.0):
    """Grid side x side; each cell gets the index of the nearest neuron.
    Without lo/hi the points fill the grid (same scale on both axes).
    Returns (idx, empty): empty marks cells farther than `gap` times the
    typical spacing between neurons, drawn blank instead of stretched."""
    from scipy.spatial import cKDTree
    if lo is None or hi is None:
        lo = pos.min(0)
        span = np.full(2, max((pos.max(0) - lo).max(), 1e-9))
    else:
        span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    p = (pos - lo) / span * (side - 1)
    tree = cKDTree(p)
    gy, gx = np.mgrid[0:side, 0:side]
    dist, idx = tree.query(np.column_stack([gy.ravel(), gx.ravel()]))
    nn = tree.query(p, k=2)[0][:, 1]
    spacing = max(float(np.median(nn)), 1.0)
    return idx.reshape(side, side), (dist > gap * spacing).reshape(side, side)


def draw(values, vor):
    idx, empty = vor
    img = gray(values[idx])
    img[empty] = BG
    return img


def gray(z, clip=2.5):
    g = ((np.clip(z, -clip, clip) / clip + 1) * 127.5).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def diverging(v):
    """v in [-1, 1] -> BGR, blue negative, red positive, dark at zero."""
    v = np.clip(v, -1, 1)
    out = np.zeros(v.shape + (3,), np.uint8)
    out[..., 2] = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    out[..., 0] = (np.clip(-v, 0, 1) * 255).astype(np.uint8)
    out[..., 1] = (np.abs(v) * 60).astype(np.uint8)
    return out


def panel(img, size, title, sub=None):
    big = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    top = 34 if sub else 22
    canvas = np.full((size + top, size, 3), BG, np.uint8)
    canvas[top:] = big
    cv2.putText(canvas, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1, cv2.LINE_AA)
    if sub:
        cv2.putText(canvas, sub, (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1,
                    cv2.LINE_AA)
    return canvas


def hstack(panels, gap=8):
    h = max(p.shape[0] for p in panels)
    cols = []
    for p in panels:
        q = np.full((h, p.shape[1], 3), BG, np.uint8)
        q[:p.shape[0]] = p
        cols += [q, np.full((h, gap, 3), BG, np.uint8)]
    return np.hstack(cols[:-1])


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    vm = os.path.expanduser("~/research/visual-motor-agent/dev/venv/bin/python")
    if os.path.exists(vm):
        try:
            return subprocess.check_output(
                [vm, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
                text=True).strip()
        except Exception:
            return None
    return None


def write_video(frames, path, fps=15):
    exe = find_ffmpeg()
    h, w = frames[0].shape[:2]
    h2, w2 = h + h % 2, w + w % 2
    if exe:
        with tempfile.TemporaryDirectory() as d:
            for i, f in enumerate(frames):
                g = np.full((h2, w2, 3), BG, np.uint8)
                g[:h, :w] = f
                cv2.imwrite(os.path.join(d, f"f{i:05d}.png"), g)
            subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(fps),
                            "-i", os.path.join(d, "f%05d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20", path], check=True)
    else:
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w2, h2))
        for f in frames:
            g = np.full((h2, w2, 3), BG, np.uint8)
            g[:h, :w] = f
            vw.write(g)
        vw.release()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--restart-every", type=int, default=100,
                    help="replay restarts the eye at a random place every N ticks (0 = never)")
    ap.add_argument("--video-ticks", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--panel", type=int, default=256)
    ap.add_argument("--out", default=None, help="output dir (default RUN/layers)")
    args = ap.parse_args()

    a, emb, state, slot_map = load_run(args.run)
    w, h = int(a["width"]), int(a["height"])
    S = w * h
    col = build_column(state, slot_map)
    m, d = col.m, col.n_outputs
    n = S + m * d
    assert emb.shape[0] == n, (emb.shape, n)
    out = args.out or os.path.join(args.run, "layers")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    vals, crops, path = replay(a, col, S, args.ticks, args.seed,
                               restart_every=args.restart_every)
    rf, centres, rf_size, sign, peak, alive = receptive_fields(vals, S, w, h)
    # live code = changes during the replay and has a receptive field (peak abs corr >= RF_FLOOR);
    # silent codes are left out of the layers and of the RF sizes
    live = alive & (peak >= RF_FLOOR)
    depth_all, level_all, _ = layers_from_wiring(slot_map, S, d)
    depth, level, pixel_share = layers_from_wiring(slot_map, S, d, live=live)
    K = m * d
    code_dim = np.arange(K) % d
    code_col = np.arange(K) // d

    # ---- measures
    meas = {"run": args.run, "ticks": args.ticks, "seed": args.seed,
            "m": m, "d": d, "levels": list(col.levels), "n_sensory": S}
    hist = state.get("history")
    if hist is not None and len(hist):
        hist = hist.numpy()
        meas["loss_history"] = hist.tolist()
        last = hist[-1]
        meas["loss_last"] = {"model": last[1], "copy_last": last[2], "window_mean": last[3],
                             "alive_share": last[4]}
        if len(last) > 5:
            meas["loss_last"]["slot_long_run_mean"] = last[5]
    used = np.zeros((m, d), int)
    for j in range(d):
        q = np.round(vals[:, S + np.arange(m) * d + j] * (col.levels[j] // 2)).astype(int)
        used[:, j] = [len(np.unique(q[:, c])) for c in range(m)]
    meas["replay_levels_used_mean"] = used.mean(0).round(2).tolist()
    meas["replay_columns_all_dims_ge3"] = float((used.min(1) >= 3).mean())
    # live versions (depth passed only through live codes; level 0 = unreached or silent)
    # are what the readings use; the *_all_wiring versions count every wire
    col_live = live.reshape(m, d).any(1)
    meas["rf_floor"] = RF_FLOOR
    meas["code_live_share"] = float(live.mean())
    meas["column_depths"] = {str(int(k)): int(v) for k, v in
                             zip(*np.unique(depth[col_live], return_counts=True))}
    meas["neuron_levels"] = {str(int(k)): int(v) for k, v in
                             zip(*np.unique(level, return_counts=True))}
    meas["column_depths_all_wiring"] = {str(int(k)): int(v) for k, v in
                                        zip(*np.unique(depth_all, return_counts=True))}
    meas["neuron_levels_all_wiring"] = {str(int(k)): int(v) for k, v in
                                        zip(*np.unique(level_all, return_counts=True))}
    meas["column_pixel_input_share_mean"] = float(pixel_share.mean())

    nb = knn(emb, 10)
    code_nb = nb[S:]
    meas["code_neighbours_that_are_pixels"] = float((code_nb < S).mean())
    meas["pixel_neighbours_that_are_codes"] = float((nb[:S] >= S).mean())
    yy, xx = np.divmod(np.arange(S), w)
    true_xy = np.stack([yy, xx], 1).astype(np.float64)
    pnb = knn(emb[:S], 10)
    dpx = np.linalg.norm(true_xy[:, None, :] - true_xy[pnb], axis=2)
    meas["v1_k10_within_3px"] = float((dpx <= 3).mean())

    cnb = knn(emb[S:], 8)
    meas["code_nb_same_dim"] = float((code_dim[cnb] == code_dim[:, None]).mean())
    meas["code_nb_same_column"] = float((code_col[cnb] == code_col[:, None]).mean())
    meas["code_nb_same_sign"] = float((sign[cnb] == sign[:, None]).mean())
    p_pos = float((sign > 0).mean())
    meas["code_sign_chance"] = p_pos ** 2 + (1 - p_pos) ** 2
    meas["code_alive_share"] = float(alive.mean())

    per_layer = {}
    max_level = int(level.max())
    for L in range(2, max_level + 1):
        idx = np.where(level[S:] == L)[0]
        if len(idx) < 4:
            continue
        r, r0 = spearman_topography(emb[S + idx], centres[idx], rng)
        per_layer[f"V{L}"] = {
            "neurons": int(len(idx)),
            "rf_size_mean": float(rf_size[idx].mean()),
            "rf_peak_mean": float(peak[idx].mean()),
            "topography_spearman": r, "topography_shuffled": r0,
            "pixel_input_share": float(pixel_share[np.unique(code_col[idx])].mean()),
        }
    r1, r10 = spearman_topography(emb[:S], true_xy, rng)
    per_layer["V1"] = {"neurons": S, "topography_spearman": r1, "topography_shuffled": r10}

    # ---- projections: whole plane; per layer its own projection (plane view)
    # and its neurons placed at their receptive-field centres (image view)
    pos_all = project(emb, args.seed)
    side_all = int(np.ceil(np.sqrt(n)))
    vor_all = voronoi_index(pos_all, side_all)
    layer_ids = sorted(set(level.tolist()) - {0})
    layer_pos, plane_vor, image_vor, layer_side = {}, {}, {}, {}
    box_lo, box_hi = np.array([0.0, 0.0]), np.array([h - 1.0, w - 1.0])
    for L in layer_ids:
        if L == 1:
            idx = np.arange(S)
            target = true_xy
        else:
            idx = S + np.where(level[S:] == L)[0]
            if len(idx) < 16:
                continue
            target = centres[idx - S]
        side = w if L == 1 else max(16, int(np.ceil(np.sqrt(len(idx)))))
        layer_pos[L] = idx
        layer_side[L] = side
        plane_vor[L] = voronoi_index(rotate_onto(project(emb[idx], args.seed), target), side)
        image_vor[L] = voronoi_index(target, side, lo=box_lo, hi=box_hi)

    zv = zscore(vals)                      # each neuron against its own replay
    zsigned = zv.copy()
    zsigned[:, S:] *= sign[None, :]        # sign-corrected codes (an FSQ sign is arbitrary)

    # ---- copies of the image: per layer, the drawn frame vs the shrunk crop
    for L, idx in layer_pos.items():
        side = layer_side[L]
        acc = {"plane": [], "plane_signed": [], "image_signed": []}
        for t in range(0, args.ticks, 10):
            c = crops[t] - crops[t].mean()
            small = cv2.resize(c, (side, side), interpolation=cv2.INTER_AREA)
            for key, zz, vor in (("plane", zv, plane_vor[L]), ("plane_signed", zsigned, plane_vor[L]),
                                 ("image_signed", zsigned, image_vor[L])):
                vidx, empty = vor
                frame = zz[t, idx][vidx][~empty]
                ref = small[~empty]
                if frame.std() > 1e-8 and ref.std() > 1e-8:
                    acc[key].append(np.corrcoef(frame, ref)[0, 1])
        per_layer.setdefault(f"V{L}", {})
        for key, v in acc.items():
            per_layer[f"V{L}"][f"copy_corr_{key}"] = float(np.mean(v)) if v else float("nan")
        per_layer[f"V{L}"]["grid_side"] = side
    meas["layers"] = per_layer

    # ---- static images
    P = args.panel
    idx_all, empty_all = vor_all
    lvl_map = np.zeros((side_all, side_all, 3), np.uint8)
    for L in layer_ids:
        lvl_map[level[idx_all] == L] = LAYER_BGR[min(L - 1, len(LAYER_BGR) - 1)]
    lvl_map[level[idx_all] == 0] = (70, 70, 70)
    lvl_map[empty_all] = BG
    dim_maps = []
    for L in sorted(layer_pos):
        if L == 1:
            continue
        idx = layer_pos[L]
        dm = np.array([DIM_BGR[j % len(DIM_BGR)] for j in code_dim[idx - S]], np.uint8)
        for name, vor in (("sorted", plane_vor[L]), ("at RF centres", image_vor[L])):
            img = dm[vor[0]]
            img[vor[1]] = BG
            dim_maps.append(panel(img, P, f"V{L} {name}: code number",
                                  "colour = which of the d numbers"))

    def frame_at(t):
        rows = [hstack([
            panel(cv2.cvtColor((crops[t] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), P,
                  "input crop", f"tick {t}"),
            panel(draw(zv[t], vor_all), P, "whole plane", "every neuron where the sort put it"),
            panel(lvl_map, P, "whole plane: layer", "grey V1, blue V2, green V3, orange V4")])]
        for L in sorted(layer_pos):
            idx = layer_pos[L]
            if L == 1:
                rows.append(hstack([panel(draw(zv[t, idx], plane_vor[L]), P, "V1 sorted",
                                          f"{len(idx)} pixels, own projection")]))
                continue
            rows.append(hstack([
                panel(draw(zv[t, idx], plane_vor[L]), P, f"V{L} sorted",
                      f"{len(idx)} codes, own projection"),
                panel(draw(zsigned[t, idx], plane_vor[L]), P, f"V{L} sorted, sign-corrected",
                      "each code flipped to its RF sign"),
                panel(draw(zsigned[t, idx], image_vor[L]), P, f"V{L} at RF centres",
                      "each code where it looks, sign-corrected")]))
        width = max(r.shape[1] for r in rows)
        out_rows = []
        for r in rows:
            out_rows += [np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), BG, np.uint8)]),
                         np.full((8, width, 3), BG, np.uint8)]
        return np.vstack(out_rows[:-1])

    t_mid = min(args.ticks - 1, args.video_ticks // 2)
    cv2.imwrite(os.path.join(out, "plane_frame.png"), frame_at(t_mid))
    if dim_maps:
        cv2.imwrite(os.path.join(out, "code_dims.png"), hstack(dim_maps))

    # RF gallery: 16 random code neurons per layer
    rows = []
    for L in sorted(layer_pos):
        if L == 1:
            continue
        idx = layer_pos[L] - S
        pick = rng.choice(idx, size=min(16, len(idx)), replace=False)
        tiles = []
        for k in pick:
            v = rf[k] / max(np.abs(rf[k]).max(), 1e-8)
            tile = cv2.resize(diverging(v.reshape(h, w)), (96, 96), interpolation=cv2.INTER_NEAREST)
            tiles += [tile, np.full((96, 4, 3), BG, np.uint8)]
        row = np.hstack(tiles[:-1])
        lab = np.full((20, row.shape[1], 3), BG, np.uint8)
        cv2.putText(lab, f"V{L}: receptive fields (red +, blue -), mean size "
                    f"{rf_size[idx].mean():.0f} px", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    TEXT, 1, cv2.LINE_AA)
        rows += [lab, row, np.full((8, row.shape[1], 3), BG, np.uint8)]
    if rows:
        wmax = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], wmax - r.shape[1], 3), BG, np.uint8)])
                for r in rows]
        cv2.imwrite(os.path.join(out, "receptive_fields.png"), np.vstack(rows))

    frames = [frame_at(t) for t in range(min(args.video_ticks, args.ticks))]
    write_video(frames, os.path.join(out, "layers.mp4"))

    np.savez_compressed(os.path.join(out, "replay.npz"), level=level, depth=depth,
                        centres=centres, rf_size=rf_size, sign=sign, path=np.array(path))
    with open(os.path.join(out, "measures.json"), "w") as f:
        json.dump(meas, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in meas.items() if k != "loss_history"}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
