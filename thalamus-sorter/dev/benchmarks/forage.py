"""FORAGE benchmark: navigate to points of interest on a virtual field.

Tests the full sensorimotor loop: perceive position + target → navigate
via motor output → collect POI → manage hunger drive.

Sensory neurons (override first N_SENSE neurons):
    2× pos_x, 2× pos_y         — agent position (normalized 0-1)
    2× target_x, 2× target_y   — nearest POI position
    2× dir_x, 2× dir_y         — direction to nearest POI (signed, clipped)
    2× hunger                   — ramps up without collection, resets on collect

Remaining neurons get random noise (background activity).

The agent moves via random walk + motor column output (if --motor-column set).
When agent position is within `collect_radius` of a POI, it's collected:
score +1, POI removed, hunger resets.

Phases:
    Phase 1 (0 to phase_ticks):       dense POIs (easy — learn mechanics)
    Phase 2 (phase_ticks to end):     sparse POIs (hard — must navigate)

Metrics:
    - Collection rate per phase
    - Direction correlation: does any column output track dir_x/dir_y?
    - Hunger stability: does the system learn to keep hunger low?

Usage:
    python main.py word2vec --signal-source forage --column-lateral ...
"""

import os
import json
import numpy as np

name = 'forage'
description = 'Foraging: navigate to POIs, collect them, manage hunger'

N_SENSE = 22  # 14 base + 4 restlessness + 4 tiredness
# 2×pos + 2×target + 2×dir + 2×hunger + 4×restless + 4×tired = 22


def add_args(parser):
    parser.add_argument("--forage-field", type=int, default=100,
                        help="Virtual field size (default: 100×100)")
    parser.add_argument("--forage-pois-dense", type=int, default=20,
                        help="Number of POIs in dense phase (default: 20)")
    parser.add_argument("--forage-pois-sparse", type=int, default=3,
                        help="Number of POIs in sparse phase (default: 3)")
    parser.add_argument("--forage-phase-ticks", type=int, default=5000,
                        help="Ticks before switching to sparse phase (default: 5000)")
    parser.add_argument("--forage-collect-radius", type=float, default=5.0,
                        help="Distance to collect a POI (default: 5.0)")
    parser.add_argument("--forage-walk-step", type=float, default=2.0,
                        help="Random walk step size (default: 2.0)")
    parser.add_argument("--forage-hunger-rate", type=float, default=0.01,
                        help="Hunger ramp per tick (default: 0.01, 100 ticks to 1.0)")
    parser.add_argument("--forage-motor-column", type=int, default=0,
                        help="Which column drives motor output (default: 0)")
    parser.add_argument("--forage-motor-scale", type=float, default=3.0,
                        help="Motor output scale in field units (default: 3.0)")


def make_signal(w, h, args):
    n = w * h
    field_size = getattr(args, 'forage_field', 100)
    n_pois_dense = getattr(args, 'forage_pois_dense', 20)
    n_pois_sparse = getattr(args, 'forage_pois_sparse', 3)
    phase_ticks = getattr(args, 'forage_phase_ticks', 5000)
    collect_radius = getattr(args, 'forage_collect_radius', 5.0)
    walk_step = getattr(args, 'forage_walk_step', 2.0)
    hunger_rate = getattr(args, 'forage_hunger_rate', 0.01)
    motor_column = getattr(args, 'forage_motor_column', 0)
    motor_scale = getattr(args, 'forage_motor_scale', 3.0)
    rng = np.random.RandomState(42)

    # Agent state
    pos = np.array([field_size / 2, field_size / 2], dtype=np.float32)
    prev_pos = pos.copy()
    hunger = np.float32(0.0)
    # Muscle feedback: 4 directions (dx+, dx-, dy+, dy-)
    restlessness = np.zeros(4, dtype=np.float32)  # ramps when muscle idle
    tiredness = np.zeros(4, dtype=np.float32)      # ramps when muscle active
    rest_rate = 0.01    # restlessness ramp per idle tick
    tire_rate = 0.005   # tiredness ramp per active tick
    recovery_rate = 0.002  # tiredness decay per idle tick
    move_threshold = 0.5  # minimum movement to count as "active"
    score = [0]
    phase_scores = [0, 0]
    _refs = {'column_mgr': None}

    # POIs
    pois = rng.rand(n_pois_dense, 2).astype(np.float32) * field_size

    # Sensory neuron indices (first N_SENSE neurons)
    idx = {
        'pos_x': [0, 1],
        'pos_y': [2, 3],
        'target_x': [4, 5],
        'target_y': [6, 7],
        'dir_x': [8, 9],
        'dir_y': [10, 11],
        'hunger': [12, 13],
        # Muscle feedback: dx+, dx-, dy+, dy-
        'restless': [14, 15, 16, 17],
        'tired': [18, 19, 20, 21],
    }

    feature_log = []

    def tick_fn(t):
        nonlocal hunger
        is_sparse = t >= phase_ticks

        # Motor control from column outputs
        motor_dx, motor_dy = 0.0, 0.0
        col_mgr = _refs['column_mgr']
        if col_mgr is not None and motor_column >= 0:
            out = col_mgr.get_outputs()[motor_column]
            motor_dx = (out[0] - out[1]) * motor_scale
            motor_dy = (out[2] - out[3]) * motor_scale
            # Confidence-gated: suppress random walk when motor is strong
            motor_mag = np.sqrt(motor_dx**2 + motor_dy**2)
            confidence = min(1.0, motor_mag / motor_scale)
            rand_scale = 1.0 - confidence
        else:
            rand_scale = 1.0

        # Move: random walk (scaled by confidence) + motor bias
        prev_pos[:] = pos
        pos[0] = np.clip(pos[0] + rng.randn() * walk_step * rand_scale + motor_dx,
                         0, field_size)
        pos[1] = np.clip(pos[1] + rng.randn() * walk_step * rand_scale + motor_dy,
                         0, field_size)

        # Actual movement for muscle feedback
        actual_dx = pos[0] - prev_pos[0]
        actual_dy = pos[1] - prev_pos[1]
        # 4 directions: dx+, dx-, dy+, dy-
        movements = [
            max(0, actual_dx),     # dx+
            max(0, -actual_dx),    # dx-
            max(0, actual_dy),     # dy+
            max(0, -actual_dy),    # dy-
        ]
        for i in range(4):
            if movements[i] > move_threshold:
                # Muscle active: tiredness ramps, restlessness resets
                restlessness[i] = 0.0
                tiredness[i] = min(1.0, tiredness[i] + tire_rate)
            else:
                # Muscle idle: restlessness ramps, tiredness recovers
                restlessness[i] = min(1.0, restlessness[i] + rest_rate)
                tiredness[i] = max(0.0, tiredness[i] - recovery_rate)

        # Find nearest POI
        if len(pois) > 0:
            dists = np.sqrt(((pois - pos) ** 2).sum(axis=1))
            nearest_idx = dists.argmin()
            nearest_dist = dists[nearest_idx]
            nearest_pos = pois[nearest_idx].copy()
            direction = (nearest_pos - pos)
            dir_norm = max(np.sqrt((direction ** 2).sum()), 1e-8)
            direction = direction / dir_norm  # unit direction

            # Collection check
            if nearest_dist < collect_radius:
                score[0] += 1
                phase_idx = 1 if is_sparse else 0
                phase_scores[phase_idx] += 1
                hunger = 0.0
                tiredness[:] = 0.0  # reward: muscles refreshed
                # Remove collected POI
                pois_list = list(range(len(pois)))
                pois_list.remove(nearest_idx)
                if len(pois_list) > 0:
                    new_pois = pois[pois_list]
                else:
                    new_pois = np.empty((0, 2), dtype=np.float32)
                # Respawn: add new POI at random location
                n_target = n_pois_sparse if is_sparse else n_pois_dense
                while len(new_pois) < n_target:
                    new_poi = rng.rand(1, 2).astype(np.float32) * field_size
                    new_pois = np.vstack([new_pois, new_poi])
                pois[:] = 0  # clear
                pois.resize(len(new_pois), 2, refcheck=False)
                pois[:] = new_pois
        else:
            nearest_pos = np.array([field_size / 2, field_size / 2])
            direction = np.array([0.0, 0.0])

        # Hunger ramps
        hunger = min(1.0, hunger + hunger_rate)

        # Build signal
        sig = rng.randn(n).astype(np.float32) * 0.05  # background noise
        norm_pos = pos / field_size
        norm_target = nearest_pos / field_size

        for i in idx['pos_x']:
            sig[i] = norm_pos[0]
        for i in idx['pos_y']:
            sig[i] = norm_pos[1]
        for i in idx['target_x']:
            sig[i] = norm_target[0]
        for i in idx['target_y']:
            sig[i] = norm_target[1]
        for i in idx['dir_x']:
            sig[i] = np.clip(direction[0], -1, 1)
        for i in idx['dir_y']:
            sig[i] = np.clip(direction[1], -1, 1)
        for i in idx['hunger']:
            sig[i] = hunger
        # Muscle feedback signals
        for i in range(4):
            sig[idx['restless'][i]] = restlessness[i]
            sig[idx['tired'][i]] = tiredness[i]

        feature_log.append((t, norm_pos[0], norm_pos[1],
                            norm_target[0], norm_target[1],
                            direction[0], direction[1],
                            hunger, float(is_sparse)))

        return sig

    metadata = {
        'region_names': ['pos_x', 'pos_y', 'target_x', 'target_y',
                         'dir_x', 'dir_y', 'hunger', 'is_sparse'],
        'feature_log': feature_log,
        'w': w, 'h': h, 'n': n,
        'field_size': field_size,
        'score': score,
        'phase_scores': phase_scores,
        'pos': pos,
        'pois': pois,
        'sensor_indices': idx,
        'phase_ticks': phase_ticks,
        'collect_radius': collect_radius,
        '_refs': _refs,  # main.py sets _refs['column_mgr'] after cluster init
    }

    print(f"  signal buffer: FORAGE synthetic "
          f"(field={field_size}, dense={n_pois_dense}, sparse={n_pois_sparse}, "
          f"phase={phase_ticks}, radius={collect_radius})")

    return tick_fn, metadata


def analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir):
    import torch

    score = metadata['score'][0]
    phase_scores = metadata['phase_scores']
    phase_ticks = metadata['phase_ticks']
    total_ticks = metadata['_total_ticks']

    print(f"  FORAGE results:")
    print(f"    Total collections: {score}")
    print(f"    Dense phase (0-{phase_ticks}): {phase_scores[0]} collections")
    sparse_ticks = max(1, total_ticks - phase_ticks)
    print(f"    Sparse phase ({phase_ticks}-{total_ticks}): {phase_scores[1]} collections")
    if phase_scores[0] > 0 and sparse_ticks > 0:
        dense_rate = phase_scores[0] / phase_ticks
        sparse_rate = phase_scores[1] / sparse_ticks
        print(f"    Collection rate: dense={dense_rate:.4f}/tick, "
              f"sparse={sparse_rate:.4f}/tick")

    # Correlate column outputs with features
    if cluster_mgr is None or not cluster_mgr.initialized:
        return
    if cluster_mgr.column_mgr is None:
        return

    print("  FORAGE analysis: sampling 500 ticks...")
    tick_fn = metadata['_tick_fn']
    col_history = []
    n_sensory = metadata['n']

    for t in range(500):
        sig_t = tick_fn(metadata['_total_ticks'] + T + t)
        col_t = tick_counter[0] % T
        signals[:n_sensory, col_t] = torch.from_numpy(sig_t).to(signals.device)
        tick_counter[0] += 1
        if cluster_mgr._signals is not None and cluster_mgr._signal_T > 0:
            cw = cluster_mgr.column_mgr.window
            indices = [(tick_counter[0] - 1 - i) % T for i in range(cw)]
            sw = cluster_mgr._signals[:, indices].cpu().numpy()
            cluster_mgr.column_mgr.tick(sw)
            col_history.append((t, cluster_mgr.column_mgr.get_outputs()))

    log = np.array(metadata['feature_log'])
    if len(log) == 0 or len(col_history) == 0:
        return

    log_ticks = set(log[:, 0].astype(int))
    aligned = []
    for tick, outputs in col_history:
        if tick in log_ticks:
            idx = np.searchsorted(log[:, 0], tick)
            if idx < len(log) and int(log[idx, 0]) == tick:
                aligned.append((log[idx, 1:], outputs))

    if len(aligned) < 10:
        return

    features = np.array([a[0] for a in aligned])
    all_outputs = np.array([a[1] for a in aligned])
    n_ticks, m, n_out = all_outputs.shape
    feature_names = ['pos_x', 'pos_y', 'target_x', 'target_y',
                     'dir_x', 'dir_y', 'hunger', 'is_sparse']

    best = {}
    for fi, fname in enumerate(feature_names):
        feat = features[:, fi]
        if feat.std() < 1e-8:
            continue
        max_corr = 0.0
        best_c, best_o = 0, 0
        mean_corr = 0.0
        count = 0
        for c in range(m):
            for o in range(n_out):
                out = all_outputs[:, c, o]
                if out.std() < 1e-8:
                    continue
                r = abs(np.corrcoef(feat, out)[0, 1])
                mean_corr += r
                count += 1
                if r > max_corr:
                    max_corr = r
                    best_c, best_o = c, o
        best[fname] = {
            'max_abs_corr': float(max_corr),
            'best_column': best_c,
            'best_output': best_o,
            'mean_abs_corr': float(mean_corr / max(count, 1)),
        }

    print(f"  FORAGE feature correlations ({n_ticks} ticks):")
    for fname, info in best.items():
        print(f"    {fname:10s}: max|r|={info['max_abs_corr']:.3f} "
              f"(col {info['best_column']}, out {info['best_output']})")

    if output_dir:
        results = {
            'score': score,
            'dense_collections': phase_scores[0],
            'sparse_collections': phase_scores[1],
            'feature_correlations': best,
        }
        path = os.path.join(output_dir, "forage_analysis.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  FORAGE analysis saved: {path}")
