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


def pulsate(value, t, period):
    """Amplitude-modulate a slow signal with a periodic carrier.

    Slow signals (hunger ramp, position drift) have near-zero temporal
    derivative, making them invisible to derivative-correlation. Pulsation
    adds a carrier wave whose amplitude encodes the value:

        output = |sin(t * 2π / period)| * value

    Different periods for different signals prevent cross-correlation
    from the carrier alone — the system must attend to amplitude.

    Args:
        value: float in [0, 1], the actual signal to encode
        t: current tick (int)
        period: carrier period in ticks (use different per signal type)

    Returns:
        float in [0, 1], pulsating signal
    """
    carrier = abs(np.sin(t * 2.0 * np.pi / period))
    return float(carrier * value)

N_SENSE = 144  # 8 neurons per signal × 18 signals
# 8×(pos_x, pos_y, target_x, target_y, dir_xp, dir_xn, dir_yp, dir_yn,
#    hunger, proximity, rest_dx+, rest_dx-, rest_dy+, rest_dy-,
#    tire_dx+, tire_dx-, tire_dy+, tire_dy-) = 144
NEURONS_PER_SIGNAL = 8


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
    parser.add_argument("--forage-motor-scale", type=float, default=0.5,
                        help="Motor output scale in field units (default: 0.5)")


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
    # Muscle feedback: 4 directions × 8 muscle fibers each
    # Each fiber has independent restlessness, tiredness, and spasms
    n_fibers = NEURONS_PER_SIGNAL  # 8 fibers per direction
    restlessness = np.zeros((4, n_fibers), dtype=np.float32)
    tiredness = np.zeros((4, n_fibers), dtype=np.float32)
    rest_rate = 0.01
    tire_rate = 0.005
    recovery_rate = 0.002
    move_threshold = 0.3
    score = [0]
    phase_scores = [0, 0]
    _refs = {'column_mgr': None, 'renderer': None, 'dsolver': None}
    base_column_lr = [None]  # captured from column_mgr on first tick
    base_embed_lr = [None]   # captured from dsolver on first tick
    field_save_every = 100

    # POIs
    pois = rng.rand(n_pois_dense, 2).astype(np.float32) * field_size

    # Sensory neuron indices: 8 neurons per signal, 18 signals = 144
    S = NEURONS_PER_SIGNAL
    idx = {}
    offset = 0
    for name in ['pos_x', 'pos_y', 'target_x', 'target_y',
                  'dir_xp', 'dir_xn', 'dir_yp', 'dir_yn',
                  'hunger', 'proximity',
                  'rest_dxp', 'rest_dxn', 'rest_dyp', 'rest_dyn',
                  'tire_dxp', 'tire_dxn', 'tire_dyp', 'tire_dyn']:
        idx[name] = list(range(offset, offset + S))
        offset += S

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

        # Per-fiber muscle spasms: each of 8 fibers per direction
        # independently gets restless and spasms. Total force = sum/n_fibers.
        spasm_forces = np.zeros(4, dtype=np.float32)
        for d in range(4):
            for f in range(n_fibers):
                if rng.rand() < restlessness[d, f] * 0.3:
                    spasm_forces[d] += walk_step * (0.5 + rng.rand() * 0.5)

        # Combine motor + spasm forces per direction
        raw_forces = np.zeros(4, dtype=np.float32)
        raw_forces[0] = max(0, motor_dx) + spasm_forces[0]   # dx+
        raw_forces[1] = max(0, -motor_dx) + spasm_forces[1]  # dx-
        raw_forces[2] = max(0, motor_dy) + spasm_forces[2]   # dy+
        raw_forces[3] = max(0, -motor_dy) + spasm_forces[3]  # dy-

        # Apply tiredness: average across fibers, tired fibers weaken force
        mean_tiredness = tiredness.mean(axis=1)  # (4,)
        effective = raw_forces * (1.0 - mean_tiredness)

        total_dx = effective[0] - effective[1]
        total_dy = effective[2] - effective[3]

        # Move (wrap around — teleport to opposite side at walls)
        prev_pos[:] = pos
        pos[0] = (pos[0] + total_dx) % field_size
        pos[1] = (pos[1] + total_dy) % field_size

        # Per-fiber muscle feedback based on force applied
        for d in range(4):
            for f in range(n_fibers):
                fiber_force = raw_forces[d] / max(1, n_fibers)
                if fiber_force > move_threshold or spasm_forces[d] > 0:
                    restlessness[d, f] = 0.0
                    tiredness[d, f] = min(1.0, tiredness[d, f] + tire_rate)
                else:
                    restlessness[d, f] = min(1.0, restlessness[d, f] + rest_rate)
                    tiredness[d, f] = max(0.0, tiredness[d, f] - recovery_rate)

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
                tiredness[:, :] = 0.0  # reward: all fibers refreshed
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

        # Hunger modulates learning rates:
        # Just ate (hunger=0) → full lr, starving (hunger=1) → lr * 0.01
        # Like glucose: no food → no energy → no plasticity
        lr_scale = max(0.01, 1.0 - hunger * 0.99)
        col_mgr = _refs['column_mgr']
        if col_mgr is not None:
            if base_column_lr[0] is None:
                base_column_lr[0] = col_mgr.lr
            col_mgr.lr = base_column_lr[0] * lr_scale
        dsolver = _refs.get('dsolver')
        if dsolver is not None:
            if base_embed_lr[0] is None:
                base_embed_lr[0] = dsolver.lr
            dsolver.lr = base_embed_lr[0] * lr_scale

        # Build signal: retina renders the field onto the neuron grid.
        # Each neuron "sees" a patch of the field. POIs appear as bright
        # gaussian spots, agent position as a separate gaussian.
        sig = np.zeros(n, dtype=np.float32)
        # Map each neuron to a field position
        neuron_field_x = (np.arange(n) % w).astype(np.float32) / (w - 1) * field_size
        neuron_field_y = (np.arange(n) // w).astype(np.float32) / (h - 1) * field_size
        # POI spots: gaussian with radius ~field_size/8
        poi_sigma2 = (field_size / 8.0) ** 2
        for pi in range(len(pois)):
            dx = neuron_field_x - pois[pi, 0]
            dy = neuron_field_y - pois[pi, 1]
            sig += np.exp(-(dx * dx + dy * dy) / (2 * poi_sigma2))
        # Agent spot: narrower gaussian
        agent_sigma2 = (field_size / 16.0) ** 2
        dx_a = neuron_field_x - pos[0]
        dy_a = neuron_field_y - pos[1]
        sig -= 0.5 * np.exp(-(dx_a * dx_a + dy_a * dy_a) / (2 * agent_sigma2))
        # Add small noise
        sig += rng.randn(n).astype(np.float32) * 0.02
        norm_pos = pos / field_size
        norm_target = nearest_pos / field_size

        # Override neurons: 8 per signal.
        # Fast signals: raw. Slow signals: pulsated.
        for i in idx['pos_x']:     sig[i] = norm_pos[0]
        for i in idx['pos_y']:     sig[i] = norm_pos[1]
        for i in idx['target_x']:  sig[i] = norm_target[0]
        for i in idx['target_y']:  sig[i] = norm_target[1]
        for i in idx['dir_xp']:    sig[i] = max(0.0, direction[0])
        for i in idx['dir_xn']:    sig[i] = max(0.0, -direction[0])
        for i in idx['dir_yp']:    sig[i] = max(0.0, direction[1])
        for i in idx['dir_yn']:    sig[i] = max(0.0, -direction[1])
        # Proximity
        if len(pois) > 0:
            prox = max(0.0, 1.0 - nearest_dist / (field_size * 0.3))
        else:
            prox = 0.0
        for i in idx['proximity']: sig[i] = pulsate(prox, t, 15)
        for i in idx['hunger']:    sig[i] = pulsate(hunger, t, 17)
        # Per-fiber restlessness and tiredness signals
        rest_names = ['rest_dxp', 'rest_dxn', 'rest_dyp', 'rest_dyn']
        tire_names = ['tire_dxp', 'tire_dxn', 'tire_dyp', 'tire_dyn']
        rest_periods = [19, 21, 23, 25]
        tire_periods = [27, 29, 31, 33]
        for d in range(4):
            for f in range(n_fibers):
                sig[idx[rest_names[d]][f]] = pulsate(restlessness[d, f], t, rest_periods[d])
                sig[idx[tire_names[d]][f]] = pulsate(tiredness[d, f], t, tire_periods[d])

        feature_log.append((t, norm_pos[0], norm_pos[1],
                            norm_target[0], norm_target[1],
                            max(0, direction[0]), max(0, -direction[0]),
                            max(0, direction[1]), max(0, -direction[1]),
                            hunger, prox,
                            restlessness.mean(), tiredness.mean(),
                            float(is_sparse)))

        # Save field visualization
        renderer = _refs.get('renderer')
        if renderer is not None and t % field_save_every == 0:
            renderer.field(t, pos.copy(), pois.copy(), field_size,
                           collect_radius=collect_radius)

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
                     'dir_xp', 'dir_xn', 'dir_yp', 'dir_yn',
                     'hunger', 'proximity', 'restless', 'tired',
                     'is_sparse']

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
