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

N_SENSE_DEFAULT = 176  # 144 base + 32 muscle contraction feedback
# 8×18 signals + 8×4 muscle contraction = 176
NEURONS_PER_SIGNAL_DEFAULT = 8
N_SIGNAL_TYPES = 22  # 18 base + 4 muscle contraction


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
    parser.add_argument("--forage-walk-step", type=float, default=0.5,
                        help="Spasm step size per fiber (default: 0.5)")
    parser.add_argument("--forage-hunger-rate", type=float, default=0.001,
                        help="Hunger ramp per tick (default: 0.001, 1000 ticks to 1.0)")
    parser.add_argument("--forage-motor-columns", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated columns for motor (8 = one per fiber, default: 0,1,2,3,4,5,6,7)")
    parser.add_argument("--forage-motor-scale", type=float, default=0.5,
                        help="Motor output scale in field units (default: 0.5)")
    parser.add_argument("--forage-neurons-per-signal", type=int, default=8,
                        help="Neurons per signal type (default: 8, use higher for larger grids)")
    parser.add_argument("--forage-visual-field", action="store_true",
                        help="Feed model a grayscale rendering of the foraging field")
    parser.add_argument("--forage-visual-res", type=int, default=32,
                        help="Visual field resolution in pixels (default: 32 → 1024 neurons)")


def make_signal(w, h, args):
    n = w * h
    field_size = getattr(args, 'forage_field', 100)
    n_pois_dense = getattr(args, 'forage_pois_dense', 20)
    n_pois_sparse = getattr(args, 'forage_pois_sparse', 3)
    phase_ticks = getattr(args, 'forage_phase_ticks', 5000)
    collect_radius = getattr(args, 'forage_collect_radius', 5.0)
    walk_step = getattr(args, 'forage_walk_step', 0.5)
    hunger_rate = getattr(args, 'forage_hunger_rate', 0.01)
    motor_cols_str = getattr(args, 'forage_motor_columns', '0,1,2,3,4,5,6,7')
    motor_columns = [int(x) for x in motor_cols_str.split(',')]
    motor_scale = getattr(args, 'forage_motor_scale', 0.5)
    rng = np.random.RandomState(42)

    # Agent state
    pos = np.array([field_size / 2, field_size / 2], dtype=np.float32)
    prev_pos = pos.copy()
    hunger = np.float32(0.0)
    state = {
        'prev_nearest_dist': None,
        'prev_nearest_idx': -1,   # track which POI we're approaching
        'hunger': [0.0],
    }
    # Neurons per signal (configurable for larger grids)
    S = getattr(args, 'forage_neurons_per_signal', NEURONS_PER_SIGNAL_DEFAULT)
    N_SENSE = S * N_SIGNAL_TYPES
    # Visual field: render foraging field as grayscale image fed as neurons
    visual_field = getattr(args, 'forage_visual_field', False)
    visual_res = getattr(args, 'forage_visual_res', 32)
    n_visual = visual_res * visual_res if visual_field else 0
    N_SENSE = N_SENSE + n_visual

    assert n >= N_SENSE, (
        f"Grid {w}x{h}={n} too small for {N_SENSE} sensory neurons "
        f"({S} per signal × {N_SIGNAL_TYPES} signals"
        f"{f' + {visual_res}x{visual_res} visual' if visual_field else ''}). "
        f"Need at least {N_SENSE} neurons, e.g. -W {int(N_SENSE**0.5)+1} -H {int(N_SENSE**0.5)+1}")

    # Muscle feedback: 4 directions × S muscle fibers each
    # Each fiber has independent restlessness, tiredness, and spasms
    n_fibers = S  # fibers per direction
    restlessness = np.zeros((4, n_fibers), dtype=np.float32)
    tiredness = np.zeros((4, n_fibers), dtype=np.float32)
    rest_rate = 0.01
    tire_rate = 0.005
    recovery_rate = 0.002
    move_threshold = 0.3  # per-fiber force for "straining" (tires muscle)
    score = [0]
    phase_scores = [0, 0]
    _refs = {'column_mgr': None, 'renderer': None, 'dsolver': None,
             'field_address': getattr(args, 'field_address', None)}
    base_column_lr = [None]  # captured from column_mgr on first tick
    base_embed_lr = [None]   # captured from dsolver on first tick
    field_save_every = 100

    # POIs — stored in state dict so tick_fn and metadata share the reference
    state['pois'] = rng.rand(n_pois_dense, 2).astype(np.float32) * field_size

    # Visual field rendering: precompute coordinate grid
    if visual_field:
        _vf_yy, _vf_xx = np.mgrid[:visual_res, :visual_res]
        _vf_xx = _vf_xx.astype(np.float32)
        _vf_yy = _vf_yy.astype(np.float32)
        _vf_scale = visual_res / field_size
        # Pixel radii: agent ~3px, POI contact circle ~3px, POI center ~1px
        _vf_agent_r = max(2.0, visual_res * 0.03)
        _vf_poi_r = max(3.0, collect_radius * _vf_scale)
        _vf_poi_center_r = max(1.0, _vf_poi_r * 0.3)
        _vf_visual_offset = N_SENSE - n_visual  # where visual neurons start in sig

    # Sensory neuron indices: S neurons per signal
    idx = {}
    offset = 0
    for name in ['pos_x', 'pos_y', 'target_x', 'target_y',
                  'dir_xp', 'dir_xn', 'dir_yp', 'dir_yn',
                  'hunger', 'proximity',
                  'rest_dxp', 'rest_dxn', 'rest_dyp', 'rest_dyn',
                  'tire_dxp', 'tire_dxn', 'tire_dyp', 'tire_dyn']:
        idx[name] = list(range(offset, offset + S))
        offset += S
    # Muscle contraction feedback: 8 fibers × 4 directions
    # Each fiber senses its own contraction (motor + spasm output)
    for name in ['contract_dxp', 'contract_dxn', 'contract_dyp', 'contract_dyn']:
        idx[name] = list(range(offset, offset + S))
        offset += S

    feature_log = []

    def tick_fn(t):
        nonlocal hunger
        is_sparse = t >= phase_ticks

        # Motor control: 8 columns, each drives one fiber per direction.
        # Force below threshold → no movement, no tiredness (idle).
        # Force above threshold → contributes to movement (scaled by
        # 1-tiredness) AND tires the fiber. Pushing all directions
        # at once exhausts the fiber completely.
        motor_forces = np.zeros(4, dtype=np.float32)
        per_fiber_force = np.zeros((4, n_fibers), dtype=np.float32)
        col_mgr = _refs['column_mgr']
        if col_mgr is not None and len(motor_columns) > 0:
            all_out = col_mgr.get_outputs()
            m_cols = all_out.shape[0]
            for f, mc in enumerate(motor_columns):
                if f < n_fibers and mc < m_cols:
                    for d in range(4):
                        force = all_out[mc, d] * motor_scale
                        per_fiber_force[d, f] = force
                        if force >= move_threshold:
                            # Above gate: contributes to movement, scaled by tiredness
                            motor_forces[d] += force * (1.0 - tiredness[d, f])

        # Per-fiber muscle spasms: each of 8 fibers per direction
        # independently gets restless and spasms. Total force = sum/n_fibers.
        # Spasm probability decays over time: halves every 5000 ticks.
        # Spasm decays to zero — agent can fully stop if no motor output.
        spasm_base = 0.0625  # = 0.5^4, equivalent to old t=20k
        spasm_floor = 0.01  # minimum spasm rate — always some random walk
        spasm_decay = max(spasm_base * (0.5 ** (t / 20000.0)), spasm_floor)
        spasm_forces = np.zeros(4, dtype=np.float32)
        for d in range(4):
            for f in range(n_fibers):
                if rng.rand() < restlessness[d, f] * 0.3 * spasm_decay:
                    raw_spasm = walk_step * (0.5 + rng.rand() * 0.5)
                    # Each fiber's spasm scaled by its own tiredness
                    spasm_forces[d] += raw_spasm * (1.0 - tiredness[d, f])

        # Combine: both motor and spasms already have per-fiber tiredness
        effective = motor_forces + spasm_forces

        total_dx = effective[0] - effective[1]
        total_dy = effective[2] - effective[3]

        # Move (clip at walls)
        prev_pos[:] = pos
        pos[0] = np.clip(pos[0] + total_dx, 0, field_size)
        pos[1] = np.clip(pos[1] + total_dy, 0, field_size)

        # Per-fiber tiredness update: uses per_fiber_force from motor above
        # Add spasm contribution
        for d in range(4):
            if spasm_forces[d] > 0:
                per_fiber_force[d, :] += spasm_forces[d] / n_fibers

        for d in range(4):
            for f in range(n_fibers):
                if per_fiber_force[d, f] >= move_threshold:
                    restlessness[d, f] = 0.0
                    tiredness[d, f] = min(1.0, tiredness[d, f] + tire_rate)
                else:
                    restlessness[d, f] = min(1.0, restlessness[d, f] + rest_rate)
                    tiredness[d, f] = max(0.0, tiredness[d, f] - recovery_rate)

        # Find nearest POI
        pois = state['pois']
        if len(pois) > 0:
            dists = np.sqrt(((pois - pos) ** 2).sum(axis=1))
            nearest_idx = dists.argmin()
            nearest_dist = dists[nearest_idx]
            nearest_pos = pois[nearest_idx].copy()
            direction = (nearest_pos - pos)
            dir_norm = max(np.sqrt((direction ** 2).sum()), 1e-8)
            direction = direction / dir_norm  # unit direction

            # Distance-based reward (disabled — too noisy)
            # col_mgr = _refs.get('column_mgr')
            # if (col_mgr is not None
            #         and state['prev_nearest_dist'] is not None
            #         and state['prev_nearest_idx'] == nearest_idx):
            #     delta_dist = state['prev_nearest_dist'] - nearest_dist
            #     if delta_dist > 0:
            #         col_mgr.set_reward(min(delta_dist / collect_radius, 0.5))
            # state['prev_nearest_dist'] = nearest_dist
            # state['prev_nearest_idx'] = nearest_idx

            col_mgr = _refs.get('column_mgr')

            # Hunger penalty (disabled — positive reward only for now)
            # if col_mgr is not None and hunger > 0.5:
            #     col_mgr.set_reward(-0.01 * hunger)

            # Collection check
            if nearest_dist < collect_radius:
                score[0] += 1
                phase_idx = 1 if is_sparse else 0
                phase_scores[phase_idx] += 1
                hunger = 0.0
                # Strong positive reward on collection
                if col_mgr is not None:
                    col_mgr.set_reward(1.0)
                # tiredness NOT reset on collection — muscles stay tired
                # Only hunger resets (reward signal for learning rate)
                # Remove collected POI, respawn to maintain count
                keep = [i for i in range(len(pois)) if i != nearest_idx]
                remaining = pois[keep] if keep else np.empty((0, 2), dtype=np.float32)
                n_target = n_pois_sparse if is_sparse else n_pois_dense
                n_spawn = max(0, n_target - len(remaining))
                if n_spawn > 0:
                    spawned = rng.rand(n_spawn, 2).astype(np.float32) * field_size
                    remaining = np.vstack([remaining, spawned]) if len(remaining) > 0 else spawned
                state['pois'] = remaining
        else:
            nearest_pos = np.array([field_size / 2, field_size / 2])
            direction = np.array([0.0, 0.0])

        # Hunger ramps: 0.001/tick → 1k ticks to reach 1.0
        hunger = min(1.0, hunger + hunger_rate)
        state['hunger'][0] = hunger

        # LR modulation: hunger scales down learning rate (disabled)
        # High hunger → low lr (stop changing, focus on acting)
        # Low hunger (just collected) → full lr (consolidate learning)
        # col_mgr = _refs['column_mgr']
        # dsolver = _refs.get('dsolver')
        # if col_mgr is not None:
        #     if base_column_lr[0] is None:
        #         base_column_lr[0] = col_mgr.lr
        #     col_mgr.lr = base_column_lr[0] * (1.0 - 0.9 * hunger)
        # if dsolver is not None:
        #     if base_embed_lr[0] is None:
        #         base_embed_lr[0] = dsolver.lr
        #     dsolver.lr = base_embed_lr[0] * (1.0 - 0.9 * hunger)

        # Live LR control from field viz (polls TCP controls port)
        if t % 100 == 0 and _refs.get('field_address'):
            try:
                import json as _json
                import socket as _socket
                host, port = _refs['field_address'].rsplit(':', 1)
                ctrl_port = int(port) + 1  # controls on port+1
                s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                s.settimeout(0.1)
                s.connect((host, ctrl_port))
                data = s.recv(1024)
                s.close()
                _ctrl = _json.loads(data.decode())
                col_mgr = _refs.get('column_mgr')
                dsolver = _refs.get('dsolver')
                if col_mgr is not None and 'column_lr' in _ctrl:
                    col_mgr.lr = float(_ctrl['column_lr'])
                if dsolver is not None and 'lr' in _ctrl:
                    dsolver.lr = float(_ctrl['lr'])
            except Exception:
                pass

        # Retina neurons get zero for now (unused)
        sig = np.zeros(n, dtype=np.float32)
        norm_pos = pos / field_size
        norm_target = nearest_pos / field_size

        # Override neurons: 8 per signal.
        # All slow-changing signals pulsated with unique periods.
        for i in idx['pos_x']:     sig[i] = pulsate(norm_pos[0], t, 11)
        for i in idx['pos_y']:     sig[i] = pulsate(norm_pos[1], t, 13)
        for i in idx['target_x']:  sig[i] = pulsate(norm_target[0], t, 37)
        for i in idx['target_y']:  sig[i] = pulsate(norm_target[1], t, 41)
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
        contract_names = ['contract_dxp', 'contract_dxn', 'contract_dyp', 'contract_dyn']
        for d in range(4):
            for f in range(n_fibers):
                sig[idx[rest_names[d]][f]] = pulsate(restlessness[d, f], t, rest_periods[d])
                sig[idx[tire_names[d]][f]] = pulsate(tiredness[d, f], t, tire_periods[d])
                # Muscle contraction: raw fiber force normalized to 0-1
                contraction = min(1.0, per_fiber_force[d, f] / max(motor_scale, 0.01))
                sig[idx[contract_names[d]][f]] = contraction

        # Hunger disruption (disabled)
        # sig += hunger * 0.5 * rng.randn(n).astype(np.float32)

        feature_log.append((t, norm_pos[0], norm_pos[1],
                            norm_target[0], norm_target[1],
                            max(0, direction[0]), max(0, -direction[0]),
                            max(0, direction[1]), max(0, -direction[1]),
                            hunger, prox,
                            restlessness.mean(), tiredness.mean(),
                            float(is_sparse)))

        # Visual field: render and write to signal buffer
        if visual_field:
            vf = np.zeros((visual_res, visual_res), dtype=np.float32)
            pois_px = pois * _vf_scale
            ax, ay = pos[0] * _vf_scale, pos[1] * _vf_scale
            # POIs: gray contact circle + white center
            for p in range(len(pois_px)):
                dist = np.sqrt((_vf_xx - pois_px[p, 0]) ** 2 +
                               (_vf_yy - pois_px[p, 1]) ** 2)
                vf = np.maximum(vf, np.where(dist < _vf_poi_r, 0.3, 0.0))
                vf = np.maximum(vf, np.where(dist < _vf_poi_center_r, 1.0, 0.0))
            # Agent: gray circle
            dist_a = np.sqrt((_vf_xx - ax) ** 2 + (_vf_yy - ay) ** 2)
            vf = np.maximum(vf, np.where(dist_a < _vf_agent_r, 0.5, 0.0))
            sig[_vf_visual_offset:_vf_visual_offset + n_visual] = vf.ravel()

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
        'state': state,  # state['pois'] — mutable ref, updated on collection
        'sensor_indices': idx,
        'phase_ticks': phase_ticks,
        'collect_radius': collect_radius,
        'state': state,  # mutable: hunger, prev_nearest_dist
        '_refs': _refs,  # main.py sets _refs['column_mgr'] after cluster init
    }

    vf_str = f", visual={visual_res}x{visual_res}" if visual_field else ""
    print(f"  signal buffer: FORAGE synthetic "
          f"(field={field_size}, dense={n_pois_dense}, sparse={n_pois_sparse}, "
          f"phase={phase_ticks}, radius={collect_radius}, "
          f"neurons/signal={S}, sensory={N_SENSE}{vf_str})")

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
