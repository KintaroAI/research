"""Training loop: shared tick/log/report loop for all render paths.

Extracted from main.py — pure loop logic with no CLI dependencies.
All configuration passed as explicit parameters.
"""

import os
import time
import numpy as np


def run_training_loop(do_tick, dsolver, max_frames, sig_channels, wlog,
                      log_every=1000, knn_report_every=1000,
                      cluster_report_every=1000, save_every=1000,
                      n_sensory=None, w=None,
                      on_save=None, on_display=None, can_break=None,
                      cluster_mgr=None, bench_metadata=None,
                      viz_every=0):
    """Shared training loop for async and sync render paths.

    Args:
        do_tick: callable returning number of pairs processed
        dsolver: DriftSolver instance
        max_frames: total ticks to run
        sig_channels: number of signal channels
        wlog: WandbLogger instance
        log_every: print tick progress every N ticks (0=disabled)
        knn_report_every: report KNN stability every N ticks
        cluster_report_every: report cluster metrics every N ticks
        save_every: call on_save every N ticks
        n_sensory: number of sensory neurons (for KNN eval)
        w: grid width (for KNN spatial accuracy)
        on_save(tick, total_pairs): called every save_every ticks
        on_display(tick): called every tick
        can_break(): return True to exit loop early
        cluster_mgr: ClusterManager instance (optional)
        bench_metadata: benchmark metadata dict (optional)
        viz_every: send viz data every N ticks (0=disabled)

    Returns:
        (t0, total_pairs)
    """
    n = dsolver.positions.shape[0]
    t0 = time.time()
    total_pairs = 0
    prev_pairs = 0
    t_log = t0

    _debug_timing = os.environ.get('DEBUG_TICK_TIMING')
    _tick_threshold_ms = 20.0  # only print ticks slower than this

    for tick in range(1, max_frames + 1):
        if _debug_timing:
            _t0_tick = time.perf_counter()

        total_pairs += do_tick()

        if _debug_timing:
            _t_dotick = time.perf_counter()

        # Live cluster maintenance
        if cluster_mgr is not None:
            if not cluster_mgr.initialized:
                if cluster_mgr.knn2_mode == 'knn':
                    if dsolver.knn_k > 0:
                        knn_np = dsolver.get_knn_lists()
                        cluster_mgr.init_clusters(dsolver.positions, knn_np)
                    else:
                        print("  Warning: knn2_mode='knn' requires --knn-track, "
                              "falling back to incremental")
                        cluster_mgr.knn2_mode = 'incremental'
                        cluster_mgr.init_clusters(dsolver.positions)
                else:
                    cluster_mgr.init_clusters(dsolver.positions)
            if cluster_mgr.initialized:
                anchors_np = dsolver._last_anchors.cpu().numpy()
                pairs = getattr(dsolver, '_last_pairs', None)
                cluster_mgr.tick(dsolver.positions, anchors_np, pairs, tick)

                if _debug_timing:
                    _t_cluster = time.perf_counter()

                # Send graph + field to viz (non-blocking, skips if busy)
                if (viz_every > 0 and tick % viz_every == 0
                        and cluster_mgr._renderer is not None
                        and not cluster_mgr._renderer._send_busy()):
                    r = cluster_mgr._renderer
                    if r.viz_address and cluster_mgr.column_mgr:
                        most_recent = cluster_mgr.cluster_ids[
                            np.arange(cluster_mgr.n), cluster_mgr.pointers]
                        lateral_adj = (cluster_mgr.column_mgr.lateral_adj
                                       if cluster_mgr.column_mgr.lateral else None)
                        knn2_viz = (cluster_mgr.knn2 if cluster_mgr.knn2_mode == 'knn'
                                    else cluster_mgr.knn2_t.cpu().numpy())
                        r.graph(tick, most_recent,
                                cluster_mgr.n_sensory,
                                cluster_mgr.column_n_outputs,
                                lateral_adj=lateral_adj,
                                column_outputs=cluster_mgr.column_mgr.get_outputs(),
                                knn2=knn2_viz,
                                centroids=cluster_mgr.centroids_t.cpu().numpy())
                    if (r.field_address and bench_metadata is not None
                            and 'pos' in bench_metadata):
                        bm = bench_metadata
                        bm_state = bm.get('state', {})
                        r.field_live(
                            tick, bm['pos'].copy(),
                            bm_state.get('pois', np.empty((0, 2))).copy(),
                            bm['field_size'],
                            hunger=float(bm_state.get('hunger', [0])[0]),
                            collect_radius=bm.get('collect_radius', 5.0),
                            score=int(bm.get('score', [0])[0]),
                            visual_field=bm_state.get('_visual_field'),
                            blocked=bm.get('blocked'))

                if _debug_timing:
                    _t_viz = time.perf_counter()
                    _total_ms = (_t_viz - _t0_tick) * 1000
                    if _total_ms > _tick_threshold_ms:
                        _dotick_ms = (_t_dotick - _t0_tick) * 1000
                        _cluster_ms = (_t_cluster - _t_dotick) * 1000
                        _viz_ms = (_t_viz - _t_cluster) * 1000
                        print(f"  SLOW tick {tick}: {_total_ms:.1f}ms "
                              f"(signal={_dotick_ms:.1f} cluster={_cluster_ms:.1f} "
                              f"viz={_viz_ms:.1f})")

                if tick % cluster_report_every == 0:
                    # Refresh knn_lists for knn mode
                    if cluster_mgr.knn2_mode == 'knn' and dsolver.knn_k > 0:
                        cluster_mgr.knn_lists = dsolver.get_knn_lists()
                    cluster_mgr.report(tick)
                    # Periodic lateral knn2 sync (not every split -- too disruptive)
                    if (cluster_mgr.column_mgr and cluster_mgr.column_mgr.lateral
                            and cluster_mgr.knn2_mode != 'knn'):
                        knn2_np = cluster_mgr.knn2_t.cpu().numpy()
                        cluster_mgr.column_mgr.sync_lateral_knn2(knn2_np)

        # Tick progress logging
        if log_every > 0 and tick % log_every == 0:
            now = time.time()
            elapsed = now - t0
            ms_tick = (now - t_log) / log_every * 1000
            pairs_per_tick = (total_pairs - prev_pairs) / log_every
            print(f"  tick {tick}/{max_frames} "
                  f"({elapsed:.1f}s, {ms_tick:.1f} ms/tick, "
                  f"pairs={total_pairs}, {pairs_per_tick:.0f}/tick)")
            wlog.log_tick(tick, elapsed, total_pairs, ms_tick, pairs_per_tick)
            t_log = now
            prev_pairs = total_pairs

        # KNN stability reporting
        if dsolver.knn_k > 0 and tick % knn_report_every == 0:
            dsolver._refresh_knn_dists()
            overlap, n_changed, top50_swaps, top90_swaps = dsolver.knn_stability()
            spatial_acc = dsolver.knn_spatial_accuracy(w, radius=3, channels=sig_channels,
                                                       n_eval=n_sensory)
            dsolver._knn_overlap_history.append((tick, overlap, spatial_acc))
            lr_str = f" lr={dsolver.lr:.6f}" if dsolver.lr_decay < 1.0 else ""
            print(f"  KNN @ tick {tick}: overlap={overlap:.3f} "
                  f"spatial={spatial_acc:.3f} "
                  f"({n_changed}/{n} changed) "
                  f"swaps: top50={top50_swaps:.1f} top90={top90_swaps:.1f}{lr_str}")
            wlog.log_knn(tick, overlap, spatial_acc, n_changed, n,
                         top50_swaps, top90_swaps,
                         lr=dsolver.lr if dsolver.lr_decay < 1.0 else None)

        if on_save is not None and tick % save_every == 0:
            on_save(tick, total_pairs)

        if on_display is not None:
            on_display(tick)

        if can_break is not None and can_break():
            break

    return t0, total_pairs
