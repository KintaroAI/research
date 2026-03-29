"""Live graph visualization server for thalamus-sorter.

Receives graph payloads from the render server via TCP and displays
the cluster/column/feedback graph in real-time using DearPyGui.

Architecture:
    Model → render server (Unix socket) → viz_server (TCP) → DearPyGui window

Usage:
    python viz_server.py [--port 9100]

The server listens for length-prefixed pickle payloads containing:
    most_recent: (n_total,) cluster assignment per neuron
    n_sensory:   int — split between sensory and feedback neurons
    n_outputs:   int — column outputs per cluster
    lateral_adj: (m, K) lateral neighbor indices (optional)
    column_outputs: (m, n_outputs) current column output probabilities

All graph structure (layers, edges, sizes) is derived on this side.
"""

import sys
import socket
import struct
import pickle
import threading
import time
import numpy as np

DEFAULT_PORT = 9100


# ---------------------------------------------------------------------------
# Protocol (same as render_server)
# ---------------------------------------------------------------------------

def _recv_msg(sock):
    """Receive length-prefixed pickled data."""
    raw_len = b''
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            return None
        raw_len += chunk
    msg_len = struct.unpack('!I', raw_len)[0]
    data = b''
    while len(data) < msg_len:
        chunk = sock.recv(min(msg_len - len(data), 65536))
        if not chunk:
            return None
        data += chunk
    return pickle.loads(data)


# ---------------------------------------------------------------------------
# Graph analysis — derive structure from raw arrays
# ---------------------------------------------------------------------------

def analyze_graph(payload):
    """Derive full graph structure from the visualization payload.

    Returns dict with:
        clusters: list of {id, size, sensory_count, feedback_count, layer, outputs}
        feedback_edges: list of (src_cluster, dst_cluster) — directed
        lateral_edges: list of (cluster_a, cluster_b) — undirected
        layers: dict layer_name → set of cluster ids
    """
    most_recent = payload['most_recent']
    n_sensory = payload['n_sensory']
    n_outputs = payload['n_outputs']
    lateral_adj = payload.get('lateral_adj')
    column_outputs = payload.get('column_outputs')

    n_total = len(most_recent)
    m = int(most_recent.max()) + 1 if len(most_recent) > 0 else 0

    # Per-cluster neuron counts
    sensory_counts = np.zeros(m, dtype=int)
    feedback_counts = np.zeros(m, dtype=int)
    for i in range(n_total):
        c = most_recent[i]
        if i < n_sensory:
            sensory_counts[c] += 1
        else:
            feedback_counts[c] += 1

    # Layer classification
    v1 = set(c for c in range(m) if sensory_counts[c] > 0)
    assigned = set(v1)
    layers = {'V1': v1}
    prev_layer = v1
    depth = 2

    while True:
        next_layer = set()
        for c in range(m):
            if c in assigned or feedback_counts[c] == 0:
                continue
            fb = np.where((most_recent == c) & (np.arange(n_total) >= n_sensory))[0]
            src = set((fb - n_sensory) // n_outputs)
            if src & prev_layer:
                next_layer.add(c)
        if not next_layer:
            break
        name = f'V{depth}'
        layers[name] = next_layer
        assigned |= next_layer
        prev_layer = next_layer
        depth += 1

    # Unassigned clusters
    unassigned = set(range(m)) - assigned
    if unassigned:
        layers['?'] = unassigned

    def _layer_of(c):
        for name, s in layers.items():
            if c in s:
                return name
        return '?'

    # Build cluster info
    clusters = []
    for c in range(m):
        size = sensory_counts[c] + feedback_counts[c]
        if size == 0:
            continue
        info = {
            'id': c,
            'size': int(size),
            'sensory': int(sensory_counts[c]),
            'feedback': int(feedback_counts[c]),
            'layer': _layer_of(c),
        }
        if column_outputs is not None:
            info['outputs'] = column_outputs[c].tolist()
            info['winner'] = int(column_outputs[c].argmax())
        clusters.append(info)

    # Feedback edges: feedback neuron f in cluster C2 came from column C1
    # Track which output indices flow on each edge
    feedback_info = {}  # (src, dst) → set of output indices
    for f in range(n_sensory, n_total):
        src_col = (f - n_sensory) // n_outputs
        out_idx = (f - n_sensory) % n_outputs
        dst_cluster = int(most_recent[f])
        if src_col != dst_cluster:
            edge = (int(src_col), dst_cluster)
            if edge not in feedback_info:
                feedback_info[edge] = set()
            feedback_info[edge].add(out_idx)
    feedback_edges = [(src, dst, sorted(outs))
                      for (src, dst), outs in feedback_info.items()]

    # Lateral edges
    lateral_edges = []
    if lateral_adj is not None:
        seen_lat = set()
        for c in range(min(m, len(lateral_adj))):
            for nb in lateral_adj[c]:
                nb = int(nb)
                if nb >= 0 and nb != c:
                    edge = (min(c, nb), max(c, nb))
                    if edge not in seen_lat:
                        seen_lat.add(edge)
                        lateral_edges.append(edge)

    # KNN edges (cluster-level nearest neighbors)
    knn2 = payload.get('knn2')
    knn_edges = []
    if knn2 is not None:
        seen_knn = set()
        for c in range(min(m, len(knn2))):
            for nb in knn2[c]:
                nb = int(nb)
                if nb >= 0 and nb != c:
                    edge = (min(c, nb), max(c, nb))
                    if edge not in seen_knn:
                        seen_knn.add(edge)
                        knn_edges.append(edge)

    return {
        'clusters': clusters,
        'feedback_edges': feedback_edges,
        'lateral_edges': lateral_edges,
        'knn_edges': knn_edges,
        'layers': {k: sorted(v) for k, v in layers.items()},
        'centroids': payload.get('centroids'),
        'tick': payload.get('tick', 0),
    }


# ---------------------------------------------------------------------------
# DearPyGui visualization
# ---------------------------------------------------------------------------

# Layer colors for nodes
LAYER_COLORS = {
    'V1': (80, 180, 80),     # green
    'V2': (80, 130, 220),    # blue
    'V3': (200, 130, 50),    # orange
    'V4': (180, 80, 180),    # purple
    'V5': (200, 200, 60),    # yellow
    'V6': (60, 200, 200),    # cyan
    '?':  (120, 120, 120),   # gray
}


def _get_layer_color(layer):
    return LAYER_COLORS.get(layer, (120, 120, 120))


class ForceLayout:
    """Force-directed graph layout with persistent positions.

    Nodes repel each other (Coulomb), connected nodes attract (spring).
    Positions persist across graph updates — new nodes get random initial
    positions, removed nodes are forgotten.
    """

    def __init__(self, canvas_w=1180, canvas_h=720, margin=40):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.margin = margin
        self.positions = {}  # cluster_id → (x, y)
        self.velocities = {}  # cluster_id → (vx, vy)

    def update(self, graph, steps=50):
        """Run force simulation for N steps, return {cluster_id: (x, y)}."""
        clusters = graph['clusters']
        feedback_edges = graph['feedback_edges']
        lateral_edges = graph['lateral_edges']

        alive = set(cl['id'] for cl in clusters)
        cx = (self.canvas_w) / 2
        cy = (self.canvas_h) / 2
        m = self.margin

        # Project centroids to 2D for initial positions
        centroid_2d = self._project_centroids(graph, alive)

        # Init new nodes, remove dead ones
        for cid in alive:
            if cid not in self.positions:
                if cid in centroid_2d:
                    self.positions[cid] = centroid_2d[cid]
                else:
                    self.positions[cid] = (
                        cx + (np.random.rand() - 0.5) * (self.canvas_w - 2 * m),
                        cy + (np.random.rand() - 0.5) * (self.canvas_h - 2 * m))
                self.velocities[cid] = (0.0, 0.0)
        for cid in list(self.positions):
            if cid not in alive:
                del self.positions[cid]
                self.velocities.pop(cid, None)

        if len(alive) < 2:
            return self.positions

        # Build edge set for attraction (knn = primary, feedback/lateral = secondary)
        knn_edges = graph.get('knn_edges', [])
        edges_strong = set()  # knn — strong attraction
        edges_weak = set()    # feedback + lateral — weak attraction
        for a, b in knn_edges:
            if a in alive and b in alive:
                edges_strong.add((a, b))
        for edge in feedback_edges:
            src, dst = edge[0], edge[1]
            if src in alive and dst in alive:
                edges_weak.add((src, dst))
        for a, b in lateral_edges:
            if a in alive and b in alive:
                edges_weak.add((a, b))

        # Simulation parameters
        # With ~100 nodes and ~800 knn edges, attraction needs to be weak
        # relative to repulsion to avoid clumping
        repulsion = 100000.0
        attraction = 0.002
        damping = 0.85
        max_speed = 25.0
        ids = sorted(alive)

        for _ in range(steps):
            forces = {cid: [0.0, 0.0] for cid in ids}

            # Repulsion (all pairs)
            for i, a in enumerate(ids):
                ax, ay = self.positions[a]
                for b in ids[i+1:]:
                    bx, by = self.positions[b]
                    dx, dy = ax - bx, ay - by
                    dist2 = max(dx*dx + dy*dy, 1.0)
                    f = repulsion / dist2
                    dist = dist2 ** 0.5
                    fx, fy = f * dx / dist, f * dy / dist
                    forces[a][0] += fx
                    forces[a][1] += fy
                    forces[b][0] -= fx
                    forces[b][1] -= fy

            # Attraction — knn edges (strong)
            for a, b in edges_strong:
                ax, ay = self.positions[a]
                bx, by = self.positions[b]
                dx, dy = bx - ax, by - ay
                dist = max((dx*dx + dy*dy) ** 0.5, 0.1)
                f = attraction * dist
                fx, fy = f * dx / dist, f * dy / dist
                forces[a][0] += fx
                forces[a][1] += fy
                forces[b][0] -= fx
                forces[b][1] -= fy

            # Attraction — feedback/lateral edges (weaker)
            for a, b in edges_weak:
                ax, ay = self.positions[a]
                bx, by = self.positions[b]
                dx, dy = bx - ax, by - ay
                dist = max((dx*dx + dy*dy) ** 0.5, 0.1)
                f = attraction * 0.3 * dist
                fx, fy = f * dx / dist, f * dy / dist
                forces[a][0] += fx
                forces[a][1] += fy
                forces[b][0] -= fx
                forces[b][1] -= fy

            # Center gravity (gentle pull toward center)
            for cid in ids:
                px, py = self.positions[cid]
                forces[cid][0] += (cx - px) * 0.001
                forces[cid][1] += (cy - py) * 0.001

            # Update velocities and positions
            m = self.margin
            for cid in ids:
                vx, vy = self.velocities[cid]
                vx = (vx + forces[cid][0]) * damping
                vy = (vy + forces[cid][1]) * damping
                speed = (vx*vx + vy*vy) ** 0.5
                if speed > max_speed:
                    vx *= max_speed / speed
                    vy *= max_speed / speed
                self.velocities[cid] = (vx, vy)
                px, py = self.positions[cid]
                px = max(m, min(self.canvas_w - m, px + vx))
                py = max(m, min(self.canvas_h - m, py + vy))
                self.positions[cid] = (px, py)

        return self.positions

    def _project_centroids(self, graph, alive):
        """Project D-dim centroids to 2D canvas positions via PCA."""
        centroids = graph.get('centroids')
        if centroids is None:
            return {}

        # Gather alive cluster centroids
        ids = sorted(alive)
        valid = [cid for cid in ids if cid < len(centroids)]
        if len(valid) < 2:
            return {}

        pts = centroids[valid]  # (n_alive, D)

        # PCA to 2D
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = centered.T @ centered / max(len(centered) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Top 2 components (eigenvalues sorted ascending)
        proj = centered @ eigvecs[:, -2:]  # (n, 2)

        # Scale to canvas
        m = self.margin
        for dim in range(2):
            pmin, pmax = proj[:, dim].min(), proj[:, dim].max()
            span = max(pmax - pmin, 1e-8)
            proj[:, dim] = m + (proj[:, dim] - pmin) / span * (self.canvas_w - 2*m if dim == 0 else self.canvas_h - 2*m)

        return {cid: (float(proj[i, 0]), float(proj[i, 1]))
                for i, cid in enumerate(valid)}

    def reset(self):
        """Clear all positions — next update will reinitialize."""
        self.positions.clear()
        self.velocities.clear()


def run_viz(port=DEFAULT_PORT):
    """Main entry point: start TCP listener + DearPyGui window."""
    try:
        import dearpygui.dearpygui as dpg
    except ImportError:
        print("ERROR: dearpygui not installed. Run: pip install dearpygui")
        sys.exit(1)

    # Shared state: raw payload (TCP thread) → analyzed graph (analysis thread)
    latest_payload = {'data': None, 'lock': threading.Lock()}
    latest_graph = {'data': None, 'lock': threading.Lock()}

    # --- TCP listener thread (fast: just receive and store) ---
    def listen_thread():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', port))
        srv.listen(4)
        srv.settimeout(1.0)
        print(f"Viz server listening on port {port}")

        while True:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = _recv_msg(conn)
                conn.close()
                if payload is not None:
                    with latest_payload['lock']:
                        latest_payload['data'] = payload
            except Exception as e:
                print(f"Viz recv error: {e}")

    listener = threading.Thread(target=listen_thread, daemon=True)
    listener.start()

    # --- Analysis thread (picks up latest payload, skips if busy) ---
    def analysis_thread():
        prev_tick = -1
        while True:
            with latest_payload['lock']:
                payload = latest_payload['data']
            if payload is not None and payload.get('tick', 0) != prev_tick:
                prev_tick = payload.get('tick', 0)
                graph = analyze_graph(payload)
                with latest_graph['lock']:
                    latest_graph['data'] = graph
            time.sleep(0.005)

    analyzer = threading.Thread(target=analysis_thread, daemon=True)
    analyzer.start()

    # --- DearPyGui setup ---
    dpg.create_context()
    dpg.create_viewport(title="Thalamus Graph", width=1200, height=800)

    # Scale UI for high-DPI
    dpg.set_global_font_scale(2.0)

    layout = ForceLayout()
    reset_flag = [False]

    def _on_reset():
        reset_flag[0] = True

    with dpg.window(label="Graph", tag="main_window"):
        with dpg.group(horizontal=True):
            dpg.add_text("Waiting for data...", tag="status_text")
            dpg.add_button(label="Rearrange", callback=_on_reset)
            dpg.add_text("", tag="fps_text")
        dpg.add_drawlist(width=1180, height=700, tag="canvas")

    dpg.set_primary_window("main_window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # --- Render loop ---
    prev_tick = -1
    current_graph = None
    frame_count = 0
    fps_time = time.time()
    needs_render = False

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

        # FPS counter
        frame_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            dpg.set_value("fps_text", f"  FPS: {fps:.0f}")
            frame_count = 0
            fps_time = now

        # Resize canvas to match viewport
        vw = dpg.get_viewport_width() - 20
        vh = dpg.get_viewport_height() - 80
        if vw > 100 and vh > 100:
            dpg.configure_item("canvas", width=vw, height=vh)
            layout.canvas_w = vw
            layout.canvas_h = vh

        with latest_graph['lock']:
            new_graph = latest_graph['data']

        # New data arrived — update visuals but keep positions
        if new_graph is not None and new_graph['tick'] != prev_tick:
            current_graph = new_graph
            prev_tick = new_graph['tick']
            # First time: project from centroids
            if not layout.positions:
                alive = set(cl['id'] for cl in current_graph['clusters'])
                positions = layout._project_centroids(current_graph, alive)
                if positions:
                    layout.positions = positions
            needs_render = True

        # Rearrange button — re-project from current centroids
        if reset_flag[0]:
            reset_flag[0] = False
            if current_graph is not None:
                alive = set(cl['id'] for cl in current_graph['clusters'])
                positions = layout._project_centroids(current_graph, alive)
                if positions:
                    layout.positions = positions
                needs_render = True

        if current_graph is not None and needs_render:
            _render_graph(dpg, current_graph, layout.positions)
            needs_render = False

    dpg.destroy_context()


def _render_graph(dpg, graph, positions):
    """Render the graph onto the DearPyGui canvas."""
    clusters = graph['clusters']
    feedback_edges = graph['feedback_edges']
    lateral_edges = graph['lateral_edges']
    layers = graph['layers']
    tick = graph['tick']

    if not clusters:
        return

    # Clear canvas
    dpg.delete_item("canvas", children_only=True)

    # Status
    layer_counts = {k: len(v) for k, v in layers.items()}
    total_fb_neurons = sum(len(outs) for _, _, outs in feedback_edges)
    status = (f"Tick {tick} | Clusters: {len(clusters)} | "
              f"Layers: {layer_counts} | "
              f"Edges: {len(feedback_edges)} fb ({total_fb_neurons} neurons), "
              f"{len(lateral_edges)} lat")
    dpg.set_value("status_text", status)

    # Build cluster output info for edge coloring
    cluster_map = {cl['id']: cl for cl in clusters}
    winner_colors = [(255, 80, 80), (80, 255, 80),
                     (80, 80, 255), (255, 255, 80)]

    # Draw edges first (behind nodes)
    # Feedback edges — colored by winning output, thickness by probability
    for src, dst, out_indices in feedback_edges:
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        cl = cluster_map.get(src)
        if cl and 'outputs' in cl:
            winner = cl['winner']
            prob = cl['outputs'][winner]
            if winner in out_indices:
                # Active edge — winner output flows here
                color = winner_colors[winner % len(winner_colors)]
                t = max(1, int(prob * 6))  # 1-6px by probability
                alpha = min(255, int(80 + prob * 175))
                dpg.draw_line((x1, y1), (x2, y2),
                             color=(color[0], color[1], color[2], alpha),
                             thickness=t, parent="canvas")
            else:
                # Inactive edge — winner doesn't flow here
                dpg.draw_line((x1, y1), (x2, y2),
                             color=(80, 80, 80, 25), thickness=1,
                             parent="canvas")
        else:
            dpg.draw_line((x1, y1), (x2, y2),
                         color=(80, 80, 80, 25), thickness=1,
                         parent="canvas")

    # Lateral edges
    for a, b in lateral_edges:
        if a in positions and b in positions:
            x1, y1 = positions[a]
            x2, y2 = positions[b]
            dpg.draw_line((x1, y1), (x2, y2),
                         color=(200, 200, 50, 60), thickness=1,
                         parent="canvas")

    # Draw nodes
    for cl in clusters:
        cid = cl['id']
        if cid not in positions:
            continue
        x, y = positions[cid]
        radius = max(4, min(20, cl['size'] ** 0.5 * 2))
        color = _get_layer_color(cl['layer'])

        dpg.draw_circle((x, y), radius, color=color, fill=color,
                        parent="canvas")

        # Winner indicator
        if 'winner' in cl:
            winner_colors = [(255, 80, 80), (80, 255, 80),
                           (80, 80, 255), (255, 255, 80)]
            wc = winner_colors[cl['winner'] % len(winner_colors)]
            dpg.draw_circle((x, y), max(2, radius * 0.4),
                          color=wc, fill=wc, parent="canvas")

    # Legend
    y_leg = 10
    for lname in sorted(layers.keys()):
        color = _get_layer_color(lname)
        dpg.draw_circle((20, y_leg + 8), 8, color=color, fill=color,
                        parent="canvas")
        dpg.draw_text((36, y_leg), f"{lname} ({len(layers[lname])})",
                     size=20, color=(220, 220, 220), parent="canvas")
        y_leg += 28


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Thalamus graph visualization")
    p.add_argument('--port', type=int, default=DEFAULT_PORT,
                   help=f"TCP port to listen on (default: {DEFAULT_PORT})")
    args = p.parse_args()
    run_viz(port=args.port)
