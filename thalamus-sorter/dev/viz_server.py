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
    feedback_edges = []
    seen = set()
    for f in range(n_sensory, n_total):
        src_col = (f - n_sensory) // n_outputs
        dst_cluster = int(most_recent[f])
        edge = (int(src_col), dst_cluster)
        if edge not in seen and src_col != dst_cluster:
            seen.add(edge)
            feedback_edges.append(edge)

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

    return {
        'clusters': clusters,
        'feedback_edges': feedback_edges,
        'lateral_edges': lateral_edges,
        'layers': {k: sorted(v) for k, v in layers.items()},
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


def run_viz(port=DEFAULT_PORT):
    """Main entry point: start TCP listener + DearPyGui window."""
    try:
        import dearpygui.dearpygui as dpg
    except ImportError:
        print("ERROR: dearpygui not installed. Run: pip install dearpygui")
        sys.exit(1)

    # Shared state between network thread and render thread
    latest_graph = {'data': None, 'lock': threading.Lock()}

    # --- TCP listener thread ---
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
                    graph = analyze_graph(payload)
                    with latest_graph['lock']:
                        latest_graph['data'] = graph
            except Exception as e:
                print(f"Viz recv error: {e}")

    listener = threading.Thread(target=listen_thread, daemon=True)
    listener.start()

    # --- DearPyGui setup ---
    dpg.create_context()
    dpg.create_viewport(title="Thalamus Graph", width=1200, height=800)

    with dpg.window(label="Graph", tag="main_window"):
        dpg.add_text("Waiting for data...", tag="status_text")
        dpg.add_drawlist(width=1180, height=720, tag="canvas")

    dpg.set_primary_window("main_window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # --- Render loop ---
    prev_tick = -1

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

        with latest_graph['lock']:
            graph = latest_graph['data']

        if graph is None or graph['tick'] == prev_tick:
            continue

        prev_tick = graph['tick']
        _render_graph(dpg, graph)

    dpg.destroy_context()


def _render_graph(dpg, graph):
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
    status = (f"Tick {tick} | Clusters: {len(clusters)} | "
              f"Layers: {layer_counts} | "
              f"Edges: {len(feedback_edges)} fb, {len(lateral_edges)} lat")
    dpg.set_value("status_text", status)

    # Layout: arrange by layer (left to right)
    canvas_w, canvas_h = 1180, 720
    margin = 60

    layer_order = ['V1'] + [f'V{i}' for i in range(2, 20)] + ['?']
    active_layers = [l for l in layer_order if l in layers and layers[l]]

    if not active_layers:
        return

    # X position per layer
    layer_x = {}
    n_layers = len(active_layers)
    for i, lname in enumerate(active_layers):
        layer_x[lname] = margin + (canvas_w - 2 * margin) * i / max(n_layers - 1, 1)

    # Y position: spread within layer
    cluster_pos = {}
    for lname in active_layers:
        cids = sorted(layers[lname])
        n = len(cids)
        for j, cid in enumerate(cids):
            y = margin + (canvas_h - 2 * margin) * (j + 0.5) / max(n, 1)
            cluster_pos[cid] = (layer_x[lname], y)

    # Draw edges first (behind nodes)
    # Feedback edges (directed, semi-transparent)
    for src, dst in feedback_edges:
        if src in cluster_pos and dst in cluster_pos:
            x1, y1 = cluster_pos[src]
            x2, y2 = cluster_pos[dst]
            dpg.draw_line((x1, y1), (x2, y2),
                         color=(100, 100, 100, 40), thickness=1,
                         parent="canvas")

    # Lateral edges (undirected, brighter)
    for a, b in lateral_edges:
        if a in cluster_pos and b in cluster_pos:
            x1, y1 = cluster_pos[a]
            x2, y2 = cluster_pos[b]
            dpg.draw_line((x1, y1), (x2, y2),
                         color=(200, 200, 50, 80), thickness=1,
                         parent="canvas")

    # Draw nodes
    for cl in clusters:
        cid = cl['id']
        if cid not in cluster_pos:
            continue
        x, y = cluster_pos[cid]
        radius = max(4, min(20, cl['size'] ** 0.5 * 2))
        color = _get_layer_color(cl['layer'])

        # Node circle
        dpg.draw_circle((x, y), radius, color=color, fill=color,
                        parent="canvas")

        # Winner indicator (small colored dot in center)
        if 'winner' in cl:
            winner_colors = [(255, 80, 80), (80, 255, 80),
                           (80, 80, 255), (255, 255, 80)]
            wc = winner_colors[cl['winner'] % len(winner_colors)]
            dpg.draw_circle((x, y), max(2, radius * 0.4),
                          color=wc, fill=wc, parent="canvas")

    # Layer labels
    for lname in active_layers:
        x = layer_x[lname]
        dpg.draw_text((x - 10, 10), lname, size=16,
                     color=(255, 255, 255), parent="canvas")


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
