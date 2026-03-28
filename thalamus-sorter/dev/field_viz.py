"""Live field visualization server for thalamus-sorter foraging.

Shows agent position, POIs, hunger, and collection events in real-time.
Receives data from the render server via TCP relay.

Usage:
    python field_viz.py [--port 9101]
"""

import sys
import socket
import struct
import pickle
import threading
import time
import numpy as np

DEFAULT_PORT = 9101


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


def run_field_viz(port=DEFAULT_PORT):
    """Main entry point: TCP listener + DearPyGui field display."""
    try:
        import dearpygui.dearpygui as dpg
    except ImportError:
        print("ERROR: dearpygui not installed. Run: pip install dearpygui")
        sys.exit(1)

    latest_data = {'data': None, 'lock': threading.Lock()}
    trail = []  # list of (x, y) for agent trail
    max_trail = 500

    # --- TCP listener ---
    def listen_thread():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', port))
        srv.listen(4)
        srv.settimeout(1.0)
        print(f"Field viz listening on port {port}")

        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = _recv_msg(conn)
                conn.close()
                if payload is not None:
                    with latest_data['lock']:
                        latest_data['data'] = payload
            except Exception as e:
                print(f"Field recv error: {e}")

    listener = threading.Thread(target=listen_thread, daemon=True)
    listener.start()

    # --- DearPyGui ---
    dpg.create_context()
    dpg.create_viewport(title="Foraging Field", width=800, height=800)
    dpg.set_global_font_scale(2.0)

    # --- Controls server for live parameter tuning ---
    import json as _json
    controls = {'lr': 0.001, 'column_lr': 0.05}
    controls_lock = threading.Lock()
    controls_port = port + 1  # 9102 by default

    def _update_controls():
        with controls_lock:
            controls['lr'] = dpg.get_value("ctrl_lr")
            controls['column_lr'] = dpg.get_value("ctrl_col_lr")

    def controls_thread():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', controls_port))
        srv.listen(4)
        srv.settimeout(1.0)
        print(f"Controls server listening on port {controls_port}")
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                with controls_lock:
                    data = _json.dumps(controls).encode()
                conn.sendall(data)
                conn.close()
            except Exception:
                pass

    ctl_thread = threading.Thread(target=controls_thread, daemon=True)
    ctl_thread.start()

    with dpg.window(label="Field", tag="field_window"):
        dpg.add_text("Waiting for data...", tag="field_status")

        # LR controls
        with dpg.group(horizontal=True):
            dpg.add_text("embed lr:")
            dpg.add_slider_float(tag="ctrl_lr", default_value=0.001,
                                 min_value=0.0, max_value=0.01,
                                 width=200, callback=lambda: _update_controls())
        with dpg.group(horizontal=True):
            dpg.add_text("col lr:  ")
            dpg.add_slider_float(tag="ctrl_col_lr", default_value=0.05,
                                 min_value=0.0, max_value=0.5,
                                 width=200, callback=lambda: _update_controls())

        dpg.add_drawlist(width=760, height=660, tag="field_canvas")

    dpg.set_primary_window("field_window", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    prev_tick = -1
    frame_count = 0
    fps_time = time.time()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

        # FPS
        frame_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now
        else:
            fps = 0

        # Resize
        vw = dpg.get_viewport_width() - 40
        vh = dpg.get_viewport_height() - 100
        canvas_size = max(100, min(vw, vh))
        dpg.configure_item("field_canvas", width=canvas_size, height=canvas_size)

        with latest_data['lock']:
            data = latest_data['data']

        if data is None or data.get('tick', 0) == prev_tick:
            continue

        prev_tick = data.get('tick', 0)

        # Extract field data
        agent_pos = data.get('agent_pos')
        pois = data.get('pois')
        field_size = data.get('field_size', 100)
        hunger = data.get('hunger', 0)
        collect_radius = data.get('collect_radius', 5.0)
        score = data.get('score', 0)

        if agent_pos is None:
            continue

        # Update trail
        trail.append((float(agent_pos[0]), float(agent_pos[1])))
        if len(trail) > max_trail:
            trail.pop(0)

        # Scale factor: field coords → canvas coords
        margin = 20
        scale = (canvas_size - 2 * margin) / field_size

        def to_canvas(x, y):
            return (margin + x * scale, margin + y * scale)

        # Clear and draw
        dpg.delete_item("field_canvas", children_only=True)

        # Status
        dpg.set_value("field_status",
                      f"Tick {prev_tick} | Score: {score} | "
                      f"Hunger: {hunger:.2f} | FPS: {fps:.0f}")

        # Background
        dpg.draw_rectangle((margin, margin),
                          (margin + field_size * scale, margin + field_size * scale),
                          color=(40, 40, 40), fill=(20, 20, 20),
                          parent="field_canvas")

        # Trail (fading)
        for i in range(1, len(trail)):
            alpha = int(40 + 160 * i / len(trail))
            x1, y1 = to_canvas(*trail[i-1])
            x2, y2 = to_canvas(*trail[i])
            dpg.draw_line((x1, y1), (x2, y2),
                         color=(100, 100, 255, alpha), thickness=1,
                         parent="field_canvas")

        # POIs
        if pois is not None and len(pois) > 0:
            for px, py in pois:
                cx, cy = to_canvas(float(px), float(py))
                r = collect_radius * scale
                dpg.draw_circle((cx, cy), r,
                               color=(50, 200, 50, 40), fill=(50, 200, 50, 20),
                               parent="field_canvas")
                dpg.draw_circle((cx, cy), 3,
                               color=(50, 255, 50), fill=(50, 255, 50),
                               parent="field_canvas")

        # Agent
        ax, ay = to_canvas(float(agent_pos[0]), float(agent_pos[1]))
        # Hunger ring — red when hungry
        hunger_color = (int(255 * hunger), int(255 * (1 - hunger)), 0)
        dpg.draw_circle((ax, ay), 8,
                       color=hunger_color, thickness=3,
                       parent="field_canvas")
        dpg.draw_circle((ax, ay), 3,
                       color=(255, 255, 255), fill=(255, 255, 255),
                       parent="field_canvas")

    dpg.destroy_context()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Foraging field visualization")
    p.add_argument('--port', type=int, default=DEFAULT_PORT,
                   help=f"TCP port (default: {DEFAULT_PORT})")
    args = p.parse_args()
    run_field_viz(port=args.port)
