"""Persistent async render server for thalamus-sorter.

Runs as a single background process, accepts render jobs via Unix socket.
Survives across training runs — check if running before spawning.

Usage:
    # Start server (normally done automatically by main.py)
    python render_server.py

    # From training code:
    from render_server import RenderClient
    client = RenderClient.connect_or_spawn()
    client.submit('grid', output_path, embeddings=emb, pixel_values=pv, ...)
    client.shutdown()  # optional — server stays alive for next run
"""

import os
import sys
import pickle
import signal
import struct
import time
import socket
import numpy as np

SOCK_PATH = f"/tmp/thalamus-render-{os.getuid()}.sock"
PID_PATH = f"/tmp/thalamus-render-{os.getuid()}.pid"
IDLE_TIMEOUT = 300  # shutdown after 5 min idle


def _send_msg(sock, data):
    """Send length-prefixed pickled data."""
    blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack('!I', len(blob)) + blob)


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
# Render functions (run in server process, no GPU needed)
# ---------------------------------------------------------------------------

def _render_grid(job):
    """Voronoi grid render: project embeddings to 2D, assign pixels."""
    from render_embeddings import project, align_to_grid, render
    import cv2

    emb = job['embeddings']
    pv = job['pixel_values']
    w, h = job['width'], job['height']
    method = job.get('method', 'pca')
    prev_2d = job.get('prev_2d', None)
    do_align = job.get('align', False)
    gpu = job.get('gpu', False)

    pos_2d = project(emb, w, h, method, prev_2d=prev_2d, gpu=gpu)
    if do_align:
        pos_2d = align_to_grid(pos_2d, w, h)

    frame = render(pos_2d, w, h, pv)
    if frame is not None:
        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U)
        cv2.imwrite(job['output_path'], normalized)

    return pos_2d  # for warm-start chaining


def _render_cluster(job):
    """Color-coded cluster map."""
    from cluster_experiments import visualize_clusters
    visualize_clusters(job['cluster_ids'], job['width'], job['height'],
                       path=job['output_path'])


def _render_cluster_signal(job):
    """Cluster signal map — mean signal per cluster."""
    from cluster_experiments import visualize_clusters_signal
    visualize_clusters_signal(job['cluster_ids'], job['signal'],
                              job['width'], job['height'],
                              sig_channels=job.get('sig_channels', 1),
                              path=job['output_path'])


def _render_embed(job):
    """Scatter plot of all neurons in embedding space."""
    from render_embeddings import render_embed
    import cv2

    frame = render_embed(
        job['embeddings'], job['n_sensory'],
        pixel_values=job.get('pixel_values'),
        cluster_ids=job.get('cluster_ids'),
        n_outputs=job.get('n_outputs', 4),
        img_size=job.get('img_size', 800),
        method=job.get('method', 'pca'))
    cv2.imwrite(job['output_path'], frame)


def _render_heatmap(job):
    """Motor position heatmap."""
    import cv2
    positions = job['positions']  # (N, 2) int
    x_max = int(positions[:, 0].max()) + 1
    y_max = int(positions[:, 1].max()) + 1

    img = np.zeros((y_max, x_max, 3), dtype=np.uint8)
    for x, y in positions:
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 25), -1)
    img = cv2.GaussianBlur(img, (15, 15), 0)
    channel = img[:, :, 2]
    p90 = np.percentile(channel[channel > 0], 90) if (channel > 0).any() else 1
    channel = np.clip(channel.astype(np.float32) / p90 * 255, 0, 255).astype(np.uint8)
    hist_img = cv2.applyColorMap(channel, cv2.COLORMAP_HOT)
    cv2.imwrite(job['output_path'], hist_img)


HANDLERS = {
    'grid': _render_grid,
    'cluster': _render_cluster,
    'cluster_signal': _render_cluster_signal,
    'embed': _render_embed,
    'heatmap': _render_heatmap,
}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run_server():
    """Main server loop. Listens on Unix socket, processes render jobs."""
    # Clean up stale socket
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCK_PATH)
    server.listen(5)
    server.settimeout(IDLE_TIMEOUT)

    # Write PID file
    with open(PID_PATH, 'w') as f:
        f.write(str(os.getpid()))

    print(f"Render server started: pid={os.getpid()}, socket={SOCK_PATH}")
    jobs_done = 0

    def cleanup(signum=None, frame=None):
        try:
            os.unlink(SOCK_PATH)
        except OSError:
            pass
        try:
            os.unlink(PID_PATH)
        except OSError:
            pass
        print(f"Render server stopped: {jobs_done} jobs done")
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        while True:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                print(f"Render server idle {IDLE_TIMEOUT}s, shutting down")
                break

            try:
                job = _recv_msg(conn)
                if job is None:
                    conn.close()
                    continue

                if job.get('type') == 'shutdown':
                    _send_msg(conn, {'status': 'ok'})
                    conn.close()
                    break

                if job.get('type') == 'ping':
                    _send_msg(conn, {'status': 'alive', 'jobs_done': jobs_done})
                    conn.close()
                    continue

                handler = HANDLERS.get(job.get('type'))
                if handler is None:
                    _send_msg(conn, {'status': 'error',
                                     'msg': f"unknown type: {job.get('type')}"})
                    conn.close()
                    continue

                t0 = time.time()
                try:
                    result = handler(job)
                    elapsed = time.time() - t0
                    jobs_done += 1
                    _send_msg(conn, {'status': 'ok', 'elapsed': elapsed})
                except Exception as e:
                    _send_msg(conn, {'status': 'error', 'msg': str(e)})

            finally:
                conn.close()

    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RenderClient:
    """Client for submitting render jobs to the server."""

    def __init__(self):
        self._prev_2d = None  # warm-start state for grid renders

    @staticmethod
    def is_server_running():
        """Check if render server is alive."""
        if not os.path.exists(PID_PATH):
            return False
        try:
            with open(PID_PATH) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)  # check if process exists
        except (OSError, ValueError):
            return False
        # Process exists, try ping
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(SOCK_PATH)
            _send_msg(sock, {'type': 'ping'})
            resp = _recv_msg(sock)
            sock.close()
            return resp is not None and resp.get('status') == 'alive'
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return False

    @staticmethod
    def spawn_server():
        """Spawn render server as a background process."""
        import subprocess
        script = os.path.join(os.path.dirname(__file__), 'render_server.py')
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True)
        # Wait for socket to appear
        for _ in range(50):
            time.sleep(0.1)
            if os.path.exists(SOCK_PATH):
                return True
        print(f"Warning: render server didn't start (pid={proc.pid})")
        return False

    @classmethod
    def connect_or_spawn(cls):
        """Connect to existing server or spawn a new one."""
        client = cls()
        if not cls.is_server_running():
            cls.spawn_server()
        return client

    def submit(self, render_type, output_path, **kwargs):
        """Submit a render job. Non-blocking (fire and forget)."""
        job = {'type': render_type, 'output_path': output_path}
        job.update(kwargs)

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(SOCK_PATH)
            _send_msg(sock, job)
            resp = _recv_msg(sock)
            sock.close()
            return resp
        except (ConnectionRefusedError, FileNotFoundError, OSError) as e:
            print(f"  render submit failed: {e}")
            return None

    def shutdown(self):
        """Ask server to shut down."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(SOCK_PATH)
            _send_msg(sock, {'type': 'shutdown'})
            _recv_msg(sock)
            sock.close()
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            pass


if __name__ == '__main__':
    run_server()
