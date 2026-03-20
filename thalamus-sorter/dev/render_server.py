"""Persistent async render server for thalamus-sorter.

Runs as a single background process, accepts render jobs via Unix socket.
Survives across training runs — check if running before spawning.
Multiple worker processes render in parallel.

Usage:
    # Start server (normally done automatically by main.py)
    python render_server.py [--workers N]

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
import threading
import queue
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future

SOCK_PATH = f"/tmp/thalamus-render-{os.getuid()}.sock"
PID_PATH = f"/tmp/thalamus-render-{os.getuid()}.pid"
IDLE_TIMEOUT = 300  # shutdown after 5 min idle
DEFAULT_WORKERS = 2
MAX_QUEUE = 100  # drop oldest jobs beyond this


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
# Render functions (run in worker processes, no GPU needed)
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


def _execute_job(job):
    """Top-level function for worker processes. Must be picklable."""
    handler = HANDLERS.get(job.get('type'))
    if handler is None:
        return {'status': 'error', 'msg': f"unknown type: {job.get('type')}"}
    try:
        handler(job)
        return {'status': 'ok', 'path': job.get('output_path', '')}
    except Exception as e:
        return {'status': 'error', 'msg': str(e),
                'path': job.get('output_path', '')}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run_server(n_workers=DEFAULT_WORKERS):
    """Main server loop. Accepts jobs via socket, dispatches to worker pool."""
    # Clean up stale socket
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCK_PATH)
    server.listen(16)
    server.settimeout(1.0)  # poll interval for idle check

    # Write PID file
    with open(PID_PATH, 'w') as f:
        f.write(str(os.getpid()))

    # Worker pool
    pool = ProcessPoolExecutor(max_workers=n_workers)
    pending = []  # list of Future objects
    jobs_submitted = 0
    jobs_done = 0
    jobs_dropped = 0
    last_activity = time.time()
    shutdown_requested = False

    print(f"Render server started: pid={os.getpid()}, workers={n_workers}, "
          f"socket={SOCK_PATH}")

    def cleanup(signum=None, frame=None):
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            os.unlink(SOCK_PATH)
        except OSError:
            pass
        try:
            os.unlink(PID_PATH)
        except OSError:
            pass
        print(f"Render server stopped: {jobs_done}/{jobs_submitted} done, "
              f"{jobs_dropped} dropped")
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        while not shutdown_requested:
            # Reap completed futures
            still_pending = []
            for f in pending:
                if f.done():
                    jobs_done += 1
                    try:
                        result = f.result()
                        if result.get('status') == 'error':
                            print(f"  render error: {result.get('msg', '?')} "
                                  f"({result.get('path', '')})")
                    except Exception as e:
                        print(f"  render exception: {e}")
                else:
                    still_pending.append(f)
            pending = still_pending

            # Accept new connections
            try:
                conn, _ = server.accept()
            except socket.timeout:
                # Check idle timeout
                if time.time() - last_activity > IDLE_TIMEOUT:
                    print(f"Render server idle {IDLE_TIMEOUT}s, shutting down")
                    break
                continue

            last_activity = time.time()

            try:
                job = _recv_msg(conn)
                if job is None:
                    conn.close()
                    continue

                jtype = job.get('type', '')

                if jtype == 'shutdown':
                    _send_msg(conn, {'status': 'ok',
                                     'jobs_done': jobs_done,
                                     'pending': len(pending)})
                    conn.close()
                    shutdown_requested = True
                    continue

                if jtype == 'ping':
                    _send_msg(conn, {'status': 'alive',
                                     'jobs_done': jobs_done,
                                     'pending': len(pending),
                                     'workers': n_workers})
                    conn.close()
                    continue

                if jtype not in HANDLERS:
                    _send_msg(conn, {'status': 'error',
                                     'msg': f"unknown type: {jtype}"})
                    conn.close()
                    continue

                # Backpressure: drop oldest if queue too large
                if len(pending) >= MAX_QUEUE:
                    jobs_dropped += 1
                    _send_msg(conn, {'status': 'dropped',
                                     'pending': len(pending)})
                    conn.close()
                    continue

                # Submit to worker pool
                future = pool.submit(_execute_job, job)
                pending.append(future)
                jobs_submitted += 1

                _send_msg(conn, {'status': 'queued',
                                 'pending': len(pending)})
                conn.close()

            except Exception as e:
                try:
                    _send_msg(conn, {'status': 'error', 'msg': str(e)})
                except Exception:
                    pass
                conn.close()

    finally:
        # Wait for remaining jobs
        if pending:
            print(f"Draining {len(pending)} pending jobs...")
            for f in pending:
                try:
                    f.result(timeout=30)
                    jobs_done += 1
                except Exception:
                    pass
        pool.shutdown(wait=True)
        cleanup()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RenderClient:
    """Client for submitting render jobs to the server. Fire-and-forget."""

    def __init__(self):
        pass

    @staticmethod
    def is_server_running():
        """Check if render server is alive."""
        if not os.path.exists(PID_PATH):
            return False
        try:
            with open(PID_PATH) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(SOCK_PATH)
            _send_msg(sock, {'type': 'ping'})
            resp = _recv_msg(sock)
            sock.close()
            return resp is not None and resp.get('status') == 'alive'
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return False

    @staticmethod
    def spawn_server(n_workers=DEFAULT_WORKERS):
        """Spawn render server as a background process."""
        import subprocess
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'render_server.py')
        proc = subprocess.Popen(
            [sys.executable, script, '--workers', str(n_workers)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True)
        for _ in range(50):
            time.sleep(0.1)
            if os.path.exists(SOCK_PATH):
                return True
        print(f"Warning: render server didn't start (pid={proc.pid})")
        return False

    @classmethod
    def connect_or_spawn(cls, n_workers=DEFAULT_WORKERS):
        """Connect to existing server or spawn a new one."""
        client = cls()
        if not cls.is_server_running():
            cls.spawn_server(n_workers=n_workers)
        return client

    def submit(self, render_type, output_path, **kwargs):
        """Submit a render job. Returns immediately after queueing."""
        job = {'type': render_type, 'output_path': output_path}
        job.update(kwargs)
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCK_PATH)
            _send_msg(sock, job)
            resp = _recv_msg(sock)
            sock.close()
            return resp
        except (ConnectionRefusedError, FileNotFoundError,
                OSError, socket.timeout) as e:
            return {'status': 'error', 'msg': str(e)}

    def shutdown(self):
        """Ask server to shut down (drains pending jobs first)."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(SOCK_PATH)
            _send_msg(sock, {'type': 'shutdown'})
            resp = _recv_msg(sock)
            sock.close()
            return resp
        except (ConnectionRefusedError, FileNotFoundError,
                OSError, socket.timeout):
            return None


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Thalamus render server")
    p.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                   help=f"Number of worker processes (default: {DEFAULT_WORKERS})")
    args = p.parse_args()
    run_server(n_workers=args.workers)
