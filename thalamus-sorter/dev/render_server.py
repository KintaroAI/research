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


def _render_signal(job):
    """Raw signal frame — normalize to grayscale/RGB image."""
    import cv2
    signal = job['signal']
    w, h = job['width'], job['height']
    sig_channels = job.get('sig_channels', 1)

    if sig_channels == 1:
        vmin, vmax = signal.min(), signal.max()
        if vmax > vmin:
            img = ((signal - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            img = np.full_like(signal, 128, dtype=np.uint8)
        img = img.reshape(h, w)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        n_pixels = h * w
        rgb = signal.reshape(n_pixels, sig_channels)
        vmin, vmax = rgb.min(), rgb.max()
        if vmax > vmin:
            rgb = ((rgb - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            rgb = np.full_like(rgb, 128, dtype=np.uint8)
        img = rgb.reshape(h, w, sig_channels)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    scale = max(1, 512 // max(w, h))
    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale),
                         interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(job['output_path'], img)


def _render_field(job):
    """2D field with agent and points of interest."""
    import cv2
    field_size = job['field_size']
    agent_pos = job['agent_pos']  # (2,) x, y
    pois = job['pois']            # (N, 2) x, y
    img_size = job.get('img_size', 400)

    img = np.full((img_size, img_size, 3), 32, dtype=np.uint8)
    scale = img_size / field_size

    # Draw POIs as green circles
    for i in range(len(pois)):
        px = int(pois[i, 0] * scale)
        py = int(pois[i, 1] * scale)
        cv2.circle(img, (px, py), 6, (0, 200, 0), -1)
        cv2.circle(img, (px, py), 6, (0, 255, 0), 1)

    # Draw agent as red circle
    ax = int(agent_pos[0] * scale)
    ay = int(agent_pos[1] * scale)
    cv2.circle(img, (ax, ay), 8, (0, 0, 200), -1)
    cv2.circle(img, (ax, ay), 8, (0, 0, 255), 2)

    # Draw collection radius
    collect_radius = job.get('collect_radius', 5.0)
    cr = int(collect_radius * scale)
    cv2.circle(img, (ax, ay), cr, (0, 0, 100), 1)

    cv2.imwrite(job['output_path'], img)


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


def _relay_graph(job):
    """Relay graph payload to visualization app via TCP. Fire-and-forget."""
    viz_addr = job.get('viz_address')
    if not viz_addr:
        return {'status': 'ok', 'msg': 'no viz_address configured'}
    try:
        host, port = viz_addr.rsplit(':', 1)
        port = int(port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        sock.connect((host, port))
        # Send the payload (minus viz_address and type)
        payload = {k: v for k, v in job.items()
                   if k not in ('type', 'output_path', 'viz_address')}
        _send_msg(sock, payload)
        sock.close()
        return {'status': 'ok', 'msg': 'relayed to viz'}
    except (ConnectionRefusedError, socket.timeout, OSError):
        return {'status': 'ok', 'msg': 'viz not running, dropped'}


HANDLERS = {
    'grid': _render_grid,
    'cluster': _render_cluster,
    'cluster_signal': _render_cluster_signal,
    'signal': _render_signal,
    'field': _render_field,
    'embed': _render_embed,
    'heatmap': _render_heatmap,
    'graph': _relay_graph,
    'field_live': _relay_graph,  # same relay mechanism, different address
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


# ---------------------------------------------------------------------------
# Renderer — high-level API for main.py
# ---------------------------------------------------------------------------

class Renderer:
    """High-level render API. Wraps RenderClient with typed methods.

    Usage:
        renderer = Renderer(output_dir, w, h)
        renderer.grid(tick, embeddings, pixel_values, method='umap')
        renderer.cluster(tick, cluster_ids)
        renderer.embed(tick, embeddings, n_sensory, cluster_ids=..., ...)
        renderer.signal(tick, signal_data)
        renderer.heatmap(positions)

    All methods are fire-and-forget via the render server.
    If server is unavailable, calls are silently dropped.
    """

    def __init__(self, output_dir, w, h, sig_channels=1, n_workers=2,
                 viz_address=None, field_address=None):
        self.output_dir = output_dir
        self.w = w
        self.h = h
        self.sig_channels = sig_channels
        self.viz_address = viz_address    # 'host:port' for graph viz relay
        self.field_address = field_address  # 'host:port' for field viz relay
        self._client = None
        self._n_workers = n_workers
        if output_dir:
            self._client = RenderClient.connect_or_spawn(n_workers=n_workers)

    def _path(self, prefix, tick, ext='png'):
        return os.path.join(self.output_dir, f"{prefix}_{tick:06d}.{ext}")

    def _submit(self, render_type, output_path, **kwargs):
        if self._client is None:
            return
        return self._client.submit(render_type, output_path, **kwargs)

    def _submit_nowait(self, render_type, output_path, **kwargs):
        """Fire-and-forget via background thread. Latest job per type wins."""
        if self._client is None:
            return
        job = {'type': render_type, 'output_path': output_path}
        job.update(kwargs)
        # Start sender thread on first use
        if not hasattr(self, '_send_thread'):
            import threading
            self._send_slots = {}  # render_type -> latest job
            self._send_lock = threading.Lock()
            self._send_event = threading.Event()
            def _sender():
                while True:
                    self._send_event.wait()
                    self._send_event.clear()
                    with self._send_lock:
                        jobs = list(self._send_slots.values())
                        self._send_slots.clear()
                    for j in jobs:
                        try:
                            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            sock.settimeout(0.5)
                            sock.connect(SOCK_PATH)
                            _send_msg(sock, j)
                            sock.close()
                        except (ConnectionRefusedError, FileNotFoundError,
                                OSError, socket.timeout):
                            pass
            self._send_thread = threading.Thread(target=_sender, daemon=True)
            self._send_thread.start()
        with self._send_lock:
            self._send_slots[render_type] = job
        self._send_event.set()

    def _send_busy(self):
        """Check if sender thread still has unsent jobs (skip data prep)."""
        if not hasattr(self, '_send_lock'):
            return False
        with self._send_lock:
            return len(self._send_slots) > 0

    def grid(self, tick, embeddings, pixel_values, method='pca',
             align=False, gpu=False):
        """Voronoi grid render of sensory embeddings."""
        self._submit('grid', self._path('frame', tick),
                     embeddings=embeddings, pixel_values=pixel_values,
                     width=self.w, height=self.h,
                     method=method, align=align, gpu=gpu)

    def cluster(self, tick, cluster_ids):
        """Color-coded cluster map."""
        self._submit('cluster', self._path('clusters', tick),
                     cluster_ids=cluster_ids,
                     width=self.w, height=self.h)

    def cluster_signal(self, tick, cluster_ids, signal):
        """Cluster signal map — mean signal per cluster."""
        self._submit('cluster_signal', self._path('clusters_sig', tick),
                     cluster_ids=cluster_ids, signal=signal,
                     width=self.w, height=self.h,
                     sig_channels=self.sig_channels)

    def signal(self, tick, signal_data):
        """Raw signal frame."""
        self._submit('signal', self._path('signal', tick),
                     signal=signal_data,
                     width=self.w, height=self.h,
                     sig_channels=self.sig_channels)

    def embed(self, tick, embeddings, n_sensory, pixel_values=None,
              cluster_ids=None, n_outputs=4, method='pca'):
        """Scatter plot of all neurons in embedding space."""
        self._submit('embed', self._path('embed', tick),
                     embeddings=embeddings, n_sensory=n_sensory,
                     pixel_values=pixel_values, cluster_ids=cluster_ids,
                     n_outputs=n_outputs, method=method)

    def field(self, tick, agent_pos, pois, field_size,
              collect_radius=5.0, img_size=400):
        """2D field with agent and POIs."""
        self._submit('field', self._path('field', tick),
                     agent_pos=agent_pos, pois=pois,
                     field_size=field_size,
                     collect_radius=collect_radius,
                     img_size=img_size)

    def heatmap(self, positions, name='motor_heatmap'):
        """Position heatmap (e.g., motor saccade positions)."""
        path = os.path.join(self.output_dir, f"{name}.png")
        self._submit('heatmap', path, positions=positions)

    def field_live(self, tick, agent_pos, pois, field_size,
                   hunger=0.0, collect_radius=5.0, score=0,
                   visual_field=None, blocked=None):
        """Send field data to field viz app via render server."""
        if self._client is None or not self.field_address:
            return
        self._submit_nowait('field_live', '',
                            tick=tick, agent_pos=agent_pos, pois=pois,
                            field_size=field_size, hunger=hunger,
                            collect_radius=collect_radius, score=score,
                            visual_field=visual_field, blocked=blocked,
                            viz_address=self.field_address)

    def graph(self, tick, most_recent, n_sensory, n_outputs,
              lateral_adj=None, column_outputs=None, knn2=None,
              centroids=None):
        """Send graph visualization payload to viz app via render server."""
        if self._client is None or not self.viz_address:
            return
        self._submit_nowait('graph', '',  # no output file — relay only
                     tick=tick,
                     most_recent=most_recent,
                     n_sensory=n_sensory,
                     n_outputs=n_outputs,
                     lateral_adj=lateral_adj,
                     column_outputs=column_outputs,
                     knn2=knn2,
                     centroids=centroids,
                     viz_address=self.viz_address)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Thalamus render server")
    p.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                   help=f"Number of worker processes (default: {DEFAULT_WORKERS})")
    args = p.parse_args()
    run_server(n_workers=args.workers)
