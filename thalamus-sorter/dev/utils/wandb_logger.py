"""Lightweight W&B logger for thalamus-sorter training.

Usage in main.py:
    from utils.wandb_logger import WandbLogger
    wlog = WandbLogger(args)       # no-op if --wandb not set or wandb not installed
    wlog.log_knn(tick, overlap, spatial_acc, n_changed, n_total, top50, top90, lr=None)
    wlog.log_done(ticks, elapsed_s, std, total_pairs)
    wlog.log_eval(pca_disp, k10_mean, within_3, within_5)
    wlog.finish()
"""

import os

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class WandbLogger:
    """Thin wrapper around wandb. Becomes a no-op if wandb is missing or disabled."""

    def __init__(self, args):
        self.enabled = getattr(args, 'wandb', False) and HAS_WANDB
        self._initialized = False
        self._args = args

        if getattr(args, 'wandb', False) and not HAS_WANDB:
            import sys
            print("WARNING: --wandb requested but wandb not installed. "
                  "Install with: pip install wandb", file=sys.stderr)

    def _ensure_init(self):
        """Lazily init wandb on first log call so config is complete."""
        if self._initialized or not self.enabled:
            return
        args = self._args

        # Build config from args
        config = {k: v for k, v in vars(args).items()
                  if k != 'func' and not k.startswith('_')}

        # Patch email in API-fetched user settings
        PUBLIC_EMAIL = 'kin@kintaroai.com'
        _wl = wandb.setup()
        _orig_load = _wl._load_user_settings
        def _patched_load(_orig=_orig_load):
            result = _orig()
            if result and 'email' in result:
                result['email'] = PUBLIC_EMAIL
            return result
        _wl._load_user_settings = _patched_load

        wandb.init(
            project=getattr(args, 'wandb_project', 'thalamus-sorter'),
            name=getattr(args, 'wandb_name', None),
            group=getattr(args, 'wandb_group', None),
            tags=getattr(args, 'wandb_tags', None),
            entity=getattr(args, 'wandb_entity', None),
            config=config,
            dir=os.environ.get('WANDB_DIR',
                               os.path.expanduser('~/.local/share/wandb')),
            settings=wandb.Settings(disable_git=True),
            save_code=False,
        )
        self._initialized = True
        self.url = wandb.run.get_url() if wandb.run else None

    @property
    def run_url(self):
        """Return the wandb run URL, or None if not initialized."""
        if not self._initialized or not self.enabled:
            return None
        return self.url

    def log_knn(self, tick, overlap, spatial_acc, n_changed, n_total,
                top50_swaps, top90_swaps, lr=None):
        if not self.enabled:
            return
        self._ensure_init()
        metrics = {
            'knn/overlap': overlap,
            'knn/spatial': spatial_acc,
            'knn/n_changed': n_changed,
            'knn/n_total': n_total,
            'knn/pct_changed': n_changed / n_total,
            'knn/top50_swaps': top50_swaps,
            'knn/top90_swaps': top90_swaps,
        }
        if lr is not None:
            metrics['lr'] = lr
        wandb.log(metrics, step=tick)

    def log_tick(self, tick, elapsed_s, total_pairs, ms_per_tick, pairs_per_tick=None):
        if not self.enabled:
            return
        self._ensure_init()
        metrics = {
            'tick/elapsed_s': elapsed_s,
            'tick/total_pairs': total_pairs,
            'tick/ms_per_tick': ms_per_tick,
        }
        if pairs_per_tick is not None:
            metrics['tick/pairs_per_tick'] = pairs_per_tick
        wandb.log(metrics, step=tick)

    def log_done(self, ticks, elapsed_s, std, total_pairs):
        if not self.enabled:
            return
        self._ensure_init()
        wandb.log({
            'summary/ticks': ticks,
            'summary/elapsed_s': elapsed_s,
            'summary/std': std,
            'summary/total_pairs': total_pairs,
        })

    def log_clusters(self, tick, n_alive, m, contiguity, diameter,
                     jumps_per_tick, total_jumps, splits, stability=None):
        if not self.enabled:
            return
        self._ensure_init()
        metrics = {
            'cluster/alive': n_alive,
            'cluster/alive_pct': n_alive / m,
            'cluster/contiguity': contiguity,
            'cluster/diameter': diameter,
            'cluster/jumps_per_tick': jumps_per_tick,
            'cluster/total_jumps': total_jumps,
            'cluster/splits': splits,
        }
        if stability is not None:
            metrics['cluster/stability'] = stability
        wandb.log(metrics, step=tick)

    def log_eval(self, pca_disp, k10_mean, within_3, within_5):
        if not self.enabled:
            return
        self._ensure_init()
        wandb.log({
            'eval/pca_disparity': pca_disp,
            'eval/k10_mean_dist': k10_mean,
            'eval/k10_within_3px': within_3,
            'eval/k10_within_5px': within_5,
        })

    def finish(self):
        if self._initialized:
            wandb.finish()
            self._initialized = False
