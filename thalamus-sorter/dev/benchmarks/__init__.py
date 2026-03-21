"""Benchmark modules for testing column/lateral feature detection.

Each benchmark is a module in this package that exports:
    name: str               — short identifier (used as --signal-source value)
    description: str        — one-line description
    add_args(parser)        — add CLI flags (optional)
    make_signal(w, h, args) — returns (tick_fn, metadata) for signal generation
    analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir)
                            — run post-training analysis, return results dict

Usage in main.py:
    from benchmarks import get_benchmark
    bench = get_benchmark(args.signal_source)
    if bench:
        tick_fn, metadata = bench.make_signal(w, h, args)
        ...after training...
        bench.analyze(metadata, cluster_mgr, signals, tick_counter, T, output_dir)
"""

import importlib
import os

_BENCHMARKS = {}


def register(module):
    """Register a benchmark module by its name attribute."""
    _BENCHMARKS[module.name] = module
    return module


def get_benchmark(name):
    """Get benchmark module by name, or None if not a benchmark."""
    if name in _BENCHMARKS:
        return _BENCHMARKS[name]
    # Try to auto-import from this package
    try:
        mod = importlib.import_module(f'.{name}', package='benchmarks')
        if hasattr(mod, 'name'):
            _BENCHMARKS[mod.name] = mod
            return mod
    except (ImportError, ModuleNotFoundError):
        pass
    return None


def list_benchmarks():
    """List all available benchmark names."""
    bench_dir = os.path.dirname(__file__)
    names = []
    for f in os.listdir(bench_dir):
        if f.endswith('.py') and f != '__init__.py':
            names.append(f[:-3])
    return sorted(names)
