"""The long-term memory gate (online.GatedMemory) driven in lifetime.py's order (task 00053): each tick a step is added
to short-term memory and the gate decides; the step's token surprise is noted one tick later, after the update that
trained on it. With a gate delay of 0 (relief_k = 1) the newest step used to be decided before it was noted. And with
several bodies, the gate counts in the metrics are summed over every body's gate (lifetime.gate_metrics).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lifetime import gate_metrics  # noqa: E402
from online import DEFAULT_ONLINE, GatedMemory  # noqa: E402

L, IMG, NT = 8, 8, 5
KINDS = np.array([0, 0, 0, 1, 2])


def online(**kw):
    on = dict(DEFAULT_ONLINE, memory="gated", stm=64, replay=400, store_warmup=0, store_base=0.0,
              store_surprise=0.2, surprise_window=200, relief_k=1)
    on.update(kw)
    return on


def add(gm, rng):
    gm.stm.add(img=rng.integers(0, 256, (IMG, IMG, 3), dtype=np.uint8), nimg=np.zeros((IMG, IMG, 3), np.uint8),
               drv=np.zeros(2, np.float32), ndrv=np.zeros(2, np.float32), pa=np.zeros(3, np.float32), pr=0.0, es=0,
               act=np.zeros(3, np.float32), rew=0.0, done=0, hit=0, cmd=0.0)


def live(gates, ce_of, steps, seed=0):
    """lifetime.py's order for each gate: note the previous step's token cross-entropies, add this step, decide."""
    rng = np.random.default_rng(seed)
    for tick in range(1, steps + 1):
        for gm in gates:
            if tick > 1:
                gm.note_tokens(ce_of(tick - 1, rng), KINDS)
            add(gm, rng)
            gm.decide(tick)


def test_a_chunk_whose_last_step_is_surprising_is_stored():
    """relief_k = 1: chunk 12 (steps 89..96) is ordinary but for its last step, 100 times as surprising as usual."""
    def ce_of(step, rng):
        if 89 <= step <= 96:
            return np.full(NT, 100.0 if step == 96 else 0.5)
        return rng.uniform(0.8, 1.2, NT)

    gm = GatedMemory(online(), L, IMG, 0, seed=1, tokens=True)
    live([gm], ce_of, 110)
    stored_ends = set(gm.store.end[:gm.store.n].tolist())
    assert 96 in stored_ends, stored_ends
    assert gm.D >= 1


def test_gate_delay_unchanged_where_it_was_already_positive():
    assert GatedMemory(online(relief_k=12), L, IMG, 0, seed=1, tokens=True).D == 11
    assert GatedMemory(online(relief_k=1), L, IMG, 0, seed=1, tokens=False).D == 0


def test_gate_metrics_sum_over_bodies():
    """Two bodies' gates share one store: the counts are the sum of both gates', the store's state is read once."""
    on = online(relief_k=12, store_base=0.5, store_surprise=0.0)
    g0 = GatedMemory(on, L, IMG, 0, seed=1, tokens=True)
    g1 = GatedMemory(on, L, IMG, 0, seed=1, store=g0.store, rng_key=(1,), tokens=True)
    live([g0, g1], lambda s, rng: rng.uniform(0.8, 1.2, NT), 200)
    c0, c1 = dict(g0.counts), dict(g1.counts)
    m = gate_metrics([g0, g1])
    assert m["chunks"] == c0["chunks"] + c1["chunks"] > 0
    assert m["stored"] == (c0["chunks"] - c0.get("dropped", 0)) + (c1["chunks"] - c1.get("dropped", 0))
    assert m["stored_base"] == c0.get("base", 0) + c1.get("base", 0)
    assert m["ltm_chunks"] == g0.store.n
    assert not g0.counts and not g1.counts   # read once: the counters restart


def test_gate_metrics_of_one_body_are_its_own():
    on = online(relief_k=12, store_base=0.5, store_surprise=0.0)
    a = GatedMemory(on, L, IMG, 0, seed=1, tokens=True)
    b = GatedMemory(on, L, IMG, 0, seed=1, tokens=True)
    live([a], lambda s, rng: rng.uniform(0.8, 1.2, NT), 150)
    live([b], lambda s, rng: rng.uniform(0.8, 1.2, NT), 150)
    ma, mb = gate_metrics([a]), b.metrics()
    assert ma.keys() == mb.keys()
    assert all(ma[k] == mb[k] or (ma[k] != ma[k] and mb[k] != mb[k]) for k in ma)
