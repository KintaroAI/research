"""Task 00078: the two halves of reference/mimic.py."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reference import mimic  # noqa: E402

V, T = 12, 8


def setup(seed=0):
    torch.manual_seed(seed)
    h1, h2 = mimic.Half(V + 1, T, d=16, layers=1), mimic.Half(V + 1, T, d=16, layers=1)
    o1, o2 = torch.optim.Adam(h1.parameters()), torch.optim.Adam(h2.parameters())
    x, y = torch.randint(0, V, (4, T)), torch.randint(0, V, (4, T))
    return h1, h2, o1, o2, x, y, torch.zeros(4, T, dtype=torch.bool)


def snap(m):
    return [p.detach().clone() for p in m.parameters()]


def same(a, m):
    return all(torch.equal(p, q) for p, q in zip(a, m.parameters()))


def test_half2_never_sees_the_world():
    """Half 2's step depends on the letters only through half 1: another target leaves it exactly the same."""
    results = []
    for shift in (0, 1):
        h1, h2, o1, o2, x, y, own = setup()
        mimic.update(h1, h2, o1, o2, x, (y + shift) % V, own, "B", 2.0, True)
        results.append(snap(h2))
    assert all(torch.equal(p, q) for p, q in zip(*results))


def test_half1_gets_nothing_from_half2():
    """Half 1's step is the same whatever half 2 is."""
    after = []
    for seed2 in (1, 2):
        h1, h2, o1, o2, x, y, own = setup()
        torch.manual_seed(seed2)
        h2b = mimic.Half(V + 1, T, d=16, layers=1)
        mimic.update(h1, h2b, o1, torch.optim.Adam(h2b.parameters()), x, y, own, "B", 2.0, False)
        after.append(snap(h1))
    assert all(torch.equal(p, q) for p, q in zip(*after))


def test_frozen_half1():
    h1, h2, o1, o2, x, y, own = setup()
    a = snap(h1)
    mimic.update(h1, h2, o1, o2, x, y, own, "A", 2.0, True)
    assert same(a, h1)


def test_half2_is_taught():
    h1, h2, o1, o2, x, y, own = setup()
    a = snap(h2)
    mimic.update(h1, h2, o1, o2, x, y, own, "A", 2.0, True)
    assert not same(a, h2)


def test_weights():
    ce = torch.tensor([0.5, 1.0, 3.0])
    own = torch.tensor([True, False, True])
    assert mimic.half1_weights(ce, own, "S", 2.0).tolist() == [0.25, 0.5, 1.0]
    assert mimic.half1_weights(ce, own, "E", 2.0).tolist() == [0.0, 1.0, 0.0]
    assert mimic.half1_weights(ce, own, "B", 2.0).tolist() == [1.0, 1.0, 1.0]


def test_own_letters_teach_arm_e_nothing():
    """All letters own: half 1 does not move."""
    h1, h2, o1, o2, x, y, own = setup()
    a = snap(h1)
    mimic.update(h1, h2, o1, o2, x, y, torch.ones_like(own), "E", 2.0, False)
    assert same(a, h1)


def step(letters, spoken, gap, starts, spaces, rate, silent=False):
    t = lambda v: torch.tensor(v)
    return mimic.world_step(t(letters), t(spoken), t(gap), t(starts), t(spaces), rate, torch.Generator(), silent, 99)


def test_speech_is_heard_only_in_gaps():
    heard, gap = step([1, 2, 3], [7, 8, 9], [True, False, True], [False] * 3, [False, False, True], 0.0)
    assert heard.tolist() == [7, 2, 3]            # in a gap: said; not: real; a space ends the gap and is real
    assert gap.tolist() == [True, False, False]


def test_silent_gap_is_blank():
    heard, _ = step([1], [7], [True], [False], [False], 0.0, silent=True)
    assert heard.tolist() == [99]


def test_gaps_begin_only_at_word_starts():
    _, gap = step([1, 2, 3], [7, 8, 9], [False] * 3, [True, False, True], [False, False, True], 1.0)
    assert gap.tolist() == [True, False, False]   # a word start; not within a word; a space is not a word


def test_omit_rate():
    assert mimic.omit_rate(14999, 15000, 5000, 0.3) == 0
    assert mimic.omit_rate(17500, 15000, 5000, 0.3) == pytest.approx(0.15)
    assert mimic.omit_rate(30000, 15000, 5000, 0.3) == pytest.approx(0.3)


def test_say_never_blank():
    h = mimic.Half(V + 1, T, d=16, layers=1)
    h.out.bias.data[V] = 100.0
    x = torch.randint(0, V, (32, T))
    assert (mimic.say(h, x, V, True, torch.Generator()) != V).all()
    assert (mimic.say(h, x, V, False, torch.Generator()) != V).all()
