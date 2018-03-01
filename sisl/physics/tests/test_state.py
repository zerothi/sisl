from __future__ import print_function, division

import pytest

pytestmark = pytest.mark.state

import math as m
import numpy as np

from sisl import State, CState


def ar(*args):
    return np.arange(*args, dtype=np.float64)


def outer(v):
    return np.outer(v, np.conjugate(v))


def couter(c, v):
    return np.outer(v * c, np.conjugate(v))


def test_state_creation1():
    state = State(ar(6))
    assert len(state) == 1
    assert state.shape == (1, 6)
    assert state.norm()[0] == pytest.approx((ar(6) ** 2).sum() ** .5)
    state_c = state.copy()
    assert len(state) == len(state_c)
    repr(state)


def test_state_norm1():
    state = State(ar(6)).normalize()
    repr(state)
    assert len(state) == 1
    assert state.norm()[0] == pytest.approx(1)


def test_state_sub1():
    state = ar(100).reshape(10, -1)
    state = State(state)
    assert len(state) == 10
    norm = state.norm()
    for i in range(len(state)):
        assert len(state.sub(i)) == 1
        assert state.sub(i).norm()[0] == norm[i]
        assert state[i].norm()[0] == norm[i]
    for i, sub in enumerate(state):
        assert len(sub) == 1
        assert sub.norm()[0] == norm[i]


def test_state_outer1():
    state = ar(100).reshape(10, -1)
    state = State(state)
    out = state.outer()
    o = out.copy()
    o.fill(0)
    for sub in state:
        o += outer(sub.state[0, :])

    assert np.allclose(out, o)


def test_state_toCState1():
    state = State(ar(6)).toCState()
    assert len(state) == 1
    assert state.c[0] == pytest.approx((ar(6) ** 2).sum() ** .5)
    assert state.norm()[0] == pytest.approx(1)


def test_cstate_creation1():
    state = CState(1, ar(6))


def test_cstate_outer1():
    state = ar(100).reshape(10, -1)
    state = CState(ar(10), state)
    out = state.outer()
    o = out.copy()
    o.fill(0)
    for sub in state:
        o += couter(sub.c[0], sub.state[0, :])

    assert np.allclose(out, o)
