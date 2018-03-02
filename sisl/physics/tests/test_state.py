from __future__ import print_function, division

import pytest

pytestmark = pytest.mark.state

import math as m
import numpy as np

from sisl import geom, State, CState


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
    assert state.norm2()[0] == pytest.approx((ar(6) ** 2).sum())
    state_c = state.copy()
    assert len(state) == len(state_c)
    repr(state)


def test_state_repr1():
    state = State(ar(6))
    repr(state)
    state = State(ar(6), parent=geom.graphene())
    repr(state)


def test_state_dkind():
    state = State(ar(6))
    assert state.dkind == 'f'
    state = State(ar(6).astype(np.complex128))
    assert state.dkind == 'c'


def test_state_norm1():
    state = State(ar(6)).normalize()
    repr(state)
    assert len(state) == 1
    assert state.norm()[0] == pytest.approx(1)
    assert state.norm2()[0] == pytest.approx(1)


def test_state_sub1():
    state = ar(100).reshape(10, -1)
    state = State(state)
    assert len(state) == 10
    norm = state.norm()
    norm2 = state.norm2()
    for i in range(len(state)):
        assert len(state.sub(i)) == 1
        assert state.sub(i).norm()[0] == norm[i]
        assert state[i].norm()[0] == norm[i]
        assert state[i].norm2()[0] == norm2[i]
    for i, sub in enumerate(state):
        assert len(sub) == 1
        assert sub.norm()[0] == norm[i]


def test_state_outer1():
    state = ar(100).reshape(10, -1)
    state = State(state)
    out = state.outer()
    o = out.copy()
    o1 = out.copy()
    o.fill(0)
    o1.fill(0)
    for i, sub in enumerate(state):
        o += outer(sub.state[0, :])
        o1 += state.outer(i)

    assert np.allclose(out, o)
    assert np.allclose(out, o1)
    o = state.outer(np.arange(len(state)))
    assert np.allclose(out, o)


def test_state_toCState1():
    state = State(ar(6)).toCState()
    assert len(state) == 1
    assert state.c[0] == pytest.approx((ar(6) ** 2).sum() ** .5)
    assert state.norm()[0] == pytest.approx(1)


def test_state_toCState2():
    state = State(ar(10).reshape(2, -1)).toCState()
    assert np.allclose(state.norm2(), 1.)
    state = State(ar(10).reshape(2, -1)).toCState(norm=[0.5, 0.5])
    assert np.allclose(state.norm2(), 0.5)
    state = State(ar(10).reshape(2, -1)).toCState(norm=[0.25, 0.75])
    assert np.allclose(state.norm2(), [0.25, 0.75])


@pytest.mark.xfail(raises=ValueError)
def test_state_toCState_fail():
    State(ar(10).reshape(2, -1)).toCState(norm=[0.2, 0.5, 0.5])


def test_cstate_creation1():
    state = CState(1, ar(6))


def test_cstate_repr1():
    state = CState(1, ar(6))
    repr(state)
    state = CState(1, ar(6), parent=geom.graphene())
    repr(state)


def test_cstate_sub1():
    state = ar(100).reshape(10, -1)
    state = State(state).toCState().copy()
    assert len(state) == 10
    norm = state.norm()
    for i in range(len(state)):
        assert len(state.sub(i)) == 1
        assert state.sub(i).norm()[0] == norm[i]
        assert state[i].norm()[0] == norm[i]
        assert state[i].c[0] == state.c[i]
    for i, sub in enumerate(state):
        assert len(sub) == 1
        assert sub.norm()[0] == norm[i]


def test_cstate_sort1():
    state = ar(100).reshape(-1, 10).T
    state = State(state).toCState()
    sort = state.sort()
    assert len(state) == len(sort)


def test_cstate_outer1():
    state = ar(100).reshape(10, -1)
    state = CState(ar(10), state)
    out = state.outer()
    o = out.copy()
    o1 = out.copy()
    o.fill(0)
    o1.fill(0)
    for i, sub in enumerate(state):
        o += couter(sub.c[0], sub.state[0, :])
        o1 += state.outer(i)

    assert np.allclose(out, o)
    assert np.allclose(out, o1)
    o = state.outer(np.arange(len(state)))
    assert np.allclose(out, o)
