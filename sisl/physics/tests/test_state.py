from __future__ import print_function, division

import pytest
import numpy as np

from sisl import geom, Coefficient, State, StateC

pytestmark = pytest.mark.state


def ar(*args):
    l = np.prod(args)
    return np.arange(l, dtype=np.float64).reshape(args)


def outer(v):
    return np.outer(v, np.conjugate(v))


def couter(c, v):
    return np.outer(v * c, np.conjugate(v))


def test_coefficient_creation1():
    c = Coefficient(ar(6))
    str(c)
    assert len(c) == 6
    assert c.shape == (6, )
    assert c.dtype == np.float64
    assert c.dkind == 'f'
    assert len(c.sub(1)) == 1
    assert np.allclose(c.sub(1).c, 1)
    assert len(c.sub([1, 4])) == 2
    assert np.allclose(c.sub([1, 4]).c, [1, 4])
    assert np.allclose(c[1, 4].c, [1, 4])


def test_coefficient_creation2():
    c = Coefficient(ar(6), geom.graphene(), k='HELLO')
    assert np.allclose(c.parent.xyz, geom.graphene().xyz)
    assert c.info['k'] == 'HELLO'


def test_coefficient_copy():
    c = Coefficient(ar(6), geom.graphene(), k='HELLO', test='test')
    cc = c.copy()
    assert cc.info['k'] == 'HELLO'
    assert cc.info['test'] == 'test'


def test_coefficient_iter():
    c = Coefficient(ar(6))
    i = 0
    for C in c:
        assert len(C) == 1
        i += 1
    assert i == 6
    for i, C in enumerate(c.iter(True)):
        assert C == c.c[i]


def test_state_creation1():
    state = State(ar(6))
    assert len(state) == 1
    assert state.shape == (1, 6)
    assert state.norm2()[0] == pytest.approx((ar(6) ** 2).sum())
    state_c = state.copy()
    assert len(state) == len(state_c)
    str(state)


def test_state_repr1():
    state = State(ar(6))
    str(state)
    state = State(ar(6), parent=geom.graphene())
    str(state)


def test_state_dkind():
    state = State(ar(6))
    assert state.dkind == 'f'
    state = State(ar(6).astype(np.complex128))
    assert state.dkind == 'c'


def test_state_norm1():
    state = State(ar(6)).normalize()
    str(state)
    assert len(state) == 1
    assert state.norm()[0] == pytest.approx(1)
    assert state.norm2()[0] == pytest.approx(1)


def test_state_sub1():
    state = ar(10, 10)
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

    for i, sub in enumerate(state.iter(True)):
        assert (sub ** 2).sum() ** .5 == norm[i]


def test_state_outer1():
    state = ar(10, 10)
    state = State(state)
    out = state.outer()
    o = out.copy()
    o.fill(0)
    for i, sub in enumerate(state):
        o += outer(sub.state[0, :])

    assert np.allclose(out, o)
    o = state.outer(state)
    assert np.allclose(out, o)


def test_state_inner1():
    state = ar(10, 10)
    state = State(state)
    inner = state.inner()
    assert np.allclose(inner, state.inner(state))
    inner = state.inner(diagonal=False)
    assert np.allclose(inner, state.inner(state, diagonal=False))


def test_state_phase_max():
    state = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    state1 = State(state)
    state2 = State(-state)
    ph1 = state1.phase()
    ph2 = state2.phase()
    assert np.allclose(ph1, ph2 + np.pi)


def test_state_phase_all():
    state = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    state1 = State(state)
    state2 = State(-state)
    ph1 = state1.phase('all')
    ph2 = state2.phase('all')
    assert np.allclose(ph1, ph2 + np.pi)


def test_state_align1():
    state = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    state1 = State(state)
    state2 = State(-state)

    # This should rotate all back
    align2 = state1.align(state2)
    assert np.allclose(state1.state, align2.state)


def test_state_rotate_1():
    state = State([[1+1.j, 1.], [0.1-0.1j, 0.1]])

    # Angles are 45 and -45
    s = state.copy()
    assert pytest.approx(np.angle(s.state[0, 0]), np.pi / 4)
    assert pytest.approx(np.angle(s.state[0, 1]), 0)
    assert pytest.approx(np.angle(s.state[1, 0]) -np.pi / 4)
    assert pytest.approx(np.angle(s.state[1, 1]), 0)

    s.rotate() # individual false
    assert pytest.approx(np.angle(s.state[0, 0]), 0)
    assert pytest.approx(np.angle(s.state[0, 1]), -np.pi / 4)
    assert pytest.approx(np.angle(s.state[1, 0]), -np.pi / 2)
    assert pytest.approx(np.angle(s.state[1, 1]), -np.pi / 4)

    s = state.copy()
    s.rotate(individual=True)
    assert pytest.approx(np.angle(s.state[0, 0]), 0)
    assert pytest.approx(np.angle(s.state[0, 1]), -np.pi / 4)
    assert pytest.approx(np.angle(s.state[1, 0]), 0)
    assert pytest.approx(np.angle(s.state[1, 1]), np.pi / 4)

    s = state.copy()
    s.rotate(np.pi / 4, individual=True)
    assert pytest.approx(np.angle(s.state[0, 0]), np.pi / 4)
    assert pytest.approx(np.angle(s.state[0, 1]), 0)
    assert pytest.approx(np.angle(s.state[1, 0]), np.pi / 4)
    assert pytest.approx(np.angle(s.state[1, 1]), np.pi / 2)

# def test_state_toStateC1():
#     state = State(ar(6)).toStateC()
#     assert len(state) == 1
#     assert state.c[0] == pytest.approx((ar(6) ** 2).sum() ** .5)
#     assert state.norm()[0] == pytest.approx(1)


# def test_state_toStateC2():
#     state = State(ar(2, 5)).toStateC()
#     assert np.allclose(state.norm2(), 1.)
#     state = State(ar(2, 5)).toStateC(norm=[0.5, 0.5])
#     assert np.allclose(state.norm2(), 0.5)
#     state = State(ar(2, 5)).toStateC(norm=[0.25, 0.75])
#     assert np.allclose(state.norm2(), [0.25, 0.75])


# @pytest.mark.xfail(raises=ValueError)
# def test_state_toStateC_fail():
#     State(ar(2, 5)).toStateC(norm=[0.2, 0.5, 0.5])


def test_cstate_creation1():
    state = StateC(ar(6), 1)
    assert len(state) == 1
    state = StateC(ar(6, 6), ar(6))
    assert len(state) == 6
    assert np.allclose(state.c, ar(6))
    str(state)
    state2 = state.copy()
    assert np.allclose(state2.c, state.c)
    assert np.allclose(state2.state, state.state)
    state2 = state.asState()
    assert np.allclose(state2.state, state.state)
    state2 = state.asCoefficient()
    assert np.allclose(state2.c, state.c)


def test_cstate_repr1():
    state = StateC(ar(6), 1)
    assert len(state) == 1
    str(state)
    state = StateC(ar(6), 1, parent=geom.graphene())
    str(state)
    assert len(state) == 1


def test_cstate_sub1():
    state = StateC(ar(10, 10), ar(10))
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
    state = StateC(ar(10, 10), ar(10))
    sort = state.sort()
    assert len(state) == len(sort)
    c = sort.c[::-1]
    sort_descending = sort.sort(False)
    assert np.allclose(c, sort_descending.c)


def test_cstate_norm1():
    state = StateC(ar(10, 10), ar(10)).normalize()
    assert len(state) == 10
    assert np.allclose(state.norm(), 1)
    assert np.allclose(state.norm2(), 1)


def test_cstate_outer1():
    state = ar(10, 10)
    state = StateC(state, ar(10))
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
