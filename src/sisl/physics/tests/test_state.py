# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Coefficient, State, StateC, geom

pytestmark = [pytest.mark.physics, pytest.mark.state]


def ar(*args):
    l = np.prod(args)
    return np.arange(l, dtype=np.float64).reshape(*args)


def ortho_matrix(n, m=None):
    if m is None:
        m = n
    max_nm = max(n, m)
    from scipy.linalg import qr

    H = np.random.randn(max_nm, max_nm) + 1j * np.random.randn(max_nm, max_nm)
    Q, _ = qr(H)
    M = Q.dot(np.conjugate(Q.T))
    return M[:n, :]


def outer(v):
    return np.outer(v, np.conjugate(v))


def couter(c, v):
    return np.outer(v * c, np.conjugate(v))


def test_coefficient_creation_simple():
    c = Coefficient(ar(6))
    str(c)
    assert len(c) == 6
    assert c.shape == (6,)
    assert c.dtype == np.float64
    assert c.dkind == "f"
    assert len(c.sub(1)) == 1
    assert np.allclose(c.sub(1).c, 1)
    assert len(c.sub([1, 4])) == 2
    assert np.allclose(c.sub([1, 4]).c, [1, 4])
    assert np.allclose(c[1, 4].c, [1, 4])


def test_coefficient_creation_info():
    c = Coefficient(ar(6), geom.graphene(), k="HELLO")
    assert np.allclose(c.parent.xyz, geom.graphene().xyz)
    assert c.info["k"] == "HELLO"


def test_coefficient_copy():
    c = Coefficient(ar(6), geom.graphene(), k="HELLO", test="test")
    cc = c.copy()
    assert cc.info["k"] == "HELLO"
    assert cc.info["test"] == "test"


def test_coefficient_sub():
    state = ar(10)
    state = Coefficient(state)
    assert len(state) == 10
    for i in range(len(state)):
        assert len(state.sub(i)) == 1
    for i, sub in enumerate(state):
        assert len(sub) == 1

    assert np.allclose(
        state.sub(np.array([False, True, False, True])).c, state.sub([1, 3]).c
    )

    sub = state.sub(np.array([False, True, False, True]))
    state.sub([1, 3], inplace=True)
    assert np.allclose(sub.c, state.c)


def test_coefficient_iter():
    c = Coefficient(ar(6))
    i = 0
    for C in c:
        assert len(C) == 1
        i += 1
    assert i == 6
    for i, C in enumerate(c.iter(True)):
        assert C == c.c[i]


def test_state_creation():
    state = State(ar(6))
    assert len(state) == 1
    assert state.shape == (1, 6)
    assert state.norm2()[0] == pytest.approx((ar(6) ** 2).sum())
    state_c = state.copy()
    assert len(state) == len(state_c)
    str(state)


def test_state_repr():
    state = State(ar(6))
    str(state)
    state = State(ar(6), parent=geom.graphene())
    str(state)


def test_state_dkind():
    state = State(ar(6))
    assert state.dkind == "f"
    state = State(ar(6).astype(np.complex128))
    assert state.dkind == "c"


def test_state_norm():
    state = State(ar(6)).normalize()
    str(state)
    assert len(state) == 1
    assert state.norm()[0] == pytest.approx(1)


def test_state_change_gauge():
    g = geom.graphene(1.42)
    state = State(ar(2, 2).astype(np.complex128), g, gauge="atomic", k=(0.1, 0.2, 0.4))
    assert len(state) == 2
    old = state.state.copy()
    state.change_gauge("lattice")
    assert not np.allclose(old, state.state)
    state.change_gauge("atomic")
    assert np.allclose(old, state.state)


def test_state_change_gauge_nc():
    g = geom.graphene(1.42)
    state = State(ar(2, 4).astype(np.complex128), g, gauge="atomic", k=(0.1, 0.2, 0.4))
    assert len(state) == 2
    old = state.state.copy()
    state.change_gauge("lattice")
    assert not np.allclose(old, state.state)
    state.change_gauge("atomic")
    assert np.allclose(old, state.state)


def test_state_sub():
    state = ar(10, 10)
    state = State(state)
    assert len(state) == 10
    norm = state.norm()
    norm2 = state.norm2()
    for i in range(len(state)):
        assert len(state.sub(i)) == 1
        assert state.sub(i).norm()[0] == norm[i]
        assert state[i].norm()[0] == norm[i]
    for i, sub in enumerate(state):
        assert len(sub) == 1
        assert sub.norm()[0] == norm[i]

    for i, sub in enumerate(state.iter(True)):
        assert (sub**2).sum() == norm2[i]

    assert np.allclose(
        state.sub(np.array([False, True, False, True])).state, state.sub([1, 3]).state
    )

    sub = state.sub(np.array([False, True, False, True]))
    state.sub([1, 3], inplace=True)
    assert np.allclose(sub.state, state.state)


def test_state_outer():
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


def test_state_outer_matrix():
    state = ar(10, 10)
    M = ar(10)
    state = State(state)
    out = state.outer(matrix=M)
    out_full = state.outer(matrix=np.diag(M))
    assert np.allclose(out, out_full)


def test_state_inner():
    state = ar(10, 10)
    state = State(state)
    inner = state.inner(diag=False)
    assert np.allclose(inner, state.inner(state, projection="matrix"))
    inner_diag = state.inner()
    assert np.allclose(np.diag(inner), inner_diag)


def test_state_inner_matrix():
    state = ar(10, 10)
    M = ar(10)
    state = State(state)
    inner = state.inner(matrix=M, diag=False)
    assert np.allclose(
        inner, state.inner(state, matrix=np.diag(M), projection="matrix")
    )
    inner_diag = state.inner(matrix=M)
    assert np.allclose(np.diag(inner), inner_diag)
    inner_diag = state.inner(matrix=np.diag(M))
    assert np.allclose(np.diag(inner), inner_diag)


def test_state_inner_differing_size():
    state1 = State(ar(8, 10))
    state2 = State(ar(4, 10))

    inner = state1.inner(state2, projection="matrix")
    assert inner.shape == (8, 4)


def test_state_inner_projections():
    g = geom.graphene(atoms=Atom(6, R=(1, 2))) * (2, 2, 1)
    n = g.no + 1
    state = State(ar(n, g.no), parent=g)

    for projs, shape in (
        (("diag", "diagonal", True), (n,)),
        (("trace", "sum"), tuple()),
        (("matrix", False), (n, n)),
        (("basis", "orbitals", "orbital"), (n, g.no)),
        (("atoms", "atom"), (n, g.na)),
    ):
        for proj in projs:
            data = state.inner(projection=proj)
            assert data.shape == shape


def test_state_norm_projections():
    g = geom.graphene(atoms=Atom(6, R=(1, 2))) * (2, 2, 1)
    n = g.no + 1
    state = State(ar(n, g.no), parent=g)
    assert state.shape[0] != state.shape[1]

    for projs, shape in (
        (("diagonal", True), (n,)),
        (("trace", "sum"), tuple()),
        (("hadamard", "basis", "orbitals", "orbital"), (n, g.no)),
        (("atoms", "atom", "hadamard:atoms"), (n, g.na)),
    ):
        for proj in projs:
            data = state.norm2(projection=proj)
            assert data.shape == shape


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
    ph1 = state1.phase("all")
    ph2 = state2.phase("all")
    assert np.allclose(ph1, ph2 + np.pi)


def test_state_align_phase():
    state = ortho_matrix(10)
    state1 = State(state)
    state2 = State(-state)

    # This should rotate all back
    align2 = state2.align_phase(state1)
    assert np.allclose(state1.state, align2.state)
    align2, idx = state2.align_phase(state1, ret_index=True)
    assert not np.allclose(state1.state, state2.state)
    state2.align_phase(state1, inplace=True)
    assert np.allclose(state1.state, state2.state)
    state2 = State(-state)
    idx = state2.align_phase(state1, inplace=True, ret_index=True)
    assert np.allclose(state1.state, state2.state)


def test_state_ipr():
    state = State(ortho_matrix(10))
    ipr = state.ipr()
    assert ipr.shape == (10,)


def test_state_align_norm():
    state = ortho_matrix(10)
    state1 = State(state)
    idx = np.arange(len(state))
    np.random.shuffle(idx)
    state2 = state1.sub(idx)

    # This should swap all back
    align2 = state2.align_norm(state1)
    assert np.allclose(state1.state, align2.state)
    # This should swap all back
    align2, idx2 = state2.align_norm(state1, ret_index=True)
    assert np.allclose(state1.state, align2.state)
    align1 = state2.sub(idx2)
    assert np.allclose(state1.state, align1.state)


def test_state_align_norm2():
    state = ortho_matrix(10)
    state1 = State(state)
    idx = np.arange(len(state))
    np.random.shuffle(idx)
    state2 = state1.sub(idx)

    # This should swap all back
    align2, idx2 = state2.align_norm(state1, ret_index=True)
    assert np.allclose(state1.state, align2.state)
    assert np.allclose(state1.state, state2.sub(idx2).state)


def test_state_rotate():
    state = State([[1 + 1.0j, 1.0], [0.1 - 0.1j, 0.1]])

    # Angles are 45 and -45
    s = state.copy()
    assert np.pi / 4 == pytest.approx(np.angle(s.state[0, 0]))
    assert 0 == pytest.approx(np.angle(s.state[0, 1]))
    assert -np.pi / 4 == pytest.approx(np.angle(s.state[1, 0]))
    assert 0 == pytest.approx(np.angle(s.state[1, 1]))

    s.rotate(inplace=True)  # individual false
    assert 0 == pytest.approx(np.angle(s.state[0, 0]))
    assert -np.pi / 4 == pytest.approx(np.angle(s.state[0, 1]))
    assert -np.pi / 2 == pytest.approx(np.angle(s.state[1, 0]))
    assert -np.pi / 4 == pytest.approx(np.angle(s.state[1, 1]))

    s = state.copy()
    s.rotate(individual=True, inplace=True)
    assert 0 == pytest.approx(np.angle(s.state[0, 0]))
    assert -np.pi / 4 == pytest.approx(np.angle(s.state[0, 1]))
    assert 0 == pytest.approx(np.angle(s.state[1, 0]))
    assert np.pi / 4 == pytest.approx(np.angle(s.state[1, 1]))

    s = state.copy()
    s = s.rotate(np.pi / 4, individual=True)
    assert np.pi / 4 == pytest.approx(np.angle(s.state[0, 0]))
    assert 0 == pytest.approx(np.angle(s.state[0, 1]))
    assert np.pi / 4 == pytest.approx(np.angle(s.state[1, 0]))
    assert np.pi / 2 == pytest.approx(np.angle(s.state[1, 1]))


def test_cstate_creation():
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


def test_cstate_repr():
    state = StateC(ar(6), 1)
    assert len(state) == 1
    str(state)
    state = StateC(ar(6), 1, parent=geom.graphene())
    str(state)
    assert len(state) == 1


def test_cstate_sub():
    state = StateC(ar(10, 10), ar(10))
    assert len(state) == 10
    norm = state.norm()
    norm2 = state.norm2()
    for i in range(len(state)):
        assert len(state.sub(i)) == 1
        assert state.sub(i).norm()[0] == norm[i]
        assert state[i].norm()[0] == norm[i]
        assert state[i].norm2()[0] == norm2[i]
        assert state[i].c[0] == state.c[i]
    for i, sub in enumerate(state):
        assert len(sub) == 1
        assert sub.norm()[0] == norm[i]

    sub = state.sub(np.array([False, True, False, True]))
    state.sub([1, 3], inplace=True)
    assert np.allclose(sub.c, state.c)
    assert np.allclose(sub.state, state.state)


def test_cstate_sort():
    state = StateC(ar(10, 10), ar(10))
    sort = state.sort()
    assert len(state) == len(sort)
    c = sort.c[::-1]
    sort_descending = sort.sort(False)
    assert np.allclose(c, sort_descending.c)


def test_cstate_norm():
    state = StateC(ar(10, 10), ar(10)).normalize()
    assert len(state) == 10
    assert np.allclose(state.norm(), 1)
