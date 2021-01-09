import pytest

import math as m
import numpy as np
import scipy as sc

from sisl import Geometry, Atom, SuperCell
from sisl.geom import fcc
from sisl.sparse_geometry import *


pytestmark = [pytest.mark.sparse, pytest.mark.sparse_geometry]


@pytest.mark.parametrize("n0", [1, 3])
@pytest.mark.parametrize("n1", [1, 4])
@pytest.mark.parametrize("n2", [1, 2])
def test_sparse_orbital_symmetric(n0, n1, n2):
    g = fcc(1., Atom(1, R=1.5)) * 2
    s = SparseOrbital(g)
    s.construct([[0.1, 1.51], [1, 2]])
    s = s.tile(n0, 0).tile(n1, 1).tile(n2, 2)
    no = s.geometry.no

    nnz = 0
    for io in range(no):
        # orbitals connecting to io
        edges = s.edges(io)
        # Figure out the transposed supercell indices of the edges
        isc = - s.geometry.o2isc(edges)
        # Convert to supercell
        IO = s.geometry.sc.sc_index(isc) * no + io
        # Figure out if 'io' is also in the back-edges
        for jo, edge in zip(IO, edges % no):
            assert jo in s.edges(edge)
            nnz += 1

    # Check that we have counted all nnz
    assert s.nnz == nnz


@pytest.mark.parametrize("n0", [1, 3])
@pytest.mark.parametrize("n1", [1, 4])
@pytest.mark.parametrize("n2", [1, 2])
@pytest.mark.parametrize("axis", [0, 1])
def test_sparse_orbital_append(n0, n1, n2, axis):
    g = fcc(1., Atom(1, R=1.98)) * 2
    dists = np.insert(g.distance(0, R=g.maxR()) + 0.001, 0, 0.001)
    connect = np.arange(dists.size, dtype=np.float64) / 5
    s = SparseOrbital(g)
    s.construct([dists, connect])
    s = s.tile(2, 0).tile(2, 1).tile(2, 2)
    s1 = s.tile(n0, 0).tile(n1, 1).tile(n2, 2)
    s2 = s1.copy()
    # Resulting full sparse-geometry
    sf = s1.tile(2, axis)
    # Ensure the test works for empty rows
    for i in range(sf.shape[0]):
        sf._csr._extend_empty(i, 11)

    # Now perform some appends and randomizations
    idx1 = np.arange(s1.na)
    idx2 = np.arange(s2.na)

    # Test 4 permutations
    for _ in range(3):
        np.random.shuffle(idx1)
        np.random.shuffle(idx2)
        sout = s1.sub(idx1).append(s2.sub(idx2), axis)
        sout.finalize()

        s = sf.sub(np.concatenate([idx1, s1.na + idx2]))
        assert sout.spsame(s)
        s.finalize()
        assert np.allclose(s._csr._D, sout._csr._D)


@pytest.mark.parametrize("n0", [1, 3])
@pytest.mark.parametrize("n1", [1, 4])
@pytest.mark.parametrize("n2", [1, 2])
@pytest.mark.parametrize("axis", [1, 2])
def test_sparse_orbital_append_scale(n0, n1, n2, axis):
    g = fcc(1., Atom(1, R=1.98)) * 2
    dists = np.insert(g.distance(0, R=g.maxR()) + 0.001, 0, 0.001)
    connect = np.arange(dists.size, dtype=np.float64) / 5
    s = SparseOrbital(g)
    s.construct([dists, connect])
    s = s.tile(2, 0).tile(2, 1).tile(2, 2)
    s1 = s.tile(n0, 0).tile(n1, 1).tile(n2, 2)
    s2 = s1.copy()
    # Resulting full sparse-geometry
    sf = s1.tile(2, axis)
    for i in range(sf.shape[0]):
        sf._csr._extend_empty(i, 11)

    # Now perform some appends and randomizations
    idx1 = np.arange(s1.na)
    idx2 = np.arange(s2.na)

    # Test 4 permutations
    for _ in range(3):
        np.random.shuffle(idx1)
        np.random.shuffle(idx2)

        s = sf.sub(np.concatenate([idx1, s1.na + idx2]))
        s.finalize()

        sout = s1.sub(idx1).append(s2.sub(idx2), axis, scale=(2., 0))
        sout = (sout + sout.transpose()) * 0.5
        assert sout.spsame(s)
        sout.finalize()
        assert np.allclose(s._csr._D, sout._csr._D)

        sout = s1.sub(idx1).append(s2.sub(idx2), axis, scale=(0., 2.))
        sout.finalize()
        # Ensure that some elements are not the same!
        assert not np.allclose(s._csr._D, sout._csr._D)
        sout = (sout + sout.transpose()) * 0.5
        assert sout.spsame(s)
        sout.finalize()
        assert np.allclose(s._csr._D, sout._csr._D)


def test_sparse_orbital_hermitian():
    g = Geometry([0] * 3, Atom(1, R=1), sc=SuperCell(1, nsc=[3, 1, 1]))

    for f in [True, False]:
        spo = SparseOrbital(g)
        spo[0, 0] = 1.

        # Create only a single coupling to the neighouring element
        spo[0, 1] = 2.

        if f:
            spo.finalize()

        assert spo.nnz == 2

        # implicit sort
        spoT = spo.transpose()
        assert spoT.finalized
        assert spoT.nnz == 2
        assert spoT[0, 0] == 1.
        assert spoT[0, 1] == 0.
        assert spoT[0, 2] == 2.

        spoH = (spo + spoT) * 0.5

        assert spoH.nnz == 3
        assert spoH[0, 0] == 1.
        assert spoH[0, 1] == 1.
        assert spoH[0, 2] == 1.


def test_sparse_orbital_sub_orbital():
    a0 = Atom(1, R=(1.1, 1.4, 1.6))
    a1 = Atom(2, R=(1.3, 1.1))
    g = Geometry([[0, 0, 0], [1, 1, 1]], [a0, a1], sc=SuperCell(2, nsc=[3, 1, 1]))
    assert g.no == 5

    spo = SparseOrbital(g)
    for io in range(g.no):
        spo[io, io] = io + 1
        spo[io, io + g.no - 1] = io - 2
        spo[io, io + 1] = io + 2
    # Ensure we have a Hermitian matrix
    spo = spo + spo.transpose()

    # Ensure sub and remove does the same
    for i in [0, a0]:
        spo_rem = spo.remove_orbital(i, 0)
        spo_sub = spo.sub_orbital(i, [1, 2])
        assert spo_rem.spsame(spo_sub)

        spo_rem = spo.remove_orbital(i, [0, 2])
        spo_sub = spo.sub_orbital(i, 1)
        assert spo_rem.spsame(spo_sub)

        spo_rem = spo.remove_orbital(i, 2)
        spo_sub = spo.sub_orbital(i, [0, 1])
        assert spo_rem.spsame(spo_sub)

        spo_rem = spo.remove_orbital(i, a0[0])
        spo_sub = spo.sub_orbital(i, [1, 2])
        assert spo_rem.spsame(spo_sub)

    for i in [1, a1]:
        spo_rem = spo.remove_orbital(i, a1[0])
        spo_sub = spo.sub_orbital(i, a1[1])
        assert spo_rem.spsame(spo_sub)

    spo_rem = spo.remove_orbital([0, 1], 0)
    spo_sub = spo.sub_orbital(0, [1, 2]).sub_orbital(1, 1)
    assert spo_rem.spsame(spo_sub)

    spo_rem = spo.remove_orbital([0, 1], 1)
    spo_sub = spo.sub_orbital(0, [0, 2]).sub_orbital(1, 0)
    assert spo_rem.spsame(spo_sub)
