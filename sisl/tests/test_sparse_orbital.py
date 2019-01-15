from __future__ import print_function, division

import pytest

import math as m
import numpy as np
import scipy as sc

from sisl import Geometry, Atom
from sisl.geom import fcc
from sisl.sparse_geometry import *


pytestmark = [pytest.mark.sparse, pytest.mark.sparse_geometry]


@pytest.mark.parametrize("n0", [1, 2, 4])
@pytest.mark.parametrize("n1", [1, 2, 4])
@pytest.mark.parametrize("n2", [1, 2, 4])
def test_sparse_orbital_symmetric(n0, n1, n2):
    g = fcc(1., Atom(1, R=1.5)) * 2
    s = SparseOrbital(g)
    s.construct([[0.1, 1.51], [1, 2]])
    s = s.tile(n0, 0).tile(n1, 1).tile(n2, 2)
    no = s.geometry.no

    nnz = no
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


@pytest.mark.parametrize("n0", [1, 2])
@pytest.mark.parametrize("n1", [1, 2])
@pytest.mark.parametrize("n2", [1, 2])
@pytest.mark.parametrize("axis", [0, 1, 2])
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

    # Now perform some appends and randomizations
    idx1 = np.arange(s1.na)
    idx2 = np.arange(s2.na)

    # Test 4 permutations
    for _ in range(3):
        np.random.shuffle(idx1)
        np.random.shuffle(idx2)
        sout = s1.sub(idx1).append(s2.sub(idx2), axis)

        s = sf.sub(np.concatenate([idx1, s1.na + idx2]))
        assert sout.spsame(s)
