# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl.messages as sm
from sisl import Atom, Cuboid, Geometry, Lattice
from sisl._core.sparse_geometry import *
from sisl.geom import fcc, graphene

pytestmark = [
    pytest.mark.sparse,
    pytest.mark.sparse_geometry,
    pytest.mark.sparse_orbital,
]


@pytest.mark.parametrize("n0", [1, 3])
@pytest.mark.parametrize("n1", [1, 4])
@pytest.mark.parametrize("n2", [1, 2])
def test_sparse_orbital_symmetric(n0, n1, n2):
    g = fcc(1.0, Atom(1, R=1.5)) * 2
    s = SparseOrbital(g)
    s.construct([[0.1, 1.51], [1, 2]])
    s = s.tile(n0, 0).tile(n1, 1).tile(n2, 2)
    no = s.geometry.no

    nnz = 0
    for io in range(no):
        # orbitals connecting to io
        edges = s.edges(io)
        # Figure out the transposed supercell indices of the edges
        isc = -s.geometry.o2isc(edges)
        # Convert to supercell
        IO = s.geometry.lattice.sc_index(isc) * no + io
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
    g = fcc(1.0, Atom(1, R=1.98)) * 2
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

    np.random.seed(42)
    shuffle = np.random.shuffle

    # Test 4 permutations
    for _ in range(3):
        shuffle(idx1)
        shuffle(idx2)
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
    g = fcc(1.0, Atom(1, R=1.98)) * 2
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

    np.random.seed(42)
    shuffle = np.random.shuffle

    # Test 4 permutations
    for _ in range(3):
        shuffle(idx1)
        shuffle(idx2)

        s = sf.sub(np.concatenate([idx1, s1.na + idx2]))
        s.finalize()

        sout = s1.sub(idx1).append(s2.sub(idx2), axis, scale=(2.0, 0))
        sout = (sout + sout.transpose()) * 0.5
        assert sout.spsame(s)
        sout.finalize()
        assert np.allclose(s._csr._D, sout._csr._D)

        sout = s1.sub(idx1).append(s2.sub(idx2), axis, scale=(0.0, 2.0))
        sout.finalize()
        # Ensure that some elements are not the same!
        assert not np.allclose(s._csr._D, sout._csr._D)
        sout = (sout + sout.transpose()) * 0.5
        assert sout.spsame(s)
        sout.finalize()
        assert np.allclose(s._csr._D, sout._csr._D)


def test_sparse_orbital_hermitian():
    g = Geometry([0] * 3, Atom(1, R=1), lattice=Lattice(1, nsc=[3, 1, 1]))

    for f in [True, False]:
        spo = SparseOrbital(g)
        spo[0, 0] = 1.0

        # Create only a single coupling to the neighouring element
        spo[0, 1] = 2.0

        if f:
            spo.finalize()

        assert spo.nnz == 2

        # implicit sort
        spoT = spo.transpose()
        assert spoT.finalized
        assert spoT.nnz == 2
        assert spoT[0, 0] == 1.0
        assert spoT[0, 1] == 0.0
        assert spoT[0, 2] == 2.0

        spoH = (spo + spoT) * 0.5

        assert spoH.nnz == 3
        assert spoH[0, 0] == 1.0
        assert spoH[0, 1] == 1.0
        assert spoH[0, 2] == 1.0


def test_sparse_orbital_sub_orbital():
    a0 = Atom(1, R=(1.1, 1.4, 1.6))
    a1 = Atom(2, R=(1.3, 1.1))
    g = Geometry([[0, 0, 0], [1, 1, 1]], [a0, a1], lattice=Lattice(2, nsc=[3, 1, 1]))
    g = g.tile(3, 0)
    assert g.no == 15

    spo = SparseOrbital(g)
    for io in range(g.no):
        spo[io, io] = io + 1
        spo[io, io + g.no - 1] = io - 2
        spo[io, io + 1] = io + 2
    # Ensure we have a Hermitian matrix
    spo = spo + spo.transpose()

    # orbitals on first atom (a0)
    rem_sub = [(0, [1, 2]), ([0, 2], 1), (2, [0, 1]), (a0[0], [1, 2])]
    for rem, sub in rem_sub:
        spo_rem = spo.remove_orbital(0, rem)
        spo_sub = spo.sub_orbital(0, sub)
        assert spo_rem.spsame(spo_sub)
        atoms = spo_rem.geometry.atoms
        assert atoms == spo_sub.geometry.atoms
        assert atoms.nspecies == 3
        assert (atoms.species == 0).sum() == 2
        assert (atoms.species == 1).sum() == 3
        assert (atoms.species == 2).sum() == 1

        spo_rem = spo.remove_orbital(a0, rem)
        spo_sub = spo.sub_orbital(a0, sub)
        assert spo_rem.spsame(spo_sub)
        atoms = spo_rem.geometry.atoms
        assert atoms == spo_sub.geometry.atoms
        assert atoms.nspecies == 2
        assert (atoms.species == 0).sum() == 3
        assert (atoms.species == 1).sum() == 3

    # orbitals on second atom (a1)
    rem_sub = [
        (0, [1]),
        (a1[0], 1),
        (0, a1[1]),
    ]
    for rem, sub in rem_sub:
        spo_rem = spo.remove_orbital(1, rem)
        spo_sub = spo.sub_orbital(1, sub)
        assert spo_rem.spsame(spo_sub)
        atoms = spo_rem.geometry.atoms
        assert atoms == spo_sub.geometry.atoms
        assert atoms.nspecies == 3
        assert (atoms.species == 0).sum() == 3
        assert (atoms.species == 1).sum() == 2
        assert (atoms.species == 2).sum() == 1

        spo_rem = spo.remove_orbital(a1, rem)
        spo_sub = spo.sub_orbital(a1, sub)
        assert spo_rem.spsame(spo_sub)
        atoms = spo_rem.geometry.atoms
        assert atoms == spo_sub.geometry.atoms
        assert atoms.nspecies == 2
        assert (atoms.species == 0).sum() == 3
        assert (atoms.species == 1).sum() == 3

    spo_rem = spo.remove_orbital([0, 1], 0)
    spo_sub = spo.sub_orbital(0, [1, 2]).sub_orbital(1, 1)
    assert spo_rem.spsame(spo_sub)

    spo_rem = spo.remove_orbital([0, 1], 1)
    spo_sub = spo.sub_orbital(0, [0, 2]).sub_orbital(1, 0)
    assert spo_rem.spsame(spo_sub)


def test_sparse_orbital_sub_orbital_nested():
    """
    Doing nested or multiple subs that exposes the same sub-atom
    should ultimately re-use the existing atom.

    However, due to the renaming of the tags
    """
    a0 = Atom(1, R=(1.1, 1.4, 1.6))
    a1 = Atom(2, R=(1.3, 1.1))
    g = Geometry([[0, 0, 0], [1, 1, 1]], [a0, a1], lattice=Lattice(2, nsc=[3, 1, 1]))
    g = g.repeat(3, 0)
    assert g.no == 15

    spo = SparseOrbital(g)
    for io in range(g.no):
        spo[io, io] = io + 1
        spo[io, io + g.no - 1] = io - 2
        spo[io, io + 1] = io + 2
    # Ensure we have a Hermitian matrix
    spo = spo + spo.transpose()

    assert spo.geometry.atoms.nspecies == 2
    sub0 = spo.remove_orbital(0, 1)
    assert sub0.geometry.atoms.nspecies == 3
    # this will *replace* since we take all atoms of a given species
    sub01 = sub0.remove_orbital(0, 1)
    assert sub01.geometry.atoms.nspecies == 3

    # while this creates the same atom as the above two steps
    # it cannot recognize it as the same atom due
    # to different tag names
    sub = sub01.sub_orbital(1, 0)
    assert sub.geometry.atoms.nspecies == 4


def test_sparse_orbital_replace_simple():
    """
    Replacing parts of a sparse-orbital matrix is quite tricky.

    Here we check a few things with graphene
    """
    gr = graphene()
    spo = SparseOrbital(gr)
    # create the sparse-orbital
    spo.construct([(0.1, 1.45), (0, 2.7)])

    # create 4x4 geoemetry
    spo44 = spo.tile(4, 0).tile(4, 1)

    # now replace every position that can be replaced
    for position in range(0, spo44.geometry.na, 2):
        # replace both atoms
        new = spo44.replace([position, position + 1], spo)
        assert np.fabs((new - spo44)._csr._D).sum() == 0.0

        # replace both atoms (reversed)
        # When swapping it is not the same
        new = spo44.replace([position, position + 1], spo, [1, 0])
        assert np.fabs((new - spo44)._csr._D).sum() != 0.0
        pvt = np.arange(spo44.na)
        pvt[position] = position + 1
        pvt[position + 1] = position
        assert np.unique(pvt).size == pvt.size
        new_pvt = spo44.sub(pvt)
        assert np.fabs((new - new_pvt)._csr._D).sum() == 0.0

        # replace first atom
        new = spo44.replace([position], spo, 0)
        assert np.fabs((new - spo44)._csr._D).sum() == 0.0

        # replace second atom
        new = spo44.replace([position + 1], spo, 1)
        assert np.fabs((new - spo44)._csr._D).sum() == 0.0


def test_sparse_orbital_replace_species_warn():
    """
    Replacing an atom with another one but retaining couplings.
    """
    C = graphene()
    spC = SparseOrbital(C)
    spC.construct([(0.1, 1.45), (0, 2.7)])

    N = graphene(atoms=C.atoms[1].copy(Z=5))
    spN = SparseOrbital(N)
    spN.construct([(0.1, 1.45), (0, 2.7)])

    # Replace
    with pytest.warns(sm.SislWarning):
        sp = spC.copy()
        sp.replace(0, spN, 0)
    with pytest.warns(sm.SislWarning):
        sp = spC.copy()
        sp.replace(1, spN, 1)


def test_sparse_orbital_replace_hole():
    """Create a big graphene flake remove a hole (1 orbital system)"""
    g = graphene(orthogonal=True)
    spo = SparseOrbital(g)
    # create the sparse-orbital
    spo.construct([(0.1, 1.45), (0, 2.7)])

    # create 10x10 geoemetry
    nx, ny = 10, 10
    big = spo.tile(10, 0).tile(10, 1)
    hole = spo.tile(6, 0).tile(6, 1)
    hole = hole.remove(hole.close(hole.center(), R=3))

    def create_sp(geom):
        spo = SparseOrbital(geom)
        # create the sparse-orbital
        spo.construct([(0.1, 1.45), (0, 2.7)])
        return spo

    # now replace every position that can be replaced
    for y in [0, 2, 3]:
        for x in [1, 2, 3]:
            cube = Cuboid(hole.lattice.cell, origin=g.lattice.offset([x, y, 0]) - 0.1)
            atoms = big.within(cube)
            assert len(atoms) == 4 * 6 * 6
            new = big.replace(atoms, hole)
            new_copy = create_sp(new.geometry)
            assert np.fabs((new - new_copy)._csr._D).sum() == 0.0


def test_sparse_orbital_replace_hole_norbs():
    """Create a big graphene flake remove a hole (multiple orbitals)"""
    a1 = Atom(5, R=(1.44, 1.44))
    a2 = Atom(7, R=(1.44, 1.44, 1.44))
    g = graphene(atoms=[a1, a2], orthogonal=True)
    spo = SparseOrbital(g)

    def func(self, ia, atoms, atoms_xyz=None):
        geom = self.geometry

        def a2o(idx):
            return geom.a2o(idx, True)

        io = a2o(ia)
        idx = self.geometry.close(ia, R=[0.1, 1.44], atoms=atoms, atoms_xyz=atoms_xyz)
        idx = list(map(a2o, idx))
        self[io, idx[0]] = 0
        for i in io:
            self[i, idx[1]] = 2.7

    # create the sparse-orbital
    spo.construct(func)

    # create 10x10 geoemetry
    nx, ny = 10, 10
    big = spo.tile(10, 0).tile(10, 1)
    hole = spo.tile(6, 0).tile(6, 1)
    hole = hole.remove(hole.close(hole.center(), R=3))

    def create_sp(geom):
        spo = SparseOrbital(geom)
        # create the sparse-orbital
        spo.construct(func)
        return spo

    # now replace every position that can be replaced
    for y in [0, 3]:
        for x in [1, 3]:
            cube = Cuboid(hole.lattice.cell, origin=g.lattice.offset([x, y, 0]) - 0.1)
            atoms = big.within(cube)
            assert len(atoms) == 4 * 6 * 6
            new = big.replace(atoms, hole)
            new_copy = create_sp(new.geometry)
            assert np.fabs((new - new_copy)._csr._D).sum() == 0.0


def test_translate_sparse_orbitals():
    # Build the geometry
    H = Atom(1, (1, 2, 3))
    graph = graphene(atoms=H)
    assert np.allclose(graph.nsc, [3, 3, 1])

    # Build a dummy matrix with onsite terms and just one coupling term
    matrix = SparseOrbital(graph)
    matrix[0, 0] = 1
    matrix[1, 1] = 2
    matrix[0, 1] = 3
    matrix[0, 4] = 4

    # Translate the second atom
    transl = matrix._translate_atoms_sc([[0, 0, 0], [1, 0, 0]])

    # Check that the new auxiliary cell is correct.
    assert np.allclose(transl.nsc, [3, 1, 1])
    assert np.allclose(transl.shape, [6, 18, 1])

    # Check coordinates
    assert np.allclose(transl.geometry[0], matrix.geometry[0])
    assert np.allclose(transl.geometry[1], matrix.geometry[1] + matrix.geometry.cell[0])

    # Assert that the matrix elements have been translated
    assert transl[0, 0] == 1
    assert transl[1, 1] == 2
    assert transl[0, 1] == 3
    assert transl[0, 6 + 4] == 4

    # Translate back to unit cell
    uc_matrix = transl.translate2uc()

    # Check the auxiliary cell, coordinates and the matrix elements
    assert np.allclose(uc_matrix.nsc, [1, 1, 1])
    assert np.allclose(uc_matrix.shape, [6, 6, 1])

    assert np.allclose(uc_matrix.geometry.xyz, matrix.geometry.xyz)

    assert uc_matrix[0, 0] == 1
    assert uc_matrix[1, 1] == 2
    assert uc_matrix[0, 1] == 3
    assert uc_matrix[0, 4] == 4

    # Instead, test atoms and axes arguments to avoid any translation.
    for kwargs in [{"atoms": [0]}, {"axes": [1, 2]}]:
        not_uc_matrix = transl.translate2uc(**kwargs)

        # Check the auxiliary cell, coordinates and the matrix elements
        assert np.allclose(not_uc_matrix.nsc, transl.nsc)
        assert np.allclose(not_uc_matrix.shape, transl.shape)

        assert np.allclose(not_uc_matrix.geometry.xyz, transl.geometry.xyz)

        assert np.allclose(not_uc_matrix._csr.todense(), transl._csr.todense())

    # Now, translate both atoms
    transl_both = uc_matrix._translate_atoms_sc([[-1, 0, 0], [1, 0, 0]])

    # Check the auxiliary cell, coordinates and the matrix elements
    assert np.allclose(transl_both.nsc, [5, 1, 1])
    assert np.allclose(transl_both.shape, [6, 6 * 5, 1])

    assert np.allclose(
        transl_both.geometry[0], uc_matrix.geometry[0] - uc_matrix.geometry.cell[0]
    )
    assert np.allclose(
        transl_both.geometry[1], uc_matrix.geometry[1] + uc_matrix.geometry.cell[0]
    )

    assert transl_both[0, 0] == 1
    assert transl_both[1, 1] == 2
    assert transl_both[0, 1] == 3
    assert transl_both[0, 6 + 4] == 4
