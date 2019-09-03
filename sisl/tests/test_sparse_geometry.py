from __future__ import print_function, division

import pytest

import math as m
import numpy as np
import scipy as sc

from sisl import Geometry, Atom
from sisl.geom import fcc, graphene
from sisl.sparse_geometry import *


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            self.g = fcc(1., Atom(1, R=1.5)) * 2
            self.s1 = SparseAtom(self.g)
            self.s2 = SparseAtom(self.g, 2)
    return t()


@pytest.mark.sparse
@pytest.mark.sparse_geometry
class TestSparseAtom(object):

    @pytest.mark.xfail(raises=ValueError)
    def test_fail_align1(self, setup):
        s = SparseAtom(setup.g * 2)
        str(s)
        setup.s1.spalign(s)

    def test_align1(self, setup):
        s = SparseAtom(setup.g)
        setup.s1.spalign(s)
        setup.s1.spalign(s._csr)

    def test_create1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        assert setup.s1.nnz == 3
        assert setup.s1[0, 1] == 1
        setup.s1[2, [1, 2, 3]] = 1
        assert setup.s1.nnz == 6
        setup.s1.empty(keep_nnz=True)
        assert setup.s1.nnz == 6
        setup.s1.empty()
        assert setup.s1.nnz == 0
        # nan produces zeros
        setup.s1[0, 0] = np.nan
        assert setup.s1.nnz == 1
        setup.s1.empty()

    def test_create2(self, setup):
        setup.s1[0, [1, 2, 3], (1, 1, 1)] = 1
        assert setup.s1[0, 2, (1, 1, 1)] == 1
        setup.s1.empty()

    def test_create3(self, setup):
        setup.s2[0, [1, 2, 3], (1, 1, 1)] = [1, 2]
        assert np.allclose(setup.s2[0, 2, (1, 1, 1)], [1, 2])
        assert setup.s2[0, 2, 0, (1, 1, 1)] == 1
        assert setup.s2[0, 2, 1, (1, 1, 1)] == 2
        setup.s2.empty()

    def test_in1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        assert setup.s1[0, 2] == 1
        assert [0, 4] not in setup.s1
        assert [0, 2] in setup.s1
        setup.s1.empty()

    def test_reset1(self, setup):
        setup.s1[0, [1, 2, 3], (1, 1, 1)] = 1
        assert setup.s1[0, 2, (1, 1, 1)] == 1
        setup.s1.reset()
        assert setup.s1.nnz == 0
        setup.s1.empty()

    def test_eliminate_zeros1(self, setup):
        setup.s1[0, [1, 2, 3]] = 0
        assert setup.s1.nnz == 3
        setup.s1.eliminate_zeros()
        assert setup.s1.nnz == 0
        setup.s1.empty()

    def test_nonzero1(self, setup):
        s2 = setup.s2.copy()
        s2[0, [0, 2, 3]] = [1, 2]
        s2[3, [1, 2, 3]] = [1, 2]

        r, c = s2.nonzero()
        assert np.allclose(r, [0, 0, 0, 3, 3, 3])
        assert np.allclose(c, [0, 2, 3, 1, 2, 3])
        c = s2.nonzero(only_col=True)
        assert np.allclose(c, [0, 2, 3, 1, 2, 3])
        r, c = s2.nonzero(atom=1)
        assert len(r) == 0
        assert len(c) == 0
        r, c = s2.nonzero(atom=0)
        assert np.allclose(r, [0, 0, 0])
        assert np.allclose(c, [0, 2, 3])
        c = s2.nonzero(atom=0, only_col=True)
        assert np.allclose(c, [0, 2, 3])

    def test_cut1(self, setup):
        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(setup.g * 2)
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.cut(2, 2).cut(2, 1).cut(2, 0)
        assert s1.spsame(s2)

        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(setup.g * [2, 1, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.cut(2, 0)
        assert s1.spsame(s2)

        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(setup.g * [1, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.cut(2, 1)
        assert s1.spsame(s2)

    def test_iter(self, setup):
        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        i = 0
        for r, c in s1:
            i += 1
        assert i == s1.nnz

    @pytest.mark.xfail(raises=ValueError)
    def test_rij_fail1(self, setup):
        s = SparseOrbital(setup.g.copy())
        s.construct([[0.1, 1.5], [1, 2]])
        s.rij(what='none')

    def test_rij_atom(self, setup):
        s = SparseAtom(setup.g.copy())
        s.construct([[0.1, 1.5], [1, 2]])
        atom = s.rij()
        assert atom.spsame(s)

    def test_rij_atom_orbital_compare(self, setup):
        sa = SparseAtom(setup.g.copy())
        so = SparseOrbital(setup.g.copy())
        sa.construct([[0.1, 1.5], [1, 2]])
        so.construct([[0.1, 1.5], [1, 2]])
        atom = sa.rij()
        orbital = so.rij()
        assert atom.spsame(orbital)
        atom = sa.rij()
        orbital = so.rij('atom')
        assert atom.spsame(orbital)
        # This only works because there is 1 orbital per atom
        orbital = so.rij()
        assert so.spsame(orbital)
        so.finalize()
        orbital = so.rij()
        assert so.spsame(orbital)

    def test_sp_orb_remove(self, setup):
        so = SparseOrbital(setup.g.copy())
        so2 = so.remove(0)
        assert so.geometry.na - 1 == so2.geometry.na
        so2 = so.remove([0])
        assert so.geometry.na - 1 == so2.geometry.na
        so2 = so.remove([1])
        assert so.geometry.na - 1 == so2.geometry.na

    def test_sp_orb_remove_atom(self):
        so = SparseOrbital(Geometry([[0] *3, [1]* 3], [Atom[1], Atom[2]], 2))
        so2 = so.remove(Atom[1])
        assert so.geometry.na - 1 == so2.geometry.na
        assert so.geometry.no -1 == so2.geometry.no

    def test_remove1(self, setup):
        for i in range(len(setup.g)):
            setup.s1.construct([[0.1, 1.5], [1, 2]])
            s1 = setup.s1.remove(i)
            setup.s1.empty()
            s2 = SparseAtom(setup.g.remove(i))
            s2.construct([[0.1, 1.5], [1, 2]])
            assert s1.spsame(s2)

    def test_sub1(self, setup):
        all = range(len(setup.g))
        for i in range(len(setup.g)):
            setup.s1.construct([[0.1, 1.5], [1, 2]])
            # my new sub
            sub = [j for j in all if i != j]
            s1 = setup.s1.sub(sub)
            setup.s1.empty()
            s2 = SparseAtom(setup.g.sub(sub))
            s2.construct([[0.1, 1.5], [1, 2]])
            assert s1.spsame(s2)
            assert len(s2) == len(sub)

    def test_construct_eta(self, setup):
        s = setup.s1.copy()
        s.construct([[0.1, 1.5], [1, 2]], eta=True)

    def test_tile1(self, setup):
        setup.s1.construct([[0.1, 1.5], [1, 2]])
        setup.s1.finalize()
        s1 = setup.s1.tile(2, 0).tile(2, 1)
        s2 = SparseAtom(setup.g * [2, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        assert s1.spsame(s2)
        s1.finalize()
        s2.finalize()
        assert np.allclose(s1._csr._D, s2._csr._D)
        s2 = s2.cut(2, 1).cut(2, 0)
        assert setup.s1.spsame(s2)
        s2.finalize()
        assert np.allclose(setup.s1._csr._D, s2._csr._D)
        s1 = s1.cut(2, 1).cut(2, 0)
        assert setup.s1.spsame(s1)
        s1.finalize()
        assert np.allclose(s1._csr._D, setup.s1._csr._D)
        setup.s1.empty()

    def test_tile2(self, setup):
        setup.s1.construct([[0.1, 1.5], [1, 2]])
        setup.s1.finalize()
        s1 = setup.s1.tile(2, 0).tile(2, 1)
        s2 = SparseAtom(setup.g * [2, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        assert s1.spsame(s2)
        s1.finalize()
        s2.finalize()
        assert np.allclose(s1._csr._D, s2._csr._D)
        s2 = s2.cut(2, 1).cut(2, 0)
        assert setup.s1.spsame(s2)
        s2.finalize()
        assert np.allclose(setup.s1._csr._D, s2._csr._D)
        s1 = s1.cut(2, 1).cut(2, 0)
        assert setup.s1.spsame(s1)
        s1.finalize()
        assert np.allclose(s1._csr._D, setup.s1._csr._D)
        setup.s1.empty()

    def test_repeat1(self, setup):
        setup.s1.construct([[0.1, 1.5], [1, 2]])
        s1 = setup.s1.repeat(2, 0).repeat(2, 1)
        setup.s1.empty()
        s2 = SparseAtom(setup.g * ([2, 2, 1], 'r'))
        s2.construct([[0.1, 1.5], [1, 2]])
        assert s1.spsame(s2)
        s1.finalize()
        s2.finalize()
        assert np.allclose(s1._csr._D, s2._csr._D)

    def test_repeat2(self, setup):
        setup.s1.construct([[0.1, 1.5], [1, 2]])
        setup.s1.finalize()
        s1 = setup.s1.repeat(2, 0).repeat(2, 1)
        setup.s1.empty()
        s2 = SparseAtom(setup.g * ([2, 2, 1], 'r'))
        s2.construct([[0.1, 1.5], [1, 2]])
        assert s1.spsame(s2)
        s1.finalize()
        s2.finalize()
        assert np.allclose(s1._csr._D, s2._csr._D)

    def test_supercell_poisition1(self, setup):
        g1 = setup.g.copy()
        g2 = setup.g.translate([100, 100, 100])
        # Just check that the atomic coordinates are not equivalent
        # up to 1 angstrom
        assert not np.allclose(g1.xyz, g2.xyz, atol=1)
        s1 = SparseAtom(g1)
        s2 = SparseAtom(g2)
        s1.construct([[0.1, 1.5], [1, 2]])
        s1.finalize()
        s2.construct([[0.1, 1.5], [1, 2]])
        s2.finalize()
        assert s1.spsame(s2)

    def test_set_nsc1(self, setup):
        g = fcc(1., Atom(1, R=3.5))
        s = SparseAtom(g)
        s.construct([[0.1, 1.5, 3.5], [1, 2, 3]])
        s.finalize()
        assert s.nnz > 1
        s.set_nsc([1, 1, 1])
        assert s.nnz == 1
        assert s[0, 0] == 1

    def test_set_nsc2(self, setup):
        g = graphene(atom=Atom(6, R=1.43))
        s = SparseAtom(g)
        s.construct([[0.1, 1.43], [1, 2]])
        s.finalize()
        assert s.nnz == 8
        s.set_nsc(a=1)
        assert s.nnz == 6
        s.set_nsc([None, 1, 1])
        assert s.nnz == 4
        assert s[0, 0] == 1

    def test_edges1(self, setup):
        g = graphene(atom=Atom(6, R=1.43))
        s = SparseAtom(g)
        s.construct([[0.1, 1.43], [1, 2]])
        assert len(s.edges(0)) == 4
        assert len(s.edges(0, exclude=[0])) == 3

    def test_op_numpy_scalar(self, setup):
        g = graphene(atom=Atom(6, R=1.43))
        S = SparseAtom(g)
        I = np.ones(1, dtype=np.complex128)[0]
        # Create initial stuff
        for i in range(10):
            j = range(i, i*2)
            S[0, j] = i
        S.finalize()

        Ssum = S._csr._D.sum()

        s = S + I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum + S.nnz

        s = S - I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum - S.nnz

        s = I + S
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum + S.nnz

        s = S * I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum

        s = I * S
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum

        s = S / I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum

        s = S ** I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum

        s = I ** S
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128

    def test_fromsp1(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        lil = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil[0, [1, 2, 3]] = 1
        lil[1, [2, 4, 1]] = 2
        s1 = SparseAtom.fromsp(g, [lil])
        assert s1.nnz == 6
        assert np.allclose(s1.shape, [g.na, g.na_s, 1])

        assert np.allclose(s1[0, [1, 2, 3]], np.ones([3], np.int32))
        assert np.allclose(s1[1, [1, 2, 4]], np.ones([3], np.int32)*2)

        # Different instantiating
        s2 = SparseAtom.fromsp(g, lil)
        assert s1.spsame(s2)

    def test_fromsp2(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        lil1 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil2 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil1[0, [1, 2, 3]] = 1
        lil2[1, [2, 4, 1]] = 2
        s1 = SparseAtom.fromsp(g, [lil1, lil2])
        assert s1.nnz == 6
        assert np.allclose(s1.shape, [g.na, g.na_s, 2])

        assert np.allclose(s1[0, [1, 2, 3], 0], np.ones([3], np.int32))
        assert np.allclose(s1[0, [1, 2, 3], 1], np.zeros([3], np.int32))
        assert np.allclose(s1[1, [1, 2, 4], 0], np.zeros([3], np.int32))
        assert np.allclose(s1[1, [1, 2, 4], 1], np.ones([3], np.int32)*2)

    @pytest.mark.xfail(raises=ValueError)
    def test_fromsp4(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        lil1 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil2 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil1[0, [1, 2, 3]] = 1
        lil2[1, [2, 4, 1]] = 2

        # Ensure that one does not mix everything.
        SparseAtom.fromsp(setup.g.copy(), [lil1, lil2])

    def test_pickle(self, setup):
        import pickle as p

        g = setup.g.repeat(2, 0).tile(2, 1)
        lil1 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil2 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil1[0, [1, 2, 3]] = 1
        lil2[1, [2, 4, 1]] = 2
        S = SparseAtom.fromsp(g, [lil1, lil2])
        n = p.dumps(S)
        s = p.loads(n)
        assert s.spsame(S)


@pytest.mark.parametrize("n0", [1, 2, 4])
@pytest.mark.parametrize("n1", [1, 2, 4])
@pytest.mark.parametrize("n2", [1, 2, 4])
def test_sparse_atom_symmetric(n0, n1, n2):
    g = fcc(1., Atom(1, R=1.5)) * 2
    s = SparseAtom(g)
    s.construct([[0.1, 1.51], [1, 2]])
    s = s.tile(n0, 0).tile(n1, 1).tile(n2, 2)
    na = s.geometry.na

    nnz = 0
    for ia in range(na):
        # orbitals connecting to ia
        edges = s.edges(ia)
        # Figure out the transposed supercell indices of the edges
        isc = - s.geometry.a2isc(edges)
        # Convert to supercell
        IA = s.geometry.sc.sc_index(isc) * na + ia
        # Figure out if 'ia' is also in the back-edges
        for ja, edge in zip(IA, edges % na):
            assert ja in s.edges(edge)
            nnz += 1

    # Check that we have counted all nnz
    assert s.nnz == nnz
