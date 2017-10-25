from __future__ import print_function, division

import pytest

import math as m
import numpy as np
import scipy as sc

from sisl import Atom
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
        print(s)
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
        setup.s1[0, 0] = np.nan
        assert setup.s1.nnz == 0
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

    @pytest.mark.xfail(raises=ValueError)
    def test_rij_fail1(self, setup):
        s = SparseAtom(setup.g.copy())
        s.construct([[0.1, 1.5], [1, 2]])
        s.rij(what='none')

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_rij_fail2(self, setup):
        s = SparseAtom(setup.g.copy())
        s.construct([[0.1, 1.5], [1, 2]])
        s.rij(what='orbital')

    def test_rij2(self, setup):
        s = SparseAtom(setup.g.copy())
        s.construct([[0.1, 1.5], [1, 2]])
        atom = s.rij()
        assert atom.spsame(s)

    def test_rij3(self, setup):
        sa = SparseAtom(setup.g.copy())
        so = SparseOrbital(setup.g.copy())
        sa.construct([[0.1, 1.5], [1, 2]])
        so.construct([[0.1, 1.5], [1, 2]])
        atom = sa.rij()
        orbital = so.rij()
        assert atom.spsame(orbital)
        atom = sa.rij('atom')
        orbital = so.rij('atom')
        assert atom.spsame(orbital)
        # This only works because there is 1 orbital per atom
        orbital = so.rij()
        assert so.spsame(orbital)
        so.finalize()
        orbital = so.rij()
        assert so.spsame(orbital)

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
        assert len(s.edges(0)) == 3
        assert len(s.edges(0, exclude=[])) == 4

    def test_fromsp1(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        csr = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr[0, [1, 2, 3]] = 1
        csr[1, [2, 4, 1]] = 2
        s1 = SparseAtom.fromsp(g, [csr])
        assert s1.nnz == 6
        assert np.allclose(s1.shape, [g.na, g.na_s, 1])

        assert np.allclose(s1[0, [1, 2, 3]], np.ones([3], np.int32))
        assert np.allclose(s1[1, [1, 2, 4]], np.ones([3], np.int32)*2)

        # Different instantiating
        s2 = SparseAtom.fromsp(g, csr)
        assert s1.spsame(s2)

    def test_fromsp2(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        csr1 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr2 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr1[0, [1, 2, 3]] = 1
        csr2[1, [2, 4, 1]] = 2
        s1 = SparseAtom.fromsp(g, [csr1, csr2])
        assert s1.nnz == 6
        assert np.allclose(s1.shape, [g.na, g.na_s, 2])

        assert np.allclose(s1[0, [1, 2, 3], 0], np.ones([3], np.int32))
        assert np.allclose(s1[0, [1, 2, 3], 1], np.zeros([3], np.int32))
        assert np.allclose(s1[1, [1, 2, 4], 0], np.zeros([3], np.int32))
        assert np.allclose(s1[1, [1, 2, 4], 1], np.ones([3], np.int32)*2)

        s2 = SparseAtom.fromsp(g, csr1, csr2)
        assert s2.nnz == 6
        assert np.allclose(s2.shape, [g.na, g.na_s, 2])

        assert np.allclose(s2[0, [1, 2, 3], 0], np.ones([3], np.int32))
        assert np.allclose(s2[0, [1, 2, 3], 1], np.zeros([3], np.int32))
        assert np.allclose(s2[1, [1, 2, 4], 0], np.zeros([3], np.int32))
        assert np.allclose(s2[1, [1, 2, 4], 1], np.ones([3], np.int32)*2)

        assert s1.spsame(s2)

    @pytest.mark.xfail(raises=ValueError)
    def test_fromsp3(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        csr1 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr2 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr1[0, [1, 2, 3]] = 1
        csr2[1, [2, 4, 1]] = 2

        # Ensure that one does not mix everything.
        SparseAtom.fromsp(g, [csr1], csr2)

    @pytest.mark.xfail(raises=ValueError)
    def test_fromsp4(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        csr1 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr2 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr1[0, [1, 2, 3]] = 1
        csr2[1, [2, 4, 1]] = 2

        # Ensure that one does not mix everything.
        SparseAtom.fromsp(setup.g.copy(), [csr1, csr2])
