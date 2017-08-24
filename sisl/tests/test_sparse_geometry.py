from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np
import scipy as sc

from sisl import Atom
from sisl.geom import fcc, graphene
from sisl.sparse_geometry import *


@attr('sparse', 'sparse_geometry')
class TestSparseAtom(object):

    def setUp(self):
        self.g = fcc(1., Atom(1, R=1.5)) * 2
        self.s1 = SparseAtom(self.g)
        self.s2 = SparseAtom(self.g, 2)

    @raises(ValueError)
    def test_fail_align1(self):
        s = SparseAtom(self.g * 2)
        self.s1.align(s)

    def test_create1(self):
        self.s1[0, [1, 2, 3]] = 1
        assert_equal(self.s1.nnz, 3)
        assert_equal(self.s1[0, 1], 1)
        self.s1[2, [1, 2, 3]] = 1
        assert_equal(self.s1.nnz, 6)
        self.s1.empty(keep=True)
        assert_equal(self.s1.nnz, 6)
        self.s1.empty()
        assert_equal(self.s1.nnz, 0)
        self.s1[0, 0] = np.nan
        assert_equal(self.s1.nnz, 0)
        self.s1.empty()

    def test_create2(self):
        self.s1[0, [1, 2, 3], (1, 1, 1)] = 1
        assert_equal(self.s1[0, 2, (1, 1, 1)], 1)
        self.s1.empty()

    def test_create3(self):
        self.s2[0, [1, 2, 3], (1, 1, 1)] = [1, 2]
        assert_true(np.allclose(self.s2[0, 2, (1, 1, 1)], [1, 2]))
        assert_equal(self.s2[0, 2, 0, (1, 1, 1)], 1)
        assert_equal(self.s2[0, 2, 1, (1, 1, 1)], 2)
        self.s2.empty()

    def test_nonzero1(self):
        s2 = self.s2.copy()
        s2[0, [0, 2, 3]] = [1, 2]
        s2[3, [1, 2, 3]] = [1, 2]

        r, c = s2.nonzero()
        assert_true(np.allclose(r, [0, 0, 0, 3, 3, 3]))
        assert_true(np.allclose(c, [0, 2, 3, 1, 2, 3]))
        c = s2.nonzero(only_col=True)
        assert_true(np.allclose(c, [0, 2, 3, 1, 2, 3]))
        r, c = s2.nonzero(atom=1)
        assert_equal(len(r), 0)
        assert_equal(len(c), 0)
        r, c = s2.nonzero(atom=0)
        assert_true(np.allclose(r, [0, 0, 0]))
        assert_true(np.allclose(c, [0, 2, 3]))
        c = s2.nonzero(atom=0, only_col=True)
        assert_true(np.allclose(c, [0, 2, 3]))

    def test_cut1(self):
        s1 = SparseAtom(self.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(self.g * 2)
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.cut(2, 2).cut(2, 1).cut(2, 0)
        assert_true(s1.spsame(s2))

        s1 = SparseAtom(self.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(self.g * [2, 1, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.cut(2, 0)
        assert_true(s1.spsame(s2))

        s1 = SparseAtom(self.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(self.g * [1, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.cut(2, 1)
        assert_true(s1.spsame(s2))

    def test_remove1(self):
        for i in range(len(self.g)):
            self.s1.construct([[0.1, 1.5], [1, 2]])
            s1 = self.s1.remove(i)
            self.s1.empty()
            s2 = SparseAtom(self.g.remove(i))
            s2.construct([[0.1, 1.5], [1, 2]])
            assert_true(s1.spsame(s2))

    def test_sub1(self):
        all = range(len(self.g))
        for i in range(len(self.g)):
            self.s1.construct([[0.1, 1.5], [1, 2]])
            # my new sub
            sub = [j for j in all if i != j]
            s1 = self.s1.sub(sub)
            self.s1.empty()
            s2 = SparseAtom(self.g.sub(sub))
            s2.construct([[0.1, 1.5], [1, 2]])
            assert_true(s1.spsame(s2))

    def test_tile1(self):
        self.s1.construct([[0.1, 1.5], [1, 2]])
        self.s1.finalize()
        s1 = self.s1.tile(2, 0).tile(2, 1)
        s2 = SparseAtom(self.g * [2, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        assert_true(s1.spsame(s2))
        s1.finalize()
        s2.finalize()
        assert_true(np.allclose(s1._csr._D, s2._csr._D))
        s2 = s2.cut(2, 1).cut(2, 0)
        assert_true(self.s1.spsame(s2))
        s2.finalize()
        assert_true(np.allclose(self.s1._csr._D, s2._csr._D))
        s1 = s1.cut(2, 1).cut(2, 0)
        assert_true(self.s1.spsame(s1))
        s1.finalize()
        assert_true(np.allclose(s1._csr._D, self.s1._csr._D))
        self.s1.empty()

    def test_tile2(self):
        self.s1.construct([[0.1, 1.5], [1, 2]])
        self.s1.finalize()
        s1 = self.s1.tile(2, 0).tile(2, 1)
        s2 = SparseAtom(self.g * [2, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        assert_true(s1.spsame(s2))
        s1.finalize()
        s2.finalize()
        assert_true(np.allclose(s1._csr._D, s2._csr._D))
        s2 = s2.cut(2, 1).cut(2, 0)
        assert_true(self.s1.spsame(s2))
        s2.finalize()
        assert_true(np.allclose(self.s1._csr._D, s2._csr._D))
        s1 = s1.cut(2, 1).cut(2, 0)
        assert_true(self.s1.spsame(s1))
        s1.finalize()
        assert_true(np.allclose(s1._csr._D, self.s1._csr._D))
        self.s1.empty()

    def test_repeat1(self):
        self.s1.construct([[0.1, 1.5], [1, 2]])
        s1 = self.s1.repeat(2, 0).repeat(2, 1)
        self.s1.empty()
        s2 = SparseAtom(self.g * ([2, 2, 1], 'r'))
        s2.construct([[0.1, 1.5], [1, 2]])
        assert_true(s1.spsame(s2))
        s1.finalize()
        s2.finalize()
        assert_true(np.allclose(s1._csr._D, s2._csr._D))

    def test_repeat2(self):
        self.s1.construct([[0.1, 1.5], [1, 2]])
        self.s1.finalize()
        s1 = self.s1.repeat(2, 0).repeat(2, 1)
        self.s1.empty()
        s2 = SparseAtom(self.g * ([2, 2, 1], 'r'))
        s2.construct([[0.1, 1.5], [1, 2]])
        assert_true(s1.spsame(s2))
        s1.finalize()
        s2.finalize()
        assert_true(np.allclose(s1._csr._D, s2._csr._D))

    def test_set_nsc1(self):
        g = fcc(1., Atom(1, R=3.5))
        s = SparseAtom(g)
        s.construct([[0.1, 1.5, 3.5], [1, 2, 3]])
        s.finalize()
        assert_true(s.nnz > 1)
        s.set_nsc([1, 1, 1])
        assert_equal(s.nnz, 1)
        assert_equal(s[0, 0], 1)

    def test_set_nsc2(self):
        g = graphene(atom=Atom(6, R=1.43))
        s = SparseAtom(g)
        s.construct([[0.1, 1.43], [1, 2]])
        s.finalize()
        assert_equal(s.nnz, 8)
        s.set_nsc(a=1)
        assert_equal(s.nnz, 6)
        s.set_nsc([None, 1, 1])
        assert_equal(s.nnz, 4)
        assert_equal(s[0, 0], 1)

    def test_fromsp1(self):
        g = self.g.repeat(2, 0).tile(2, 1)
        csr = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr[0, [1, 2, 3]] = 1
        csr[1, [2, 4, 1]] = 2
        s1 = SparseAtom.fromsp(g, [csr])
        assert_equal(s1.nnz, 6)
        assert_true(np.allclose(s1.shape, [g.na, g.na_s, 1]))

        assert_true(np.allclose(s1[0, [1, 2, 3]], np.ones([3], np.int32)))
        assert_true(np.allclose(s1[1, [1, 2, 4]], np.ones([3], np.int32)*2))

        # Different instantiating
        s2 = SparseAtom.fromsp(g, csr)
        assert_true(s1.spsame(s2))

    def test_fromsp2(self):
        g = self.g.repeat(2, 0).tile(2, 1)
        csr1 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr2 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr1[0, [1, 2, 3]] = 1
        csr2[1, [2, 4, 1]] = 2
        s1 = SparseAtom.fromsp(g, [csr1, csr2])
        assert_equal(s1.nnz, 6)
        assert_true(np.allclose(s1.shape, [g.na, g.na_s, 2]))

        assert_true(np.allclose(s1[0, [1, 2, 3], 0], np.ones([3], np.int32)))
        assert_true(np.allclose(s1[0, [1, 2, 3], 1], np.zeros([3], np.int32)))
        assert_true(np.allclose(s1[1, [1, 2, 4], 0], np.zeros([3], np.int32)))
        assert_true(np.allclose(s1[1, [1, 2, 4], 1], np.ones([3], np.int32)*2))

        s2 = SparseAtom.fromsp(g, csr1, csr2)
        assert_equal(s2.nnz, 6)
        assert_true(np.allclose(s2.shape, [g.na, g.na_s, 2]))

        assert_true(np.allclose(s2[0, [1, 2, 3], 0], np.ones([3], np.int32)))
        assert_true(np.allclose(s2[0, [1, 2, 3], 1], np.zeros([3], np.int32)))
        assert_true(np.allclose(s2[1, [1, 2, 4], 0], np.zeros([3], np.int32)))
        assert_true(np.allclose(s2[1, [1, 2, 4], 1], np.ones([3], np.int32)*2))

        assert_true(s1.spsame(s2))

    @raises(ValueError)
    def test_fromsp3(self):
        g = self.g.repeat(2, 0).tile(2, 1)
        csr1 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr2 = sc.sparse.csr_matrix((g.na, g.na_s), dtype=np.int32)
        csr1[0, [1, 2, 3]] = 1
        csr2[1, [2, 4, 1]] = 2

        # Ensure that one does not mix everything.
        SparseAtom.fromsp(g, [csr1], csr2)
