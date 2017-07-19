from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np
import scipy as sc

from sisl import Atom
from sisl.geom import fcc
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
        s1 = self.s1.tile(2, 0).tile(2, 1)
        s2 = SparseAtom(self.g * [2, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        assert_true(s1.spsame(s2))
        s2 = s2.cut(2, 1).cut(2, 0)
        assert_true(self.s1.spsame(s2))
        s1 = s1.cut(2, 1).cut(2, 0)
        assert_true(self.s1.spsame(s1))
        self.s1.empty()

    def test_repeat1(self):
        self.s1.construct([[0.1, 1.5], [1, 2]])
        s1 = self.s1.repeat(2, 0).repeat(2, 1)
        self.s1.empty()
        s2 = SparseAtom(self.g * ([2, 2, 1], 'r'))
        s2.construct([[0.1, 1.5], [1, 2]])
        assert_true(s1.spsame(s2))
