from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl.sparse import SparseCSR

class TestSparseCSR(object):

    def setUp(self):
        self.s1 = SparseCSR((10,100), dtype=np.int32)
        self.s2 = SparseCSR((10,100,2))

    def test_init1(self):
        assert_equal(self.s1.dtype, np.int32)
        assert_equal(self.s2.dtype, np.float64)

    def test_init2(self):
        SparseCSR((10,100))
        for d in [np.int32, np.float64, np.complex128]:
            s = SparseCSR((10,100), dtype=d)
            assert_equal(s.shape, (10, 100, 1))
            assert_equal(s.dim, 1)
            assert_equal(s.dtype, d)
            for k in [1, 2]:
                s = SparseCSR((10,100,k), dtype=d)
                assert_equal(s.shape, (10, 100, k))
                assert_equal(s.dim, k)
                s = SparseCSR((10,100), dim=k, dtype=d)
                assert_equal(s.shape, (10, 100, k))
                assert_equal(s.dim, k)
                s = SparseCSR((10,100, 3), dim=k, dtype=d)
                assert_equal(s.shape, (10, 100, 3))
                assert_equal(s.dim, 3)

    def test_create1(self):
        self.s1[0,[1,2,3]] = 1.
        assert_equal(self.s1.nnz, 3)
        self.s1[2,[1,2,3]] = 1.
        assert_equal(self.s1.nnz, 6)
        self.s1.empty(keep=True)
        assert_equal(self.s1.nnz, 6)
        self.s1.empty()
        assert_equal(self.s1.nnz, 0)
        
    def test_create2(self):
        for i in range(10):
            j = range(i*4, i*4+3)
            self.s1[0, j] = i
            assert_equal(len(self.s1), (i+1)*3)
            for jj in j:
                assert_equal(self.s1[0, jj], i)
                assert_equal(self.s1[1, jj], 0)
        self.s1.empty()

    def test_finalize1(self):
        self.s1[0,[1,2,3]] = 1.
        self.s1[2,[1,2,3]] = 1.
        assert_false(self.s1.finalized)
        self.s1.finalize()
        assert_true(self.s1.finalized)
        self.s1.empty(keep=True)
        assert_true(self.s1.finalized)
        self.s1.empty()
        assert_false(self.s1.finalized)

    def test_op1(self):
        for i in range(10):
            j = range(i*4, i*4+3)
            self.s1[0, j] = i

            # i+
            self.s1 += 1
            for jj in j:
                assert_equal(self.s1[0, jj], i+1)
                assert_equal(self.s1[1, jj], 0)

            # i-
            self.s1 -= 1
            for jj in j:
                assert_equal(self.s1[0, jj], i)
                assert_equal(self.s1[1, jj], 0)

            # i*
            self.s1 *= 2
            for jj in j:
                assert_equal(self.s1[0, jj], i*2)
                assert_equal(self.s1[1, jj], 0)

            # /
            # This is truedivision which
            # does not allow the returned value
            # to be integer, 
            #self.s1 /= 2
            #for jj in j:
            #    assert_equal(self.s1[0, jj], i)
            #    assert_equal(self.s1[1, jj], 0)

            # i**
            self.s1 **= 2
            for jj in j:
                assert_equal(self.s1[0, jj], 4*i**2)
                assert_equal(self.s1[1, jj], 0)


    @attr('only')
    def test_op2(self):
        for i in range(10):
            j = range(i*4, i*4+3)
            self.s1[0, j] = i

            # +
            s = self.s1 + 1
            for jj in j:
                assert_equal(s[0, jj], i+1)
                assert_equal(self.s1[0, jj], i)
                assert_equal(s[1, jj], 0)

            # -
            s = self.s1 - 1
            for jj in j:
                assert_equal(s[0, jj], i-1)
                assert_equal(self.s1[0, jj], i)
                assert_equal(s[1, jj], 0)

            # *
            s = self.s1 * 2
            for jj in j:
                assert_equal(s[0, jj], i*2)
                assert_equal(self.s1[0, jj], i)
                assert_equal(s[1, jj], 0)

            # /
            # This is truedivision which
            # does not allow the returned value
            # to be integer, 
            #s = self.s1 / 2
            #for jj in j:
            #    assert_equal(s[0, jj], i*2)
            #    assert_equal(self.s1[0, jj], i)
            #    assert_equal(s[1, jj], 0)

            # **
            s = self.s1 ** 2
            for jj in j:
                assert_equal(s[0, jj], i**2)
                assert_equal(self.s1[0, jj], i)
                assert_equal(s[1, jj], 0)
