from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np
import scipy as sc

from sisl.sparse import SparseCSR


@attr('sparse')
class TestSparseCSR(object):

    def setUp(self):
        self.s1 = SparseCSR((10, 100), dtype=np.int32)
        self.s2 = SparseCSR((10, 100, 2))

    def test_init1(self):
        assert_equal(self.s1.dtype, np.int32)
        assert_equal(self.s2.dtype, np.float64)
        assert_true(np.allclose(self.s1.data, self.s1.data))
        assert_true(np.allclose(self.s2.data, self.s2.data))

    def test_init2(self):
        SparseCSR((10, 100))
        for d in [np.int32, np.float64, np.complex128]:
            s = SparseCSR((10, 100), dtype=d)
            assert_equal(s.shape, (10, 100, 1))
            assert_equal(s.dim, 1)
            assert_equal(s.dtype, d)
            for k in [1, 2]:
                s = SparseCSR((10, 100, k), dtype=d)
                assert_equal(s.shape, (10, 100, k))
                assert_equal(s.dim, k)
                s = SparseCSR((10, 100), dim=k, dtype=d)
                assert_equal(s.shape, (10, 100, k))
                assert_equal(s.dim, k)
                s = SparseCSR((10, 100, 3), dim=k, dtype=d)
                assert_equal(s.shape, (10, 100, 3))
                assert_equal(s.dim, 3)

    def test_init3(self):
        csr = sc.sparse.csr_matrix((10, 10), dtype=np.int32)
        csr[0, 1] = 1
        csr[0, 2] = 2
        sp = SparseCSR(csr)
        assert_equal(sp.dtype, np.int32)
        assert_equal(sp.shape, (10, 10, 1))
        assert_equal(len(sp), 2)
        assert_equal(sp[0, 1], 1)
        assert_equal(sp[0, 2], 2)
        sp = SparseCSR(csr, dtype=np.float64)
        assert_equal(sp.shape, (10, 10, 1))
        assert_equal(sp.dtype, np.float64)
        assert_equal(len(sp), 2)
        assert_equal(sp[0, 1], 1)
        assert_equal(sp[0, 2], 2)

    def test_init4(self):
        csr = sc.sparse.csr_matrix((10, 10), dtype=np.int32)
        csr[0, 1] = 1
        csr[0, 2] = 2
        print(csr.indices, csr.indptr)
        sp = SparseCSR((csr.data, csr.indices, csr.indptr))
        assert_equal(sp.dtype, np.int32)
        assert_equal(sp.shape, (10, 10, 1))
        assert_equal(len(sp), 2)
        assert_equal(sp[0, 1], 1)
        assert_equal(sp[0, 2], 2)
        sp = SparseCSR((csr.data, csr.indices, csr.indptr), dtype=np.float64)
        assert_equal(sp.shape, (10, 10, 1))
        assert_equal(sp.dtype, np.float64)
        assert_equal(len(sp), 2)
        assert_equal(sp[0, 1], 1)
        assert_equal(sp[0, 2], 2)

    def test_create1(self):
        self.s1[0, [1, 2, 3]] = 1
        assert_equal(self.s1.nnz, 3)
        self.s1[2, [1, 2, 3]] = 1
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

    def test_create3(self):
        for i in range(10):
            j = range(i*4, i*4+3)
            self.s1[0, j] = i
            assert_equal(len(self.s1), (i+1)*3)
            self.s1[0, range((i+1)*4, (i+1)*4+3)] = None
            assert_equal(len(self.s1), (i+1)*3)
            for jj in j:
                assert_equal(self.s1[0, jj], i)
                assert_equal(self.s1[1, jj], 0)
        self.s1.empty()

    def test_finalize1(self):
        self.s1[0, [1, 2, 3]] = 1
        self.s1[2, [1, 2, 3]] = 1.
        assert_false(self.s1.finalized)
        self.s1.finalize()
        assert_true(self.s1.finalized)
        self.s1.empty(keep=True)
        assert_true(self.s1.finalized)
        self.s1.empty()
        assert_false(self.s1.finalized)

    def test_delitem1(self):
        self.s1[0, [1, 2, 3]] = 1
        assert_equal(len(self.s1), 3)
        del self.s1[0, 1]
        assert_equal(len(self.s1), 2)
        assert_equal(self.s1[0, 1], 0)
        assert_equal(self.s1[0, 2], 1)
        assert_equal(self.s1[0, 3], 1)
        self.s1[0, [1, 2, 3]] = 1
        del self.s1[0, [1, 3]]
        assert_equal(len(self.s1), 1)
        assert_equal(self.s1[0, 1], 0)
        assert_equal(self.s1[0, 2], 1)
        assert_equal(self.s1[0, 3], 0)
        self.s1.empty()

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

            # //
            self.s1 //= 2
            for jj in j:
                assert_equal(self.s1[0, jj], i)
                assert_equal(self.s1[1, jj], 0)

            # i**
            self.s1 **= 2
            for jj in j:
                assert_equal(self.s1[0, jj], i**2)
                assert_equal(self.s1[1, jj], 0)

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

            # - (r)
            s = 1 - self.s1
            for jj in j:
                assert_equal(s[0, jj], 1 - i)
                assert_equal(self.s1[0, jj], i)
                assert_equal(s[1, jj], 0)

            # *
            s = self.s1 * 2
            for jj in j:
                assert_equal(s[0, jj], i*2)
                assert_equal(self.s1[0, jj], i)
                assert_equal(s[1, jj], 0)

            # //
            s = s // 2
            for jj in j:
                assert_equal(s[0, jj], i)
                assert_equal(self.s1[0, jj], i)
                assert_equal(s[1, jj], 0)

            # **
            s = self.s1 ** 2
            for jj in j:
                assert_equal(s[0, jj], i**2)
                assert_equal(self.s1[0, jj], i)
                assert_equal(s[1, jj], 0)

            # ** (r)
            s = 2 ** self.s1
            for jj in j:
                assert_equal(s[0, jj], 2 ** self.s1[0, jj])
                assert_equal(self.s1[0, jj], i)
                assert_equal(s[1, jj], 0)

    def test_op3(self):
        S = SparseCSR((10, 100), dtype=np.int32)
        # Create initial stuff
        for i in range(10):
            j = range(i*4, i*4+3)
            S[0, j] = i

        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(S, '__{}__'.format(op))
            s = func(1)
            assert_equal(s.dtype, np.int32)
            s = func(1.)
            assert_equal(s.dtype, np.float64)
            if op != 'pow':
                s = func(1.j)
                assert_equal(s.dtype, np.complex128)

        S = S.copy(dtype=np.float64)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(S, '__{}__'.format(op))
            s = func(1)
            assert_equal(s.dtype, np.float64)
            s = func(1.)
            assert_equal(s.dtype, np.float64)
            if op != 'pow':
                s = func(1.j)
                assert_equal(s.dtype, np.complex128)

        S = S.copy(dtype=np.complex128)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(S, '__{}__'.format(op))
            s = func(1)
            assert_equal(s.dtype, np.complex128)
            s = func(1.)
            assert_equal(s.dtype, np.complex128)
            if op != 'pow':
                s = func(1.j)
                assert_equal(s.dtype, np.complex128)

    def test_op4(self):
        S = SparseCSR((10, 100), dtype=np.int32)
        # Create initial stuff
        for i in range(10):
            j = range(i*4, i*4+3)
            S[0, j] = i

        s = 1 + S
        assert_equal(s.dtype, np.int32)
        s = 1. + S
        assert_equal(s.dtype, np.float64)
        s = 1.j + S
        assert_equal(s.dtype, np.complex128)

        s = 1 - S
        assert_equal(s.dtype, np.int32)
        s = 1. - S
        assert_equal(s.dtype, np.float64)
        s = 1.j - S
        assert_equal(s.dtype, np.complex128)

        s = 1 * S
        assert_equal(s.dtype, np.int32)
        s = 1. * S
        assert_equal(s.dtype, np.float64)
        s = 1.j * S
        assert_equal(s.dtype, np.complex128)

        s = 1 ** S
        assert_equal(s.dtype, np.int32)
        s = 1. ** S
        assert_equal(s.dtype, np.float64)
        s = 1.j ** S
        assert_equal(s.dtype, np.complex128)
