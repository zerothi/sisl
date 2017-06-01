from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np
import scipy as sc

from sisl.sparse import *


@attr('sparse')
class TestSparseCSR(object):

    def setUp(self):
        self.s1 = SparseCSR((10, 100), dtype=np.int32)
        self.s1d = SparseCSR((10, 100))
        self.s2 = SparseCSR((10, 100, 2))

    @raises(ValueError)
    def test_fail_init1(self):
        s = SparseCSR((10, 100, 20, 20), dtype=np.int32)

    @raises(ValueError)
    def test_fail_align1(self):
        s1 = SparseCSR((10, 100), dtype=np.int32)
        s2 = SparseCSR((20, 100), dtype=np.int32)
        s1.spalign(s2)

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

    def test_extend1(self):
        csr = SparseCSR((10, 10), nnzpr=1, dtype=np.int32)
        csr[1, 1] = 3
        csr[2, 2] = 4
        csr[0, 1] = 1
        csr[0, 2] = 2
        assert_equal(csr[0, 1], 1)
        assert_equal(csr[0, 2], 2)
        assert_equal(csr[1, 1], 3)
        assert_equal(csr[2, 2], 4)

    def test_create1(self):
        self.s1d[0, [1, 2, 3]] = 1
        assert_equal(self.s1d.nnz, 3)
        self.s1d[2, [1, 2, 3]] = 1
        assert_equal(self.s1d.nnz, 6)
        self.s1d.empty(keep=True)
        assert_equal(self.s1d.nnz, 6)
        self.s1d.empty()
        assert_equal(self.s1d.nnz, 0)
        self.s1d[0, 0] = np.nan
        assert_equal(self.s1d.nnz, 0)
        self.s1d.empty()

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

    def test_iterator1(self):
        self.s1[0, [1, 2, 3]] = 1
        self.s1[2, [1, 2, 4]] = 1.
        e = [[1, 2, 3], [], [1, 2, 4]]
        for i, j in self.s1:
            assert_true(j in e[i])

        for i, j in self.s1.iter_nnz(0):
            assert_equal(i, 0)
            assert_true(j in e[0])
        for i, j in self.s1.iter_nnz(2):
            assert_equal(i, 2)
            assert_true(j in e[2])

        for i, j in ispmatrix(self.s1):
            assert_true(j in e[i])

        for i, j in ispmatrix(self.s1, map_col = lambda x: x):
            assert_true(j in e[i])
        for i, j in ispmatrix(self.s1, map_row = lambda x: x):
            assert_true(j in e[i])

        for i, j, d in ispmatrixd(self.s1):
            assert_true(j in e[i])
            assert_true(d == 1.)

        self.s1.empty()

    def test_iterator2(self):
        e = [[1, 2, 3], [], [1, 2, 4]]
        self.s1[0, [1, 2, 3]] = 1
        self.s1[2, [1, 2, 4]] = 1.
        a = self.s1.tocsr()
        for func in ['csr', 'csc', 'coo', 'lil']:
            a = getattr(a, 'to' + func)()
            for r, c in ispmatrix(a):
                assert_true(r in [0, 2])
                assert_true(c in e[r])

        for func in ['csr', 'csc', 'coo', 'lil']:
            a = getattr(a, 'to' + func)()
            for r, c, d in ispmatrixd(a):
                assert_true(r in [0, 2])
                assert_true(c in e[r])
                assert_true(d == 1.)

        self.s1.empty()

    def test_iterator3(self):
        e = [[1, 2, 3], [], [1, 2, 4]]
        self.s1[0, [1, 2, 3]] = 1
        self.s1[2, [1, 2, 4]] = 1.
        a = self.s1.tocsr()
        for func in ['csr', 'csc', 'coo', 'lil']:
            a = getattr(a, 'to' + func)()
            for r, c in ispmatrix(a):
                assert_true(r in [0, 2])
                assert_true(c in e[r])

        # number of mapped values
        nvals = 2
        for func in ['csr', 'lil']:
            a = getattr(a, 'to' + func)()
            n = 0
            for r, c in ispmatrix(a, lambda x: x%2, lambda x: x%2):
                assert_true(r == 0)
                assert_true(c in [0, 1])
                n += 1
            assert_true(n == nvals)

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
        del self.s1[range(2), 0]
        for i in range(2):
            assert_equal(self.s1[i, 0], 0)
        self.s1.empty()

    def test_eliminate_zeros1(self):
        self.s1[0, [1, 2, 3]] = 1
        self.s1[1, [1, 2, 3]] = 0
        assert_equal(len(self.s1), 6)
        self.s1.eliminate_zeros()
        assert_equal(len(self.s1), 3)
        assert_equal(self.s1[1, 1], 0)
        assert_equal(self.s1[1, 2], 0)
        assert_equal(self.s1[1, 3], 0)
        self.s1.empty()

    def test_same1(self):
        self.s1[0, [1, 2, 3]] = 1
        self.s2[0, [1, 2, 3]] = (1, 1)
        assert_true(self.s1.spsame(self.s2))
        self.s2[1, 1] = (1, 1)
        assert_false(self.s1.spsame(self.s2))
        self.s1.spalign(self.s2)
        assert_true(self.s1.spsame(self.s2))

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

    def test_op5(self):
        S1 = SparseCSR((10, 100), dtype=np.int32)
        S2 = SparseCSR((10, 100), dtype=np.int32)
        S3 = SparseCSR((10, 100), dtype=np.int32)
        # Create initial stuff
        for i in range(10):
            j = range(i*4, i*4+3)
            S1[0, j] = i
            S2[0, j] = i

            if i < 5:
                S3[0, j] = i

        S = S1 * S2
        assert_true(np.allclose(S._D, (S2**2)._D))

        S = S * S
        assert_true(np.allclose(S._D, (S2**4)._D))

        S = S1 + S2
        assert_true(np.allclose(S._D, S1._D + S2._D))

        S = S1 - S2
        assert_true(np.allclose(S._D, S1._D - S2._D))
