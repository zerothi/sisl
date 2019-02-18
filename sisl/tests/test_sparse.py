from __future__ import print_function, division

import pytest

import math as m
import numpy as np
import scipy as sc

from sisl.utils.ranges import array_arange
from sisl.sparse import *
from sisl.sparse import indices


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            self.s1 = SparseCSR((10, 100), dtype=np.int32)
            self.s1d = SparseCSR((10, 100))
            self.s2 = SparseCSR((10, 100, 2))
    return t()


@pytest.mark.sparse
class TestSparseCSR(object):

    def test_indices(self):
        search = np.array([1, 4, 5, 0], np.int32)
        val = np.array([0, 5], np.int32)
        idx = indices(search, val, 0)
        assert np.allclose(idx, [3, 2])
        val = np.array([0, 1, 5], np.int32)
        idx = indices(search, val, 0)
        assert np.allclose(idx, [3, 0, 2])
        idx = indices(search, val, 20)
        assert np.allclose(idx, [23, 20, 22])
        val = np.array([-1, 0, 1, 5, 10], np.int32)
        idx = indices(search, val, 20)
        assert np.allclose(idx, [-1, 23, 20, 22, -1])

    @pytest.mark.xfail(raises=ValueError)
    def test_fail_init1(self):
        SparseCSR((10, 100, 20, 20), dtype=np.int32)

    @pytest.mark.xfail(raises=ValueError)
    def test_fail_init_shape0(self):
        SparseCSR((0, 10, 10), dtype=np.int32)

    @pytest.mark.xfail(raises=ValueError)
    def test_fail_init2(self):
        data = np.empty([2, 2], np.float64)
        indices = np.arange(2)
        indptr = np.arange(2)
        s = SparseCSR((data, indices, indptr, indptr), shape=(100, 20, 20))

    @pytest.mark.xfail(raises=ValueError)
    def test_fail_align1(self):
        s1 = SparseCSR((10, 100), dtype=np.int32)
        s2 = SparseCSR((20, 100), dtype=np.int32)
        s1.align(s2)

    def test_set_get1(self):
        s1 = SparseCSR((10, 10, 3), dtype=np.int32)
        s1[1, 1] = [1, 2, 3]
        s1[1, 2] = [1, 2, 3]
        assert s1.nnz == 2
        assert np.all(s1[1, 1] == [1, 2, 3])
        assert np.all(s1[1, [1, 2]] == [[1, 2, 3], [1, 2, 3]])
        assert np.all(s1[1, [1, 2], 1] == [2, 2])
        assert np.all(s1[1, [1, 6], 1] == [2, 0])

    def test_init1(self, setup):
        str(setup.s1)
        assert setup.s1.dtype == np.int32
        assert setup.s2.dtype == np.float64
        assert np.allclose(setup.s1.data, setup.s1.data)
        assert np.allclose(setup.s2.data, setup.s2.data)

    def test_init2(self, setup):
        SparseCSR((10, 100))
        for d in [np.int32, np.float64, np.complex128]:
            s = SparseCSR((10, 100), dtype=d)
            assert s.shape, (10, 100 == 1)
            assert s.dim == 1
            assert s.dtype == d
            for k in [1, 2]:
                s = SparseCSR((10, 100, k), dtype=d)
                assert s.shape, (10, 100 == k)
                assert s.dim == k
                s = SparseCSR((10, 100), dim=k, dtype=d)
                assert s.shape, (10, 100 == k)
                assert s.dim == k
                s = SparseCSR((10, 100, 3), dim=k, dtype=d)
                assert s.shape, (10, 100 == 3)
                assert s.dim == 3

    def test_init3(self, setup):
        lil = sc.sparse.lil_matrix((10, 10), dtype=np.int32)
        lil[0, 1] = 1
        lil[0, 2] = 2
        sp = SparseCSR(lil)
        assert sp.dtype == np.int32
        assert sp.shape, (10, 10 == 1)
        assert sp.nnz == 2
        assert sp[0, 1] == 1
        assert sp[0, 2] == 2
        sp = SparseCSR(lil, dtype=np.float64)
        assert sp.shape, (10, 10 == 1)
        assert sp.dtype == np.float64
        assert sp.nnz == 2
        assert sp[0, 1] == 1
        assert sp[0, 2] == 2

    def test_init4(self, setup):
        lil = sc.sparse.lil_matrix((10, 10), dtype=np.int32)
        lil[0, 1] = 1
        lil[0, 2] = 2
        csr = lil.tocsr()
        sp = SparseCSR((csr.data, csr.indices, csr.indptr))
        assert sp.dtype == np.int32
        assert sp.shape, (10, 10 == 1)
        assert sp.nnz == 2
        assert sp[0, 1] == 1
        assert sp[0, 2] == 2
        sp = SparseCSR((csr.data, csr.indices, csr.indptr), dtype=np.float64)
        assert sp.shape, (10, 10 == 1)
        assert sp.dtype == np.float64
        assert sp.nnz == 2
        assert sp[0, 1] == 1
        assert sp[0, 2] == 2

    def test_extend1(self, setup):
        csr = SparseCSR((10, 10), nnzpr=1, dtype=np.int32)
        csr[1, 1] = 3
        csr[2, 2] = 4
        csr[0, 1] = 1
        csr[0, 2] = 2
        csr[0, []] = 2
        assert csr[0, 1] == 1
        assert csr[0, 2] == 2
        assert csr[1, 1] == 3
        assert csr[2, 2] == 4
        assert csr.nnz == 4
        assert np.allclose(csr[0, [0, 1, 2]], [0, 1, 2])

    def test_diag1(self, setup):
        csr = SparseCSR((10, 10), nnzpr=1, dtype=np.int32)
        csr1 = csr.diags(10)
        assert csr1.shape[0] == 10
        assert csr1.shape[1] == 10
        assert csr1.shape[2] == 1
        assert csr1.nnz == 10
        csr2 = csr.diags(10)
        csr3 = csr.diags(np.zeros(10, np.int32))
        assert csr2.spsame(csr3)

    def test_diag2(self, setup):
        csr = SparseCSR((10, 10), dim=3, nnzpr=1, dtype=np.int32)
        csr1 = csr.diags(10, dim=1)
        csr2 = csr.diags(10, dim=2)
        assert csr1.shape[2] == 1
        assert csr2.shape[2] == 2
        assert csr1.nnz == 10
        assert csr2.nnz == 10
        assert csr1.spsame(csr2)

    def test_diag3(self, setup):
        csr = SparseCSR((10, 10), dim=3, nnzpr=1, dtype=np.int32)
        csr1 = csr.diags(10, dim=1, dtype=np.int16)
        csr2 = csr.diags(10, dim=2, dtype=np.float64)
        assert csr1.shape[2] == 1
        assert csr2.shape[2] == 2
        assert csr1.nnz == 10
        assert csr2.nnz == 10
        assert csr1.dtype == np.int16
        assert csr2.dtype == np.float64

    def test_create1(self, setup):
        setup.s1d[0, [1, 2, 3]] = 1
        assert setup.s1d.nnz == 3
        setup.s1d[2, [1, 2, 3]] = 1
        assert setup.s1d.nnz == 6
        setup.s1d.empty(keep_nnz=True)
        assert setup.s1d.nnz == 6
        setup.s1d.empty()
        assert setup.s1d.nnz == 0
        setup.s1d[0, 0] = np.nan
        # nan produces zeros
        assert setup.s1d.nnz == 1
        setup.s1d.empty()

    def test_create2(self, setup):
        assert len(setup.s1) == setup.s1.shape[0]
        for i in range(10):
            j = range(i*4, i*4+3)
            setup.s1[0, j] = i
            assert setup.s1.nnz == (i+1)*3
            for jj in j:
                assert setup.s1[0, jj] == i
                assert setup.s1[1, jj] == 0
        setup.s1.empty()

    def test_create3(self, setup):
        for i in range(10):
            j = range(i*4, i*4+3)
            setup.s1[0, j] = i
            assert setup.s1.nnz == (i+1)*3
            setup.s1[0, range((i+1)*4, (i+1)*4+3)] = None
            assert setup.s1.nnz == (i+1)*3
            for jj in j:
                assert setup.s1[0, jj] == i
                assert setup.s1[1, jj] == 0
        setup.s1.empty()

    def test_finalize1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        setup.s1[2, [1, 2, 3]] = 1.
        setup.s1[1, [3, 2, 1]] = 1.
        assert not setup.s1.finalized
        p = setup.s1.ptr.view()
        n = setup.s1.ncol.view()
        # Assert that the ordering is good
        assert np.allclose(setup.s1.col[p[1]:p[1]+n[1]], [3, 2, 1])
        setup.s1.finalize()
        # This also asserts that we do not change the memory-locations
        # of the pointers and ncol
        assert np.allclose(setup.s1.col[p[1]:p[1]+n[1]], [1, 2, 3])
        assert setup.s1.finalized
        setup.s1.empty(keep_nnz=True)
        assert setup.s1.finalized
        setup.s1.empty()
        assert not setup.s1.finalized

    def test_finalize2(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        setup.s1[2, [1, 2, 3]] = 1.
        setup.s1[1, [3, 2, 1]] = 1.
        assert not setup.s1.finalized
        p = setup.s1.ptr.view()
        n = setup.s1.ncol.view()
        # Assert that the ordering is good
        assert np.allclose(setup.s1.col[p[1]:p[1]+n[1]], [3, 2, 1])
        setup.s1.finalize(False)
        # This also asserts that we do not change the memory-locations
        # of the pointers and ncol
        assert np.allclose(setup.s1.col[p[1]:p[1]+n[1]], [3, 2, 1])
        assert not setup.s1.finalized
        assert len(setup.s1.col) == 9

    def test_iterator1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        setup.s1[2, [1, 2, 4]] = 1.
        e = [[1, 2, 3], [], [1, 2, 4]]
        for i, j in setup.s1:
            assert j in e[i]

        for i, j in setup.s1.iter_nnz(0):
            assert i == 0
            assert j in e[0]
        for i, j in setup.s1.iter_nnz(2):
            assert i == 2
            assert j in e[2]

        for i, j in ispmatrix(setup.s1):
            assert j in e[i]

        for i, j in ispmatrix(setup.s1, map_col = lambda x: x):
            assert j in e[i]
        for i, j in ispmatrix(setup.s1, map_row = lambda x: x):
            assert j in e[i]

        for i, j, d in ispmatrixd(setup.s1):
            assert j in e[i]
            assert d == 1.

        setup.s1.empty()

    def test_iterator2(self, setup):
        e = [[1, 2, 3], [], [1, 2, 4]]
        setup.s1[0, [1, 2, 3]] = 1
        setup.s1[2, [1, 2, 4]] = 1.
        a = setup.s1.tocsr()
        for func in ['csr', 'csc', 'coo', 'lil']:
            a = getattr(a, 'to' + func)()
            for r, c in ispmatrix(a):
                assert r in [0, 2]
                assert c in e[r]

        for func in ['csr', 'csc', 'coo', 'lil']:
            a = getattr(a, 'to' + func)()
            for r, c, d in ispmatrixd(a):
                assert r in [0, 2]
                assert c in e[r]
                assert d == 1.

        setup.s1.empty()

    def test_iterator3(self, setup):
        e = [[1, 2, 3], [], [1, 2, 4]]
        setup.s1[0, [1, 2, 3]] = 1
        setup.s1[2, [1, 2, 4]] = 1.
        a = setup.s1.tocsr()
        for func in ['csr', 'csc', 'coo', 'lil']:
            a = getattr(a, 'to' + func)()
            for r, c in ispmatrix(a):
                assert r in [0, 2]
                assert c in e[r]

        # number of mapped values
        nvals = 2
        for func in ['csr', 'lil']:
            a = getattr(a, 'to' + func)()
            n = 0
            for r, c in ispmatrix(a, lambda x: x%2, lambda x: x%2):
                assert r == 0
                assert c in [0, 1]
                n += 1
            assert n == nvals

    def test_delitem1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        assert setup.s1.nnz == 3
        del setup.s1[0, 1]
        assert setup.s1.nnz == 2
        assert setup.s1[0, 1] == 0
        assert setup.s1[0, 2] == 1
        assert setup.s1[0, 3] == 1
        setup.s1[0, [1, 2, 3]] = 1
        del setup.s1[0, [1, 3]]
        assert setup.s1.nnz == 1
        assert setup.s1[0, 1] == 0
        assert setup.s1[0, 2] == 1
        assert setup.s1[0, 3] == 0
        del setup.s1[range(2), 0]
        assert setup.s1.nnz == 1
        for i in range(2):
            assert setup.s1[i, 0] == 0
        del setup.s1[range(2), range(3), 0]
        assert setup.s1.nnz == 0
        setup.s1.empty()

    def test_contains1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        assert setup.s1.nnz == 3
        assert [0, 1] in setup.s1
        assert [0, [1, 3]] in setup.s1
        setup.s1.empty()

    def test_sub1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        assert setup.s1.nnz == 3
        s1 = setup.s1.sub([0, 1])
        assert s1.nnz == 1
        assert s1.shape[0] == 2
        assert s1.shape[1] == 2
        setup.s1.empty()

    def test_remove1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        assert setup.s1.nnz == 3
        s1 = setup.s1.remove([1])
        assert s1.nnz == 2
        assert s1.shape[0] == setup.s1.shape[0] - 1
        assert s1.shape[1] == setup.s1.shape[1] - 1
        setup.s1.empty()

    def test_eliminate_zeros1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        setup.s1[1, [1, 2, 3]] = 0
        assert setup.s1.nnz == 6
        setup.s1.eliminate_zeros()
        assert setup.s1.nnz == 3
        assert setup.s1[1, 1] == 0
        assert setup.s1[1, 2] == 0
        assert setup.s1[1, 3] == 0
        setup.s1.empty()

    def test_eliminate_zeros_tolerance(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        setup.s1[1, [1, 2, 3]] = 2
        assert setup.s1.nnz == 6
        setup.s1.eliminate_zeros()
        assert setup.s1.nnz == 6
        setup.s1.eliminate_zeros(1)
        assert setup.s1.nnz == 3
        assert setup.s1[0, 1] == 0
        assert setup.s1[0, 2] == 0
        assert setup.s1[0, 3] == 0
        setup.s1.empty()

    def test_eliminate_zeros_tolerance_ndim(self, setup):
        s = SparseCSR((3, 3, 3))
        s[1, [0, 1, 2]] = 0.1
        s[1, [0, 1, 2], 1] = 0.05
        assert s.nnz == 3
        s.eliminate_zeros()
        assert s.nnz == 3
        s.eliminate_zeros(0.075)
        assert s.nnz == 3
        assert s[1, 1, 0] == 0.1
        assert s[1, 1, 1] == 0.05
        assert s[1, 1, 2] == 0.1
        s.eliminate_zeros(0.2)
        assert s.nnz == 0

    def test_same1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        setup.s2[0, [1, 2, 3]] = (1, 1)
        assert setup.s1.spsame(setup.s2)
        setup.s2[1, 1] = (1, 1)
        assert not setup.s1.spsame(setup.s2)
        setup.s1.align(setup.s2)
        assert setup.s1.spsame(setup.s2)

    def test_delete_col1(self, setup):
        s1 = setup.s1.copy()
        nc = s1.shape[1]
        s1[0, [1, 3]] = 1
        s1[1, [1, 2, 3]] = 1
        assert s1.nnz == 5
        s1.delete_columns(2)
        assert s1.nnz == 4
        assert s1.shape[1] == nc - 1
        assert np.all(s1.col[array_arange(s1.ptr, n=s1.ncol)] < 3)
        s1.delete_columns(2, True)
        assert s1.nnz == 2
        assert s1.shape[1] == nc - 1
        assert np.all(s1.col[array_arange(s1.ptr, n=s1.ncol)] < 2)
        # Delete a non-existing column
        s1.delete_columns(100000, True)
        assert s1.nnz == 2
        assert s1.shape[1] == nc - 1

    def test_delete_col2(self, setup):
        s1 = setup.s1.copy()
        nc = s1.shape[1]
        s1[1, [1, 2, 3]] = 1
        assert s1.nnz == 3
        s1.delete_columns(3)
        assert s1.nnz == 2
        assert s1.shape[1] == nc - 1
        s1.delete_columns(2, True)
        assert s1.nnz == 1
        assert s1.shape[1] == nc - 1

    def test_delete_col3(self, setup):
        s1 = setup.s1.copy()
        s2 = setup.s1.copy()
        nc = s1.shape[1]
        for i in range(10):
            s1[i, [3, 2, 1]] = 1
            s2[i, 2] = 1
        s1.finalize()
        s2.finalize()
        assert s1.nnz == 10*3
        s1.delete_columns([3, 1], keep_shape=True)
        assert s1.ptr[-1] == s1.nnz
        assert s2.ptr[-1] == s2.nnz
        assert s1.nnz == 10 * 1
        assert s1.spsame(s2)

    def test_delete_col4(self, setup):
        s1 = SparseCSR((10, 100), dtype=np.int32)
        s2 = SparseCSR((10, 98), dtype=np.int32)
        nc = s1.shape[1]
        for i in range(10):
            s1[i, [3, 2, 1]] = 1
            s2[i, 1] = 1
        s1.finalize()
        s2.finalize()
        assert s1.nnz == 10*3
        s1.delete_columns([3, 1])
        assert s1.ptr[-1] == s1.nnz
        assert s2.ptr[-1] == s2.nnz
        assert s1.nnz == 10 * 1
        assert s1.spsame(s2)

    def test_delete_col5(self, setup):
        s1 = setup.s1.copy()
        nc = s1.shape[1]
        s1[1, [1, 2, 3]] = 1
        assert s1.nnz == 3
        s1.delete_columns(2)
        assert s1.nnz == 2
        assert s1.shape[1] == nc - 1
        assert np.all(s1.col[array_arange(s1.ptr, n=s1.ncol)] < 3)
        s1.delete_columns(2, True)
        assert s1.nnz == 1
        assert s1.shape[1] == nc - 1
        assert np.all(s1.col[array_arange(s1.ptr, n=s1.ncol)] < 2)
        # Delete a non-existing column
        s1.delete_columns(100000, True)
        assert s1.nnz == 1
        assert s1.shape[1] == nc - 1

    def test_delete_col6(self, setup):
        s1 = setup.s1.copy()
        nc = s1.shape[1]
        for i in range(3):
            s1[i, [1, 2, 3]] = 1
        assert s1.nnz == 9
        s1.delete_columns(2)
        assert s1.nnz == 6
        assert s1.shape[1] == nc - 1

    def test_translate_col1(self, setup):
        s1 = setup.s1.copy()
        s1[1, 1] = 1
        s1[1, 2] = 2
        s1[1, 3] = 3
        assert s1.nnz == 3
        s1.translate_columns([1, 3], [3, 1])
        assert s1.nnz == 3
        assert s1[1, 1] == 3
        assert s1[1, 3] == 1

    def test_translate_col2(self, setup):
        s1 = setup.s1.copy()
        s1[1, 1] = 1
        s1[1, 2] = 2
        s1[1, 3] = 3
        assert s1.nnz == 3
        s1.translate_columns([1, 3], [3, s1.shape[1] + 100])
        assert s1.nnz == 2
        assert s1[1, 3] == 1
        assert s1[1, 1] == 0

    def test_translate_col3(self, setup):
        s1 = setup.s1.copy()
        for i in range(3):
            s1[i, 1] = 1
            s1[i, 2] = 2
            s1[i, 3] = 3
        assert s1.nnz == 9
        s1.translate_columns([1, 3], [3, 1])
        assert s1.nnz == 9
        for i in range(3):
            assert s1[i, 1] == 3
            assert s1[i, 3] == 1
        s1.translate_columns([1, 3], [3, 1])
        assert s1.nnz == 9
        for i in range(3):
            assert s1[i, 1] == 1
            assert s1[i, 3] == 3

    def test_translate_col4(self, setup):
        s1 = setup.s1.copy()
        nc = s1.shape[1]
        for i in range(3):
            s1[i, 1] = 1
            s1[i, 2] = 2
            s1[i, 3] = 3
        assert s1.nnz == 9
        s1.translate_columns([1, 3], [nc, 1])
        assert s1.nnz == 6
        assert s1.shape[1] == nc
        for i in range(3):
            assert s1[i, 1] == 3
            assert s1[i, 2] == 2
            assert s1[i, 3] == 0
        s1.translate_columns([1, 3], [3, 1])
        assert s1.nnz == 6
        for i in range(3):
            assert s1[i, 1] == 0
            assert s1[i, 3] == 3

    def test_edges1(self, setup):
        s1 = setup.s1.copy()
        s1[1, 1] = 1
        s1[1, 2] = 2
        s1[1, 3] = 3
        assert np.all(s1.edges(1, exclude=[]) == [1, 2, 3])
        assert np.all(s1.edges(1) == [2, 3])
        assert np.all(s1.edges(1, exclude=2) == [1, 3])
        assert len(s1.edges(2)) == 0

    def test_nonzero1(self, setup):
        s1 = setup.s1.copy()
        s1[2, 1] = 1
        s1[1, 1] = 1
        s1[1, 2] = 2
        s1[1, 3] = 3
        assert s1.nnz == 4
        r, c = s1.nonzero()
        assert np.all(r == [1, 1, 1, 2])
        assert np.all(c == [1, 2, 3, 1])
        c = s1.nonzero(only_col=True)
        assert np.all(c == [1, 2, 3, 1])
        c = s1.nonzero(row=1, only_col=True)
        assert np.all(c == [1, 2, 3])
        c = s1.nonzero(row=2, only_col=True)
        assert np.all(c == [1])
        c = s1.nonzero(row=[0, 1], only_col=True)
        assert np.all(c == [1, 2, 3])
        r, c = s1.nonzero(row=[0, 1])
        assert np.all(r == [1, 1, 1])
        assert np.all(c == [1, 2, 3])

    def test_op1(self, setup):
        for i in range(10):
            j = range(i*4, i*4+3)
            setup.s1[0, j] = i

            # i+
            setup.s1 += 1
            for jj in j:
                assert setup.s1[0, jj] == i+1
                assert setup.s1[1, jj] == 0

            # i-
            setup.s1 -= 1
            for jj in j:
                assert setup.s1[0, jj] == i
                assert setup.s1[1, jj] == 0

            # i*
            setup.s1 *= 2
            for jj in j:
                assert setup.s1[0, jj] == i*2
                assert setup.s1[1, jj] == 0

            # //
            setup.s1 //= 2
            for jj in j:
                assert setup.s1[0, jj] == i
                assert setup.s1[1, jj] == 0

            # i**
            setup.s1 **= 2
            for jj in j:
                assert setup.s1[0, jj] == i**2
                assert setup.s1[1, jj] == 0

    def test_op2(self, setup):
        for i in range(10):
            j = range(i*4, i*4+3)
            setup.s1[0, j] = i

            # +
            s = setup.s1 + 1
            for jj in j:
                assert s[0, jj] == i+1
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0

            # -
            s = setup.s1 - 1
            for jj in j:
                assert s[0, jj] == i-1
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0

            # - (r)
            s = 1 - setup.s1
            for jj in j:
                assert s[0, jj] == 1 - i
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0

            # *
            s = setup.s1 * 2
            for jj in j:
                assert s[0, jj] == i*2
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0

            # //
            s = s // 2
            for jj in j:
                assert s[0, jj] == i
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0

            # **
            s = setup.s1 ** 2
            for jj in j:
                assert s[0, jj] == i**2
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0

            # ** (r)
            s = 2 ** setup.s1
            for jj in j:
                assert s[0, jj], 2 ** setup.s1[0 == jj]
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0

    def test_op_csr(self, setup):
        csr = sc.sparse.csr_matrix((10, 100), dtype=np.int32)
        for i in range(10):
            j = range(i + 2)
            setup.s1[0, j] = i

            csr[0, 0] = 1

            # +
            s = setup.s1 + csr
            for jj in j:
                if jj == 0: continue
                assert s[0, jj] == i
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0
            assert s[0, 0] == i + 1

            # -
            s = setup.s1 - csr
            for jj in j:
                if jj == 0: continue
                assert s[0, jj] == i
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0
            assert s[0, 0] == i - 1

            # - (r)
            s = csr - setup.s1
            for jj in j:
                if jj == 0: continue
                assert s[0, jj] == -i
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0
            assert s[0, 0] == 1 - i

            csr[0, 0] = 2

            # *
            s = setup.s1 * csr
            for jj in j:
                if jj == 0: continue
                assert s[0, jj] == 0
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0
            assert s[0, 0] == i * 2

            # //
            s = s // csr
            assert s[0, 0] == i

            # **
            s = setup.s1 ** csr
            for jj in j:
                if jj == 0: continue
                assert s[0, jj] == 1
                assert setup.s1[0, jj] == i
                assert s[1, jj] == 0
            assert s[0, 0] == i ** 2

    def test_op3(self, setup):
        S = SparseCSR((10, 100), dtype=np.int32)
        # Create initial stuff
        for i in range(10):
            j = range(i*4, i*4+3)
            S[0, j] = i

        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(S, '__{}__'.format(op))
            s = func(1)
            assert s.dtype == np.int32
            s = func(1.)
            assert s.dtype == np.float64
            if op != 'pow':
                s = func(1.j)
                assert s.dtype == np.complex128

        S = S.copy(dtype=np.float64)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(S, '__{}__'.format(op))
            s = func(1)
            assert s.dtype == np.float64
            s = func(1.)
            assert s.dtype == np.float64
            if op != 'pow':
                s = func(1.j)
                assert s.dtype == np.complex128

        S = S.copy(dtype=np.complex128)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(S, '__{}__'.format(op))
            s = func(1)
            assert s.dtype == np.complex128
            s = func(1.)
            assert s.dtype == np.complex128
            if op != 'pow':
                s = func(1.j)
                assert s.dtype == np.complex128

    def test_op4(self, setup):
        S = SparseCSR((10, 100), dtype=np.int32)
        # Create initial stuff
        for i in range(10):
            j = range(i*4, i*4+3)
            S[0, j] = i

        s = 1 + S
        assert s.dtype == np.int32
        s = 1. + S
        assert s.dtype == np.float64
        s = 1.j + S
        assert s.dtype == np.complex128

        s = 1 - S
        assert s.dtype == np.int32
        s = 1. - S
        assert s.dtype == np.float64
        s = 1.j - S
        assert s.dtype == np.complex128

        s = 1 * S
        assert s.dtype == np.int32
        s = 1. * S
        assert s.dtype == np.float64
        s = 1.j * S
        assert s.dtype == np.complex128

        s = 1 ** S
        assert s.dtype == np.int32
        s = 1. ** S
        assert s.dtype == np.float64
        s = 1.j ** S
        assert s.dtype == np.complex128

    def test_op5(self, setup):
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
        assert np.allclose(S._D, (S2**2)._D)

        S = S * S
        assert np.allclose(S._D, (S2**4)._D)

        S = S1 + S2
        assert np.allclose(S._D, S1._D + S2._D)

        S = S1 - S2
        assert np.allclose(S._D, S1._D - S2._D)

    def test_op_numpy_scalar(self, setup):
        S = SparseCSR((10, 100), dtype=np.float32)
        I = np.ones(1, dtype=np.complex64)[0]
        # Create initial stuff
        for i in range(10):
            j = range(i*4, i*4+3)
            S[0, j] = i
        S.finalize()

        Ssum = S._D.sum()

        s = S + I
        assert isinstance(s, SparseCSR)
        assert s.dtype == np.complex64
        assert s._D.sum() == Ssum + S.nnz

        s = S - I
        assert isinstance(s, SparseCSR)
        assert s.dtype == np.complex64
        assert s._D.sum() == Ssum - S.nnz

        s = I + S
        assert isinstance(s, SparseCSR)
        assert s.dtype == np.complex64
        assert s._D.sum() == Ssum + S.nnz

        s = S * I
        assert isinstance(s, SparseCSR)
        assert s.dtype == np.complex64
        assert s._D.sum() == Ssum

        s = I * S
        assert isinstance(s, SparseCSR)
        assert s.dtype == np.complex64
        assert s._D.sum() == Ssum

        s = S / I
        assert isinstance(s, SparseCSR)
        assert s.dtype == np.complex64
        assert s._D.sum() == Ssum

        s = S ** I
        assert isinstance(s, SparseCSR)
        assert s.dtype == np.complex64
        assert s._D.sum() == Ssum

        s = I ** S
        assert isinstance(s, SparseCSR)
        assert s.dtype == np.complex64

    def test_sum1(self, setup):
        S1 = SparseCSR((10, 10, 2), dtype=np.int32)
        S1[0, 0] = [1, 2]
        S1[2, 0] = [1, 2]
        S1[2, 2] = [1, 2]

        S2 = S1.sum(-1)
        assert S1.spsame(S2)
        assert S2[0, 0] == 3
        assert S2[0, 1] == 0
        assert S2[2, 0] == 3
        assert S2[2, 2] == 3

        assert S1.sum() == 1 * 3 + 2 * 3

        S = S1.sum(0)
        v = np.zeros([S1.shape[0], S1.shape[2]], np.int32)
        v[0] = [1, 2]
        v[2] = [2, 4]
        assert np.allclose(S, v)

        v = np.zeros([S1.shape[0]], np.int32)
        v[0] = 3
        v[2] = 6
        assert np.allclose(S.sum(1), v)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_sum2(self, setup):
        S1 = SparseCSR((10, 10, 2), dtype=np.int32)
        S1[0, 0] = [1, 2]
        S1[2, 0] = [1, 2]
        S1[2, 2] = [1, 2]
        S1.sum(1)

    def test_pickle(self, setup):
        import pickle as p
        S = SparseCSR((10, 10, 2), dtype=np.int32)
        S[0, 0] = [1, 2]
        S[2, 0] = [1, 2]
        S[2, 2] = [1, 2]
        n = p.dumps(S)
        s = p.loads(n)
        assert s.spsame(S)


@pytest.mark.xfail(raises=IndexError)
@pytest.mark.parametrize("i", [-1, 10])
def test_sparse_row_out_of_bounds(i):
    S = SparseCSR((10, 10, 1), dtype=np.int32)
    S[i, 0] = 1


@pytest.mark.xfail(raises=IndexError)
@pytest.mark.parametrize("j", [-1, 10])
def test_sparse_column_out_of_bounds(j):
    S = SparseCSR((10, 10, 1), dtype=np.int32)
    S[0, j] = 1
