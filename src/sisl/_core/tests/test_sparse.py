# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import operator
import sys

import numpy as np
import pytest
import scipy as sc

from sisl._array import array_arange
from sisl._core.sparse import *
from sisl._core.sparse import indices

pytestmark = [
    pytest.mark.sparse,
    pytest.mark.filterwarnings("ignore", category=sc.sparse.SparseEfficiencyWarning),
]


@pytest.fixture
def s1():
    return SparseCSR((10, 100), dtype=np.int32)


@pytest.fixture
def s1d():
    return SparseCSR((10, 100))


@pytest.fixture
def s2():
    return SparseCSR((10, 100, 2))


def test_indices():
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


def test_fail_init1():
    with pytest.raises(ValueError):
        SparseCSR((10, 100, 20, 20), dtype=np.int32)


def test_fail_init_shape0():
    with pytest.raises(ValueError):
        SparseCSR((0, 10, 10), dtype=np.int32)


def test_fail_init2():
    data = np.empty([2, 2], np.float64)
    indices = np.arange(2)
    indptr = np.arange(2)
    with pytest.raises(ValueError):
        SparseCSR((data, indices, indptr, indptr), shape=(100, 20, 20))


def test_init_csr_inputs():
    data = np.empty([2, 2], np.float64)
    indices = np.arange(2)
    indptr = np.arange(3)
    SparseCSR((data, indices, indptr))


def test_csr_matrix_int64_fail():
    # fixes #901
    s = sc.sparse.csr_matrix((2, 2))
    s[0, 0] = 1
    s[1, 1] = 1
    # this will also break scipy, but here for testing purposes:
    s.indices = s.indices.astype(np.int64)
    s.indptr = s.indptr.astype(np.int64)
    sp = SparseCSR.fromsp(s)
    assert sp.col.dtype == np.int32
    assert sp.ncol.dtype == np.int32
    assert sp.ptr.dtype == np.int32


def test_fail_align1():
    s1 = SparseCSR((10, 100), dtype=np.int32)
    s2 = SparseCSR((20, 100), dtype=np.int32)
    with pytest.raises(ValueError):
        s1.align(s2)


def test_set_get1():
    s1 = SparseCSR((10, 10, 3), dtype=np.int32)
    s1[1, 1] = [1, 2, 3]
    s1[1, 2] = [1, 2, 3]
    assert s1.nnz == 2
    assert np.all(s1[1, 1] == [1, 2, 3])
    assert np.all(s1[1, [1, 2]] == [[1, 2, 3], [1, 2, 3]])
    assert np.all(s1[1, [1, 2], 1] == [2, 2])
    assert np.all(s1[1, [1, 6], 1] == [2, 0])


def test_init1(s1, s2, s1d):
    str(s1)
    assert s1.dtype == np.int32
    assert s1d.dtype == np.float64
    assert s2.dtype == np.float64


def test_init2():
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


def test_init3():
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


def test_init4():
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


def test_diagonal_1d():
    s = SparseCSR((10, 100), dtype=np.int32)
    for i in range(s.shape[0]):
        s[i, i] = i
    d = s.diagonal()
    assert np.allclose(d, np.arange(s.shape[0]))


def test_diagonal_2d():
    s = SparseCSR((10, 100, 2), dtype=np.int32)
    d1 = np.arange(s.shape[0])
    d2 = np.arange(s.shape[0]) + 4

    for i in range(s.shape[0]):
        s[i, i] = (d1[i], d2[i])
    d = s.diagonal()

    assert np.allclose(d[:, 0], d1)
    assert np.allclose(d[:, 1], d2)


def test_extend1():
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


def test_diags_1():
    csr = SparseCSR((10, 10), nnzpr=1, dtype=np.int32)
    csr1 = csr.diags(10)
    assert csr1.shape[0] == 10
    assert csr1.shape[1] == 10
    assert csr1.shape[2] == 1
    assert csr1.nnz == 10
    assert np.allclose(csr1.toarray()[..., 0], np.diag(np.full(10, 10)))
    csr2 = csr.diags(10)
    csr3 = csr.diags(np.zeros(10, np.int32))
    assert csr2.spsame(csr3)


def test_diags_2():
    csr = SparseCSR((10, 10), dim=3, nnzpr=1, dtype=np.int32)
    csr1 = csr.diags(10, dim=1)
    csr2 = csr.diags(10, dim=2, dtype=np.int16)
    assert csr1.shape[2] == 1
    assert csr2.shape[2] == 2
    assert csr1.nnz == 10
    assert csr2.nnz == 10
    assert csr1.spsame(csr2)
    # TODO fix diags such that it won't affect
    assert csr1.dtype == np.int64
    assert csr2.dtype == np.int16


def test_diags_offsets():
    csr = SparseCSR((10, 10), dtype=np.int32)
    m = csr.diags(1)
    assert m.nnz == 10
    assert np.sum(m) == 10
    m = csr.diags(1, offsets=1)
    assert m.nnz == 9
    assert np.sum(m) == 9
    m = csr.diags(1, offsets=2)
    assert m.nnz == 8
    assert np.sum(m) == 8
    m = csr.diags(1, offsets=-2)
    assert m.nnz == 8
    assert np.sum(m) == 8


def test_diags_multiple_diagonals():
    csr = SparseCSR((10, 10), dim=2, dtype=np.int32)
    m = csr.diags([[1, 2]])
    assert np.allclose(m[0, 0], [1, 2])


def test_create1(s1d):
    s1d[0, [1, 2, 3]] = 1
    assert s1d.nnz == 3
    s1d[2, [1, 2, 3]] = 1
    assert s1d.nnz == 6
    s1d.empty(keep_nnz=True)
    assert s1d.nnz == 6
    s1d.empty()
    assert s1d.nnz == 0
    s1d[0, 0] = np.nan
    # nan produces zeros
    assert s1d.nnz == 1
    s1d.empty()


def test_create2(s1):
    assert len(s1) == s1.shape[0]
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        s1[0, j] = i
        assert s1.nnz == (i + 1) * 3
        for jj in j:
            assert s1[0, jj] == i
            assert s1[1, jj] == 0


def test_create3(s1):
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        s1[0, j] = i
        assert s1.nnz == (i + 1) * 3
        s1[0, range((i + 1) * 4, (i + 1) * 4 + 3)] = None
        assert s1.nnz == (i + 1) * 3
        for jj in j:
            assert s1[0, jj] == i
            assert s1[1, jj] == 0


def test_create_1d_bcasting_data_1d(s1):
    s2 = s1.copy()
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        s2[0, j] = i
        s2[1, j] = i
        s2[2, j] = i
        s2[3, j] = i

    s3 = s1.copy()
    for i in range(10):
        j = np.arange(i * 4, i * 4 + 3).reshape(1, -1)
        s3[np.arange(4).reshape(-1, 1), j] = i

    assert s2.spsame(s3)
    assert np.sum(s2 - s3) == 0


@pytest.mark.xfail(
    sys.platform.startswith("win"), reason="Unknown windows error in b-casting"
)
def test_create_1d_bcasting_data_2d(s1):
    s2 = s1.copy()
    data = np.random.randint(1, 100, (4, 3))
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        s2[0, j] = data[0, :]
        s2[1, j] = data[1, :]
        s2[2, j] = data[2, :]
        s2[3, j] = data[3, :]

    s3 = s1.copy()
    for i in range(10):
        j = np.arange(i * 4, i * 4 + 3).reshape(1, -1)
        s3[np.arange(4).reshape(-1, 1), j] = data

    assert s2.spsame(s3)
    assert np.sum(s2 - s3) == 0


def test_create_1d_diag(s1):
    s2 = s1.copy()
    data = np.random.randint(1, 100, len(s1))
    d = np.arange(len(s1))
    for i in d:
        s2[i, i] = data[i]

    s3 = s1.copy()
    s3[d, d] = data

    assert s2.spsame(s3)
    assert np.sum(s2 - s3) == 0


def test_create_2d_diag_0d(s2):
    S1 = s2.copy()
    data = 1
    d = np.arange(len(s2))
    for i in d:
        S1[i, i] = data
    S2 = s2.copy()
    S2[d, d] = data

    S3 = s2.copy()
    S3[d, d, 0] = data
    S3[d, d, 1] = data

    assert S1.spsame(S2)
    assert np.sum(S1 - S2) == 0
    assert S1.spsame(S3)
    assert np.sum(S1 - S3) == 0


def test_create_2d_diag_1d(s2):
    s1 = s2.copy()
    s3 = s2.copy()

    data = np.random.randint(1, 100, len(s1))
    d = np.arange(len(s1))
    for i in d:
        s1[i, i] = data[i]

    s2[d, d] = data
    s3[d, d, 0] = data
    s3[d, d, 1] = data

    assert s1.spsame(s2)
    assert np.sum(s1 - s2) == 0
    assert s1.spsame(s3)
    assert np.sum(s1 - s3) == 0


def test_create_2d_diag_2d(s2):
    s1 = s2.copy()
    s3 = s2.copy()

    data = np.random.randint(1, 100, len(s1) * 2).reshape(-1, 2)
    d = np.arange(len(s1))
    for i in d:
        s1[i, i] = data[i]

    s2[d, d] = data
    s3[d, d, 0] = data[:, 0]
    s3[d, d, 1] = data[:, 1]

    assert s1.spsame(s2)
    assert np.sum(s1 - s2) == 0
    assert s1.spsame(s3)
    assert np.sum(s1 - s3) == 0


def test_create_2d_data_2d(s2):
    s1 = s2.copy()
    s3 = s2.copy()

    # matrix assignment
    I = np.arange(len(s1) // 2)
    data = np.random.randint(1, 100, I.size**2).reshape(I.size, I.size)
    for i in I:
        s1[i, I, 0] = data[i]
        s1[i, I, 1] = data[i]

    I.shape = (-1, 1)

    s2[I, I.T, 0] = data
    s2[I, I.T, 1] = data

    s3[I, I.T] = data[:, :, None]

    assert s1.spsame(s2)
    assert np.sum(s1 - s2) == 0
    assert s1.spsame(s3)
    assert np.sum(s1 - s3) == 0


def test_create_2d_data_3d(s2):
    s1 = s2.copy()

    # matrix assignment
    I = np.arange(len(s1) // 2)
    data = np.random.randint(1, 100, I.size**2 * 2).reshape(I.size, I.size, 2)
    for i in I:
        s1[i, I] = data[i]

    I.shape = (-1, 1)
    s2[I, I.T] = data

    assert s1.spsame(s2)
    assert np.sum(s1 - s2) == 0


def test_copy_dims(s2):
    s2[2, 2] = [2, 3]
    s1 = s2.copy(dims=0)
    assert np.allclose(s2._D[:, 0], s1._D[:, 0])
    s1 = s2.copy(dims=1)
    assert np.allclose(s2._D[:, 1], s1._D[:, 0])

    s1 = s2.copy(dims=[1, 0])
    assert np.allclose(s2._D[:, 1], s1._D[:, 0])
    assert np.allclose(s2._D[:, 0], s1._D[:, 1])


def test_fail_data_3d_to_1d(s2):
    # matrix assignment
    I = np.arange(len(s2) // 2).reshape(-1, 1)
    data = np.random.randint(1, 100, I.size * 2).reshape(I.size, 1, 2)
    with pytest.raises(ValueError):
        s2[I, I.T] = data


def test_fail_data_2d_to_2d(s2):
    # matrix assignment
    I = np.arange(len(s2) // 2).reshape(-1, 1)
    data = np.random.randint(1, 100, I.size**2).reshape(I.size, I.size)
    with pytest.raises(ValueError):
        s2[I, I.T] = data


def test_fail_data_2d_to_3d(s2):
    # matrix assignment
    I = np.arange(len(s2) // 2).reshape(-1, 1)
    data = np.random.randint(1, 100, I.size**2).reshape(I.size, I.size)
    with pytest.raises(ValueError):
        s2[I, I.T, [0, 1]] = data


def test_finalize1(s1):
    s1[0, [1, 2, 3]] = 1
    s1[2, [1, 2, 3]] = 1.0
    s1[1, [3, 2, 1]] = 1.0
    assert not s1.finalized
    p = s1.ptr.view()
    n = s1.ncol.view()
    # Assert that the ordering is good
    assert np.allclose(s1.col[p[1] : p[1] + n[1]], [3, 2, 1])
    s1.finalize()

    assert s1.finalized
    s1.empty(keep_nnz=True)
    assert s1.finalized
    s1.empty()
    assert not s1.finalized


def test_finalize2(s1):
    s1[0, [1, 2, 3]] = 1
    s1[2, [1, 2, 3]] = 1.0
    s1[1, [3, 2, 1]] = 1.0
    assert not s1.finalized
    p = s1.ptr.view()
    n = s1.ncol.view()
    # Assert that the ordering is good
    assert np.allclose(s1.col[p[1] : p[1] + n[1]], [3, 2, 1])
    s1.finalize(False)

    assert not s1.finalized
    assert len(s1.col) == 9


def test_iterator1(s1):
    s1[0, [1, 2, 3]] = 1
    s1[2, [1, 2, 4]] = 1.0
    e = [[1, 2, 3], [], [1, 2, 4]]
    for i, j in s1:
        assert j in e[i]

    for i, j in s1.iter_nnz(0):
        assert i == 0
        assert j in e[0]
    for i, j in s1.iter_nnz(2):
        assert i == 2
        assert j in e[2]

    for i, j in ispmatrix(s1):
        assert j in e[i]

    for i, j in ispmatrix(s1, map_col=lambda x: x):
        assert j in e[i]
    for i, j in ispmatrix(s1, map_row=lambda x: x):
        assert j in e[i]

    for i, j, d in ispmatrixd(s1):
        assert j in e[i]
        assert d == 1.0


def test_iterator2(s1):
    e = [[1, 2, 3], [], [1, 2, 4]]
    s1[0, [1, 2, 3]] = 1
    s1[2, [1, 2, 4]] = 1.0
    a = s1.tocsr()
    for func in ["csr", "csc", "coo", "lil"]:
        a = getattr(a, "to" + func)()
        for r, c in ispmatrix(a):
            assert r in [0, 2]
            assert c in e[r]

    for func in ["csr", "csc", "coo", "lil"]:
        a = getattr(a, "to" + func)()
        for r, c, d in ispmatrixd(a):
            assert r in [0, 2]
            assert c in e[r]
            assert d == 1.0


def test_iterator3(s1):
    e = [[1, 2, 3], [], [1, 2, 4]]
    s1[0, [1, 2, 3]] = 1
    s1[2, [1, 2, 4]] = 1.0
    a = s1.tocsr()
    for func in ["csr", "csc", "coo", "lil"]:
        a = getattr(a, "to" + func)()
        for r, c in ispmatrix(a):
            assert r in [0, 2]
            assert c in e[r]

    # number of mapped values
    nvals = 2
    for func in ["csr", "lil"]:
        a = getattr(a, "to" + func)()
        n = 0
        for r, c in ispmatrix(a, lambda x: x % 2, lambda x: x % 2):
            assert r == 0
            assert c in [0, 1]
            n += 1
        assert n == nvals


def test_delitem_simple(s1):
    s1[0, [1, 2, 3]] = 1
    assert s1.nnz == 3
    del s1[0, 1]
    assert s1.nnz == 2
    assert s1[0, 1] == 0
    assert s1[0, 2] == 1
    assert s1[0, 3] == 1
    s1[0, [1, 2, 3]] = 1
    del s1[0, [1, 3]]
    assert s1.nnz == 1
    assert s1[0, 1] == 0
    assert s1[0, 2] == 1
    assert s1[0, 3] == 0
    del s1[range(2), 0]
    assert s1.nnz == 1
    for i in range(2):
        assert s1[i, 0] == 0


def test_delitem_order():
    s = SparseCSR((10, 10), dtype=np.int32)
    s[0, 3] = 3
    s[0, 2] = 2
    s[0, 1] = 1
    del s[0, 2]
    assert s.nnz == 2
    assert s[0, 1] == 1
    assert s[0, 3] == 3


def test_contains1(s1):
    s1[0, [1, 2, 3]] = 1
    assert s1.nnz == 3
    assert [0, 1] in s1
    assert [0, [1, 3]] in s1


def test_sub1(s1):
    s1[0, [1, 2, 3]] = 1
    assert s1.nnz == 3
    s1 = s1.sub([0, 1])
    assert s1.nnz == 1
    assert s1.shape[0] == 2
    assert s1.shape[1] == 2


def test_remove1(s1):
    s1[0, [1, 2, 3]] = 1
    assert s1.nnz == 3
    s2 = s1.remove([1])
    assert s2.nnz == 2
    assert s2.shape[0] == s1.shape[0] - 1
    assert s2.shape[1] == s1.shape[1] - 1


def test_eliminate_zeros1(s1):
    s1[0, [1, 2, 3]] = 1
    s1[1, [1, 2, 3]] = 0
    assert s1.nnz == 6
    s1.eliminate_zeros()
    assert s1.nnz == 3
    assert s1[1, 1] == 0
    assert s1[1, 2] == 0
    assert s1[1, 3] == 0


def test_eliminate_zeros_tolerance(s1):
    s1[0, [1, 2, 3]] = 1
    s1[1, [1, 2, 3]] = 2
    assert s1.nnz == 6
    s1.eliminate_zeros()
    assert s1.nnz == 6
    s1.eliminate_zeros(1)
    assert s1.nnz == 3
    assert s1[0, 1] == 0
    assert s1[0, 2] == 0
    assert s1[0, 3] == 0


def test_eliminate_zeros_tolerance_ndim():
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


def test_spsame(s1, s2):
    s1[0, [1, 2, 3]] = 1
    s2[0, [1, 2, 3]] = (1, 1)
    assert s1.spsame(s2)
    s2[1, 1] = (1, 1)
    assert not s1.spsame(s2)
    s1.align(s2)
    assert s1.spsame(s2)


@pytest.mark.xfail(reason="same index assignment in single statement")
def test_set_same_index(s1):
    s1[0, [1, 1]] = 1
    assert s1.nnz == 1


def test_delete_col1(s1):
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


def test_delete_col2(s1):
    nc = s1.shape[1]
    s1[1, [1, 2, 3]] = 1
    assert s1.nnz == 3
    s1.delete_columns(3)
    assert s1.nnz == 2
    assert s1.shape[1] == nc - 1
    s1.delete_columns(2, True)
    assert s1.nnz == 1
    assert s1.shape[1] == nc - 1


def test_delete_col3(s1):
    s2 = s1.copy()
    nc = s1.shape[1]
    for i in range(10):
        s1[i, [3, 2, 1]] = 1
        s2[i, 2] = 1
    s1.finalize()
    s2.finalize()
    assert s1.nnz == 10 * 3
    s1.delete_columns([3, 1], keep_shape=True)
    assert s1.ptr[-1] == s1.nnz
    assert s2.ptr[-1] == s2.nnz
    assert s1.nnz == 10 * 1
    assert s1.spsame(s2)


def test_delete_col4():
    s1 = SparseCSR((10, 100), dtype=np.int32)
    s2 = SparseCSR((10, 98), dtype=np.int32)
    nc = s1.shape[1]
    for i in range(10):
        s1[i, [3, 2, 1]] = 1
        s2[i, 1] = 1
    s1.finalize()
    s2.finalize()
    assert s1.nnz == 10 * 3
    s1.delete_columns([3, 1])
    assert s1.ptr[-1] == s1.nnz
    assert s2.ptr[-1] == s2.nnz
    assert s1.nnz == 10 * 1
    assert s1.spsame(s2)


def test_delete_col5(s1):
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


def test_delete_col6(s1):
    nc = s1.shape[1]
    for i in range(3):
        s1[i, [1, 2, 3]] = 1
    assert s1.nnz == 9
    s1.delete_columns(2)
    assert s1.nnz == 6
    assert s1.shape[1] == nc - 1


def test_translate_col1(s1):
    s1[1, 1] = 1
    s1[1, 2] = 2
    s1[1, 3] = 3
    assert s1.nnz == 3
    s1.translate_columns([1, 3], [3, 1])
    assert s1.nnz == 3
    assert s1[1, 1] == 3
    assert s1[1, 3] == 1


def test_translate_col2(s1):
    s1[1, 1] = 1
    s1[1, 2] = 2
    s1[1, 3] = 3
    assert s1.nnz == 3
    s1.translate_columns([1, 3], [3, s1.shape[1] + 100])
    assert s1.nnz == 2
    assert s1[1, 3] == 1
    assert s1[1, 1] == 0


def test_translate_col3(s1):
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


def test_translate_col4(s1):
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


def test_edges1(s1):
    s1[1, 1] = 1
    s1[1, 2] = 2
    s1[1, 3] = 3
    assert np.all(s1.edges(1, exclude=[]) == [1, 2, 3])
    assert np.all(s1.edges(1, exclude=[1]) == [2, 3])
    assert np.all(s1.edges(1, exclude=2) == [1, 3])
    assert len(s1.edges(2)) == 0


def test_nonzero1(s1):
    s1[2, 1] = 1
    s1[1, 1] = 1
    s1[1, 2] = 2
    s1[1, 3] = 3
    assert s1.nnz == 4
    r, c = s1.nonzero()
    assert np.all(r == [1, 1, 1, 2])
    assert np.all(c == [1, 2, 3, 1])
    c = s1.nonzero(only_cols=True)
    assert np.all(c == [1, 2, 3, 1])
    c = s1.nonzero(rows=1, only_cols=True)
    assert np.all(c == [1, 2, 3])
    c = s1.nonzero(rows=2, only_cols=True)
    assert np.all(c == [1])
    c = s1.nonzero(rows=[0, 1], only_cols=True)
    assert np.all(c == [1, 2, 3])
    r, c = s1.nonzero(rows=[0, 1])
    assert np.all(r == [1, 1, 1])
    assert np.all(c == [1, 2, 3])


def test_op1(s1):
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        s1[0, j] = i

        # i+
        s1 += 1
        for jj in j:
            assert s1[0, jj] == i + 1
            assert s1[1, jj] == 0

        # i-
        s1 -= 1
        for jj in j:
            assert s1[0, jj] == i
            assert s1[1, jj] == 0

        # i*
        s1 *= 2
        for jj in j:
            assert s1[0, jj] == i * 2
            assert s1[1, jj] == 0

        # //
        s1 //= 2
        for jj in j:
            assert s1[0, jj] == i
            assert s1[1, jj] == 0

        # i**
        s1 **= 2
        for jj in j:
            assert s1[0, jj] == i**2
            assert s1[1, jj] == 0


def test_op2(s1):
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        s1[0, j] = i

        # +
        s = s1 + 1
        for jj in j:
            assert s[0, jj] == i + 1
            assert s1[0, jj] == i
            assert s[1, jj] == 0

        # -
        s = s1 - 1
        for jj in j:
            assert s[0, jj] == i - 1
            assert s1[0, jj] == i
            assert s[1, jj] == 0

        # - (r)
        s = 1 - s1
        for jj in j:
            assert s[0, jj] == 1 - i
            assert s1[0, jj] == i
            assert s[1, jj] == 0

        # *
        s = s1 * 2
        for jj in j:
            assert s[0, jj] == i * 2
            assert s1[0, jj] == i
            assert s[1, jj] == 0

        # *
        s = np.multiply(s1, 2)
        for jj in j:
            assert s[0, jj] == i * 2
            assert s1[0, jj] == i
            assert s[1, jj] == 0
        # *
        s.empty()
        np.multiply(s1, 2, out=s)
        for jj in j:
            assert s[0, jj] == i * 2
            assert s1[0, jj] == i
            assert s[1, jj] == 0

        # //
        s = s // 2
        for jj in j:
            assert s[0, jj] == i
            assert s1[0, jj] == i
            assert s[1, jj] == 0

        # **
        s = s1**2
        for jj in j:
            assert s[0, jj] == i**2
            assert s1[0, jj] == i
            assert s[1, jj] == 0

        # ** (r)
        s = 2**s1
        for jj in j:
            assert s[0, jj], 2 ** s1[0 == jj]
            assert s1[0, jj] == i
            assert s[1, jj] == 0


def test_op_csr(s1):
    csr = sc.sparse.csr_matrix((10, 100), dtype=np.int32)
    for i in range(10):
        j = range(i + 2)
        s1[0, j] = i

        csr[0, 0] = 1

        # +
        s = s1 + csr
        for jj in j:
            if jj == 0:
                continue
            assert s[0, jj] == i
            assert s1[0, jj] == i
            assert s[1, jj] == 0
        assert s[0, 0] == i + 1

        # -
        s = s1 - csr
        for jj in j:
            if jj == 0:
                continue
            assert s[0, jj] == i
            assert s1[0, jj] == i
            assert s[1, jj] == 0
        assert s[0, 0] == i - 1

        # - (r)
        s = csr - s1
        for jj in j:
            if jj == 0:
                continue
            assert s[0, jj] == -i
            assert s1[0, jj] == i
            assert s[1, jj] == 0
        assert s[0, 0] == 1 - i

        csr[0, 0] = 2

        # *
        s = s1 * csr
        for jj in j:
            if jj == 0:
                continue
            assert s[0, jj] == 0
            assert s1[0, jj] == i
            assert s[1, jj] == 0
        assert s[0, 0] == i * 2

        # //
        s = s // csr
        assert s[0, 0] == i

        # **
        s = s1**csr
        for jj in j:
            if jj == 0:
                continue
            assert s[0, jj] == 1
            assert s1[0, jj] == i
            assert s[1, jj] == 0
        assert s[0, 0] == i**2


def test_op3():
    S = SparseCSR((10, 100), dtype=np.int32)
    # Create initial stuff
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        S[0, j] = i

    for op in ("add", "sub", "mul", "pow"):
        func = getattr(S, f"__{op}__")
        s = func(1)
        assert s.dtype == np.int32
        s = func(1.0)
        assert s.dtype == np.float64
        if op != "pow":
            s = func(1.0j)
            assert s.dtype == np.complex128

    S = S.copy(dtype=np.float64)
    for op in ("add", "sub", "mul", "pow"):
        func = getattr(S, f"__{op}__")
        s = func(1)
        assert s.dtype == np.float64
        s = func(1.0)
        assert s.dtype == np.float64
        if op != "pow":
            s = func(1.0j)
            assert s.dtype == np.complex128

    S = S.copy(dtype=np.complex128)
    for op in ("add", "sub", "mul", "pow"):
        func = getattr(S, f"__{op}__")
        s = func(1)
        assert s.dtype == np.complex128
        s = func(1.0)
        assert s.dtype == np.complex128
        if op != "pow":
            s = func(1.0j)
            assert s.dtype == np.complex128


def test_op4():
    S = SparseCSR((10, 100), dtype=np.int32)
    # Create initial stuff
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        S[0, j] = i

    s = 1 + S
    assert s.dtype == np.int32
    s = 1.0 + S
    assert s.dtype == np.float64
    s = 1.0j + S
    assert s.dtype == np.complex128

    s = 1 - S
    assert s.dtype == np.int32
    s = 1.0 - S
    assert s.dtype == np.float64
    s = 1.0j - S
    assert s.dtype == np.complex128

    s = 1 * S
    assert s.dtype == np.int32
    s = 1.0 * S
    assert s.dtype == np.float64
    s = 1.0j * S
    assert s.dtype == np.complex128

    s = 1**S
    assert s.dtype == np.int32
    s = 1.0**S
    assert s.dtype == np.float64
    s = 1.0j**S
    assert s.dtype == np.complex128


def binary():
    op = operator
    ops = [
        op.mod,
        op.mul,
        op.add,
        op.sub,
        op.floordiv,
        op.truediv,
    ]
    return ops


def binary_int():
    op = operator
    return [op.pow]


def unary():
    op = operator
    ops = [
        op.abs,
        op.neg,
        op.pos,
    ]
    return ops


@pytest.fixture(scope="module")
def matrix_sisl_csr():
    matrix = []

    def add3(m):
        matrix.append(m)
        m = m.copy()
        m.finalize(sort=False)
        matrix.append(m)
        m = m.copy()
        m.finalize(sort=True)
        matrix.append(m)

    # diagonal
    m = SparseCSR((10, 80), dtype=np.int32)
    for i in range(10):
        m[i, i] = 1
    add3(m)

    # not completely full (some empty ncol)

    m = SparseCSR((10, 80), dtype=np.int32)
    m[0, [1, 0]] = [12, 2]
    m[2, 2] = 1
    m[4, [4, 0]] = [2, 3]
    m[5, [5, 0]] = [3, 5]
    add3(m)

    # all more than 1 coupling, and not sorted
    m = SparseCSR((10, 80), dtype=np.int32)
    m[0, [1, 0]] = [11, 0]
    m[1, [10, 50, 20]] = [14, 20, 43]
    m[2, [4, 7, 3, 1]] = [2, 5, 4, 10]
    m[3, [4, 7, 3, 1]] = [2, 5, 4, 10]
    m[4, [1, 5, 3, 21]] = [2, 5, 4, 10]
    m[5, [53, 52, 3, 21]] = [31, 6, 7, 12]
    m[6, [43, 32, 1, 6]] = [3, 6, 7, 16]
    m[7, [65, 44, 3, 2]] = [3, 6, 73, 31]
    m[8, [66, 45, 6, 8]] = [4, 3, 12, 357]
    m[9, [55, 44, 33, 22]] = [4, 3, 11, 27]
    add3(m)
    return matrix


@pytest.fixture(scope="module")
def matrix_csr_matrix():
    matrix = []

    csr_matrix = sc.sparse.csr_matrix

    def add3(m):
        matrix.append(m)
        m = m.copy()
        m.sort_indices()
        matrix.append(m)

    # diagonal
    m = csr_matrix((10, 80), dtype=np.int32)
    for i in range(10):
        m[i, i] = 1
    add3(m)

    # not completely full (some empty ncol)

    m = csr_matrix((10, 80), dtype=np.int32)
    m[0, [1, 0]] = [11, 3]
    m[2, 2] = 10
    m[4, [4, 0]] = [22, 40]
    m[5, [5, 0]] = [33, 51]
    add3(m)

    # all more than 1 coupling, and not sorted
    m = csr_matrix((10, 80), dtype=np.int32)
    m[0, [1, 0]] = [12, 4]
    m[1, [10, 50, 20]] = [11, 20, 401]
    m[2, [4, 7, 3, 1]] = [2, 5, 4, 10]
    m[3, [4, 7, 3, 1]] = [2, 5, 4, 10]
    m[4, [1, 5, 3, 21]] = [2, 5, 4, 10]
    m[5, [53, 52, 3, 21]] = [3, 6, 7, 14]
    m[6, [43, 32, 1, 6]] = [3, 6, 7, 11]
    m[7, [65, 44, 3, 2]] = [3, 6, 7, 12]
    m[8, [66, 45, 6, 8]] = [4, 3, 15, 7]
    m[9, [55, 44, 33, 22]] = [4, 3, 14, 7]
    add3(m)
    return matrix


@pytest.mark.parametrize("op", binary())
def test_op_binary(matrix_sisl_csr, matrix_csr_matrix, op):
    for m in matrix_sisl_csr:
        mD = m.toarray()[..., 0]
        if op not in (operator.add, operator.sub):
            v = op(m, 3)
            assert np.allclose(op(mD, 3), v.toarray()[..., 0])

        if op not in (operator.truediv, operator.floordiv):
            for m2 in matrix_sisl_csr:
                m2D = m2.toarray()[..., 0]
                v = op(m, m2)
                assert np.allclose(op(mD, m2D), v.toarray()[..., 0])

            for m2 in matrix_csr_matrix:
                m2D = m2.toarray()
                v = op(m, m2)
                assert np.allclose(op(mD, m2D), v.toarray()[..., 0])


def test_op5():
    S1 = SparseCSR((10, 100), dtype=np.int32)
    S2 = SparseCSR((10, 100), dtype=np.int32)
    S3 = SparseCSR((10, 100), dtype=np.int32)
    # Create initial stuff
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        S1[0, j] = i
        S2[0, j] = i

        if i < 5:
            S3[0, j] = i

    S = S1 * S2
    assert np.allclose(S.todense(), (S2**2).todense())

    S = S * S
    assert np.allclose(S.todense(), (S2**4).todense())

    S = S1 + S2
    S1 += S2
    assert np.allclose(S.todense(), S1.todense())

    S = S1 - S2
    S1 -= S2
    assert np.allclose(S.todense(), S1.todense())

    S = S1 + 2
    S -= 2
    assert np.allclose(S.todense(), S1.todense())

    S = S1 * 2
    S //= 2
    assert np.allclose(S.todense(), S1.todense())

    S = S1 / 2.0
    S *= 2
    assert np.allclose(S.todense(), S1.todense())


def test_op_numpy_scalar():
    S = SparseCSR((10, 100), dtype=np.float32)
    I = np.ones(1, dtype=np.complex64)[0]
    # Create initial stuff
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
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

    s = S**I
    assert isinstance(s, SparseCSR)
    assert s.dtype == np.complex64
    assert s._D.sum() == Ssum

    s = I**S
    assert isinstance(s, SparseCSR)
    assert s.dtype == np.complex64

    s = 1j * S
    assert isinstance(s, SparseCSR)
    assert s.dtype == (1j * np.array([1j], dtype=np.complex64)).dtype

    s = np.exp(s)
    assert isinstance(s, SparseCSR)
    assert s.dtype == np.exp(1j * np.array([1j], dtype=np.complex64)).dtype


def test_op_sparse_dim():
    S = SparseCSR((10, 100, 2), dtype=np.float32)
    assert S.shape == (10, 100, 2)
    I = np.ones(1, dtype=np.complex64)[0]
    # Create initial stuff
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        S[0, j] = i
    S.finalize()

    # Try and add different values to the last 2 dimensions
    s = S + [2, 2]
    assert np.allclose(s._D, (S + 2)._D)
    s = S + [1, 2]
    assert np.allclose(s._D[:, 0], (S + 1)._D[:, 0])

    # check both
    ss = S + 1
    assert np.any(np.isclose(s._D[:, 0], ss._D[:, 0]))
    assert not np.any(np.isclose(s._D[:, 1], ss._D[:, 1]))

    ss = S + 2
    assert not np.any(np.isclose(s._D[:, 0], ss._D[:, 0]))
    assert np.all(np.isclose(s._D[:, 1], ss._D[:, 1]))


def test_sparse_transpose():
    S = SparseCSR((10, 100, 2), dtype=np.float32)
    assert S.shape == (10, 100, 2)
    I = np.ones(1, dtype=np.complex64)[0]
    # Create initial stuff
    for i in range(10):
        j = range(i * 4, i * 4 + 3)
        S[0, j] = i
    S.finalize()

    s = S.transpose(False)
    assert s.shape == (100, 10, 2)

    s = s.transpose(False)
    assert s.shape == (10, 100, 2)


def test_op_reduce():
    S1 = SparseCSR((10, 11, 2), dtype=np.int32)
    S1[0, 0] = [1, 2]
    S1[2, 0] = [1, 2]
    S1[2, 2] = [1, 2]

    S2 = np.sum(S1, axis=-1)
    assert S1.spsame(S2)
    assert S2[0, 0] == 3
    assert S2[0, 1] == 0
    assert S2[2, 0] == 3
    assert S2[2, 2] == 3

    assert np.sum(S1) == 1 * 3 + 2 * 3

    S = np.sum(S1, axis=0)
    v = np.zeros([S1.shape[1], S1.shape[2]], np.int32)
    v[0] = [2, 4]
    v[2] = [1, 2]
    assert np.allclose(S, v)

    v = v.sum(1)
    assert np.allclose(np.sum(S, 1), v)

    S = np.sum(S1, axis=1)
    v = np.zeros([S1.shape[0], S1.shape[2]], np.int32)
    v[0] = [1, 2]
    v[2] = [2, 4]
    assert np.allclose(S, v)


def test_unfinalized_math():
    S1 = SparseCSR((4, 4, 1))
    S2 = SparseCSR((4, 4, 1))
    S1[0, 0] = 2.0
    S1[1, 2] = 3.0
    S2[2, 3] = 4.0
    S2[2, 2] = 3.0
    S2[0, 0] = 4

    for i in range(3):
        assert np.allclose(S1.todense() + S2.todense(), (S1 + S2).todense())
        assert np.allclose(S1.todense() * S2.todense(), (S1 * S2).todense())
        sin = np.sin(S1.todense()) + np.sin(S2.todense())
        sins = (np.sin(S1) + np.sin(S2)).todense()
        assert np.allclose(sin, sins)
        sins = np.sin(S1).todense() + np.sin(S2).todense()
        assert np.allclose(sin, sins)

        if i == 0:
            assert not S1.finalized
            assert not S2.finalized
            S1.finalize()
        elif i == 1:
            assert S1.finalized
            assert not S2.finalized
            S2.finalize()
        elif i == 2:
            assert S1.finalized
            assert S2.finalized


def test_pickle():
    import pickle as p

    S = SparseCSR((10, 10, 2), dtype=np.int32)
    S[0, 0] = [1, 2]
    S[2, 0] = [1, 2]
    S[2, 2] = [1, 2]
    n = p.dumps(S)
    s = p.loads(n)
    assert s.spsame(S)


@pytest.mark.parametrize("i", [-1, 10])
def test_sparse_row_out_of_bounds(i):
    S = SparseCSR((10, 10, 1), dtype=np.int32)
    with pytest.raises(IndexError):
        S[i, 0] = 1


@pytest.mark.parametrize("j", [-1, 10])
def test_sparse_column_out_of_bounds(j):
    S = SparseCSR((10, 10, 1), dtype=np.int32)
    with pytest.raises(IndexError):
        S[0, j] = 1


def test_fromsp_csr():
    csr1 = sc.sparse.random(10, 100, 0.01, random_state=24812)
    csr2 = sc.sparse.random(10, 100, 0.02, random_state=24813)

    csr = SparseCSR.fromsp([csr1, csr2])
    csr_1 = csr.tocsr(0)
    csr_2 = csr.tocsr(1)

    assert np.abs(csr1 - csr_1).sum() == 0.0
    assert np.abs(csr2 - csr_2).sum() == 0.0


def test_transform1():
    csr1 = sc.sparse.random(10, 100, 0.01, random_state=24812)
    csr2 = sc.sparse.random(10, 100, 0.02, random_state=24813)
    csr = SparseCSR.fromsp([csr1, csr2])

    # real 1x2 matrix, dtype=np.complex128
    matrix = [[0.3, 0.7]]
    tr = csr.transform(matrix=matrix, dtype=np.complex128)

    assert tr.shape[:2] == csr.shape[:2]
    assert tr.shape[2] == len(matrix)
    assert np.abs(tr.tocsr(0) - 0.3 * csr1 - 0.7 * csr2).sum() == 0.0


def test_transform2():
    csr1 = sc.sparse.random(10, 100, 0.01, random_state=24812)
    csr2 = sc.sparse.random(10, 100, 0.02, random_state=24813)
    csr = SparseCSR.fromsp([csr1, csr2])

    # real 2x2 matrix, dtype=np.float64
    matrix = [[0.3, 0], [0, 0.7]]
    tr = csr.transform(matrix=matrix, dtype=np.float64)

    assert tr.shape[:2] == csr.shape[:2]
    assert tr.shape[2] == len(matrix)
    assert np.abs(tr.tocsr(0) - 0.3 * csr1).sum() == 0.0
    assert np.abs(tr.tocsr(1) - 0.7 * csr2).sum() == 0.0


def test_transform3():
    csr1 = sc.sparse.random(10, 100, 0.01, random_state=24812)
    csr2 = sc.sparse.random(10, 100, 0.02, random_state=24813)
    csr = SparseCSR.fromsp([csr1, csr2])

    # real 3x2 matrix
    matrix = [[0.3, 0], [0, 0.7], [0.1, 0.2]]
    tr = csr.transform(matrix=matrix)

    assert tr.shape[:2] == csr.shape[:2]
    assert tr.shape[2] == len(matrix)
    assert np.abs(tr.tocsr(0) - 0.3 * csr1).sum() == 0.0
    assert np.abs(tr.tocsr(1) - 0.7 * csr2).sum() == 0.0
    assert np.abs(tr.tocsr(2) - 0.1 * csr1 - 0.2 * csr2).sum() == 0.0


def test_transform4():
    csr1 = sc.sparse.random(10, 100, 0.01, random_state=24812)
    csr2 = sc.sparse.random(10, 100, 0.02, random_state=24813)
    csr = SparseCSR.fromsp([csr1, csr2])

    # complex 1x2 matrix
    matrix = [[0.3j, 0.7j]]
    tr = csr.transform(matrix=matrix)

    assert tr.shape[:2] == csr.shape[:2]
    assert tr.shape[2] == len(matrix)
    assert np.abs(tr.tocsr(0) - 0.3j * csr1 - 0.7j * csr2).sum() == 0.0


def test_transform_fail():
    csr1 = sc.sparse.random(10, 100, 0.01, random_state=24812)
    csr2 = sc.sparse.random(10, 100, 0.02, random_state=24813)
    csr = SparseCSR.fromsp((csr1, csr2))

    # complex 1x3 matrix
    matrix = [[0.3j, 0.7j, 1.0]]
    with pytest.raises(ValueError):
        csr.transform(matrix=matrix)


@pytest.mark.slow
def test_fromsp_csr_large():
    csr1 = sc.sparse.random(10000, 10, 0.1, format="csr", random_state=23583)
    csr2 = csr1.copy()

    print_time = False

    from time import time

    # Add some more stuff
    row = csr1[9948]
    indices = row.indices
    if len(indices) == 0:
        indices = np.arange(3)
    csr2[9948, (indices + 1) % 10] = 1.0
    assert csr1.getnnz() != csr2.getnnz()

    t0 = time()
    csr = SparseCSR.fromsp([csr1, csr2])
    if print_time:
        print(f"timing: fromsp {time() - t0}")
    csr_1 = csr.tocsr(0)
    csr_2 = csr.tocsr(1)

    assert np.abs(csr1 - csr_1).sum() == 0.0
    assert np.abs(csr2 - csr_2).sum() == 0.0

    csr_ = SparseCSR(csr1.shape + (2,), nnzpr=1)

    t0 = time()
    for ic, c in enumerate([csr1, csr2]):
        ptr = c.indptr

        # Loop stuff
        for r in range(c.shape[0]):
            idx = csr_._extend(r, c.indices[ptr[r] : ptr[r + 1]])
            csr_._D[idx, ic] += c.data[ptr[r] : ptr[r + 1]]
    if print_time:
        print(f"timing: 2 x ptr[]:ptr[] {time() - t0}")

    dcsr = csr - csr_

    assert np.abs(dcsr.tocsr(0)).sum() == 0.0
    assert np.abs(dcsr.tocsr(1)).sum() == 0.0

    csr_ = SparseCSR(csr1.shape + (2,), nnzpr=1)

    t0 = time()
    for ic, c in enumerate([csr1, csr2]):
        ptr = c.indptr

        # Loop stuff
        for r in range(c.shape[0]):
            sl = slice(ptr[r], ptr[r + 1])
            idx = csr_._extend(r, c.indices[sl])
            csr_._D[idx, ic] += c.data[sl]
    if print_time:
        print(f"timing: slice(ptr[]:ptr[]) {time() - t0}")
