# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import copy
import itertools
import pickle

import numpy as np
import pytest

from sisl_toolbox.btd import BlockArray

pytestmark = [pytest.mark.btd]

BLOCKS = [2, 3, 1, 4]
# 3D array blocks
BLOCKS3 = ([2, 1, 3], [1, 3], [2, 2, 1])


def _ba(blocks=(BLOCKS, BLOCKS), dtype=np.float64, order="C", pattern="btd", seed=1):
    """Block array with random blocks, and its dense array

    pattern is one of btd (block indices differ by at most 1), upper (2D only) or full
    """
    rng = np.random.default_rng(seed)
    ba = BlockArray(blocks, dtype=dtype, order=order)
    BI = ba.block_indexer
    for key in itertools.product(*(range(len(b)) for b in blocks)):
        if pattern == "btd" and max(key) - min(key) > 1:
            continue
        if pattern == "upper" and key[1] < key[0]:
            continue
        M = rng.random([b[k] for b, k in zip(blocks, key)])
        if np.dtype(dtype).kind == "c":
            M = M + 1j * rng.random(M.shape)
        BI[key] = M
    return ba, ba.toarray()


def _same(b1, b2):
    return len(b1) == len(b2) and all(np.array_equal(x, y) for x, y in zip(b1, b2))


def test_block_array_init():
    ba = BlockArray([[2, 3], [2, 3]])
    assert isinstance(ba, np.ndarray)
    assert ba.shape == (5, 5)
    assert len(ba) == 5
    assert ba.ndim == 2
    assert ba.size == 25
    assert ba.dtype == np.float64
    assert ba.order == "C"
    assert len(ba.blocks) == 2
    assert np.array_equal(ba.blocks[0], [2, 3])
    assert len(list(ba.block_indexer)) == 0
    assert np.allclose(ba.toarray(), 0)


def test_block_array_init_nd():
    ba = BlockArray(BLOCKS3)
    assert ba.shape == (6, 4, 5)
    assert ba.ndim == 3
    assert np.shape(ba) == (6, 4, 5)
    assert np.ndim(ba) == 3
    ba = BlockArray([[2, 3]])
    assert ba.shape == (5,)


@pytest.mark.parametrize("dtype", [np.float32, np.complex128])
def test_block_array_dtype_forced(dtype):
    ba = BlockArray([[2, 3], [2, 3]], dtype=dtype)
    assert ba.dtype == dtype
    BI = ba.block_indexer
    BI[1, 1] = np.ones([3, 3], dtype=np.float64)
    assert BI[1, 1].dtype == dtype
    # undefined blocks have the correct dtype
    assert BI[0, 1].dtype == dtype
    assert ba.toarray().dtype == dtype


def test_block_array_dtype_no_complex_to_real():
    ba = BlockArray([[2, 3], [2, 3]], dtype=np.float64)
    with pytest.raises(TypeError):
        ba.block_indexer[1, 1] = np.zeros([3, 3], dtype=np.complex128)
    with pytest.raises(TypeError):
        ba[0, 0] = 1j


def test_block_array_shape_check():
    ba = BlockArray([[2, 3], [2, 3]])
    with pytest.raises(ValueError):
        ba.block_indexer[0, 1] = np.zeros([2, 2])
    with pytest.raises(ValueError):
        # a single axis must be a 1D array
        BlockArray([2, 3])
    with pytest.raises(ValueError):
        BlockArray([[2, -3]])
    with pytest.raises(ValueError):
        BlockArray([[2, 3]], order="K")


@pytest.mark.parametrize("blocks", [(BLOCKS, BLOCKS), BLOCKS3])
@pytest.mark.parametrize("order", ["C", "F"])
def test_block_array_order(blocks, order):
    ba, D = _ba(blocks, order=order)
    flag = "C_CONTIGUOUS" if order == "C" else "F_CONTIGUOUS"
    for key in ba.block_indexer:
        assert ba.block_indexer[key].flags[flag]
    # indices follow the memory layout of the blocks
    idx = ba.indices()
    values = np.concatenate(
        [ba.block_indexer[key].ravel(order="K") for key in ba.block_indexer]
    )
    assert np.array_equal(D[idx], values)


def test_block_array_indexer():
    ba = BlockArray([[2, 3], [2, 3]])
    BI = ba.block_indexer
    assert len(BI) == 2
    M = np.arange(6.0).reshape(2, 3)
    BI[0, 1] = M
    # no copy for matching dtype and order
    assert BI[0, 1] is M
    assert (0, 1) in BI
    assert (1, 0) not in BI
    assert list(BI) == [(0, 1)]
    zero = BI[1, 0]
    assert zero.shape == (3, 2)
    zero[:] = 1
    # undefined blocks are not stored
    assert (1, 0) not in BI
    del BI[0, 1]
    assert (0, 1) not in BI
    with pytest.raises(KeyError):
        del BI[0, 1]
    with pytest.raises(ValueError):
        BI[0]

    # 1D arrays may use an integer block index
    ba = BlockArray([[2, 3]])
    ba.block_indexer[1] = [1, 2, 3]
    assert np.array_equal(ba.toarray(), [0, 0, 1, 2, 3])


@pytest.mark.parametrize("blocks", [(BLOCKS, BLOCKS), BLOCKS3])
def test_block_array_indices(blocks):
    ba, D = _ba(blocks)
    index_sl = ba.indices(defined=True)
    empty_index_sl = ba.indices(defined=False)
    assert len(index_sl) == ba.ndim
    assert len(index_sl[0]) + len(empty_index_sl[0]) == ba.size
    full = np.zeros(ba.shape, dtype=ba.dtype)
    full[index_sl] = ba[index_sl]
    assert np.allclose(full, D)
    assert np.allclose(full[empty_index_sl], 0)
    # all indices are unique
    lin = np.concatenate(
        [np.ravel_multi_index(idx, ba.shape) for idx in (index_sl, empty_index_sl)]
    )
    assert len(np.unique(lin)) == ba.size


def test_block_array_indices_empty():
    ba = BlockArray([[2, 3], [2, 3]])
    rows, cols = ba.indices()
    assert len(rows) == len(cols) == 0
    rows, cols = ba.indices(defined=False)
    assert len(rows) == 25


KEYS2 = [
    (0, 0),
    (3, 9),
    (-1, -2),
    (slice(None), 4),
    (7, slice(None)),
    (slice(1, 8), slice(3, 9)),
    (slice(None, None, -2), slice(1, None, 3)),
    (slice(2, 7), [9, 0, 4, 4]),
    ([5, 1, 1, 8], slice(None)),
    ([0, 3, 9], [1, 5, 2]),
    ([[0], [4], [9]], [[1, 3]]),
    (np.arange(10) % 3 == 0, slice(None)),
    (Ellipsis, 3),
    (2, Ellipsis),
    (4,),
    (np.array([[1, 2], [3, 4]]), 5),
    (slice(None), None),
    ([], slice(None)),
]

KEYS3 = [
    (0, 0, 0),
    (5, 3, 4),
    (-1, 0, -2),
    (slice(None), 1, slice(None)),
    (2,),
    (slice(1, 5), slice(None, None, -1), slice(0, 4, 2)),
    (Ellipsis, 2),
    (1, Ellipsis, 3),
    ([0, 5, 2], slice(None), slice(None)),
    (slice(None), [3, 0], slice(None)),
    (slice(None), [3, 0], [1, 4]),
    # non-adjacent index arrays (broadcast dimensions first)
    ([0, 5, 2], slice(1, 3), [4, 0, 1]),
    ([[0], [5]], slice(None), [1, 3, 4]),
    (2, slice(None), [1, 4]),
    ([[1, 2]], [[0], [3]], 2),
    ([[0], [3], [5]], [[[1]], [[2]]], 0),
    np.ix_([4, 0, 1], [2, 3], [0, 4]),
    (slice(None), np.arange(4) > 1, None),
]


@pytest.mark.parametrize("key", KEYS2)
def test_block_array_getitem(key):
    ba, D = _ba(dtype=np.complex128)
    if len(key) == 1:
        key = key[0]
    v = ba[key]
    if isinstance(v, BlockArray):
        v = v.toarray()
    ref = D[key]
    assert v.shape == np.shape(ref)
    assert np.array_equal(v, ref)


@pytest.mark.parametrize("key", KEYS3)
def test_block_array_getitem_nd(key):
    ba, D = _ba(BLOCKS3)
    if len(key) == 1:
        key = key[0]
    v = ba[key]
    if isinstance(v, BlockArray):
        v = v.toarray()
    ref = D[key]
    assert v.shape == np.shape(ref)
    assert np.array_equal(v, ref)


@pytest.mark.parametrize("blocks", [(BLOCKS, BLOCKS), BLOCKS3])
def test_block_array_getitem_mask(blocks):
    ba, D = _ba(blocks)
    mask = D > 0.5
    assert np.array_equal(ba[mask], D[mask])


def test_block_array_getitem_errors():
    ba, _ = _ba()
    with pytest.raises(IndexError):
        ba[10, 0]
    with pytest.raises(IndexError):
        ba[0, [-11]]
    with pytest.raises(IndexError):
        ba[0, 0, 0]
    with pytest.raises(IndexError):
        ba[0.5, 0]


def test_block_array_getitem_blocks():
    ba, D = _ba()
    c = np.cumsum([0] + BLOCKS)
    # complete blocks returns a block array sharing blocks
    sub = ba[c[1] : c[3], c[1] : c[3]]
    assert isinstance(sub, BlockArray)
    assert np.array_equal(sub.blocks[0], BLOCKS[1:3])
    assert np.array_equal(sub.toarray(), D[c[1] : c[3], c[1] : c[3]])
    assert sub.block_indexer[0, 1] is ba.block_indexer[1, 2]

    full = ba[:, :]
    assert isinstance(full, BlockArray)
    assert np.array_equal(full.toarray(), D)

    # different complete blocks along the axes
    sub = ba[c[1] : c[3], c[2] :]
    assert isinstance(sub, BlockArray)
    assert np.array_equal(sub.blocks[1], BLOCKS[2:])
    assert np.array_equal(sub.toarray(), D[c[1] : c[3], c[2] :])

    # a permutation of complete blocks copies the blocks
    perm = np.concatenate([np.arange(c[2], c[3]), np.arange(c[0], c[2])])
    sub = ba[np.ix_(perm, perm)]
    assert isinstance(sub, BlockArray)
    assert np.array_equal(sub.blocks[0], [BLOCKS[2], BLOCKS[0], BLOCKS[1]])
    assert np.array_equal(sub.toarray(), D[np.ix_(perm, perm)])
    assert not np.shares_memory(sub.block_indexer[1, 1], ba.block_indexer[0, 0])

    # incomplete blocks are dense
    assert not isinstance(ba[c[1] + 1 : c[3], c[1] : c[3]], BlockArray)
    # integer indices remove an axis
    assert not isinstance(ba[0, :], BlockArray)


def test_block_array_getitem_blocks_nd():
    ba, D = _ba(BLOCKS3)
    sub = ba[2:, :, :4]
    assert isinstance(sub, BlockArray)
    assert sub.shape == (4, 4, 4)
    assert np.array_equal(sub.toarray(), D[2:, :, :4])
    assert sub.block_indexer[0, 0, 1] is ba.block_indexer[1, 0, 1]


def test_block_array_setitem():
    ba = BlockArray([BLOCKS, BLOCKS])
    D = np.zeros(ba.shape)
    rng = np.random.default_rng(3)

    ba[1:3, 4:8] = 1.0
    D[1:3, 4:8] = 1.0
    ba[5, :] = np.arange(10)
    D[5, :] = np.arange(10)
    v = rng.random([3, 2])
    ba[[0, 9, 4], 2:4] = v
    D[[0, 9, 4], 2:4] = v
    ba[[1, 8], [8, 1]] = [3.0, 4.0]
    D[[1, 8], [8, 1]] = [3.0, 4.0]
    ba[-1, -1] = 2
    D[-1, -1] = 2
    assert np.array_equal(ba.toarray(), D)

    # assigning zeros does not allocate blocks
    ba = BlockArray([BLOCKS, BLOCKS])
    ba[:, :] = 0
    ba[[0, 9], [9, 0]] = 0
    assert len(list(ba.block_indexer)) == 0


def test_block_array_setitem_nd():
    ba = BlockArray(BLOCKS3)
    D = np.zeros(ba.shape)
    rng = np.random.default_rng(3)
    for key in KEYS3:
        if any(k is None for k in key):
            continue
        v = rng.random(np.shape(D[key]))
        ba[key] = v
        D[key] = v
        assert np.array_equal(ba.toarray(), D)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("blocks", [(BLOCKS, BLOCKS), BLOCKS3])
def test_block_array_ufunc_structural(blocks, dtype):
    ba, D = _ba(blocks, dtype=dtype)
    defined = list(ba.block_indexer)
    for res, ref in (
        (-ba, -D),
        (ba * 2, D * 2),
        (2 * ba, 2 * D),
        (ba / 4, D / 4),
        (np.conj(ba), np.conj(D)),
        (ba.conj(), D.conj()),
        (abs(ba), abs(D)),
        (ba + ba, D + D),
        (ba - ba, D - D),
        (ba**2, D**2),
    ):
        assert isinstance(res, BlockArray)
        assert list(res.block_indexer) == defined
        assert np.allclose(res.toarray(), ref)
    assert (abs(ba)).dtype == np.float64


def test_block_array_ufunc_pattern():
    btd, Dbtd = _ba(pattern="btd")
    up, Dup = _ba(pattern="upper", seed=2)
    # multiply only retains the blocks defined in both
    res = btd * up
    assert np.allclose(res.toarray(), Dbtd * Dup)
    assert set(res.block_indexer) == set(btd.block_indexer) & set(up.block_indexer)
    # addition retains the blocks defined in any
    res = btd + up
    assert np.allclose(res.toarray(), Dbtd + Dup)
    assert set(res.block_indexer) == set(btd.block_indexer) | set(up.block_indexer)


def test_block_array_ufunc_dense():
    ba, D = _ba()
    # not retaining zeros returns dense arrays
    for res, ref in (
        (ba + 1, D + 1),
        (np.exp(ba), np.exp(D)),
        (1 / (ba + 2), 1 / (D + 2)),
    ):
        assert not isinstance(res, BlockArray)
        assert np.allclose(res, ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.allclose(ba / ba, D / D, equal_nan=True)
    v = np.arange(10.0)
    assert np.allclose(ba * v, D * v)
    assert np.allclose(ba.sum(), D.sum())
    assert np.allclose(ba.sum(axis=0), D.sum(axis=0))
    assert np.allclose(ba.mean(), D.mean())
    assert np.allclose(np.max(ba), D.max())
    assert np.allclose(ba, D)
    bd = ba.tobd()
    assert np.allclose(np.linalg.inv(bd), np.linalg.inv(bd.toarray()))
    assert np.allclose(ba.ravel(), D.ravel())
    assert (ba > 0.5).dtype == bool
    assert np.array_equal((ba > 0.5).toarray(), D > 0.5)


def test_block_array_ufunc_inplace():
    ba, D = _ba()
    BI = ba.block_indexer
    M = BI[1, 1]
    ba *= 2
    D *= 2
    assert BI[1, 1] is M
    assert np.allclose(ba.toarray(), D)
    up, Dup = _ba(pattern="upper", seed=2)
    ba += up
    D += Dup
    assert np.allclose(ba.toarray(), D)
    with pytest.raises(TypeError):
        ba += 1
    with pytest.raises(Exception):
        ba *= 1j


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_block_array_matmul(dtype):
    a, Da = _ba(dtype=dtype)
    b, Db = _ba(pattern="upper", seed=2)
    res = a @ b
    assert isinstance(res, BlockArray)
    assert res.dtype == np.result_type(dtype, np.float64)
    assert np.allclose(res.toarray(), Da @ Db)
    assert np.allclose(a.dot(b).toarray(), Da @ Db)

    rng = np.random.default_rng(4)
    for x in (rng.random(10), rng.random([10, 3])):
        assert np.allclose(a @ x, Da @ x)
        assert np.allclose(x.T @ a, x.T @ Da)
    with pytest.raises(ValueError):
        a @ np.ones(3)


def test_block_array_matmul_non_square():
    a, Da = _ba(([2, 3], [1, 4, 2]), pattern="full")
    b, Db = _ba(([1, 4, 2], [3, 3]), pattern="full", seed=2)
    res = a @ b
    assert isinstance(res, BlockArray)
    assert np.array_equal(res.blocks[0], [2, 3])
    assert np.array_equal(res.blocks[1], [3, 3])
    assert np.allclose(res.toarray(), Da @ Db)
    # different blocks along the contracted axis (dense)
    c, Dc = _ba(([4, 3], [3, 3]), pattern="full", seed=2)
    assert np.allclose(a @ c, Da @ Dc)


def test_block_array_matmul_stacked():
    a, Da = _ba(([2, 1], [2, 3], [1, 4]))
    b, Db = _ba(([2, 1], [1, 4], [3, 2]), seed=2)
    res = a @ b
    assert isinstance(res, BlockArray)
    assert res.shape == (3, 5, 5)
    assert np.allclose(res.toarray(), Da @ Db)


@pytest.mark.parametrize("blocks", [(BLOCKS, BLOCKS), BLOCKS3])
def test_block_array_transpose(blocks):
    ba, D = _ba(blocks, dtype=np.complex128, pattern="full")
    T = ba.T
    assert T.order == "F"
    assert np.array_equal(T.toarray(), D.T)
    # views of the blocks
    key = tuple(len(b) - 1 for b in blocks)
    assert np.shares_memory(T.block_indexer[key[::-1]], ba.block_indexer[key])
    assert np.array_equal(np.transpose(ba).toarray(), D.T)
    assert np.allclose(ba.T.conj().toarray(), D.conj().T)
    # transposed indices still fill the dense array
    full = np.zeros(T.shape, dtype=T.dtype)
    full[T.indices()] = T[T.indices()]
    assert np.array_equal(full, D.T)

    for axes in itertools.permutations(range(ba.ndim)):
        assert np.array_equal(ba.transpose(axes).toarray(), D.transpose(axes))
        assert np.array_equal(ba.transpose(*axes).toarray(), D.transpose(axes))


def test_block_array_diagonal():
    ba, D = _ba(pattern="full")
    d = ba.diagonal()
    assert isinstance(d, BlockArray)
    assert np.array_equal(d.blocks[0], BLOCKS)
    assert np.allclose(d.toarray(), D.diagonal())
    assert np.allclose(np.diagonal(ba), D.diagonal())
    for offset in (-4, -1, 1, 3, 12):
        assert np.allclose(ba.diagonal(offset), D.diagonal(offset))
        assert np.allclose(ba.trace(offset), D.trace(offset))
    assert np.allclose(ba.trace(), D.trace())
    assert np.allclose(np.trace(ba), D.trace())

    # different blocks along the axes
    ba, D = _ba(([2, 3], [1, 4]), pattern="full")
    assert not isinstance(ba.diagonal(), BlockArray)
    assert np.allclose(ba.diagonal(), D.diagonal())


def test_block_array_diagonal_nd():
    blocks = ([2, 1, 3], [3, 2], [2, 1, 3])
    ba, D = _ba(blocks)
    for axis1, axis2 in itertools.permutations(range(3), 2):
        for offset in (0, 1, -2):
            d = ba.diagonal(offset, axis1, axis2)
            ref = D.diagonal(offset, axis1, axis2)
            if isinstance(d, BlockArray):
                d = d.toarray()
            assert d.shape == ref.shape
            assert np.allclose(d, ref)
            assert np.allclose(
                ba.trace(offset, axis1, axis2), D.trace(offset, axis1, axis2)
            )
    # same blocks along axis 0 and 2 gives a block array
    assert isinstance(ba.diagonal(0, 0, 2), BlockArray)
    assert isinstance(ba.diagonal(0, 2, 0), BlockArray)
    with pytest.raises(ValueError):
        ba.diagonal(0, 1, 1)


def test_block_array_real_imag():
    ba, D = _ba(dtype=np.complex128)
    assert ba.real.dtype == np.float64
    assert np.array_equal(ba.real.toarray(), D.real)
    assert np.array_equal(np.imag(ba).toarray(), D.imag)
    r, Dr = _ba()
    assert np.array_equal(r.imag.toarray(), np.zeros_like(Dr))


def test_block_array_copy():
    ba, D = _ba(BLOCKS3)
    key = next(iter(ba.block_indexer))
    for c in (ba.copy(), copy.copy(ba), copy.deepcopy(ba), np.copy(ba)):
        assert isinstance(c, BlockArray)
        assert np.array_equal(c.toarray(), D)
        assert not np.shares_memory(c.block_indexer[key], ba.block_indexer[key])
    c = ba.astype(np.complex64)
    assert c.dtype == np.complex64
    assert c.block_indexer[key].dtype == np.complex64
    assert np.allclose(c.toarray(), D)


def test_block_array_pickle():
    ba, D = _ba(BLOCKS3, dtype=np.complex128, order="F")
    p = pickle.loads(pickle.dumps(ba))
    assert isinstance(p, BlockArray)
    assert p.dtype == ba.dtype
    assert p.order == "F"
    assert _same(p.blocks, ba.blocks)
    assert list(p.block_indexer) == list(ba.block_indexer)
    assert np.array_equal(p.toarray(), D)


def test_block_array_band():
    ba, D = _ba(pattern="full")
    bd = ba.tobd()
    btd = ba.tobtd()
    assert set(bd.block_indexer) == {(i, i) for i in range(len(BLOCKS))}
    assert set(btd.block_indexer) == {
        (i, j) for i, j in ba.block_indexer if abs(i - j) <= 1
    }
    assert btd.block_indexer[0, 1] is ba.block_indexer[0, 1]


def test_block_array_band_nd():
    ba, D = _ba(([2, 1], [2, 3, 1], [1, 4, 2]), pattern="full")
    all_keys = set(itertools.product(range(2), range(3), range(3)))
    assert set(ba.tobtd().block_indexer) == {
        k for k in all_keys if max(k) - min(k) <= 1
    }
    assert set(ba.tobd().block_indexer) == {k for k in all_keys if max(k) == min(k)}
    # only along the last two axes (stacked BTD matrices)
    assert set(ba.tobtd(axes=(-2, -1)).block_indexer) == {
        k for k in all_keys if abs(k[1] - k[2]) <= 1
    }
    assert set(ba.tobd(axes=(1, 2)).block_indexer) == {
        k for k in all_keys if k[1] == k[2]
    }


def test_block_array_unsupported():
    ba, _ = _ba()
    with pytest.raises(NotImplementedError):
        ba.fill(0)
    with pytest.raises(NotImplementedError):
        ba.sort()
