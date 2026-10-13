# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from sisl import Atom, Bloch, Hamiltonian, geom

pytestmark = [pytest.mark.physics, pytest.mark.bloch]


def get_H():
    s = geom.sc(1.0, Atom(1, 1.001))
    H = Hamiltonian(s)
    H.construct([(0.1, 1.001), (0.1, 0.5)])
    return H.tile(2, 0).tile(2, 1).tile(2, 2)


@pytest.mark.parametrize("nx", [1, 5])
@pytest.mark.parametrize("ny", [1, 4])
@pytest.mark.parametrize("nz", [1, 6])
def test_bloch_create(nx, ny, nz):
    b = Bloch([nx, ny, nz])
    assert len(b) == nx * ny * nz
    assert len(b.unfold_points([0] * 3)) == nx * ny * nz


def test_bloch_method():
    b = Bloch([1] * 3)
    assert "Bloch" in str(b)


def test_bloch_call():
    b = Bloch([2] * 3)
    H = get_H()

    # Manual
    k_unfold = b.unfold_points([0] * 3)
    m = b.unfold(np.stack([H.Hk(k, format="array") for k in k_unfold]), k_unfold)
    m1 = b.unfold([H.Hk(k, format="array") for k in k_unfold], k_unfold)

    assert np.allclose(m, m1)
    assert np.allclose(m, b(H.Hk, [0] * 3, format="array"))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("nx", [1, 4])
@pytest.mark.parametrize("ny", [1, 5])
@pytest.mark.parametrize("nz", [1, 6])
def test_bloch_H_same(nx, ny, nz, dtype):
    H = get_H()
    b = Bloch([nx, ny, nz])

    if dtype == np.complex64:
        atol = 1e-6
    else:
        atol = 1e-8

    HB = H.tile(nx, 0).tile(ny, 1).tile(nz, 2)

    KX = [0, 0.1, 0.4358923]
    KY = [0, 0.128359, -0.340925]
    KZ = [0, 0.445]
    for kx, ky, kz in product(KX, KY, KZ):
        K = [kx, ky, kz]
        k_unfold = b.unfold_points(K)

        HK = [H.Hk(k, format="array", dtype=dtype) for k in k_unfold]
        H_unfold = b.unfold(HK, k_unfold)
        H_big = HB.Hk(K, format="array", dtype=dtype)

        assert np.allclose(H_big, H_big.T.conj(), atol=atol)
        assert np.allclose(H_unfold, H_unfold.T.conj(), atol=atol)
        assert np.allclose(H_unfold, H_big, atol=atol)


def _block_mask(b, n, cells_list):
    """Dense boolean mask of the blocks coupling the cells within each list of cells"""
    B = b.bloch
    N = len(b)
    mask = np.zeros([N, N], dtype=bool)
    for cells in cells_list:
        cells = np.asarray(cells).reshape(-1, 3)
        idx = (cells[:, 2] * B[1] + cells[:, 1]) * B[0] + cells[:, 0]
        mask[np.ix_(idx, idx)] = True
    return np.kron(mask, np.ones([n, n], dtype=bool))


def _all_cells(b):
    B = b.bloch
    return np.stack(
        np.meshgrid(*[np.arange(x) for x in B], indexing="ij"), axis=-1
    ).reshape(-1, 3)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("bloch", [[3, 1, 1], [1, 4, 1], [2, 1, 3], [3, 2, 2]])
def test_bloch_parts_all(bloch, dtype):
    H = get_H()
    b = Bloch(bloch)
    atol = 1e-5 if dtype == np.complex64 else 1e-10
    K = [0.1, -0.23, 0.31]
    full = b(H.Hk, K, format="array", dtype=dtype)
    parts = b(H.Hk, K, format="array", dtype=dtype, parts=_all_cells(b))
    assert parts.dtype == full.dtype
    assert parts.shape == full.shape
    assert np.allclose(parts.toarray(), full, atol=atol)


@pytest.mark.parametrize("bloch", [[4, 1, 1], [2, 3, 1], [2, 2, 3]])
def test_bloch_parts_segments(bloch):
    """Two segment calculations patched together equals the full expansion"""
    H = get_H()
    n = H.no
    b = Bloch(bloch)
    K = [0.2, 0.13, -0.41]
    k_unfold = b.unfold_points(K)
    HK = np.stack([H.Hk(k, format="array") for k in k_unfold])
    full = b.unfold(HK, k_unfold)

    cells = _all_cells(b)
    S1 = cells[: len(cells) // 2]
    S2 = cells[len(cells) // 2 :]
    ba1 = b.unfold(HK, k_unfold, parts=S1)
    ba2 = b.unfold(HK, k_unfold, parts=S2)

    patched = (ba1 + ba2).toarray()
    mask = _block_mask(b, n, [S1, S2])
    assert np.allclose(patched[mask], full[mask])
    assert np.all(patched[~mask] == 0)

    # adding the cross-couplings yields the full matrix
    ba12 = b.unfold(HK, k_unfold, parts=cells)
    cross = _block_mask(b, n, [cells]) & ~mask
    patched[cross] = ba12.toarray()[cross]
    assert np.allclose(patched, full)


def test_bloch_parts_rim():
    """Remove the internal cells of a 5x5 Bloch expansion"""
    H = get_H()
    n = H.no
    b = Bloch([5, 5, 1])
    K = [0.1, 0.3, 0.0]
    full = b(H.Hk, K, format="array")

    cells = _all_cells(b)
    inner = np.all((cells[:, :2] >= 2) & (cells[:, :2] < 3), axis=1)
    rim = cells[~inner]
    ba = b(H.Hk, K, format="array", parts=rim)
    assert len(list(ba.block_indexer)) == len(rim) ** 2

    D = ba.toarray()
    mask = _block_mask(b, n, [rim])
    assert np.allclose(D[mask], full[mask])
    assert np.all(D[~mask] == 0)


def test_bloch_parts_tuple():
    H = get_H()
    b = Bloch([3, 3, 2])
    K = [0.1, 0.3, 0.2]
    # [0, 0, 1], [1, 1, 1], [2, 2, 1]
    ba = b(H.Hk, K, format="array", parts=(np.arange(3), np.arange(3), [1]))
    ba_ = b(H.Hk, K, format="array", parts=[[0, 0, 1], [1, 1, 1], [2, 2, 1]])
    assert np.allclose(ba.toarray(), ba_.toarray())

    # broadcasted: [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
    ba = b(
        H.Hk,
        K,
        format="array",
        parts=(np.arange(2).reshape(-1, 1), np.arange(2).reshape(1, -1), [1]),
    )
    ba_ = b(H.Hk, K, format="array", parts=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    assert np.allclose(ba.toarray(), ba_.toarray())


def test_bloch_parts_errors():
    b = Bloch([3, 1, 2])
    k = b.unfold_points([0] * 3)
    M = np.zeros([len(b), 2, 2], dtype=np.complex128)
    with pytest.raises(ValueError):
        b.unfold(M, k, parts=[[3, 0, 0]])
    with pytest.raises(ValueError):
        b.unfold(M, k, parts=[[0, 0, -3]])
    with pytest.raises(ValueError):
        b.unfold(M, k, parts=[[-4, 0, 0]])
    with pytest.raises(ValueError):
        b.unfold(M, k, parts=[[0, 0]])
    with pytest.raises(ValueError):
        b.unfold(M, k, parts=([0], [0], [0], [0]))


def test_bloch_parts_bz_average():
    from sisl import MonkhorstPack

    H = get_H()
    b = Bloch([2, 1, 3])
    bz = MonkhorstPack(H, [2, 2, 1])
    parts = [[0, 0, 0], [1, 0, 2]]

    full = bz.apply.average(lambda k: b(H.Hk, k, format="array"))()
    ba = bz.apply.average(lambda k: b(H.Hk, k, format="array", parts=parts))()
    assert np.allclose(
        b.unfold_finalize(ba).toarray()[_block_mask(b, H.no, [parts])],
        full[_block_mask(b, H.no, [parts])],
    )


@pytest.mark.parametrize("bloch", [[3, 1, 1], [2, 3, 1], [3, 2, 4]])
def test_bloch_parts_random(bloch):
    """Non-symmetric matrices, ensures the phase convention is the same as the full unfold"""
    rng = np.random.default_rng(1234)
    b = Bloch(bloch)
    k_unfold = b.unfold_points([0.13, -0.27, 0.41])
    M = rng.random([len(b), 3, 2]) + 1j * rng.random([len(b), 3, 2])
    full = b.unfold(M, k_unfold)
    cells = _all_cells(b)
    rng.shuffle(cells)
    S = cells[: max(2, len(cells) // 2)]
    D = b.unfold(M, k_unfold, parts=S).toarray()
    B = b.bloch
    N = len(b)
    idx = (S[:, 2] * B[1] + S[:, 1]) * B[0] + S[:, 0]
    mask = np.zeros([N, N], dtype=bool)
    mask[np.ix_(idx, idx)] = True
    mask = np.kron(mask, np.ones([3, 2], dtype=bool))
    assert np.allclose(D[mask], full[mask])
    assert np.all(D[~mask] == 0)


def test_bloch_parts_wrap():
    """Negative indices count from the end"""
    H = get_H()
    b = Bloch([4, 1, 3])
    K = [0.1, 0.0, 0.2]
    ba = b(H.Hk, K, format="array", parts=[[-2, 0, -1], [0, -1, 0]])
    ba_ = b(H.Hk, K, format="array", parts=[[2, 0, 2], [0, 0, 0]])
    assert set(ba.block_indexer) == set(ba_.block_indexer)
    assert np.allclose(ba.toarray(), ba_.toarray())

    ba = b(H.Hk, K, format="array", parts=([-1, 0], [0], [-3, 2]))
    ba_ = b(H.Hk, K, format="array", parts=([3, 0], [0], [0, 2]))
    assert set(ba.block_indexer) == set(ba_.block_indexer)
    assert np.allclose(ba.toarray(), ba_.toarray())


def test_bloch_parts_tuple_missing():
    """Missing lattice vectors take all unit-cells"""
    b = Bloch([5, 3, 2])
    assert np.array_equal(
        b._parts_cells(([-2],)),
        [[3, i, j] for i in range(3) for j in range(2)],
    )
    assert np.array_equal(
        b._parts_cells(([0, 4], [-1])),
        [[i, 2, j] for i in (0, 4) for j in range(2)],
    )
    assert np.array_equal(b._parts_cells(()), np.unique(_all_cells(b), axis=0))

    H = get_H()
    K = [0.1, 0.2, 0.3]
    ba = b(H.Hk, K, format="array", parts=([-2],))
    ba_ = b(H.Hk, K, format="array", parts=([3], np.arange(3).reshape(-1, 1), [0, 1]))
    assert set(ba.block_indexer) == set(ba_.block_indexer)
    assert np.allclose(ba.toarray(), ba_.toarray())
