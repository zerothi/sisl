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


def _random_M(rng, N, n0, n1, dtype=np.complex128):
    return (rng.random([N, n0, n1]) + 1j * rng.random([N, n0, n1])).astype(dtype)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("bloch", [[1, 1, 1], [4, 1, 1], [2, 3, 1], [3, 2, 4]])
def test_bloch_indices(bloch, dtype):
    """Sub-matrix equals the elements of the full unfolding

    Uses non-Hermitian matrices to ensure the phase convention is the same.
    """
    rng = np.random.default_rng(42)
    b = Bloch(bloch)
    atol = 1e-5 if dtype == np.complex64 else 1e-12
    k_unfold = b.unfold_points([0.13, -0.27, 0.41])
    N = len(b)

    # rectangular (rows, cols): unsorted, duplicates, negative indices
    n0, n1 = 5, 3
    M = _random_M(rng, N, n0, n1, dtype)
    full = b.unfold(M, k_unfold)
    rows = rng.integers(-N * n0, N * n0, size=7)
    rows[1] = rows[0]
    cols = rng.integers(-N * n1, N * n1, size=6)
    G = b.unfold(M, k_unfold, indices=(rows, cols))
    assert G.dtype == full.dtype
    assert G.shape == (len(rows), len(cols))
    assert np.allclose(G, full[np.ix_(rows, cols)], atol=atol)

    # square
    M = _random_M(rng, N, n1, n1, dtype)
    full = b.unfold(M, k_unfold)
    indices = rng.permutation(N * n1)[: max(1, N * n1 // 2)]
    G = b.unfold(M, k_unfold, indices=indices)
    assert np.allclose(G, full[np.ix_(indices, indices)], atol=atol)

    # all
    G = b.unfold(M, k_unfold, indices=np.arange(N * n1))
    assert np.allclose(G, full, atol=atol)


@pytest.mark.parametrize("bloch", [[4, 1, 1], [2, 3, 1], [2, 2, 3]])
def test_bloch_indices_segments(bloch):
    """Two segment calculations patched together equals the full unfolding"""
    H = get_H()
    b = Bloch(bloch)
    K = [0.2, 0.13, -0.41]
    k_unfold = b.unfold_points(K)
    HK = np.stack([H.Hk(k, format="array") for k in k_unfold])
    full = b.unfold(HK, k_unfold)

    n = full.shape[0]
    S1 = np.arange(n // 2)
    S2 = np.arange(n // 2, n)
    patched = np.empty_like(full)
    for r in (S1, S2):
        for c in (S1, S2):
            patched[np.ix_(r, c)] = b.unfold(HK, k_unfold, indices=(r, c))
    assert np.allclose(patched, full)


def test_bloch_indices_rim():
    """Only the outer rim (2 unit-cells wide) of a 5x5 Bloch expansion"""
    H = get_H()
    n = H.no
    b = Bloch([5, 5, 1])
    K = [0.1, 0.3, 0.0]
    full = b(H.Hk, K, format="array")

    cell = np.arange(25)
    i0, i1 = cell % 5, cell // 5
    rim = cell[(np.minimum(i0, 4 - i0) < 2) | (np.minimum(i1, 4 - i1) < 2)]
    assert len(rim) == 24
    indices = (rim.reshape(-1, 1) * n + np.arange(n)).ravel()
    G = b(H.Hk, K, format="array", indices=indices)
    assert np.allclose(G, full[np.ix_(indices, indices)])


def test_bloch_indices_call():
    H = get_H()
    b = Bloch([3, 1, 2])
    K = [0.1, 0.2, 0.3]
    full = b(H.Hk, K, format="array")
    rows = np.array([0, 5, H.no + 1, -1, 3 * H.no + 2])
    cols = np.arange(0, full.shape[1], 7)
    G = b(H.Hk, K, format="array", indices=(rows, cols))
    assert np.allclose(G, full[np.ix_(rows, cols)])
    G = b(H.Hk, K, format="array", indices=rows)
    assert np.allclose(G, full[np.ix_(rows, rows)])


def test_bloch_indices_errors():
    b = Bloch([3, 1, 2])
    k = b.unfold_points([0] * 3)
    M = np.zeros([len(b), 2, 3], dtype=np.complex128)
    # non-square requires (rows, cols)
    with pytest.raises(ValueError):
        b.unfold(M, k, indices=[0])
    with pytest.raises(ValueError):
        b.unfold(M, k, indices=([0], [0], [0]))
    with pytest.raises(ValueError):
        b.unfold(M, k, indices=([len(b) * 2], [0]))
    with pytest.raises(ValueError):
        b.unfold(M, k, indices=([0], [-len(b) * 3 - 1]))
