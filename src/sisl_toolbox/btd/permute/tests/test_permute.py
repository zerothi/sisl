# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.sparse as sps

import sisl
from sisl_toolbox.btd import PermuteBTD, btd_partition
from sisl_toolbox.btd.permute import (
    bandwidth,
    btd_blocks,
    cuthill_mckee,
    electrode_ordering,
    generalized_gibbs_poole_stockmeyer,
    gibbs_poole_stockmeyer,
    reduce_bandwidth,
    refine_bandwidth,
    symmetric_graph,
)

pytestmark = [pytest.mark.btd]

methods = ["cm", "rcm", "gps", "ggps"]


def _graph(A, dtype=np.int32):
    A = sps.csr_array(A)
    return symmetric_graph(A.indptr.astype(dtype), A.indices.astype(dtype))


def _is_perm(perm, n):
    return np.array_equal(np.sort(perm), np.arange(n))


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_symmetric_graph(dtype):
    A = sps.random(50, 50, density=0.1, random_state=1, format="csr") + sps.eye(50)
    ptr, col = _graph(A, dtype)
    assert ptr.dtype == dtype
    assert col.dtype == dtype
    G = sps.csr_array((np.ones(len(col)), col, ptr), shape=A.shape)
    S = (A + A.T).astype(bool).astype(float)
    S.setdiag(0)
    S.eliminate_zeros()
    # unique entries, symmetric, no diagonal
    assert G.max() == 1
    assert (G.astype(bool) != S.astype(bool)).nnz == 0


@pytest.mark.parametrize("func", [cuthill_mckee, gibbs_poole_stockmeyer])
@pytest.mark.parametrize("seed", range(5))
def test_scrambled_banded(func, seed):
    n = 200
    rng = np.random.default_rng(seed)
    B = sps.diags([1] * 5, range(-2, 3), shape=(n, n), format="csr")
    p = rng.permutation(n)
    ptr, col = _graph(B[p][:, p])
    assert bandwidth(ptr, col) > 2
    perm = func(ptr, col)
    assert _is_perm(perm, n)
    assert bandwidth(ptr, col, perm) == 2


@pytest.mark.parametrize(
    "func", [cuthill_mckee, gibbs_poole_stockmeyer, generalized_gibbs_poole_stockmeyer]
)
def test_disconnected_and_isolated(func):
    # two chains and two isolated nodes
    A = sps.lil_matrix((10, 10))
    for i, j in [(0, 5), (5, 2), (2, 7), (1, 8), (8, 4)]:
        A[i, j] = 1
    ptr, col = _graph(A)
    perm = func(ptr, col)
    assert _is_perm(perm, 10)
    assert bandwidth(ptr, col, perm) == 1


def test_empty():
    ptr = np.zeros(1, dtype=np.int32)
    col = np.zeros(0, dtype=np.int32)
    assert len(cuthill_mckee(ptr, col)) == 0
    assert len(gibbs_poole_stockmeyer(ptr, col)) == 0
    assert len(generalized_gibbs_poole_stockmeyer(ptr, col)) == 0


@pytest.mark.parametrize("max_roots", [None, 0, 3])
def test_generalized_gibbs_poole_stockmeyer(max_roots):
    # triangulated grid, scrambled
    nx, ny = 12, 20
    idx = np.arange(nx * ny).reshape(nx, ny)
    i = np.concatenate([idx[:-1].ravel(), idx[:, :-1].ravel(), idx[:-1, :-1].ravel()])
    j = np.concatenate([idx[1:].ravel(), idx[:, 1:].ravel(), idx[1:, 1:].ravel()])
    p = np.random.default_rng(0).permutation(nx * ny)
    A = sps.csr_array((np.ones(len(i)), (p[i], p[j])), shape=(nx * ny, nx * ny))
    ptr, col = _graph(A)
    perm = generalized_gibbs_poole_stockmeyer(ptr, col, max_roots=max_roots)
    assert _is_perm(perm, nx * ny)
    # level structures of width nx + 1 at most yields bandwidth <= 2 nx + 1
    assert bandwidth(ptr, col, perm) <= 2 * nx + 1


@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("method", methods)
def test_reduce_bandwidth_supercell(method, refine):
    g = sisl.geom.graphene().tile(10, 0).tile(8, 1)
    g = g.sub(np.random.default_rng(0).permutation(g.na))
    H = sisl.Hamiltonian(g)
    H.construct([(0.1, 1.44), (0, -2.7)])
    csr = H._csr
    # columns extend into the supercell
    assert csr.shape[1] > csr.shape[0]

    perm = reduce_bandwidth(csr, method, refine=refine)
    assert _is_perm(perm, g.no)

    # compare against the folded pattern
    A = H.tocsr()
    A = sps.csr_array((A.data, A.indices % g.no, A.indptr), shape=(g.no, g.no))
    ptr, col = _graph(A)
    assert bandwidth(ptr, col, perm) < bandwidth(ptr, col) // 2


def _supercell_hamiltonian():
    g = sisl.geom.graphene().tile(6, 0).tile(5, 1)
    g = g.sub(np.random.default_rng(0).permutation(g.na))
    H = sisl.Hamiltonian(g)
    H.construct([(0.1, 1.44), (0, -2.7)])
    return H


@pytest.mark.parametrize(
    "convert",
    [sps.csr_array, sps.coo_array, sps.csc_array, sps.csr_matrix],
    ids=["csr_array", "coo_array", "csc_array", "csr_matrix"],
)
def test_reduce_bandwidth_scipy(convert):
    H = _supercell_hamiltonian()
    A = H.tocsr()
    # columns extend into the supercell
    assert A.shape[1] > A.shape[0]
    perm = reduce_bandwidth(convert(A), "gps")
    assert np.array_equal(perm, reduce_bandwidth(H._csr, "gps"))


def _csr_index_dtype(A, dtype):
    # scipy down-casts the indices on construction, so assign them afterwards
    A = sps.csr_array(A)
    A.indices = A.indices.astype(dtype)
    A.indptr = A.indptr.astype(dtype)
    return A


@pytest.mark.parametrize("dtype", [np.int64])
@pytest.mark.parametrize("method", ["cm", "gps", "ggps"])
def test_reduce_bandwidth_dtype(method, dtype):
    H = _supercell_hamiltonian()
    A = _csr_index_dtype(H.tocsr(), dtype)
    assert A.indices.dtype == dtype
    perm = reduce_bandwidth(A, method)
    assert perm.dtype == dtype
    assert np.array_equal(perm, reduce_bandwidth(H._csr, method))


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_reduce_bandwidth_electrode_dtype(dtype):
    H = _supercell_hamiltonian()
    A = _csr_index_dtype(H.tocsr(), dtype)
    x = H.geometry.xyz[:, 0]
    L = np.flatnonzero(x < x.min() + 1)
    R = np.flatnonzero(x > x.max() - 1)
    perm = reduce_bandwidth(A, "electrode", start=L, end=R)
    assert perm.dtype == dtype
    assert np.array_equal(perm, reduce_bandwidth(H._csr, "electrode", start=L, end=R))


@pytest.mark.parametrize("func", [cuthill_mckee, gibbs_poole_stockmeyer])
def test_permute_int64_identical(func):
    A = sps.random(150, 150, density=0.02, random_state=3)
    ptr32, col32 = _graph(A, np.int32)
    ptr64, col64 = _graph(A, np.int64)
    perm32 = func(ptr32, col32)
    perm64 = func(ptr64, col64)
    assert perm64.dtype == np.int64
    assert np.array_equal(perm32, perm64)
    ref32 = refine_bandwidth(ptr32, col32, perm32)
    ref64 = refine_bandwidth(ptr64, col64, perm64)
    assert ref64.dtype == np.int64
    assert np.array_equal(ref32, ref64)
    for cost in ["speed", "memory", "speed-diagonal"]:
        assert np.array_equal(
            btd_blocks(ptr32, col32, ref32, cost), btd_blocks(ptr64, col64, ref64, cost)
        )


def test_reduce_bandwidth_errors():
    with pytest.raises(TypeError):
        reduce_bandwidth(np.eye(4))
    with pytest.raises(ValueError):
        reduce_bandwidth(sps.csr_array((4, 6)))


def test_reduce_bandwidth_method_error():
    H = sisl.Hamiltonian(sisl.geom.graphene())
    with pytest.raises(ValueError):
        reduce_bandwidth(H._csr, "unknown")


@pytest.mark.parametrize("func", [cuthill_mckee, gibbs_poole_stockmeyer])
@pytest.mark.parametrize("seed", range(5))
def test_refine_bandwidth(func, seed):
    A = sps.random(150, 150, density=0.02, random_state=seed)
    ptr, col = _graph(A)
    perm = func(ptr, col)
    ref = refine_bandwidth(ptr, col, perm)
    assert _is_perm(ref, 150)
    assert bandwidth(ptr, col, ref) <= bandwidth(ptr, col, perm)
    # the input is not modified
    assert np.array_equal(perm, func(ptr, col))


def test_refine_bandwidth_improves():
    # a path graph numbered with one long edge
    n = 10
    ptr, col = _chain(n)
    perm = np.r_[1 : n - 1, 0, n - 1]
    assert bandwidth(ptr, col, perm) == n - 2
    assert bandwidth(ptr, col, refine_bandwidth(ptr, col, perm)) < n - 2


def test_refine_bandwidth_error():
    ptr, col = _chain(4)
    with pytest.raises(ValueError):
        refine_bandwidth(ptr, col, [0, 1, 1, 3])


def _chain(n):
    return _graph(sps.diags([1], [1], shape=(n, n)))


@pytest.mark.parametrize(
    "start,end",
    [([0], [19]), ([5], [15]), ([5, 6], [14, 15]), ([3, 4], None)],
)
def test_electrode_ordering_chain(start, end):
    ptr, col = _chain(20)
    perm = electrode_ordering(ptr, col, start, end)
    assert _is_perm(perm, 20)
    if end is None:
        # both sides of the electrode share the levels
        assert bandwidth(ptr, col, perm) == 2
    else:
        assert bandwidth(ptr, col, perm) == 1


@pytest.mark.parametrize("refine", [False, True])
def test_electrode_ordering_device(refine):
    # device along x with a detached flake
    dev = sisl.geom.graphene(orthogonal=True).tile(10, 0).tile(3, 1)
    dev = dev.sub(np.random.default_rng(1).permutation(dev.na))
    flake = sisl.geom.graphene().tile(2, 0).tile(2, 1).translate([0, 0, 50])
    g = dev.add(flake)
    g.set_nsc([1, 3, 1])
    H = sisl.Hamiltonian(g)
    H.construct([(0.1, 1.44), (0, -2.7)])
    x = g.xyz[: dev.na, 0]
    L = np.flatnonzero(x < x.min() + 2)
    R = np.flatnonzero(x > x.max() - 2)

    perm = reduce_bandwidth(H._csr, "electrode", start=L, end=R, refine=refine)
    assert _is_perm(perm, g.no)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(g.no)
    # electrodes at the ends of the device, the flake after
    assert np.array_equal(np.sort(inv[L]), np.arange(len(L)))
    assert np.array_equal(np.sort(inv[R]), np.arange(dev.na - len(R), dev.na))
    assert inv[dev.na :].min() == dev.na


def test_electrode_ordering_errors():
    ptr, col = _chain(10)
    with pytest.raises(ValueError):
        electrode_ordering(ptr, col, [1, 2], [2, 3])
    with pytest.raises(ValueError):
        electrode_ordering(ptr, col, [])
    with pytest.raises(ValueError):
        electrode_ordering(ptr, col, [10])
    # disconnected electrodes
    ptr, col = _graph(sps.block_diag([sps.diags([1], [1], shape=(5, 5))] * 2))
    with pytest.raises(ValueError):
        electrode_ordering(ptr, col, [0], [9])


def _is_btd(ptr, col, perm, blocks):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    b = np.repeat(np.arange(len(blocks)), blocks)
    row = np.repeat(np.arange(len(perm)), np.diff(ptr))
    return np.abs(b[inv[row]] - b[inv[col]]).max(initial=0) <= 1


def _btd_cost(blocks, cost):
    b = np.asarray(blocks, dtype=float)
    if cost == "memory":
        return (b**2).sum() + 2 * (b[:-1] * b[1:]).sum()
    if cost == "speed":
        return (b**3).sum() + (b[:-1] ** 2 * b[1:] + b[:-1] * b[1:] ** 2).sum()
    return (b**3).sum()


@pytest.mark.parametrize("cost", ["speed", "memory", "speed-diagonal"])
@pytest.mark.parametrize("seed", range(20))
def test_btd_blocks_optimal(seed, cost):
    # compare against all partitions
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 11))
    ptr, col = _graph(sps.random(n, n, density=0.25, random_state=seed))
    perm = rng.permutation(n).astype(np.int32)
    blocks = btd_blocks(ptr, col, perm, cost)
    assert blocks.sum() == n
    assert _is_btd(ptr, col, perm, blocks)

    best = np.inf
    for cuts in itertools.product([False, True], repeat=n - 1):
        b = np.diff(np.r_[0, np.flatnonzero(cuts) + 1, n])
        if _is_btd(ptr, col, perm, b):
            best = min(best, _btd_cost(b, cost))
    assert _btd_cost(blocks, cost) == best


def test_btd_blocks_chain():
    # a chain is tri-diagonal, i.e. blocks of 1
    ptr, col = _chain(10)
    assert np.array_equal(btd_blocks(ptr, col, np.arange(10)), np.ones(10))


def test_btd_blocks_error():
    ptr, col = _chain(4)
    with pytest.raises(ValueError):
        btd_blocks(ptr, col, [0, 1, 1, 3])
    with pytest.raises(ValueError):
        btd_blocks(ptr, col, [0, 1, 2, 3], "unknown")


@pytest.mark.parametrize("csr", [True, False])
@pytest.mark.parametrize("cost", ["speed", "memory", "speed-diagonal"])
@pytest.mark.parametrize("method", methods)
def test_btd_partition(method, cost, csr):

    g = sisl.geom.graphene(orthogonal=True).tile(12, 0).tile(3, 1)
    g = g.sub(np.random.default_rng(0).permutation(g.na))
    g.set_nsc([1, 3, 1])
    H = sisl.Hamiltonian(g)
    H.construct([(0.1, 1.44), (0, -2.7)])

    permbtd = btd_partition(H._csr if csr else H, method, cost)
    assert _is_perm(permbtd.permute, g.no)
    assert permbtd.blocks.sum() == g.no
    A = H.tocsr()
    A = sps.csr_array((A.data, A.indices % g.no, A.indptr), shape=(g.no, g.no))
    ptr, col = _graph(A)
    assert _is_btd(ptr, col, permbtd.permute, permbtd.blocks)
    # sorted within each block
    for p in np.split(permbtd.permute, np.cumsum(permbtd.blocks)[:-1]):
        assert np.all(np.diff(p) > 0)
    # a ribbon along x has many small blocks
    assert len(permbtd.blocks) > 5


def test_permute_btd_validation():
    p = PermuteBTD([2, 0, 1], [1, 2])
    assert isinstance(p.permute, np.ndarray)
    assert isinstance(p.blocks, np.ndarray)
    with pytest.raises(ValueError):
        PermuteBTD([2, 0, 1], [1, 1])
    with pytest.raises(ValueError):
        PermuteBTD([[2, 0, 1]], [3])


def test_permute_btd_split_blocks():
    p = PermuteBTD([[2], [0, 1, 3]])
    assert np.array_equal(p.permute, [2, 0, 1, 3])
    assert np.array_equal(p.blocks, [1, 3])

    ref = PermuteBTD([2, 0, 1, 3], [1, 3])
    blocks = np.split(ref.permute, np.cumsum(ref.blocks)[:-1])
    p = PermuteBTD(blocks)
    assert np.array_equal(p.permute, ref.permute)
    assert np.array_equal(p.blocks, ref.blocks)

    with pytest.raises(ValueError):
        PermuteBTD([])
    with pytest.raises(ValueError):
        PermuteBTD([[[2, 0]], [1]])
