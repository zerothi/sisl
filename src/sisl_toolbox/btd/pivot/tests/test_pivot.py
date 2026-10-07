# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.sparse as sps

import sisl
from sisl_toolbox.btd import btd_partition
from sisl_toolbox.btd.pivot import (
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


def _graph(A):
    A = sps.csr_matrix(A)
    ptr = A.indptr.astype(np.int32)
    return symmetric_graph(
        ptr, np.diff(ptr).astype(np.int32), A.indices.astype(np.int32)
    )


def _is_perm(perm, n):
    return np.array_equal(np.sort(perm), np.arange(n))


def test_symmetric_graph():
    A = sps.random(50, 50, density=0.1, random_state=1, format="csr") + sps.eye(50)
    ptr, col = _graph(A)
    G = sps.csr_matrix((np.ones(len(col)), col, ptr), shape=A.shape)
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


def test_reverse_cuthill_mckee():
    ptr, col = _graph(sps.random(60, 60, density=0.05, random_state=2))
    perm = cuthill_mckee(ptr, col)
    assert np.array_equal(cuthill_mckee(ptr, col, reverse=True), perm[::-1])


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
    A = sps.csr_matrix((np.ones(len(i)), (p[i], p[j])), shape=(nx * ny, nx * ny))
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
    A = sps.csr_matrix((A.data, A.indices % g.no, A.indptr), shape=(g.no, g.no))
    ptr, col = _graph(A)
    assert bandwidth(ptr, col, perm) < bandwidth(ptr, col) // 2


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


@pytest.mark.parametrize("cost", ["speed", "memory", "speed-diagonal"])
@pytest.mark.parametrize("method", methods)
def test_btd_partition(method, cost):

    g = sisl.geom.graphene(orthogonal=True).tile(12, 0).tile(3, 1)
    g = g.sub(np.random.default_rng(0).permutation(g.na))
    g.set_nsc([1, 3, 1])
    H = sisl.Hamiltonian(g)
    H.construct([(0.1, 1.44), (0, -2.7)])

    permbtd = btd_partition(H._csr, method, cost)
    assert _is_perm(permbtd.permute, g.no)
    assert permbtd.blocks.sum() == g.no
    A = H.tocsr()
    A = sps.csr_matrix((A.data, A.indices % g.no, A.indptr), shape=(g.no, g.no))
    ptr, col = _graph(A)
    assert _is_btd(ptr, col, permbtd.permute, permbtd.blocks)
    # sorted within each block
    for p in np.split(permbtd.permute, np.cumsum(permbtd.blocks)[:-1]):
        assert np.all(np.diff(p) > 0)
    # a ribbon along x has many small blocks
    assert len(permbtd.blocks) > 5
