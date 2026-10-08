# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Self

import numpy as np
import numpy.typing as npt
import scipy.sparse as ssp

from sisl import SparseCSR
from sisl._array import array_arange
from sisl._core.sparse_geometry import _SparseGeometry
from sisl._internal import set_module

from .permute import _reduce_bandwidth, btd_blocks, csr_graph

__all__ = ["BlockMatrixIndexer", "BlockMatrix", "PermuteBTD", "btd_partition"]


@set_module("sisl_toolbox.btd")
@dataclass
class PermuteBTD:
    """Permutation table and block sizes of a block-tridiagonal matrix

    Parameters
    ----------
    permute :
        the permutation table, ``permute[new] = old``.
        If `blocks` is not passed, this should be a sequence of the permutation
        indices for each block (e.g. ``np.split(permute, np.cumsum(blocks)[:-1])``),
        the permutation table is then their concatenation, and the block sizes
        are their lengths.
    blocks :
        the block sizes, ``blocks.sum() == len(permute)``

    Examples
    --------
    The two forms are equivalent:

    >>> PermuteBTD([2, 0, 1, 3], [1, 3])
    >>> PermuteBTD([[2], [0, 1, 3]])
    """

    permute: np.ndarray
    blocks: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.blocks is None:
            permute = [np.asarray(p) for p in self.permute]
            if len(permute) == 0 or any(p.ndim != 1 for p in permute):
                raise ValueError(
                    f"{self.__class__.__name__} permute must be a non-empty "
                    "sequence of 1D arrays when blocks is not passed"
                )
            self.blocks = np.array([len(p) for p in permute])
            self.permute = np.concatenate(permute)
        self.permute = np.asarray(self.permute)
        self.blocks = np.asarray(self.blocks)
        if self.permute.ndim != 1 or self.blocks.ndim != 1:
            raise ValueError(
                f"{self.__class__.__name__} permute and blocks must be 1D arrays"
            )
        if self.blocks.sum() != len(self.permute):
            raise ValueError(
                f"{self.__class__.__name__} blocks must sum to the length of permute "
                f"({self.blocks.sum()} != {len(self.permute)})"
            )


def _graph(
    csr: SparseCSR | _SparseGeometry,
    dense: Optional[Sequence[npt.ArrayLike]] = None,
) -> ssp.csr_matrix:
    """Symmetric graph of `csr` (without the diagonal), with `dense` blocks added

    Each array in `dense` are rows that are fully coupled (a clique).
    """
    if isinstance(csr, _SparseGeometry):
        csr = csr._csr
    ptr, col = csr_graph(csr)
    n = len(ptr) - 1
    graph = ssp.csr_matrix((np.ones(len(col), dtype=np.int8), col, ptr), shape=(n, n))
    return _add_dense(graph, dense)


def _add_dense(
    graph: ssp.csr_matrix, dense: Optional[Sequence[npt.ArrayLike]] = None
) -> ssp.csr_matrix:
    """Add the fully coupled rows (cliques) in `dense` to `graph` (without the diagonal)"""
    if dense:
        n = graph.shape[0]
        rows = []
        cols = []
        for idx in dense:
            idx = np.unique(np.asarray(idx).ravel())
            rows.append(np.repeat(idx, len(idx)))
            cols.append(np.tile(idx, len(idx)))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        # remove the diagonal
        off = rows != cols
        rows, cols = rows[off], cols[off]
        graph = graph + ssp.csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
        )
        # only the pattern is used
        graph.data[:] = 1
    graph = graph.tocsr()
    graph.sort_indices()
    return graph


def _subgraph(graph: ssp.csr_matrix, idx: np.ndarray) -> ssp.csr_matrix:
    """The graph of the nodes `idx` (in that order)"""
    graph = graph[idx][:, idx].tocsr()
    graph.sort_indices()
    return graph


def _graph_ptr_col(graph: ssp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """``(ptr, col)`` arrays of `graph` as required by the `permute` routines"""
    ptr = np.ascontiguousarray(graph.indptr, dtype=np.int32)
    col = np.ascontiguousarray(graph.indices, dtype=np.int32)
    return ptr, col


@set_module("sisl_toolbox.btd")
def btd_partition(
    csr: SparseCSR | _SparseGeometry,
    method: str = "gps",
    cost: str = "speed",
    refine: bool = True,
    dense: Optional[Sequence[npt.ArrayLike]] = None,
    **kwargs,
) -> PermuteBTD:
    """Permutation table and block sizes of the block-tridiagonal matrix of minimal cost

    The permutation table reduces the bandwidth of `csr` (see
    `~sisl_toolbox.btd.permute.reduce_bandwidth`), and the permuted matrix is partitioned
    into the block-tridiagonal matrix that minimizes the `cost` model,
    see `~sisl_toolbox.btd.permute.btd_blocks`.
    The indices within each block are sorted, which improves memory access
    when assembling and post-processing the blocks.

    Parameters
    ----------
    csr :
        the sparse matrix, columns may extend into supercells.
        For a sparse geometry object (e.g. `~sisl.Hamiltonian`) its sparse
        pattern is used.
    method :
        the bandwidth reduction method, see `~sisl_toolbox.btd.permute.reduce_bandwidth`
    cost : {"speed", "memory", "speed-diagonal"}
        the cost model of the block-tridiagonal matrix, see `~sisl_toolbox.btd.permute.btd_blocks`
    refine :
        refine the permutation table, see `~sisl_toolbox.btd.permute.reduce_bandwidth`
    dense :
        rows of `csr` that are fully coupled, one array per dense block.
        E.g. the orbitals of an electrode self-energy.
        The dense blocks are added to the graph, so each dense block is
        guaranteed to span at most 2 neighboring blocks.
    **kwargs :
        passed to the routine of `method`

    Returns
    -------
    PermuteBTD : holding both the block-sizes and the permutation table.
    """
    ptr, col = _graph_ptr_col(_graph(csr, dense))
    perm = _reduce_bandwidth(ptr, col, method, refine, **kwargs)
    blocks = btd_blocks(ptr, col, perm, cost)
    # sorting within a block retains the block-tridiagonal structure
    # but speeds up memory access by limiting permutation.
    perm = perm[np.lexsort((perm, np.repeat(np.arange(len(blocks)), blocks)))]
    return PermuteBTD(perm, blocks)


def _btd_blocks_sub(
    csr: SparseCSR | _SparseGeometry,
    permute: npt.ArrayLike,
    cost: str = "speed",
    dense: Optional[Sequence[npt.ArrayLike]] = None,
) -> np.ndarray:
    """Block sizes of the block-tridiagonal matrix of a sub-set of `csr` in the order of `permute`

    Contrary to `~sisl_toolbox.btd.permute.btd_blocks`, `permute` need not
    contain all rows of `csr`, only the rows in `permute` are partitioned.

    Parameters
    ----------
    csr :
        the sparse matrix, columns may extend into supercells.
        For a sparse geometry object (e.g. `~sisl.Hamiltonian`) its sparse
        pattern is used.
    permute :
        the rows of `csr` in the order of the block-tridiagonal matrix
    cost : {"speed", "memory", "speed-diagonal"}
        the cost model of the block-tridiagonal matrix, see `~sisl_toolbox.btd.permute.btd_blocks`
    dense :
        rows of `csr` that are fully coupled, see `btd_partition`

    Returns
    -------
    blocks : the block sizes, ``blocks.sum() == len(permute)``
    """
    permute = np.asarray(permute).ravel()
    # the graph in the order of `permute`, it remains symmetric without the diagonal
    ptr, col = _graph_ptr_col(_subgraph(_graph(csr, dense), permute))
    return btd_blocks(ptr, col, np.arange(len(permute)), cost)


def _device_permute(
    graph: ssp.csr_matrix,
    orbs: np.ndarray,
    dense: Sequence[np.ndarray],
    method: str = "gps",
    refine: bool = True,
) -> np.ndarray:
    """Bandwidth reducing permutation table of the sub-set `orbs` of `graph`

    Parameters
    ----------
    graph :
        the (symmetric) graph of all orbitals, see `_graph`
    orbs :
        the orbitals of the device region
    dense :
        orbitals (in `graph`) that are fully coupled, e.g. the electrode
        self-energies, all must be in `orbs`.
    method :
        the bandwidth reduction method, see `~sisl_toolbox.btd.permute.reduce_bandwidth`.
        For ``"electrode"`` the (at most) 2 `dense` blocks are placed at
        the start and end.
    refine :
        refine the permutation table

    Returns
    -------
    permute : orbitals in `graph` (``orbs[...]``)
    """
    orbs = np.asarray(orbs).ravel()
    local = np.full(graph.shape[0], -1, dtype=np.int64)
    local[orbs] = np.arange(len(orbs))
    dense = [local[d] for d in dense]
    assert all((d >= 0).all() for d in dense)
    ptr, col = _graph_ptr_col(_add_dense(_subgraph(graph, orbs), dense))
    kwargs = {}
    if method.lower() == "electrode":
        if len(dense) > 2:
            raise ValueError("method='electrode' requires at most 2 electrodes")
        kwargs = dict(zip(("start", "end"), dense))
    perm = _reduce_bandwidth(ptr, col, method, refine, **kwargs)
    return orbs[perm]


def _downfold_permute(
    graph: ssp.csr_matrix,
    geometry,
    elec_atoms: np.ndarray,
    device_atoms: np.ndarray,
    exclude_atoms: np.ndarray,
    cost: str = "speed",
    refine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Downfolding region of an electrode, its permutation table and BTD blocks

    The downfolding region are the atoms connected to the electrode,
    which are not in the device region, nor `exclude_atoms`.
    The last block are the orbitals of the device region that couples to
    the downfolding region.
    The downfolding region is ordered by a bandwidth reduction starting
    from the electrode, the device orbitals are placed last.

    Parameters
    ----------
    graph :
        the (symmetric) graph of all orbitals in `geometry`, see `_graph`
    geometry :
        the geometry of the full system
    elec_atoms :
        atoms of the electrode
    device_atoms :
        atoms of the device region
    exclude_atoms :
        atoms that are not part of the downfolding (e.g. buffer
        atoms and other electrodes).
        A downfolding region coupling to these raises an error.
    cost :
        the cost model of the block-tridiagonal matrix
    refine :
        refine the permutation table

    Returns
    -------
    orbs_down : the orbitals of the downfolding region (in BTD order)
    blocks_down : the BTD blocks of `orbs_down`, the last block are device orbitals
    """
    na = geometry.na
    elec_atoms = np.asarray(elec_atoms).ravel()

    # atomic graph
    coo = graph.tocoo()
    agraph = ssp.csr_matrix(
        (
            np.ones(coo.nnz, dtype=np.int8),
            (geometry.o2a(coo.row), geometry.o2a(coo.col)),
        ),
        shape=(na, na),
    )

    is_device = np.zeros(na, dtype=bool)
    is_device[device_atoms] = True
    is_exclude = np.zeros(na, dtype=bool)
    is_exclude[exclude_atoms] = True
    if (is_device[elec_atoms] | is_exclude[elec_atoms]).any():
        raise ValueError("electrode atoms overlap with the device or excluded atoms")

    # breadth first search through the atoms that are not device atoms
    region = np.zeros(na, dtype=bool)
    region[elec_atoms] = True
    front = elec_atoms
    while len(front) > 0:
        neighbors = np.unique(agraph[front].indices)
        if is_exclude[neighbors].any():
            raise ValueError(
                "downfolding region couples to excluded atoms "
                f"{np.flatnonzero(is_exclude & np.isin(np.arange(na), neighbors)).tolist()}"
            )
        front = neighbors[~(is_device[neighbors] | region[neighbors])]
        region[front] = True

    orbs_region = geometry.a2o(np.flatnonzero(region), all=True)
    orbs_elec = geometry.a2o(elec_atoms, all=True)

    # device orbitals coupling to the downfolding region
    is_device_orb = np.zeros(geometry.no, dtype=bool)
    is_device_orb[geometry.a2o(device_atoms, all=True)] = True
    orbs_dev = np.unique(graph[orbs_region].indices)
    orbs_dev = orbs_dev[is_device_orb[orbs_dev]]
    if len(orbs_dev) == 0:
        raise ValueError("downfolding region does not couple to the device region")

    # the local graph, the electrode and the device orbitals are dense
    nr = len(orbs_region)
    nodes = np.concatenate([orbs_region, orbs_dev])
    elec_local = np.searchsorted(orbs_region, orbs_elec)
    dev_local = np.arange(nr, len(nodes))
    sub = _add_dense(_subgraph(graph, nodes), [elec_local, dev_local])
    ptr, col = _graph_ptr_col(sub)
    perm = _reduce_bandwidth(
        ptr, col, "electrode", refine, start=elec_local, end=dev_local
    )
    # ensure the device orbitals are last
    is_dev = perm >= nr
    perm = np.concatenate([perm[~is_dev], perm[is_dev]])

    # blocks of the downfolding region (without the device orbitals)
    perm_r = perm[:nr]
    ptr, col = _graph_ptr_col(_subgraph(sub, perm_r))
    blocks = btd_blocks(ptr, col, np.arange(nr), cost)

    # only the last block may couple to the device orbitals
    couples = np.flatnonzero(np.diff(sub[perm_r][:, dev_local].tocsr().indptr))
    first = np.searchsorted(np.cumsum(blocks), couples[0], side="right")
    blocks = np.concatenate([blocks[:first], [blocks[first:].sum()], [len(orbs_dev)]])

    return nodes[perm], blocks


def _blocks_in_range(blocks_cum0: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Blocks spanned by the range ``[indices.min(), indices.max()]``

    Parameters
    ----------
    blocks_cum0 :
        cumulative sum of the block sizes, starting with 0
    indices :
        matrix indices
    """
    block1 = (indices.min() < blocks_cum0[1:]).nonzero()[0][0]
    block2 = (indices.max() < blocks_cum0[1:]).nonzero()[0][0]
    return np.arange(block1, block2 + 1)


@set_module("sisl_toolbox.btd")
class BlockMatrixIndexer:
    def __init__(self, bm: BlockMatrix):
        self._bm = bm

    def __len__(self) -> int:
        """Number of blocks in this block matrix"""
        return len(self._bm.blocks)

    def __iter__(self):
        """Loop contained indices in the BlockMatrix"""
        yield from self._bm._M.keys()

    def _assert_key(self, key) -> None:
        if not isinstance(key, tuple):
            raise ValueError(
                f"{self.__class__.__name__} index deletion must be done with a tuple."
            )

    def __delitem__(self, key) -> None:
        self._assert_key(key)
        del self._bm._M[key]

    def __contains__(self, key) -> bool:
        self._assert_key(key)
        return key in self._bm._M

    def __getitem__(self, key):
        self._assert_key(key)
        M = self._bm._M.get(key)
        if M is None:
            i, j = key
            # the data-type is probably incorrect.. :(
            return np.zeros([self._bm.blocks[i], self._bm.blocks[j]])
        return M

    def __setitem__(self, key, M) -> None:
        self._assert_key(key)
        # Check that the shapes coincide for the block specified
        s = (self._bm.blocks[key[0]], self._bm.blocks[key[1]])
        assert (
            M.shape == s
        ), f"Could not assign matrix of shape {M.shape} into matrix of shape {s}"
        self._bm._M[key] = M

    def __getattr__(self, attr):
        return getattr(self._bm, attr)


@set_module("sisl_toolbox.btd")
class BlockMatrix:
    """Container class that holds a block matrix.

    A block matrix will always be a square matrix.
    """

    def __init__(self, blocks: npt.ArrayLike):
        self._M = {}
        self._blocks = np.asarray(blocks, dtype=int)
        self._blocks_cum0 = np.empty([len(self.blocks) + 1], dtype=self.blocks.dtype)
        self._blocks_cum0[0] = 0
        self._blocks_cum0[1:] = np.cumsum(self.blocks)
        self._index_blocks = np.arange(len(self))
        # Correct the blocks
        min_index = 0
        for block, max_index in enumerate(self._blocks_cum0[1:]):
            self._index_blocks[min_index:max_index] = block
            min_index = max_index

    @property
    def blocks(self) -> np.ndarray:
        """Size of the blocks that define this block-matrix"""
        return self._blocks

    def __len__(self) -> int:
        """The length of the first dimension of the matrix"""
        return np.sum(self._blocks)

    @property
    def shape(self) -> tuple:
        n = len(self)
        return (n, n)

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the matrix, promoted from all stored blocks

        Defaults to ``float64`` (the default data-type of numpy arrays)
        when no blocks are stored.
        """
        if self._M:
            return np.result_type(*(M.dtype for M in self._M.values()))
        return np.dtype(np.float64)

    def _get_blocks(self, indices, in_range: bool = False):
        """Returns the blocks that an index belongs to"""
        if in_range:
            return _blocks_in_range(self._blocks_cum0, indices)
        return np.unique(self._index_blocks[indices])

    def _get_block_indices(self, blocks):
        """Convert a list of blocks into a list of indices for each block

        If a block is supplied multiple times, its indices will be repeated.
        """
        b_cum0 = self._blocks_cum0
        indices = array_arange(b_cum0[blocks], b_cum0[1:][blocks])
        # Split into separate lists
        return np.split(indices, np.cumsum(self.blocks[blocks]))

    def _get_block_slices(self, blocks):
        """Convert a list of blocks into a list of indices for each block

        If a block is supplied multiple times, its indices will be repeated.
        """
        bc0 = self._blocks_cum0
        return [slice(bc0[b], bc0[b + 1]) for b in blocks]

    def copy(self) -> Self:
        """Create a copy of this block matrix"""
        new = self.__class__(self.blocks)
        # Copy values
        for k, v in self._M.items():
            new._M[k] = np.copy(v)
        return new

    def asformat(self, format):
        """Convert to a particular format"""
        return getattr(self, f"to{format}")()

    def toarray(self) -> np.ndarray:
        """Convert to a dense matrix, where non-defined blocks will be zero"""
        BI = self.block_indexer
        nb = len(BI)
        # stack stuff together
        return np.concatenate(
            [np.concatenate([BI[i, j] for i in range(nb)], axis=0) for j in range(nb)],
            axis=1,
        )

    def tobtd(self) -> BlockMatrix:
        """Return only the block tridiagonal part of the matrix"""
        ret = self.__class__(self.blocks)
        sBI = self.block_indexer
        rBI = ret.block_indexer
        nb = len(sBI)
        for j in range(nb):
            for i in range(max(0, j - 1), min(j + 2, nb)):
                rBI[i, j] = sBI[i, j]
        return ret

    def tobd(self) -> BlockMatrix:
        """Return only the block diagonal part of the matrix"""
        ret = self.__class__(self.blocks)
        sBI = self.block_indexer
        rBI = ret.block_indexer
        nb = len(sBI)
        for i in range(nb):
            rBI[i, i] = sBI[i, i]
        return ret

    def diagonal(self) -> np.ndarray:
        """Returns the diagonal of the matrix"""
        BI = self.block_indexer
        return np.concatenate([BI[b, b].diagonal() for b in range(len(BI))])

    @property
    def block_indexer(self) -> BlockMatrixIndexer:
        """Get the indexer for this block matrix

        The indexer allows manipulating the matrix using indices of the blocks,
        rather than indices of the matrix elements.
        """
        return BlockMatrixIndexer(self)
