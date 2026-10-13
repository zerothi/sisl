# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import scipy.sparse as ssp

from sisl import SparseCSR
from sisl._array import array_arange
from sisl._core.sparse_geometry import _SparseGeometry
from sisl._internal import set_module

from .permute import _reduce_bandwidth, btd_blocks, csr_graph

__all__ = [
    "BlockArrayIndexer",
    "BlockArray",
    "PermuteBTD",
    "btd_partition",
    "downfold_partition",
]


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
    csr: SparseCSR | _SparseGeometry | ssp.sparray,
    dense: Optional[Sequence[npt.ArrayLike]] = None,
) -> ssp.csr_array:
    """Symmetric graph of `csr` (without the diagonal), with `dense` blocks added

    Each array in `dense` are rows that are fully coupled (a clique).
    """
    if isinstance(csr, _SparseGeometry):
        csr = csr._csr
    ptr, col = csr_graph(csr)
    n = len(ptr) - 1
    graph = ssp.csr_array((np.ones(len(col), dtype=np.int8), col, ptr), shape=(n, n))
    return _add_dense(graph, dense)


def _add_dense(
    graph: ssp.csr_array, dense: Optional[Sequence[npt.ArrayLike]] = None
) -> ssp.csr_array:
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
        graph = graph + ssp.csr_array(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
        )
        # only the pattern is used
        graph.data[:] = 1
    graph = graph.tocsr()
    graph.sort_indices()
    return graph


def _subgraph(graph: ssp.csr_array, idx: np.ndarray) -> ssp.csr_array:
    """The graph of the nodes `idx` (in that order)"""
    graph = graph[idx][:, idx].tocsr()
    graph.sort_indices()
    return graph


def _graph_ptr_col(graph: ssp.csr_array) -> tuple[np.ndarray, np.ndarray]:
    """``(ptr, col)`` arrays of `graph` as required by the `permute` routines"""
    return graph.indptr, graph.indices


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
        The rows are the nodes of the partitioning, i.e. use the atomic
        sparse pattern (``H.toSparseAtom()``) to partition atoms, and ``H``
        to partition orbitals.
    method :
        the bandwidth reduction method, see `~sisl_toolbox.btd.permute.reduce_bandwidth`
    cost : {"speed", "memory", "speed-diagonal"}
        the cost model of the block-tridiagonal matrix, see `~sisl_toolbox.btd.permute.btd_blocks`
    refine :
        refine the permutation table, see `~sisl_toolbox.btd.permute.reduce_bandwidth`
    dense :
        rows of `csr` that are fully coupled, one array per dense block.
        E.g. the nodes (atoms or orbitals) of an electrode self-energy.
        The dense blocks are added to the graph, so each dense block is
        guaranteed to span at most 2 neighboring blocks.
    **kwargs :
        passed to the routine of `method`

    Returns
    -------
    PermuteBTD : holding both the block-sizes and the permutation table.
    """
    ptr, col = _graph_ptr_col(_graph(csr, dense))
    perm = _reduce_bandwidth(ptr, col, method, refine, False, **kwargs)
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
    graph: ssp.csr_array,
    nodes: np.ndarray,
    dense: Sequence[np.ndarray],
    method: str = "gps",
    refine: bool = True,
) -> np.ndarray:
    """Bandwidth reducing permutation table of the sub-set `nodes` of `graph`

    The nodes of `graph` may be atoms or orbitals, all indices are nodes.

    Parameters
    ----------
    graph :
        the (symmetric) graph of all nodes, see `_graph`
    nodes :
        the nodes of the device region
    dense :
        nodes that are fully coupled, e.g. the electrode
        self-energies, all must be in `nodes`.
    method :
        the bandwidth reduction method, see `~sisl_toolbox.btd.permute.reduce_bandwidth`.
        For ``"electrode"`` the (at most) 2 `dense` blocks are placed at
        the start and end.
    refine :
        refine the permutation table

    Returns
    -------
    permute : nodes in `graph` (``nodes[...]``)
    """
    nodes = np.asarray(nodes).ravel()
    local = np.full(graph.shape[0], -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    dense = [local[d] for d in dense]
    assert all((d >= 0).all() for d in dense)
    ptr, col = _graph_ptr_col(_add_dense(_subgraph(graph, nodes), dense))
    kwargs = {}
    if method.lower() == "electrode":
        if len(dense) > 2:
            raise ValueError("method='electrode' requires at most 2 electrodes")
        kwargs = dict(zip(("start", "end"), dense))
    perm = _reduce_bandwidth(ptr, col, method, refine, False, **kwargs)
    return nodes[perm]


def _downfold_permute(
    graph: ssp.csr_array,
    elec: np.ndarray,
    device: np.ndarray,
    exclude: np.ndarray,
    cost: str = "speed",
    refine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Downfolding region of an electrode, its permutation table and BTD blocks

    The downfolding region are the nodes (rows) connected to the electrode,
    which are not in the device region, nor `exclude`.
    The electrode may be partially in the device region, then only the
    electrode nodes outside the device region are downfolded.
    The last block are the nodes of the device region that couples to
    the downfolding region (including the electrode nodes in the device region,
    which couples densely through the electrode self-energy).
    The downfolding region is ordered by a bandwidth reduction starting
    from the electrode, the device nodes are placed last.

    Parameters
    ----------
    graph :
        the (symmetric) graph of all nodes (atoms or orbitals), see `_graph`
    elec :
        nodes of the electrode, at least one must not be in the device region
    device :
        nodes of the device region
    exclude :
        nodes that are not part of the downfolding (e.g. buffer
        and other electrodes).
        A downfolding region coupling to these raises an error.
    cost :
        the cost model of the block-tridiagonal matrix
    refine :
        refine the permutation table

    Returns
    -------
    down : the nodes of the downfolding region (in BTD order)
    blocks : the BTD blocks of `down`, the last block are device nodes
    """
    no = graph.shape[0]
    node_elec = np.unique(np.asarray(elec).ravel())

    is_device = np.zeros(no, dtype=bool)
    is_device[device] = True
    is_exclude = np.zeros(no, dtype=bool)
    is_exclude[exclude] = True
    if is_exclude[node_elec].any():
        raise ValueError("electrode overlaps with the excluded indices")
    # electrode nodes to be downfolded (not in the device region)
    node_elec_r = node_elec[~is_device[node_elec]]
    if len(node_elec_r) == 0:
        raise ValueError("electrode is fully in the device region")

    # breadth first search through the nodes that are not device nodes
    region = np.zeros(no, dtype=bool)
    region[node_elec_r] = True
    front = node_elec_r
    while len(front) > 0:
        neighbors = np.unique(graph[front].indices)
        if is_exclude[neighbors].any():
            raise ValueError(
                "downfolding region couples to excluded indices "
                f"{neighbors[is_exclude[neighbors]].tolist()}"
            )
        front = neighbors[~(is_device[neighbors] | region[neighbors])]
        region[front] = True
    node_region = np.flatnonzero(region)

    # device nodes coupling to the downfolding region, and the electrode
    # nodes in the device region (coupled through the self-energy)
    node_dev = np.unique(graph[node_region].indices)
    node_dev = np.union1d(
        node_dev[is_device[node_dev]], node_elec[is_device[node_elec]]
    )
    if len(node_dev) == 0:
        raise ValueError("downfolding region does not couple to the device region")

    # the local graph, the electrode and the device nodes are dense
    nr = len(node_region)
    nodes = np.concatenate([node_region, node_dev])
    elec_r_local = np.searchsorted(node_region, node_elec_r)
    elec_d_local = nr + np.searchsorted(node_dev, node_elec[is_device[node_elec]])
    elec_local = np.concatenate([elec_r_local, elec_d_local])
    dev_local = np.arange(nr, len(nodes))
    sub = _add_dense(_subgraph(graph, nodes), [elec_local, dev_local])
    ptr, col = _graph_ptr_col(sub)
    perm = _reduce_bandwidth(
        ptr, col, "electrode", refine, False, start=elec_r_local, end=dev_local
    )
    # ensure the device nodes are last
    is_dev = perm >= nr
    perm = np.concatenate([perm[~is_dev], perm[is_dev]])

    # blocks of the downfolding region (without the device nodes)
    perm_r = perm[:nr]
    ptr, col = _graph_ptr_col(_subgraph(sub, perm_r))
    blocks = btd_blocks(ptr, col, np.arange(nr), cost)

    # only the last block may couple to the device nodes.
    # Electrode nodes in the device region couples densely to all electrode
    # nodes, so the downfolded electrode nodes end up in the last block.
    couples = np.flatnonzero(np.diff(sub[perm_r][:, dev_local].tocsr().indptr))
    first = np.searchsorted(np.cumsum(blocks), couples[0], side="right")
    blocks = np.concatenate([blocks[:first], [blocks[first:].sum()], [len(node_dev)]])

    return nodes[perm], blocks


@set_module("sisl_toolbox.btd")
def downfold_partition(
    csr: SparseCSR | _SparseGeometry,
    elecs: Sequence[npt.ArrayLike],
    device: Optional[npt.ArrayLike] = None,
    buffer: Optional[npt.ArrayLike] = None,
    cost: str = "speed",
    refine: bool = True,
) -> list[Optional[PermuteBTD]]:
    """Downfolding regions of the electrodes, their permutation tables and BTD blocks

    The downfolding region of an electrode is the connected sub-graph of `csr`
    containing the electrode rows, which is disconnected from the device region
    (it stops at the device rows).
    The last block are the rows of the device region that couples to
    the downfolding region, i.e. where the downfolded self-energy lives.
    An electrode may be partially in the device region, then only its rows
    outside the device region are downfolded, and its rows in the device
    region are part of the last block.
    The downfolding region is ordered by a bandwidth reduction starting
    from the electrode, the device rows are placed last.

    All indices are rows of `csr`, i.e. use the atomic sparse pattern
    (``H.toSparseAtom()``) with atomic indices, or ``H`` with orbital indices.

    Parameters
    ----------
    csr :
        the sparse matrix of the full system, columns may extend into supercells.
        For a sparse geometry object (e.g. `~sisl.Hamiltonian`) its sparse
        pattern is used.
    elecs :
        rows of each electrode
    device :
        rows of the device region. Electrodes whose rows are all in the
        device region are not downfolded, electrodes partially in the
        device region are partially downfolded.
        Defaults to all rows not in the electrodes nor in `buffer`.
    buffer :
        rows that are removed from the calculation
    cost : {"speed", "memory", "speed-diagonal"}
        the cost model of the block-tridiagonal matrix, see `~sisl_toolbox.btd.permute.btd_blocks`
    refine :
        refine the permutation tables, see `~sisl_toolbox.btd.permute.reduce_bandwidth`

    Returns
    -------
    list :
        for each electrode, a `PermuteBTD` with the rows of the
        downfolding region (including the electrode and device rows),
        or None for electrodes in the device region.

    Raises
    ------
    ValueError
        if a downfolding region couples to the buffer or other (downfolded)
        electrode rows.
    """
    graph = _graph(csr)
    n = graph.shape[0]
    elecs = [np.asarray(idx).ravel() for idx in elecs]
    if buffer is None:
        buffer = np.array([], dtype=np.int64)
    buffer = np.asarray(buffer).ravel()

    if device is None:
        is_device = np.ones(n, dtype=bool)
        is_device[buffer] = False
        for idx in elecs:
            is_device[idx] = False
    else:
        is_device = np.zeros(n, dtype=bool)
        is_device[np.asarray(device).ravel()] = True
    device = np.flatnonzero(is_device)

    # the electrode rows that are downfolded
    elecs_r = [idx[~is_device[idx]] for idx in elecs]

    ret = []
    for i, idx in enumerate(elecs):
        if len(elecs_r[i]) == 0:
            # fully in the device region
            ret.append(None)
            continue
        exclude = np.concatenate(
            [buffer] + [e for j, e in enumerate(elecs_r) if j != i]
        )
        try:
            down, blocks = _downfold_permute(graph, idx, device, exclude, cost, refine)
        except ValueError as e:
            raise ValueError(f"downfold_partition electrode {i}: {e}") from None
        ret.append(PermuteBTD(down, blocks))
    return ret


def _split_pivot(value: str) -> tuple[str, str]:
    """Split a pivoting method ``[atom+|orb+]<method>`` into the space and method

    The space is ``"atoms"`` (default) or ``"orbitals"``, the method is lower-case.
    """
    method = value.strip().lower()
    for prefix, space in (
        ("atom+", "atoms"),
        ("atoms+", "atoms"),
        ("orb+", "orbitals"),
        ("orbs+", "orbitals"),
    ):
        if method.startswith(prefix):
            return space, method[len(prefix) :]
    return "atoms", method


def _permute_a2o(geometry, permute: PermuteBTD) -> PermuteBTD:
    """Convert an atomic permutation table and BTD blocks to orbitals

    Each atom is replaced by its orbitals (in order), and the block sizes
    are the number of orbitals of the atoms in each block.
    """
    atoms = permute.permute
    starts = np.append(0, np.cumsum(permute.blocks)[:-1])
    blocks = np.add.reduceat(geometry.orbitals[atoms], starts)
    return PermuteBTD(geometry.a2o(atoms, all=True), blocks)


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


def _is_int(key) -> bool:
    """Whether `key` is a scalar integer index"""
    return isinstance(key, (int, np.integer)) and not isinstance(key, bool)


def _normalize_axes(axes, ndim: int) -> tuple[int, ...]:
    """Non-negative, bounds checked and unique `axes`"""
    out = tuple(operator.index(a) for a in axes)
    out = tuple(a + ndim if a < 0 else a for a in out)
    if any(a < 0 or a >= ndim for a in out):
        raise ValueError(f"axes {axes} out of bounds for array of dimension {ndim}")
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in {axes}")
    return out


def _as_slice(idx: np.ndarray) -> slice | np.ndarray:
    """`idx` as a slice if it is a contiguous ascending range, else `idx`"""
    n = len(idx)
    if idx[-1] - idx[0] == n - 1 and (np.diff(idx) == 1).all():
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def _ix(*idx):
    """Orthogonal index of slices and 1D arrays"""
    if sum(not isinstance(i, slice) for i in idx) <= 1:
        # a single index array is orthogonal to the slices
        return idx
    return np.ix_(
        *(np.arange(i.start, i.stop) if isinstance(i, slice) else i for i in idx)
    )


def _same_blocks(blocks1, blocks2) -> bool:
    """Whether two sequences of (per axis) block sizes are equal"""
    return len(blocks1) == len(blocks2) and all(
        np.array_equal(b1, b2) for b1, b2 in zip(blocks1, blocks2)
    )


def _densify(obj):
    """Convert all `BlockArray` in the (nested) `obj` to dense arrays"""
    if isinstance(obj, BlockArray):
        return obj.toarray()
    if isinstance(obj, (tuple, list)):
        return type(obj)(_densify(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _densify(v) for k, v in obj.items()}
    return obj


# whether a block is defined (vectorized over the grid of blocks)
_is_defined = np.frompyfunc(lambda M: M is not None, 1, 1)

# flag of contiguous arrays in a memory order
_CONTIGUOUS = {"C": "C_CONTIGUOUS", "F": "F_CONTIGUOUS"}

# ufuncs where a structural zero in any argument yields a structural zero
_ANNIHILATE = (np.multiply, np.logical_and, np.bitwise_and)


@set_module("sisl_toolbox.btd")
class BlockArrayIndexer:
    """Indexer of a `BlockArray` using block indices

    ``BI[i, j, ...]`` is the block of the indices in block ``i`` along the first
    axis, block ``j`` along the second axis, etc.
    Undefined blocks are returned as new zero arrays (which are *not*
    stored in the array).
    Assigning a block stores it (no copy) if it has the data-type and memory order
    of the array, otherwise it is converted.
    """

    __slots__ = ("_ba", "_grid")

    def __init__(self, ba: BlockArray):
        self._ba = ba
        self._grid = ba._grid

    def __len__(self) -> int:
        """Number of blocks along the first axis"""
        return self._grid.shape[0]

    def __iter__(self):
        """Loop the block indices of the defined blocks"""
        for key in zip(*self._ba._defined().nonzero()):
            yield tuple(map(int, key))

    def _key(self, key) -> tuple[int, ...]:
        """`key` as a tuple of block indices (one integer per axis)"""
        if type(key) is not tuple:
            key = (key,)
        if len(key) != self._grid.ndim:
            raise ValueError(
                f"{self.__class__.__name__} indexing must be done with a tuple of "
                f"{self._grid.ndim} block indices."
            )
        return tuple(map(operator.index, key))

    def __delitem__(self, key) -> None:
        key = self._key(key)
        if self._grid[key] is None:
            raise KeyError(key)
        self._grid[key] = None

    def __contains__(self, key) -> bool:
        return self._grid[self._key(key)] is not None

    def _block(self, key: tuple):
        """Block `key` (None if undefined)

        numpy validates the integers, the key is only checked when it
        does not return a single block (e.g. slices or too few indices).
        """
        M = self._grid[key]
        if M is not None and M.dtype.hasobject:
            # a sub-grid of blocks
            self._key(key)
            raise ValueError(
                f"{self.__class__.__name__} indexing must be done with integer block indices."
            )
        return M

    def __getitem__(self, key) -> np.ndarray:
        if type(key) is not tuple:
            key = (key,)
        # the hot path, the checks of _block are inlined
        M = self._grid[key]
        if M is None:
            ba = self._ba
            return np.zeros(ba._block_shape(key), dtype=ba._dtype, order=ba._order)
        if M.dtype.hasobject:
            return self._block(key)
        return M

    def __setitem__(self, key, M) -> None:
        if type(key) is not tuple:
            key = (key,)
        self._block(key)
        self._grid[key] = self._ba._asblock(M, key)

    def __getattr__(self, attr):
        return getattr(self._ba, attr)


@set_module("sisl_toolbox.btd")
class BlockArray(np.ndarray):
    r"""N-dimensional array split into blocks where undefined blocks are (structurally) zero

    Each axis is split into blocks of the sizes in `blocks`, e.g. a
    block-tridiagonal matrix is a 2D array with the same blocks along
    both axes, ``BlockArray([blocks, blocks])``.

    The array is a `numpy.ndarray` subclass whose elements are the blocks.
    The underlying array is a grid of objects (one dimension per axis), each either
    a dense `numpy.ndarray` (a defined block) or `None` (an undefined, zero, block).
    `shape`, `dtype` and `len` refer to the (dense) array, i.e.
    ``shape == tuple(b.sum() for b in blocks)``.

    Blocks are accessed and assigned through `block_indexer` with block indices,
    the array can be extended with new blocks at any time.
    All blocks have the data-type `dtype` and memory order `order`.

    Indexing the array (``ba[i, j, ...]``) uses array indices and follows
    the numpy semantics, returning dense arrays. If the index along each
    axis is a sequence of complete blocks a `BlockArray` of these blocks
    is returned instead (the blocks are shared when indexing with slices,
    otherwise copied).
    Assigning elements allocates the blocks that are needed (unless the
    assigned values are zero).

    Element-wise ufuncs (``-ba``, ``ba * 2``, ``ba + ba``, ``np.conj(ba)``, ...)
    and matrix multiplications (``ba @ ba``, and ``ba @ x``, ``x @ ba`` for 2D arrays)
    only operate on the defined blocks, when the undefined blocks remain zero.
    The undefined blocks are exact zeros, so `np.multiply` only stores the
    blocks defined in all arguments (like `scipy.sparse`).
    Any other operation returns the result of the dense array (`toarray`),
    e.g. ``ba + 1``.

    Parameters
    ----------
    blocks :
        the size of the blocks along each axis (one 1D array per axis)
    dtype :
        data-type of the array, all blocks are converted to this data-type
    order : {"C", "F"}
        memory order of the blocks

    Notes
    -----
    `numpy.asarray` returns the grid of blocks, use `toarray` for the dense array.

    Examples
    --------
    >>> ba = BlockArray([[2, 3], [2, 3]], dtype=np.float64)
    >>> ba.block_indexer[1, 1] = np.ones([3, 3])
    >>> ba.shape
    (5, 5)
    >>> full = np.zeros(ba.shape, dtype=ba.dtype)
    >>> full[ba.indices()] = ba[ba.indices()]
    >>> np.allclose(full, ba.toarray())
    True
    """

    def __new__(
        cls,
        blocks: Sequence[npt.ArrayLike],
        dtype: npt.DTypeLike = np.float64,
        order: Literal["C", "F"] = "C",
    ):
        blocks = tuple(np.asarray(b, dtype=np.intp) for b in blocks)
        if len(blocks) == 0 or any(b.ndim != 1 or (b < 0).any() for b in blocks):
            raise ValueError(
                f"{cls.__name__} blocks must be a sequence (one per axis) of 1D "
                "arrays of non-negative sizes"
            )
        order = order.upper()
        if order not in ("C", "F"):
            raise ValueError(f"{cls.__name__} order must be 'C' or 'F'")
        # np.empty fills object arrays with None
        obj = np.empty([len(b) for b in blocks], dtype=object).view(cls)
        obj._set_meta(blocks, np.dtype(dtype), order)
        return obj

    def _set_meta(self, blocks: tuple, dtype: np.dtype, order: str) -> None:
        self._blocks = blocks
        self._blocks_cum0 = tuple(np.concatenate(([0], np.cumsum(b))) for b in blocks)
        # block of each index (per axis)
        self._index_blocks = tuple(np.repeat(np.arange(len(b)), b) for b in blocks)
        self._shape = tuple(int(c[-1]) for c in self._blocks_cum0)
        # block sizes as Python ints (fast scalar look-ups)
        self._block_sizes = tuple(b.tolist() for b in blocks)
        self._dtype = dtype
        self._order = order

    def __array_finalize__(self, obj) -> None:
        if obj is None or not hasattr(obj, "_blocks"):
            return
        self._set_meta(obj._blocks, obj._dtype, obj._order)

    @property
    def _grid(self) -> np.ndarray:
        """The object array of blocks (`None` for undefined blocks)"""
        return self.view(np.ndarray)

    def _defined(self) -> np.ndarray:
        """Boolean array (of the grid shape) of the defined blocks"""
        return _is_defined(self._grid).astype(bool)

    def _iter_defined(self):
        """Loop ``(key, block)`` of the defined blocks"""
        grid = self._grid
        for key in zip(*self._defined().nonzero()):
            yield key, grid[key]

    def _block_shape(self, key) -> tuple[int, ...]:
        """Shape of block `key` (block indices)"""
        return tuple(map(list.__getitem__, self._block_sizes, key))

    def _block_slices(self, key) -> tuple[slice, ...]:
        """Slices of the indices of block `key` (block indices)"""
        return tuple(slice(c[k], c[k + 1]) for c, k in zip(self._blocks_cum0, key))

    def _asblock(self, M, key) -> np.ndarray:
        """`M` as block `key` (block indices) with the data-type and order of the array"""
        shape = self._block_shape(key)
        if (
            type(M) is np.ndarray
            and M.dtype == self._dtype
            and M.shape == shape
            and M.flags[_CONTIGUOUS[self._order]]
        ):
            return M
        M = np.asarray(M)
        if M.shape != shape:
            raise ValueError(
                f"{self.__class__.__name__} could not assign array of shape "
                f"{M.shape} into block of shape {shape}"
            )
        # numpy raises for casts that lose data (e.g. complex -> float)
        return M.astype(self._dtype, order=self._order, casting="same_kind", copy=False)

    def _map(self, func, grid=None, blocks=None, dtype=None, order=None) -> BlockArray:
        """New block array with the blocks ``func(M)`` of the defined blocks ``M`` in `grid`"""
        new = BlockArray(
            self._blocks if blocks is None else blocks,
            self._dtype if dtype is None else dtype,
            self._order if order is None else order,
        )
        if grid is None:
            grid = self._grid
        new._grid[...] = np.frompyfunc(lambda M: None if M is None else func(M), 1, 1)(
            grid
        )
        return new

    @property
    def blocks(self) -> tuple[np.ndarray, ...]:
        """Size of the blocks along each axis"""
        return self._blocks

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the (dense) array"""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the array"""
        return self._dtype

    @property
    def order(self) -> str:
        """Memory order of the blocks"""
        return self._order

    @property
    def size(self) -> int:
        """Number of elements in the (dense) array"""
        return math.prod(self._shape)

    @property
    def itemsize(self) -> int:
        return self._dtype.itemsize

    @property
    def nbytes(self) -> int:
        """Number of bytes of the defined blocks"""
        return sum(M.nbytes for _, M in self._iter_defined())

    def __len__(self) -> int:
        """The length of the first dimension of the array"""
        return self._shape[0]

    def __repr__(self) -> str:
        grid = self._grid
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} "
            f"shape={self.shape}, blocks={grid.shape}, "
            f"defined={self._defined().sum()}/{grid.size}, "
            f"dtype={self._dtype}, order={self._order!r}>"
        )

    __str__ = __repr__

    def __iter__(self):
        """Loop the first dimension of the dense array"""
        return iter(self.toarray())

    def __bool__(self) -> bool:
        return bool(self.toarray())

    def __reduce__(self):
        # ndarray.__reduce__ would store the (overridden) dtype in its state
        return _rebuild_block_array, (
            self._blocks,
            self._dtype,
            self._order,
            self._grid,
        )

    def __copy__(self) -> BlockArray:
        return self.copy()

    def __deepcopy__(self, memo) -> BlockArray:
        return self.copy()

    def copy(self) -> BlockArray:
        """Create a copy of this block array (all blocks are copied)"""
        return self._map(lambda M: M.copy(order=self._order))

    def astype(
        self, dtype, order=None, casting="unsafe", copy: bool = True
    ) -> BlockArray:
        """Block array with blocks converted to `dtype` (and `order`)"""
        order = self._order if order is None or order in "KA" else order.upper()
        return self._map(
            lambda M: M.astype(dtype, order=order, casting=casting, copy=copy),
            dtype=np.dtype(dtype),
            order=order,
        )

    @property
    def T(self) -> BlockArray:
        """Array with reversed axes, see `transpose`"""
        return self.transpose()

    def transpose(self, *axes) -> BlockArray:
        """Array with permuted axes, the blocks are views of the transposed blocks

        Reversing the axes swaps the memory order ("C" <-> "F"), for other
        permutations the blocks are not contiguous.
        """
        if len(axes) == 1 and (axes[0] is None or isinstance(axes[0], (tuple, list))):
            axes = axes[0]
        nd = self.ndim
        if not axes:
            axes = tuple(range(nd))[::-1]
        axes = _normalize_axes(axes, nd)
        if len(axes) != nd:
            raise ValueError("axes don't match array")
        order = self._order
        if nd > 1 and axes == tuple(range(nd))[::-1]:
            order = "F" if order == "C" else "C"
        return self._map(
            lambda M: M.transpose(axes),
            grid=self._grid.transpose(axes),
            blocks=tuple(self._blocks[a] for a in axes),
            order=order,
        )

    @property
    def real(self) -> BlockArray:
        """Real part of the array, the blocks are views"""
        return self._map(lambda M: M.real, dtype=np.empty(0, self._dtype).real.dtype)

    @property
    def imag(self) -> BlockArray:
        """Imaginary part of the array, the blocks are views"""
        if self._dtype.kind != "c":
            return BlockArray(self._blocks, self._dtype, self._order)
        return self._map(lambda M: M.imag, dtype=np.empty(0, self._dtype).real.dtype)

    def conj(self) -> BlockArray:
        return np.conjugate(self)

    conjugate = conj

    def dot(self, other, out=None):
        if out is not None:
            return np.dot(self.toarray(), _densify(other), out=out)
        return self @ other

    def asformat(self, format):
        """Convert to a particular format"""
        return getattr(self, f"to{format}")()

    def toarray(self) -> np.ndarray:
        """Convert to a dense array, where non-defined blocks will be zero"""
        D = np.zeros(self._shape, dtype=self._dtype, order=self._order)
        for key, M in self._iter_defined():
            D[self._block_slices(key)] = M
        return D

    def _band(self, width: int, axes) -> BlockArray:
        """Block array (sharing blocks) of the blocks within `width` of the block diagonal"""
        idx = np.indices(self._grid.shape)
        if axes is not None:
            idx = idx[list(_normalize_axes(axes, self.ndim))]
        mask = idx.max(0) - idx.min(0) <= width
        ret = BlockArray(self._blocks, self._dtype, self._order)
        ret._grid[mask] = self._grid[mask]
        return ret

    def tobtd(self, axes: Optional[Sequence[int]] = None) -> BlockArray:
        """Return only the block tridiagonal part of the array (blocks are shared)

        Parameters
        ----------
        axes :
            the axes where the block indices may differ by at most 1, defaults to all axes
        """
        return self._band(1, axes)

    def tobd(self, axes: Optional[Sequence[int]] = None) -> BlockArray:
        """Return only the block diagonal part of the array (blocks are shared)

        Parameters
        ----------
        axes :
            the axes where the block indices are equal, defaults to all axes
        """
        return self._band(0, axes)

    def diagonal(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1
    ) -> BlockArray | np.ndarray:
        """Diagonal of the array along `axis1` and `axis2`, see `numpy.diagonal`

        The two axes are removed and the diagonal is the last axis.
        If `offset` is 0 and the two axes have the same blocks, the
        diagonal is a `BlockArray` (copies of the block diagonals), otherwise
        a dense array.
        """
        nd = self.ndim
        a1, a2 = _normalize_axes((axis1, axis2), nd)
        keep = [a for a in range(nd) if a not in (a1, a2)]
        if offset == 0 and np.array_equal(self._blocks[a1], self._blocks[a2]):
            return self._map(
                lambda M: M.diagonal(0, a1, a2).copy(order=self._order),
                grid=self._grid.diagonal(0, a1, a2),
                blocks=tuple(self._blocks[a] for a in keep) + (self._blocks[a1],),
            )

        n1, n2 = self._shape[a1], self._shape[a2]
        if offset >= 0:
            i1 = np.arange(max(0, min(n1, n2 - offset)))
            i2 = i1 + offset
        else:
            i2 = np.arange(max(0, min(n1 + offset, n2)))
            i1 = i2 - offset
        # the advanced indices are the last (adjacent) axes, so the diagonal is last
        T = self.transpose(keep + [a1, a2])
        return T._get_fancy(T._fancy_key([slice(None)] * len(keep) + [i1, i2]))

    def trace(
        self, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype=None, out=None
    ):
        """Sum along the diagonal of the array, see `numpy.trace`"""
        return self.diagonal(offset, axis1, axis2).sum(axis=-1, dtype=dtype, out=out)

    def indices(self, defined: bool = True) -> tuple[np.ndarray, ...]:
        """Array indices (one array per axis) of the defined (or undefined) blocks

        The indices are ordered by blocks (row-major over the blocks), and
        within each block in the memory order of the blocks.

        Parameters
        ----------
        defined :
            whether to return the indices of the defined blocks, or of
            the undefined (zero) blocks

        Examples
        --------
        >>> full = np.zeros(ba.shape, dtype=ba.dtype)
        >>> full[ba.indices()] = ba[ba.indices()]
        >>> np.allclose(full[ba.indices(defined=False)], 0)
        True
        """
        mask = self._defined()
        if not defined:
            mask = ~mask
        out = [[] for _ in range(self.ndim)]
        for key in zip(*mask.nonzero()):
            idx = np.indices(self._block_shape(key))
            for o, ix, c, k in zip(out, idx, self._blocks_cum0, key):
                o.append(ix.ravel(order=self._order) + c[k])
        if len(out[0]) == 0:
            return tuple(np.empty(0, dtype=np.intp) for _ in out)
        return tuple(np.concatenate(o) for o in out)

    @property
    def block_indexer(self) -> BlockArrayIndexer:
        """Get the indexer for this block array

        The indexer allows manipulating the array using indices of the blocks,
        rather than indices of the array elements.
        """
        return BlockArrayIndexer(self)

    # Indexing with array indices

    def _normalize(self, idx, axis: int) -> np.ndarray:
        """Non-negative, bounds checked, indices of an index array `idx` (of any shape) along `axis`"""
        idx = np.asarray(idx)
        n = self._shape[axis]
        if idx.dtype == bool:
            if idx.ndim != 1 or len(idx) != n:
                raise IndexError(
                    f"boolean index did not match axis {axis} with size {n}"
                )
            return idx.nonzero()[0]
        if idx.size == 0:
            return idx.astype(np.intp)
        if idx.dtype.kind not in "iu":
            raise IndexError(
                "only integers, slices (`:`), ellipsis (`...`) and integer or "
                "boolean arrays are valid indices"
            )
        idx = idx.astype(np.intp, copy=False)
        if idx.min() < -n or idx.max() >= n:
            raise IndexError(f"index out of bounds for axis {axis} with size {n}")
        return np.where(idx < 0, idx + n, idx)

    def _split_key(self, key) -> Optional[list]:
        """`key` as a list of normalized indices (one per axis), or None if it is not handled

        Slices are retained, all other indices are normalized integer arrays
        (0D arrays for integers).
        """
        nd = self.ndim
        if not isinstance(key, tuple):
            if (
                isinstance(key, np.ndarray)
                and key.dtype == bool
                and key.ndim == nd
                and nd > 1
            ):
                if key.shape != self.shape:
                    raise IndexError("boolean index did not match the array shape")
                return list(key.nonzero())
            key = (key,)
        if any(k is None for k in key):
            return None
        if any(
            isinstance(k, np.ndarray) and k.dtype == bool and k.ndim != 1 for k in key
        ):
            return None
        ellipsis = [i for i, k in enumerate(key) if k is Ellipsis]
        if len(ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if ellipsis:
            i = ellipsis[0]
            key = key[:i] + (slice(None),) * (nd + 1 - len(key)) + key[i + 1 :]
        if len(key) > nd:
            raise IndexError(
                f"too many indices for {self.__class__.__name__}: it is "
                f"{nd}-dimensional, but {len(key)} were indexed"
            )
        key = list(key) + [slice(None)] * (nd - len(key))
        return [
            k if isinstance(k, slice) else self._normalize(k, axis)
            for axis, k in enumerate(key)
        ]

    def _orthogonal_key(self, keys: list) -> Optional[tuple]:
        """1D indices per axis if `keys` are orthogonal (as `np.ix_`), otherwise None"""
        nd = len(keys)
        idx = []
        for axis, k in enumerate(keys):
            if isinstance(k, slice) or k.ndim != nd:
                return None
            if any(s != 1 for a, s in enumerate(k.shape) if a != axis):
                return None
            idx.append(k.ravel())
        return tuple(idx)

    def _fancy_key(self, keys: list) -> tuple[np.ndarray, ...]:
        """Broadcastable index arrays (one per axis) equivalent to numpy indexing with `keys`

        `keys` contains at least one index array.
        The broadcast shape of the index arrays is placed at the first
        index array when all index arrays are adjacent, otherwise first.
        """
        nd = len(keys)
        adv = [a for a, k in enumerate(keys) if not isinstance(k, slice)]
        sl = [a for a, k in enumerate(keys) if isinstance(k, slice)]
        bshape = np.broadcast_shapes(*(keys[a].shape for a in adv))
        nb = len(bshape)
        if adv[-1] - adv[0] == len(adv) - 1:
            start = sum(a < adv[0] for a in sl)
        else:
            start = 0
        ndim = len(sl) + nb

        idx = [None] * nd
        for r, a in enumerate(sl):
            pos = r if r < start else r + nb
            shape = [1] * ndim
            ar = np.arange(*keys[a].indices(self._shape[a]))
            shape[pos] = len(ar)
            idx[a] = ar.reshape(shape)
        for a in adv:
            k = keys[a]
            shape = (1,) * start + (1,) * (nb - k.ndim) + k.shape
            idx[a] = k.reshape(shape + (1,) * (ndim - len(shape)))
        return tuple(idx)

    def _groups(self, idx: np.ndarray, axis: int) -> list:
        """Split the indices `idx` along `axis` into the blocks they belong to

        Returns a list of ``(block, positions in idx, indices local to the block)``
        where the positions and local indices are slices when contiguous.
        """
        if len(idx) == 0:
            return []
        c = self._blocks_cum0[axis]
        bidx = self._index_blocks[axis][idx]
        order = np.argsort(bidx, kind="stable")
        bounds = np.flatnonzero(np.diff(bidx[order])) + 1
        groups = []
        for pos in np.split(order, bounds):
            b = bidx[pos[0]]
            groups.append((b, _as_slice(pos), _as_slice(idx[pos] - c[b])))
        return groups

    def _fancy_groups(self, idx: list):
        """Loop ``(key, sel)`` for the elements ``sel`` (of the 1D arrays `idx`) in block `key`"""
        if len(idx[0]) == 0:
            return
        shape = self._grid.shape
        key = np.ravel_multi_index(
            tuple(ib[x] for ib, x in zip(self._index_blocks, idx)), shape
        )
        order = np.argsort(key, kind="stable")
        bounds = np.flatnonzero(np.diff(key[order])) + 1
        for sel in np.split(order, bounds):
            yield np.unravel_index(int(key[sel[0]]), shape), sel

    def _block_sequence(self, idx: np.ndarray, axis: int) -> Optional[np.ndarray]:
        """Blocks of `idx` (along `axis`) if it is a sequence of complete blocks, otherwise None"""
        if len(idx) == 0:
            return None
        c = self._blocks_cum0[axis]
        bidx = self._index_blocks[axis][idx]
        # the blocks that start in idx
        seq = bidx[idx == c[bidx]]
        full = array_arange(c[seq], c[seq + 1])
        if len(full) == len(idx) and (full == idx).all():
            return seq
        return None

    def _block_select(self, idx: tuple, share: bool) -> Optional[BlockArray]:
        """Block array of the indices `idx` (per axis) if they are sequences of complete blocks"""
        seqs = []
        for axis, x in enumerate(idx):
            seq = self._block_sequence(x, axis)
            if seq is None:
                return None
            seqs.append(seq)
        func = (lambda M: M) if share else (lambda M: M.copy(order=self._order))
        return self._map(
            func,
            grid=self._grid[np.ix_(*seqs)],
            blocks=tuple(b[s] for b, s in zip(self._blocks, seqs)),
        )

    def __getitem__(self, key):
        keys = self._split_key(key)
        if keys is None:
            return self.toarray()[key]

        if all(not isinstance(k, slice) and k.ndim == 0 for k in keys):
            bkey = tuple(ib[k] for ib, k in zip(self._index_blocks, keys))
            M = self._grid[bkey]
            if M is None:
                return self._dtype.type(0)
            return M[tuple(k - c[b] for k, c, b in zip(keys, self._blocks_cum0, bkey))]

        if all(isinstance(k, slice) or k.ndim == 0 for k in keys):
            # basic indexing
            idx = tuple(
                (np.arange(*k.indices(n)) if isinstance(k, slice) else k.reshape(1))
                for k, n in zip(keys, self._shape)
            )
            if all(isinstance(k, slice) for k in keys):
                ba = self._block_select(idx, True)
                if ba is not None:
                    return ba
            out = self._get_orthogonal(idx)
            return out[tuple(slice(None) if isinstance(k, slice) else 0 for k in keys)]

        idx = self._orthogonal_key(keys)
        if idx is not None:
            ba = self._block_select(idx, False)
            if ba is not None:
                return ba
            return self._get_orthogonal(idx)
        return self._get_fancy(self._fancy_key(keys))

    def _get_orthogonal(self, idx: tuple) -> np.ndarray:
        """Dense array of the elements in ``np.ix_(*idx)``"""
        out = np.zeros(tuple(len(x) for x in idx), dtype=self._dtype, order=self._order)
        grid = self._grid
        groups = [self._groups(x, axis) for axis, x in enumerate(idx)]
        for combo in itertools.product(*groups):
            M = grid[tuple(g[0] for g in combo)]
            if M is not None:
                out[_ix(*(g[1] for g in combo))] = M[_ix(*(g[2] for g in combo))]
        return out

    def _get_fancy(self, idx: tuple) -> np.ndarray:
        """Elements of the index arrays `idx` (one per axis, broadcast together)"""
        idx = np.broadcast_arrays(*idx)
        shape = idx[0].shape
        idx = [x.ravel() for x in idx]
        out = np.zeros(len(idx[0]), dtype=self._dtype)
        cum = self._blocks_cum0
        grid = self._grid
        for key, sel in self._fancy_groups(idx):
            M = grid[key]
            if M is not None:
                out[sel] = M[tuple(x[sel] - c[k] for x, c, k in zip(idx, cum, key))]
        return out.reshape(shape)

    def __setitem__(self, key, value) -> None:
        keys = self._split_key(key)
        if keys is None:
            raise IndexError(
                f"{self.__class__.__name__} does not support index {key!r}"
            )
        value = np.asarray(value)
        if not np.can_cast(value.dtype, self._dtype, "same_kind"):
            raise TypeError(
                f"{self.__class__.__name__} cannot cast values from {value.dtype} to {self._dtype}"
            )

        if all(isinstance(k, slice) or k.ndim == 0 for k in keys):
            # basic indexing
            idx = tuple(
                (np.arange(*k.indices(n)) if isinstance(k, slice) else k.reshape(1))
                for k, n in zip(keys, self._shape)
            )
            shape = tuple(len(x) for x, k in zip(idx, keys) if isinstance(k, slice))
            value = np.broadcast_to(value, shape).reshape(tuple(len(x) for x in idx))
            self._set_orthogonal(idx, value)
        else:
            idx = np.broadcast_arrays(*self._fancy_key(keys))
            value = np.broadcast_to(value, idx[0].shape)
            self._set_fancy([x.ravel() for x in idx], value.ravel())

    def _new_block(self, key) -> np.ndarray:
        """Allocate a zero block `key` and store it in the array"""
        M = np.zeros(self._block_shape(key), dtype=self._dtype, order=self._order)
        self._grid[key] = M
        return M

    def _set_orthogonal(self, idx: tuple, value: np.ndarray) -> None:
        grid = self._grid
        groups = [self._groups(x, axis) for axis, x in enumerate(idx)]
        for combo in itertools.product(*groups):
            key = tuple(g[0] for g in combo)
            V = value[_ix(*(g[1] for g in combo))]
            M = grid[key]
            if M is None:
                if not V.any():
                    continue
                M = self._new_block(key)
            M[_ix(*(g[2] for g in combo))] = V

    def _set_fancy(self, idx: list, value: np.ndarray) -> None:
        cum = self._blocks_cum0
        grid = self._grid
        for key, sel in self._fancy_groups(idx):
            V = value[sel]
            M = grid[key]
            if M is None:
                if not V.any():
                    continue
                M = self._new_block(key)
            M[tuple(x[sel] - c[k] for x, c, k in zip(idx, cum, key))] = V

    # numpy protocols

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if method == "__call__" and "where" not in kwargs:
            if ufunc is np.matmul:
                ret = _matmul(*inputs, out=out, **kwargs)
            elif ufunc.nout == 1 and ufunc.signature is None:
                ret = _elementwise(ufunc, inputs, out, kwargs)
            else:
                ret = NotImplemented
            if ret is not NotImplemented:
                return ret

        # dense fallback
        if out is not None:
            if any(isinstance(o, BlockArray) for o in out):
                raise TypeError(
                    f"{ufunc.__name__} does not retain the zero blocks, the result "
                    f"cannot be stored in a {self.__class__.__name__}"
                )
            kwargs["out"] = out
        return getattr(ufunc, method)(*_densify(inputs), **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        handled = _ARRAY_FUNCTIONS.get(func)
        if handled is not None:
            return handled(*args, **kwargs)
        return func(*_densify(args), **_densify(kwargs))


def _rebuild_block_array(blocks, dtype, order, grid) -> BlockArray:
    """Unpickle a `BlockArray`"""
    ba = BlockArray(blocks, dtype, order)
    ba._grid[...] = grid
    return ba


def _elementwise(ufunc, inputs, out, kwargs):
    """Element-wise `ufunc` on the defined blocks, NotImplemented if the zero blocks are not retained"""
    bas = [x for x in inputs if isinstance(x, BlockArray)]
    if not bas:
        return NotImplemented
    ref = bas[0]
    for x in inputs:
        if isinstance(x, BlockArray):
            if not _same_blocks(x._blocks, ref._blocks):
                return NotImplemented
        elif np.ndim(x) != 0:
            return NotImplemented
    if out is not None:
        if (
            len(out) != 1
            or not isinstance(out[0], BlockArray)
            or not _same_blocks(out[0]._blocks, ref._blocks)
        ):
            return NotImplemented
        out = out[0]

    # value of the undefined blocks
    zero = [np.zeros(1, x._dtype) if isinstance(x, BlockArray) else x for x in inputs]
    with np.errstate(all="ignore"):
        zero = ufunc(*zero, **kwargs)
    if zero[0] != 0:
        return NotImplemented

    reduce = np.logical_and if ufunc in _ANNIHILATE else np.logical_or
    pattern = reduce.reduce([x._defined() for x in bas])

    if out is None:
        ret = BlockArray(ref._blocks, zero.dtype, ref._order)
        kwargs.setdefault("order", ref._order)
    else:
        ret = out
    grid = ret._grid
    grids = [x._grid if isinstance(x, BlockArray) else None for x in inputs]
    for key in zip(*pattern.nonzero()):
        args = []
        for x, g in zip(inputs, grids):
            if g is None:
                args.append(x)
                continue
            M = g[key]
            if M is None:
                M = np.zeros(ref._block_shape(key), dtype=x._dtype)
            args.append(M)
        M = grid[key] if out is not None else None
        if M is None:
            M = ufunc(*args, **kwargs)
            grid[key] = M if out is None else ret._asblock(M, key)
        else:
            ufunc(*args, out=M, **kwargs)
    if out is not None:
        grid[~pattern] = None
    return ret


def _matmul(a, b, out=None, **kwargs):
    """Matrix multiplication only using the defined blocks"""
    if out is not None or kwargs:
        return NotImplemented

    if isinstance(a, BlockArray) and isinstance(b, BlockArray):
        # stacked matrices, the leading axes must have the same blocks
        if (
            a.ndim < 2
            or a.ndim != b.ndim
            or not _same_blocks(a._blocks[:-2], b._blocks[:-2])
            or not np.array_equal(a._blocks[-1], b._blocks[-2])
        ):
            return NotImplemented
        ret = BlockArray(
            a._blocks[:-1] + (b._blocks[-1],),
            np.result_type(a._dtype, b._dtype),
            a._order,
        )
        ga, gb, grid = a._grid, b._grid, ret._grid
        ma, mb = a._defined(), b._defined()
        for L in np.ndindex(*ga.shape[:-2]):
            for k in range(ga.shape[-1]):
                J = mb[L + (k,)].nonzero()[0]
                for i in ma[L + (slice(None), k)].nonzero()[0]:
                    A = ga[L + (i, k)]
                    for j in J:
                        key = L + (i, j)
                        M = grid[key]
                        if M is None:
                            grid[key] = ret._asblock(A @ gb[L + (k, j)], key)
                        else:
                            M += A @ gb[L + (k, j)]
        return ret

    if isinstance(a, BlockArray):
        b = np.asarray(b)
        if a.ndim != 2 or b.ndim not in (1, 2) or len(b) != a._shape[1]:
            return NotImplemented
        cr, ck = a._blocks_cum0
        ret = np.zeros(
            (a._shape[0],) + b.shape[1:], dtype=np.result_type(a._dtype, b.dtype)
        )
        for (i, k), A in a._iter_defined():
            ret[cr[i] : cr[i + 1]] += A @ b[ck[k] : ck[k + 1]]
        return ret

    a = np.asarray(a)
    if b.ndim != 2 or a.ndim not in (1, 2) or a.shape[-1] != b._shape[0]:
        return NotImplemented
    ck, cc = b._blocks_cum0
    ret = np.zeros(
        a.shape[:-1] + (b._shape[1],), dtype=np.result_type(a.dtype, b._dtype)
    )
    for (k, j), B in b._iter_defined():
        ret[..., cc[j] : cc[j + 1]] += a[..., ck[k] : ck[k + 1]] @ B
    return ret


_ARRAY_FUNCTIONS = {
    np.shape: lambda a: a.shape,
    np.ndim: lambda a: a.ndim,
    np.size: lambda a, axis=None: a.size if axis is None else a.shape[axis],
    np.copy: lambda a, *args, **kwargs: a.copy(),
    np.transpose: lambda a, axes=None: a.transpose(axes),
    np.diagonal: lambda a, offset=0, axis1=0, axis2=1: a.diagonal(offset, axis1, axis2),
    np.trace: lambda a, offset=0, axis1=0, axis2=1, dtype=None, out=None: a.trace(
        offset, axis1, axis2, dtype, out
    ),
    np.real: lambda a: a.real,
    np.imag: lambda a: a.imag,
}


def _dense_method(name: str):
    def method(self, *args, **kwargs):
        return getattr(self.toarray(), name)(*args, **kwargs)

    method.__name__ = name
    method.__doc__ = f"Equivalent to ``self.toarray().{name}(...)``"
    return method


def _unsupported_method(name: str):
    def method(self, *args, **kwargs):
        raise NotImplementedError(
            f"{BlockArray.__name__}.{name} is not supported, use toarray() or block_indexer"
        )

    method.__name__ = name
    return method


# The ndarray methods operate on the grid of blocks, they are replaced
# by their dense equivalents, or disabled when they would modify the array.
for _name in (
    "all",
    "any",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "choose",
    "clip",
    "compress",
    "cumprod",
    "cumsum",
    "flatten",
    "item",
    "max",
    "mean",
    "min",
    "nonzero",
    "prod",
    "ptp",
    "ravel",
    "repeat",
    "reshape",
    "round",
    "searchsorted",
    "squeeze",
    "std",
    "sum",
    "swapaxes",
    "take",
    "tobytes",
    "tofile",
    "tolist",
    "var",
):
    setattr(BlockArray, _name, _dense_method(_name))
for _name in (
    "byteswap",
    "fill",
    "itemset",
    "partition",
    "put",
    "resize",
    "setfield",
    "sort",
):
    setattr(BlockArray, _name, _unsupported_method(_name))
del _name
