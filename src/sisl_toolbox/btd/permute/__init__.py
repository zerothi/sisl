# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Permutation tables that reduce the bandwidth of sparse matrices

The returned permutation tables are defined as ``perm[new] = old``, i.e.
``M[perm][:, perm]`` is the reduced bandwidth matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import issparse

from sisl import SparseCSR
from sisl._array import array_arange
from sisl._core._sparse import fold_csr_array
from sisl_toolbox.btd.permute._permute import (
    btd_blocks,
    cuthill_mckee,
    electrode_ordering,
    generalized_gibbs_poole_stockmeyer,
    gibbs_poole_stockmeyer,
    refine_bandwidth,
    symmetric_graph,
)

if TYPE_CHECKING:
    from scipy.sparse import sparray, spmatrix

__all__ = [
    "reduce_bandwidth",
    "csr_graph",
    "btd_blocks",
    "bandwidth",
    "cuthill_mckee",
    "gibbs_poole_stockmeyer",
    "generalized_gibbs_poole_stockmeyer",
    "electrode_ordering",
    "refine_bandwidth",
    "symmetric_graph",
]


def bandwidth(ptr: np.ndarray, col: np.ndarray, perm: np.ndarray | None = None) -> int:
    """Bandwidth of the CSR graph ``(ptr, col)`` after permutation with `perm`"""
    row = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    if len(row) == 0:
        return 0
    if perm is not None:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        row, col = inv[row], inv[col]
    return int(np.abs(row - col).max())


def _csr_ptr_col(csr) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous CSR pattern ``(ptr, col)`` of `csr` with a common integer data-type"""
    if isinstance(csr, SparseCSR):
        ncol = csr.ncol
        col = csr.col[array_arange(csr.ptr[:-1], n=ncol)]
        ptr = np.zeros(len(ncol) + 1, dtype=col.dtype)
        np.cumsum(ncol, out=ptr[1:])
    elif issparse(csr):
        csr = csr.tocsr()
        ptr, col = csr.indptr, csr.indices
    else:
        raise TypeError(
            f"{csr_graph.__name__} requires a SparseCSR or a scipy sparse matrix/array, "
            f"got {type(csr).__name__}"
        )
    nr, nc = csr.shape[:2]
    if nr == 0 or nc % nr != 0:
        raise ValueError(
            f"{csr_graph.__name__} requires the number of columns to be a multiple "
            f"of the number of rows, got shape={(nr, nc)}"
        )
    return ptr, col


def csr_graph(csr: SparseCSR | sparray | spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric graph ``(ptr, col)`` of the pattern of `csr`

    The columns are folded into the first ``csr.shape[0]`` columns
    (``col % csr.shape[0]``), and the pattern is symmetrized (:math:`A + A^T`)
    without the diagonal, see `symmetric_graph`.

    The returned arrays have the integer data-type of the indices of `csr`.
    """
    ptr, col = _csr_ptr_col(csr)
    # the folded columns are sorted per row, this determines the tie-breaking
    # in the orderings
    ptr, _, col = fold_csr_array(ptr, np.diff(ptr), col)
    return symmetric_graph(ptr, col)


def reduce_bandwidth(
    csr: SparseCSR | sparray | spmatrix,
    method: str = "gps",
    refine: bool = True,
    reverse: bool = False,
    **kwargs,
) -> np.ndarray:
    """Permutation table that reduces the bandwidth of `csr`

    The bandwidth is reduced for the graph of `csr`, see `csr_graph`.

    Parameters
    ----------
    csr :
        the sparse matrix, columns may extend into supercells (the number of
        columns must be a multiple of the number of rows)
    method : {"cm", "gps", "ggps", "electrode"}
        - cm: Cuthill-McKee
        - gps: Gibbs-Poole-Stockmeyer
        - ggps: generalized Gibbs-Poole-Stockmeyer
        - electrode: level structures rooted at electrodes, requires the
          ``start`` (and optionally ``end``) node indices

        One may prefix with `rev-` for forcing ``reverse=True``.
    refine :
        apply `refine_bandwidth` to the permutation table of `method`,
        for ``method="electrode"`` the electrode nodes keep their positions
    reverse :
        reverse the permutation table, which may improve the layout.
    **kwargs :
        passed to the routine of `method`

    Returns
    -------
    perm : the permutation table, ``perm[new] = old``, with the integer
        data-type of the indices of `csr`
    """
    return _reduce_bandwidth(*csr_graph(csr), method, refine, reverse, **kwargs)


def _reduce_bandwidth(
    ptr, col, method: str, refine: bool, reverse: bool, **kwargs
) -> np.ndarray:
    """`reduce_bandwidth` for the graph ``(ptr, col)``"""
    method = method.lower()
    if method.startswith(("rev-", "r-")):
        reverse = True
        method = method.split("-", maxsplit=1)[-1]
    if method == "cm":
        perm = cuthill_mckee(ptr, col, **kwargs)
    elif method == "rcm":
        perm = cuthill_mckee(ptr, col, **kwargs)
        # force reverse
        reverse = True
    elif method == "gps":
        perm = gibbs_poole_stockmeyer(ptr, col, **kwargs)
    elif method == "ggps":
        perm = generalized_gibbs_poole_stockmeyer(ptr, col, **kwargs)
    elif method == "electrode":
        perm = electrode_ordering(ptr, col, **kwargs)
    else:
        raise ValueError(
            f"{reduce_bandwidth.__name__} got unknown method={method!r}, "
            "must be one of [rev-]?[rcm, cm, gps, ggps, electrode]"
        )
    if reverse:
        perm = perm[::-1]

    if refine:
        fixed = None
        if method == "electrode":
            fixed = np.concatenate(
                [np.ravel(kwargs.get(key, [])) for key in ("start", "end")]
            )
        perm = refine_bandwidth(ptr, col, perm, fixed=fixed)
    return perm
