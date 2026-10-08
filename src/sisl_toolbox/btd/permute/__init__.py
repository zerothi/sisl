# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Permutation tables that reduce the bandwidth of sparse matrices

The returned permutation tables are defined as ``perm[new] = old``, i.e.
``M[perm][:, perm]`` is the reduced bandwidth matrix.
"""

from __future__ import annotations

import numpy as np

from sisl import SparseCSR
from sisl._core._sparse import fold_csr_matrix
from sisl_toolbox.btd.permute._permute import (
    btd_blocks,
    cuthill_mckee,
    electrode_ordering,
    generalized_gibbs_poole_stockmeyer,
    gibbs_poole_stockmeyer,
    refine_bandwidth,
    symmetric_graph,
)

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


def csr_graph(csr: SparseCSR) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric graph ``(ptr, col)`` of the pattern of `csr`

    The columns are folded into the first ``csr.shape[0]`` columns
    (``col % csr.shape[0]``), and the pattern is symmetrized (:math:`A + A^T`)
    without the diagonal, see `symmetric_graph`.
    """
    ptr, ncol, col = fold_csr_matrix(csr.ptr, csr.ncol, csr.col)
    return symmetric_graph(ptr, ncol, col)


def reduce_bandwidth(
    csr: SparseCSR, method: str = "gps", refine: bool = True, **kwargs
) -> np.ndarray:
    """Permutation table that reduces the bandwidth of `csr`

    The bandwidth is reduced for the graph of `csr`, see `csr_graph`.

    Parameters
    ----------
    csr :
        the sparse matrix, columns may extend into supercells
    method : {"rcm", "cm", "gps", "ggps", "electrode"}
        - rcm: reverse Cuthill-McKee
        - cm: Cuthill-McKee
        - gps: Gibbs-Poole-Stockmeyer
        - ggps: generalized Gibbs-Poole-Stockmeyer
        - electrode: level structures rooted at electrodes, requires the
          ``start`` (and optionally ``end``) orbital indices
    refine :
        apply `refine_bandwidth` to the permutation table of `method`,
        for ``method="electrode"`` the electrode orbitals keep their positions
    **kwargs :
        passed to the routine of `method`

    Returns
    -------
    perm : the permutation table, ``perm[new] = old``
    """
    return _reduce_bandwidth(*csr_graph(csr), method, refine, **kwargs)


def _reduce_bandwidth(ptr, col, method: str, refine: bool, **kwargs) -> np.ndarray:
    """`reduce_bandwidth` for the graph ``(ptr, col)``"""
    method = method.lower()
    if method in ("cm", "rcm"):
        perm = cuthill_mckee(ptr, col, reverse=method == "rcm", **kwargs)
    elif method == "gps":
        perm = gibbs_poole_stockmeyer(ptr, col, **kwargs)
    elif method == "ggps":
        perm = generalized_gibbs_poole_stockmeyer(ptr, col, **kwargs)
    elif method == "electrode":
        perm = electrode_ordering(ptr, col, **kwargs)
    else:
        raise ValueError(
            f"{reduce_bandwidth.__name__} got unknown method={method!r}, "
            "must be one of [rcm, cm, gps, ggps, electrode]"
        )
    if refine:
        fixed = None
        if method == "electrode":
            fixed = np.concatenate(
                [np.ravel(kwargs.get(key, [])) for key in ("start", "end")]
            )
        perm = refine_bandwidth(ptr, col, perm, fixed=fixed)
    return perm
