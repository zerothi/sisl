# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import sisl._array as _a
from sisl._ufuncs import register_sisl_dispatch
from sisl.typing import GaugeType, KPoint

from ._matrix_ddk import matrix_ddk, matrix_ddk_nc, matrix_ddk_nc_diag, matrix_ddk_so
from ._matrix_dk import matrix_dk, matrix_dk_nc, matrix_dk_nc_diag, matrix_dk_so
from ._matrix_k import matrix_k, matrix_k_nc, matrix_k_nc_diag, matrix_k_so
from .sparse import SparseOrbitalBZ, SparseOrbitalBZSpin

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(SparseOrbitalBZ, module="sisl.physics")
def matrix_at_k(
    M: SparseOrbitalBZ,
    k: KPoint = (0, 0, 0),
    *args,
    dtype=None,
    gauge: GaugeType = "cell",
    format: str = "csr",
    **kwargs,
):
    r"""Fold the matrix with the :math:`\mathbf k` phase applied in the `gauge` specified

    Fold the auxilliary supercell matrix elements into the primary cell by adding a phase factor:

    When `gauge` is ``cell`` the matrix is folded like:

    .. math::
        \mathbf M(\mathbf k) = \sum_{\mathbf{k}} \mathbf M^{\mathrm{sc}_i} e^{i \mathbf R_{\mathrm{sc}_i} \cdot \mathbf k}

    when `gauge` is ``orbital`` the matrix is folded with the interatomic distances, like:

    .. math::
        \mathbf M(\mathbf k) = \sum_{\mathbf{k}} \mathbf M^{\mathrm{sc}_i} e^{i (\mathbf r_i - \mathbf r_j) \cdot \mathbf k}

    Parameters
    ----------
    k :
       k-point (default is Gamma point)
    dtype : numpy.dtype, optional
       default to `numpy.complex128`
    gauge : {'cell', 'orbital'}
       chosen gauge, either the lattice gauge (``cell``), or the interatomic distance gauge (``orbital``).
    format : {"csr", "array", "coo", ...}
       the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
       however if one always requires operations on dense matrices, one can always
       return in `numpy.ndarray` (`"array"`).
       Prefixing with "sc:", or simply "sc" returns the matrix in supercell format
       with phases. This is useful for e.g. bond-current calculations where individual
       hopping + phases are required.
    *args, **kwargs:
        other arguments applicable depending on the exact matrix class.
        For details do, ``sisl.matrix_at_k.dispatch(<object>)``
    """
    k = _a.asarrayd(k).ravel()
    return matrix_k(gauge, M, 0, M.lattice, k, dtype, format)


@register_sisl_dispatch(SparseOrbitalBZ, module="sisl.physics")
def overlap_at_k(
    S: SparseOrbitalBZ,
    k: KPoint = (0, 0, 0),
    *args,
    dtype=None,
    gauge: GaugeType = "cell",
    format: str = "csr",
    **kwargs,
):
    r"""Fold the overlap matrix with the :math:`\mathbf k` phase applied in the `gauge` specified

    Fold the auxilliary supercell overlap matrix elements into the primary cell by adding a phase factor:

    When `gauge` is ``cell`` the overlap matrix is folded like:

    .. math::
        \mathbf S(\mathbf k) = \sum_{\mathbf{k}} \mathbf S^{\mathrm{sc}_i} e^{i \mathbf R_{\mathrm{sc}_i} \cdot \mathbf k}

    when `gauge` is ``orbital`` the overlap matrix is folded with the interatomic distances, like:

    .. math::
        \mathbf S(\mathbf k) = \sum_{\mathbf{k}} \mathbf S^{\mathrm{sc}_i} e^{i (\mathbf r_i - \mathbf r_j) \cdot \mathbf k}

    Parameters
    ----------
    k :
       k-point (default is Gamma point)
    dtype : numpy.dtype, optional
       default to `numpy.complex128`
    gauge : {'cell', 'orbital'}
       chosen gauge, either the lattice gauge (``cell``), or the interatomic distance gauge (``orbital``).
    format : {"csr", "array", "coo", ...}
       the returned format of the overlap matrix, defaulting to the `scipy.sparse.csr_matrix`,
       however if one always requires operations on dense matrices, one can always
       return in `numpy.ndarray` (`"array"`).
       Prefixing with "sc:", or simply "sc" returns the overlap matrix in supercell format
       with phases. This is useful for e.g. COOP calculations where individual
       overlaps + phases are required.
    """
    if S.orthogonzal:
        if dtype is None:
            dtype = np.float64
        nr = len(S)
        nc = nr
        if "sc:" in format:
            format = format[3:]
            nc = S.n_s * nr
        elif "sc" == format:
            format = "csr"
            nc = S.n_s * nr
        if format in ("array", "matrix", "dense"):
            S = np.zeros([nr, nc], dtype=dtype)
            np.fill_diagonal(S, 1.0)
        else:
            S = sps.csr_matrix((nr, nc), dtype=dtype)
            S.setdiag(1.0)
            S = S.asformat(format)
        return S

    k = _a.asarrayd(k).ravel()
    return matrix_k(gauge, S, S.S_idx, S.lattice, k, dtype, format)
