# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np

cimport numpy as cnp

from sisl._core._dtypes cimport floats_st, int_sp_st

from ._common import comply_gauge
from ._matrix_phase import *
from ._matrix_phase3 import *
from ._phase import *

__all__ = [
    "matrix_dk",
    "matrik_dk_nc",
    "matrik_dk_diag",
    "matrik_dk_so",
    "matrix_dk_nambu"
]


def phase_dk(gauge, M, sc, cnp.ndarray[floats_st] k, dtype):
    # dtype *must* be passed through phase_dtype
    gauge = comply_gauge(gauge)

    # This is the differentiated matrix with respect to k
    # See phase.pyx, we are using exp(i k.R/r)
    #  i R

    if gauge == "atomic":
        M.finalize()
        rij = M.Rij()._csr._D
        iRs = (1j * rij * phase_rij(rij, sc, k, dtype).reshape(-1, 1)).astype(dtype, copy=False)
        del rij
        p_opt = 0

    elif gauge == "lattice":
        iRs = phase_rsc(sc, k, dtype).reshape(-1, 1)
        iRs = (1j * np.dot(sc.sc_off, sc.cell) * iRs).astype(dtype, copy=False)
        p_opt = 1

    else:
        raise ValueError("phase_dk: gauge must be in [lattice, atomic]")

    assert p_opt >= 0, "Not implemented"

    return p_opt, iRs


def matrix_dk(gauge, M, const int_sp_st idx, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        return phase3_array(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)

    # Default must be something else.
    d1, d2, d3 = phase3_csr(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)
    return d1.asformat(format), d2.asformat(format), d3.asformat(format)


def matrix_dk_nc(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        return phase3_array_nc(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)

    # Default must be something else.
    d1, d2, d3 = phase3_csr_nc(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
    return d1.asformat(format), d2.asformat(format), d3.asformat(format)


def matrix_dk_diag(gauge, M, const int_sp_st idx, const int_sp_st per_row,
                   sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = phase_dk(gauge, M, sc, k, dtype)

    phx = iRs[:, 0].copy()
    phy = iRs[:, 1].copy()
    phz = iRs[:, 2].copy()
    del iRs

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        x = phase_array_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phx, p_opt,
        per_row)
        y = phase_array_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phy, p_opt,
        per_row)
        z = phase_array_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phz, p_opt,
        per_row)

    else:
        x = phase_csr_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phx, p_opt, per_row).asformat(format)
        y = phase_csr_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phy, p_opt, per_row).asformat(format)
        z = phase_csr_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phz, p_opt, per_row).asformat(format)

    return x, y, z


def matrix_dk_so(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        return phase3_array_so(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)

    # Default must be something else.
    d1, d2, d3 = phase3_csr_so(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
    return d1.asformat(format), d2.asformat(format), d3.asformat(format)


def matrix_dk_nambu(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        return phase3_array_nambu(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)

    # Default must be something else.
    d1, d2, d3 = phase3_csr_nambu(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
    return d1.asformat(format), d2.asformat(format), d3.asformat(format)
