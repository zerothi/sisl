# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np

cimport numpy as cnp

from sisl._core._dtypes cimport floats_st

from ._common import comply_gauge
from ._matrix_phase import *
from ._matrix_phase3 import *
from ._phase import *

__all__ = ["matrix_dk", "matrik_dk_nc", "matrik_dk_nc_diag", "matrik_dk_so"]


def _phase_dk(gauge, M, sc, cnp.ndarray[floats_st] k, dtype):
    # dtype *must* be passed through phase_dtype
    gauge = comply_gauge(gauge)

    # This is the differentiated matrix with respect to k
    # See _phase.pyx, we are using exp(i k.R/r)
    #  i R
    if gauge == 'cell':
        iRs = phase_rsc(sc, k, dtype).reshape(-1, 1)
        iRs = (1j * np.dot(sc.sc_off, sc.cell) * iRs).astype(dtype, copy=False)
        p_opt = 1

    elif gauge == 'atom':
        M.finalize()
        rij = M.Rij()._csr._D
        iRs = (1j * rij * phase_rij(rij, sc, k, dtype).reshape(-1, 1)).astype(dtype, copy=False)
        del rij
        p_opt = 0

    return p_opt, iRs


def matrix_dk(gauge, M, const int idx, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = _phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        return _phase3_array(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)

    # Default must be something else.
    d1, d2, d3 = _phase3_csr(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)
    return d1.asformat(format), d2.asformat(format), d3.asformat(format)


def matrix_dk_nc(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = _phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        return _phase3_array_nc(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)

    # Default must be something else.
    d1, d2, d3 = _phase3_csr_nc(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
    return d1.asformat(format), d2.asformat(format), d3.asformat(format)


def matrix_dk_nc_diag(gauge, M, const int idx, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = _phase_dk(gauge, M, sc, k, dtype)

    phx = iRs[:, 0].copy()
    phy = iRs[:, 1].copy()
    phz = iRs[:, 2].copy()
    del iRs

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        x = _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phx, p_opt)
        y = _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phy, p_opt)
        z = _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phz, p_opt)

    else:
        x = _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phx, p_opt).asformat(format)
        y = _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phy, p_opt).asformat(format)
        z = _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phz, p_opt).asformat(format)

    return x, y, z


def matrix_dk_so(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = _phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        return _phase3_array_so(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)

    # Default must be something else.
    d1, d2, d3 = _phase3_csr_so(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
    return d1.asformat(format), d2.asformat(format), d3.asformat(format)
