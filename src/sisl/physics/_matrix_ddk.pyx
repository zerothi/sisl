# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np
cimport numpy as cnp

from ._common import comply_gauge
from sisl._core._dtypes cimport floats_st
from ._matrix_phase3 import *
from ._phase import *

__all__ = ["matrix_ddk", "matrix_ddk_nc", "matrix_ddk_nc_diag", "matrix_ddk_so"]


def _phase_ddk(gauge, M, sc, cnp.ndarray[floats_st] k, dtype):
    # dtype *must* be passed through phase_dtype
    gauge = comply_gauge(gauge)

    # This is the differentiated matrix with respect to k
    # See _phase.pyx, we are using exp(i k.R/r)
    #  (i R) * (i R) = - R**2
    # And since we have a double differentiation we can have
    # two dependent variables
    # We always do the Voigt representation
    #  Rd = dx^2, dy^2, dz^2, dzy, dxz, dyx
    if gauge == 'cell':
        phases = phase_rsc(sc, k, dtype).reshape(-1, 1)
        Rs = np.dot(sc.sc_off, sc.cell)
        Rd = - (Rs * Rs * phases).astype(dtype, copy=False)
        Ro = - (np.roll(Rs, 1, axis=1) * phases).astype(dtype, copy=False) # z, x, y
        Ro *= np.roll(Rs, -1, axis=1) # y, z, x
        del phases, Rs
        p_opt = 1

    elif gauge == 'atom':
        M.finalize()
        rij = M.Rij()._csr._D
        phases = phase_rij(rij, sc, k, dtype).reshape(-1, 1)
        Rd = - (rij * rij * phases).astype(dtype, copy=False)
        Ro = - (np.roll(rij, 1, axis=1) * phases).astype(dtype, copy=False) # z, x, y
        Ro *= np.roll(rij, -1, axis=1) # y, z, x
        del rij, phases
        p_opt = 0

    return p_opt, Rd, Ro


def matrix_ddk(gauge, M, const int idx, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype)
    p_opt, Rd, Ro = _phase_ddk(gauge, M, sc, k, dtype)

    # Return list
    dd = [None, None, None, None, None, None]

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        dd[:3] = _phase3_array(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
        dd[3:] = _phase3_array(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)

    else:
        # Default must be something else.
        dd[:3] = _phase3_csr(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
        dd[3:] = _phase3_csr(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)
        dd[0] = dd[0].asformat(format)
        dd[1] = dd[1].asformat(format)
        dd[2] = dd[2].asformat(format)
        dd[3] = dd[3].asformat(format)
        dd[4] = dd[4].asformat(format)
        dd[5] = dd[5].asformat(format)

    return dd


def matrix_ddk_nc(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, Rd, Ro = _phase_ddk(gauge, M, sc, k, dtype)

    # Return list
    dd = [None, None, None, None, None, None]

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        dd[:3] = _phase3_array_nc(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
        dd[3:] = _phase3_array_nc(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)

    else:
        # Default must be something else.
        dd[:3] = _phase3_csr_nc(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
        dd[3:] = _phase3_csr_nc(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)
        dd[0] = dd[0].asformat(format)
        dd[1] = dd[1].asformat(format)
        dd[2] = dd[2].asformat(format)
        dd[3] = dd[3].asformat(format)
        dd[4] = dd[4].asformat(format)
        dd[5] = dd[5].asformat(format)

    return dd


def matrix_ddk_nc_diag(gauge, M, const int idx, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, Rd, Ro = _phase_ddk(gauge, M, sc, k, dtype)

    # We need the phases to be consecutive in memory
    Rxx = Rd[:, 0].copy()
    Ryy = Rd[:, 1].copy()
    Rzz = Rd[:, 2].copy()
    del Rd
    Rzy = Ro[:, 0].copy()
    Rxz = Ro[:, 1].copy()
    Ryx = Ro[:, 2].copy()
    del Ro

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        dxx = _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rxx, p_opt)
        dyy = _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ryy, p_opt)
        dzz = _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rzz, p_opt)
        dzy = _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rzy, p_opt)
        dxz = _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rxz, p_opt)
        dyx = _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ryx, p_opt)

    else:
        dxx = _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rxx, p_opt).asformat(format)
        dyy = _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ryy, p_opt).asformat(format)
        dzz = _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rzz, p_opt).asformat(format)
        dzy = _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rzy, p_opt).asformat(format)
        dxz = _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rxz, p_opt).asformat(format)
        dyx = _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ryx, p_opt).asformat(format)

    return dxx, dyy, dzz, dzy, dxz, dyx


def matrix_ddk_so(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, Rd, Ro = _phase_ddk(gauge, M, sc, k, dtype)

    # Return list
    dd = [None, None, None, None, None, None]

    csr = M._csr

    if format in ("array", "matrix", "dense"):
        dd[:3] = _phase3_array_so(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
        dd[3:] = _phase3_array_so(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)

    else:
        # Default must be something else.
        dd[:3] = _phase3_csr_so(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
        dd[3:] = _phase3_csr_so(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)
        dd[0] = dd[0].asformat(format)
        dd[1] = dd[1].asformat(format)
        dd[2] = dd[2].asformat(format)
        dd[3] = dd[3].asformat(format)
        dd[4] = dd[4].asformat(format)
        dd[5] = dd[5].asformat(format)

    return dd
