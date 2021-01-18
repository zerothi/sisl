cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np

from ._phase import *
from ._matrix_phase3 import *
from ._matrix_phase3_nc import *
from ._matrix_phase_nc_diag import *
from ._matrix_phase3_so import *

_dot = np.dot
_roll = np.roll

__all__ = ["matrix_ddk", "matrix_ddk_nc", "matrix_ddk_nc_diag", "matrix_ddk_so"]


def _phase_ddk(gauge, M, sc, np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype):
    # dtype *must* be passed through phase_dtype

    # This is the differentiated matrix with respect to k
    # See _phase.pyx, we are using exp(i k.R/r)
    #  (i R) * (i R) = - R**2
    # And since we have a double differentiation we can have
    # two dependent variables
    # We always do the Voigt representation
    #  Rd = dx^2, dy^2, dz^2, dzy, dxz, dyx
    if gauge == 'R':
        phases = phase_rsc(sc, k, dtype).reshape(-1, 1)
        Rs = _dot(sc.sc_off, sc.cell)
        Rd = - (Rs * Rs * phases).astype(dtype, copy=False)
        Ro = - (_roll(Rs, 1, axis=1) * phases).astype(dtype, copy=False) # z, x, y
        Ro *= _roll(Rs, -1, axis=1) # y, z, x
        del phases, Rs
        p_opt = 1

    elif gauge == 'r':
        M.finalize()
        rij = M.Rij()._csr._D
        phases = phase_rij(rij, sc, k, dtype).reshape(-1, 1)
        Rd = - (rij * rij * phases).astype(dtype, copy=False)
        Ro = - (_roll(rij, 1, axis=1) * phases).astype(dtype, copy=False) # z, x, y
        Ro *= _roll(rij, -1, axis=1) # y, z, x
        del rij, phases
        p_opt = 1

    return p_opt, Rd, Ro


def matrix_ddk(gauge, M, const int idx, sc,
               np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype)
    p_opt, Rd, Ro = _phase_ddk(gauge, M, sc, k, dtype)
    return _matrix_ddk(M._csr, idx, Rd, Ro, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_ddk(csr, const int idx, Rd, Ro, dtype, format, p_opt):

    # Return list
    dd = [None, None, None, None, None, None]

    if dtype == np.complex128:

        if format in ["array", "matrix", "dense"]:
            dd[:3] = _phase3_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
            dd[3:] = _phase3_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)

        else:
            # Default must be something else.
            dd[:3] = _phase3_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
            dd[3:] = _phase3_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    elif dtype == np.float64:
        if format in ["array", "matrix", "dense"]:
            dd[:3] = _phase3_array_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
            dd[3:] = _phase3_array_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)
        else:
            dd[:3] = _phase3_csr_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
            dd[3:] = _phase3_csr_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    elif dtype == np.complex64:
        if format in ["array", "matrix", "dense"]:
            dd[:3] = _phase3_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
            dd[3:] = _phase3_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)
        else:
            dd[:3] = _phase3_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
            dd[3:] = _phase3_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    elif dtype == np.float32:
        if format in ["array", "matrix", "dense"]:
            dd[:3] = _phase3_array_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
            dd[3:] = _phase3_array_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)
        else:
            dd[:3] = _phase3_csr_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd, p_opt)
            dd[3:] = _phase3_csr_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro, p_opt)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    else:
        raise ValueError("matrix_ddk: currently only supports dtype in [float32, float64, complex64, complex128].")

    return dd


def matrix_ddk_nc(gauge, M, sc,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, Rd, Ro = _phase_ddk(gauge, M, sc, k, dtype)
    return _matrix_ddk_nc(M._csr, Rd, Ro, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_ddk_nc(csr, Rd, Ro, dtype, format, p_opt):

    # Return list
    dd = [None, None, None, None, None, None]

    if dtype == np.complex128:

        if format in ["array", "matrix", "dense"]:
            dd[:3] = _phase3_nc_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
            dd[3:] = _phase3_nc_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)

        else:
            # Default must be something else.
            dd[:3] = _phase3_nc_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
            dd[3:] = _phase3_nc_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    elif dtype == np.complex64:
        if format in ["array", "matrix", "dense"]:
            dd[:3] = _phase3_nc_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
            dd[3:] = _phase3_nc_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)
        else:
            dd[:3] = _phase3_nc_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
            dd[3:] = _phase3_nc_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    else:
        raise ValueError("matrix_ddk_nc: currently only supports dtype in [complex64, complex128].")

    return dd


def matrix_ddk_nc_diag(gauge, M, const int idx, sc,
                       np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, Rd, Ro = _phase_ddk(gauge, M, sc, k, dtype)

    Rxx = Rd[:, 0].copy()
    Ryy = Rd[:, 1].copy()
    Rzz = Rd[:, 2].copy()
    del Rd
    Rzy = Ro[:, 0].copy()
    Rxz = Ro[:, 1].copy()
    Ryx = Ro[:, 2].copy()
    del Ro

    # Get each of them
    dxx = _matrix_ddk_nc_diag(M._csr, idx, Rxx, dtype, format, p_opt)
    dyy = _matrix_ddk_nc_diag(M._csr, idx, Ryy, dtype, format, p_opt)
    dzz = _matrix_ddk_nc_diag(M._csr, idx, Rzz, dtype, format, p_opt)
    dzy = _matrix_ddk_nc_diag(M._csr, idx, Rzy, dtype, format, p_opt)
    dxz = _matrix_ddk_nc_diag(M._csr, idx, Rxz, dtype, format, p_opt)
    dyx = _matrix_ddk_nc_diag(M._csr, idx, Ryx, dtype, format, p_opt)
    return dxx, dyy, dzz, dzy, dxz, dyx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_ddk_nc_diag(csr, const int idx, phases, dtype, format, p_opt):

    if dtype == np.complex128:

        if format in ["array", "matrix", "dense"]:
            return _phase_nc_diag_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)

        # Default must be something else.
        return _phase_nc_diag_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)

    elif dtype == np.complex64:
        if format in ["array", "matrix", "dense"]:
            return _phase_nc_diag_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)
        return _phase_nc_diag_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)

    raise ValueError("matrix_ddk_nc_diag: only supports dtype in [complex64, complex128].")


def matrix_ddk_so(gauge, M, sc,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, Rd, Ro = _phase_ddk(gauge, M, sc, k, dtype)
    return _matrix_ddk_so(M._csr, Rd, Ro, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_ddk_so(csr, Rd, Ro, dtype, format, p_opt):

    # Return list
    dd = [None, None, None, None, None, None]

    if dtype == np.complex128:

        if format in ["array", "matrix", "dense"]:
            dd[:3] = _phase3_so_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
            dd[3:] = _phase3_so_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)

        else:
            # Default must be something else.
            dd[:3] = _phase3_so_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
            dd[3:] = _phase3_so_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    elif dtype == np.complex64:
        if format in ["array", "matrix", "dense"]:
            dd[:3] = _phase3_so_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
            dd[3:] = _phase3_so_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)
        else:
            dd[:3] = _phase3_so_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, Rd, p_opt)
            dd[3:] = _phase3_so_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, Ro, p_opt)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    else:
        raise ValueError("matrix_ddk_so: currently only supports dtype in [complex64, complex128].")

    return dd
