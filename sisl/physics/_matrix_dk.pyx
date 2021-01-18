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

__all__ = ["matrix_dk", "matrik_dk_nc", "matrik_dk_nc_diag", "matrik_dk_so"]


def _phase_dk(gauge, M, sc, np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype):
    # dtype *must* be passed through phase_dtype

    # This is the differentiated matrix with respect to k
    # See _phase.pyx, we are using exp(i k.R/r)
    #  i R
    if gauge == 'R':
        iRs = phase_rsc(sc, k, dtype).reshape(-1, 1)
        iRs = (1j * _dot(sc.sc_off, sc.cell) * iRs).astype(dtype, copy=False)
        p_opt = 1

    elif gauge == 'r':
        M.finalize()
        rij = M.Rij()._csr._D
        iRs = (1j * rij * phase_rij(rij, sc, k, dtype).reshape(-1, 1)).astype(dtype, copy=False)
        del rij
        p_opt = 0

    return p_opt, iRs


def matrix_dk(gauge, M, const int idx, sc,
              np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = _phase_dk(gauge, M, sc, k, dtype)
    return _matrix_dk(M._csr, idx, iRs, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_dk(csr, const int idx, iRs, dtype, format, p_opt):

    if dtype == np.complex128:

        if format in ["array", "matrix", "dense"]:
            return _phase3_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)

        # Default must be something else.
        d1, d2, d3 = _phase3_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)

    elif dtype == np.complex64:
        if format in ["array", "matrix", "dense"]:
            return _phase3_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)
        d1, d2, d3 = _phase3_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)

    raise ValueError("matrix_dk: currently only supports dtype in [complex64, complex128].")


def matrix_dk_nc(gauge, M, sc,
                 np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = _phase_dk(gauge, M, sc, k, dtype)
    return _matrix_dk_nc(M._csr, iRs, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_dk_nc(csr, iRs, dtype, format, p_opt):

    if dtype == np.complex128:

        if format in ["array", "matrix", "dense"]:
            return _phase3_nc_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)

        # Default must be something else.
        d1, d2, d3 = _phase3_nc_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)

    elif dtype == np.complex64:
        if format in ["array", "matrix", "dense"]:
            return _phase3_nc_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
        d1, d2, d3 = _phase3_nc_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)

    raise ValueError("matrix_dk_nc: currently only supports dtype in [complex64, complex128].")


def matrix_dk_nc_diag(gauge, M, const int idx, sc,
                      np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = _phase_dk(gauge, M, sc, k, dtype)

    phx = iRs[:, 0].copy()
    phy = iRs[:, 1].copy()
    phz = iRs[:, 2].copy()
    del iRs

    # Get each of them
    x = _matrix_dk_nc_diag(M._csr, idx, phx, dtype, format, p_opt)
    y = _matrix_dk_nc_diag(M._csr, idx, phy, dtype, format, p_opt)
    z = _matrix_dk_nc_diag(M._csr, idx, phz, dtype, format, p_opt)
    return x, y, z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_dk_nc_diag(csr, const int idx, phases, dtype, format, p_opt):

    if dtype == np.complex128:

        if format in ["array", "matrix", "dense"]:
            return _phase_nc_diag_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)

        # Default must be something else.
        return _phase_nc_diag_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)

    elif dtype == np.complex64:
        if format in ["array", "matrix", "dense"]:
            return _phase_nc_diag_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)
        return _phase_nc_diag_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)

    raise ValueError("matrix_dk_nc_diag: only supports dtype in [complex64, complex128].")


def matrix_dk_so(gauge, M, sc,
                 np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, iRs = _phase_dk(gauge, M, sc, k, dtype)
    return _matrix_dk_so(M._csr, iRs, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_dk_so(csr, iRs, dtype, format, p_opt):

    if dtype == np.complex128:

        if format in ["array", "matrix", "dense"]:
            return _phase3_so_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)

        # Default must be something else.
        d1, d2, d3 = _phase3_so_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)

    elif dtype == np.complex64:
        if format in ["array", "matrix", "dense"]:
            return _phase3_so_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
        d1, d2, d3 = _phase3_so_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, iRs, p_opt)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)

    raise ValueError("matrix_dk_so: currently only supports dtype in [complex64, complex128].")
