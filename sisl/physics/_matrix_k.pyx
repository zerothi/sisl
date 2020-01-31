#!python
#cython: language_level=2
cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np

from ._phase import *
from ._matrix_phase import *
from ._matrix_phase_nc import *
from ._matrix_phase_nc_diag import *
from ._matrix_phase_so import *

__all__ = ['matrix_k', 'matrix_k_nc', 'matrix_k_so', 'matrix_k_nc_diag']


def matrix_k(gauge, M, const int idx, sc,
             np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype)
    if gauge == 'R':
        phases = phase_rsc(sc, k, dtype)
        p_opt = 1

    elif gauge == 'r':
        M.finalize()
        phases = phase_rij(M.Rij()._csr._D, sc, k, dtype)
        p_opt = 0

    return _matrix_k(M._csr, idx, phases, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_k(csr, const int idx, phases, dtype, format, p_opt):

    if dtype == np.complex128:

        if format in ['array', 'matrix', 'dense']:
            return _phase_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)

        # Default must be something else.
        return _phase_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)

    elif dtype == np.float64:
        if format in ['array', 'matrix', 'dense']:
            return _array_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx)
        return _csr_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx).asformat(format)

    elif dtype == np.complex64:
        if format in ['array', 'matrix', 'dense']:
            return _phase_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)
        return _phase_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)

    elif dtype == np.float32:
        if format in ['array', 'matrix', 'dense']:
            return _array_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx)
        return _csr_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx).asformat(format)

    raise ValueError('matrix_k: currently only supports dtype in [float32, float64, complex64, complex128].')


def matrix_k_nc(gauge, M, sc,
                np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    if gauge == 'R':
        phases = phase_rsc(sc, k, dtype)
        p_opt = 1
    elif gauge == 'r':
        M.finalize()
        phases = phase_rij(M.Rij()._csr._D, sc, k, dtype)
        p_opt = 0
    return _matrix_k_nc(M._csr, phases, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_k_nc(csr, phases, dtype, format, p_opt):

    if csr._D.shape[1] < 4:
        raise ValueError('matrix_k_nc requires input matrix to have 4 components')

    if dtype == np.complex128:

        if format in ['array', 'matrix', 'dense']:
            return _phase_nc_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt)

        # Default must be something else.
        return _phase_nc_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt).asformat(format)

    elif dtype == np.complex64:
        if format in ['array', 'matrix', 'dense']:
            return _phase_nc_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt)
        return _phase_nc_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt).asformat(format)

    raise ValueError('matrix_k_nc: only supports dtype in [complex64, complex128].')


def matrix_k_so(gauge, M, sc,
                np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    if gauge == 'R':
        phases = phase_rsc(sc, k, dtype)
        p_opt = 1
    elif gauge == 'r':
        M.finalize()
        phases = phase_rij(M.Rij()._csr._D, sc, k, dtype)
        p_opt = 0
    return _matrix_k_so(M._csr, phases, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_k_so(csr, phases, dtype, format, p_opt):

    if csr._D.shape[1] < 8:
        raise ValueError('matrix_sc_so requires input matrix to have 8 components')

    if dtype == np.complex128:

        if format in ['array', 'matrix', 'dense']:
            return _phase_so_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt)

        # Default must be something else.
        return _phase_so_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt).asformat(format)

    elif dtype == np.complex64:
        if format in ['array', 'matrix', 'dense']:
            return _phase_so_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt)
        return _phase_so_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt).asformat(format)

    raise ValueError('matrix_sc_so: only supports dtype in [complex64, complex128].')


def matrix_k_nc_diag(gauge, M, const int idx, sc,
                     np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    if gauge == 'R':
        phases = phase_rsc(sc, k, dtype)
        p_opt = 1
    elif gauge == 'r':
        M.finalize()
        phases = phase_rij(M.Rij()._csr._D, sc, k, dtype)
        p_opt = 0
    return _matrix_k_nc_diag(M._csr, idx, phases, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_k_nc_diag(csr, const int idx, phases, dtype, format, p_opt):

    if dtype == np.complex128:

        if format in ['array', 'matrix', 'dense']:
            return _phase_nc_diag_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)

        # Default must be something else.
        return _phase_nc_diag_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)

    elif dtype == np.complex64:
        if format in ['array', 'matrix', 'dense']:
            return _phase_nc_diag_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)
        return _phase_nc_diag_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)

    raise ValueError('matrix_k_nc_diag: only supports dtype in [complex64, complex128].')
