# Import libc functions
cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np

from _matrix_k_dtype import *
from _matrix_k_nc_dtype import *
from _matrix_k_so_dtype import *
from _matrix_diag_k_nc_dtype import *

_dot = np.dot

def matrix_k(gauge, M, const int idx, sc,
             np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    if gauge == 'R':
        return _matrix_k_R(M._csr, idx, sc, k, dtype, format)
    elif gauge == 'r':
        # The current gauge implementation will recreate the matrix for every k-point
        M.finalize()
        xij = M.Rij()
    raise ValueError('Currently only R gauge has been implemented in matrix_k.')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline int is_gamma(const double[::1] k) nogil:
    if fabs(k[0]) > 0.0000001:
        return 0
    if fabs(k[1]) > 0.0000001:
        return 0
    if fabs(k[2]) > 0.0000001:
        return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_k_R(csr, const int idx, sc,
                np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    """ Setup the k-point matrix """

    # Figure out if this is a Gamma point or not
    if is_gamma(k):
        if dtype is None:
            dtype = np.float64
        phases = np.ones(sc.sc_off.shape[0], dtype=dtype)
    else:
        if dtype == np.float32:
            dtype = np.complex64
        elif dtype == np.float64:
            dtype = np.complex128
        elif dtype is None:
            dtype = np.complex128
        phases = np.exp(-1j * _dot(_dot(_dot(sc.rcell, k), sc.cell), sc.sc_off.T)).astype(dtype, copy=False)

    if dtype == np.complex128:
        
        if format == 'array':
            return _k_R_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_k_R_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases))
        
        # Default must be something else.
        return _k_R_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases).asformat(format)
    
    elif dtype == np.float64:
        if format == 'array':
            return _k_R_array_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_k_R_array_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx))
        return _k_R_csr_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx).asformat(format)

    elif dtype == np.complex64:
        if format == 'array':
            return _k_R_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_k_R_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases))
        return _k_R_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases).asformat(format)

    elif dtype == np.float32:
        if format == 'array':
            return _k_R_array_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_k_R_array_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx))
        return _k_R_csr_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx).asformat(format)

    raise ValueError('matrix_k_R: currently only supports dtype in [float32, float64, complex64, complex128].')

    
def matrix_k_nc(gauge, M, sc,
                np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    if gauge == 'R':
        return _matrix_k_R_nc(M._csr, sc, k, dtype, format)
    raise ValueError('Currently only R gauge has been implemented in matrix_k_nc.')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_k_R_nc(csr, sc,
                   np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    """ Setup the k-point matrix """

    # Figure out if this is a Gamma point or not
    if dtype is None:
        dtype = np.complex128
    if is_gamma(k):
        phases = np.ones(sc.sc_off.shape[0], dtype=dtype)
    else:
        phases = np.exp(-1j * _dot(_dot(_dot(sc.rcell, k), sc.cell), sc.sc_off.T)).astype(dtype, copy=False)

    if csr._D.shape[1] < 4:
        raise ValueError('matrix_k_R_nc requires input matrix to have 4 components')

    if dtype == np.complex128:
        
        if format == 'array':
            return _k_R_nc_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_k_R_nc_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases))
        
        # Default must be something else.
        return _k_R_nc_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases).asformat(format)
    
    elif dtype == np.complex64:
        if format == 'array':
            return _k_R_nc_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_k_R_nc_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases))
        return _k_R_nc_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases).asformat(format)

    raise ValueError('matrix_k_R_nc: only supports dtype in [complex64, complex128].')


def matrix_k_so(gauge, M, sc,
                np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    if gauge == 'R':
        return _matrix_k_R_so(M._csr, sc, k, dtype, format)
    raise ValueError('Currently only R gauge has been implemented in matrix_k_so.')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_k_R_so(csr, sc,
                   np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    """ Setup the k-point matrix """

    # Figure out if this is a Gamma point or not
    if dtype is None:
        dtype = np.complex128
    if is_gamma(k):
        phases = np.ones(sc.sc_off.shape[0], dtype=dtype)
    else:
        phases = np.exp(-1j * _dot(_dot(_dot(sc.rcell, k), sc.cell), sc.sc_off.T)).astype(dtype, copy=False)

    if csr._D.shape[1] < 8:
        raise ValueError('matrix_k_R_so requires input matrix to have 8 components')

    if dtype == np.complex128:
        
        if format == 'array':
            return _k_R_so_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_k_R_so_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases))
        
        # Default must be something else.
        return _k_R_so_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, phases).asformat(format)
    
    elif dtype == np.complex64:
        if format == 'array':
            return _k_R_so_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_k_R_so_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases))
        return _k_R_so_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, phases).asformat(format)

    raise ValueError('matrix_k_R_so: only supports dtype in [complex64, complex128].')


def matrix_diag_k_nc(gauge, M, const int idx, sc,
                     np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    if gauge == 'R':
        return _matrix_diag_k_R_nc(M._csr, idx, sc, k, dtype, format)
    raise ValueError('Currently only R gauge has been implemented in matrix_diag_k_nc.')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_diag_k_R_nc(csr, const int idx, sc,
                        np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    """ Setup the k-point matrix """

    # Figure out if this is a Gamma point or not
    if dtype is None:
        dtype = np.complex128
    if is_gamma(k):
        phases = np.ones(sc.sc_off.shape[0], dtype=dtype)
    else:
        phases = np.exp(-1j * _dot(_dot(_dot(sc.rcell, k), sc.cell), sc.sc_off.T)).astype(dtype, copy=False)

    if dtype == np.complex128:
        
        if format == 'array':
            return _diag_k_R_nc_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_diag_k_R_nc_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases))
        
        # Default must be something else.
        return _diag_k_R_nc_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases).asformat(format)
    
    elif dtype == np.complex64:
        if format == 'array':
            return _diag_k_R_nc_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases)
        elif format == 'matrix' or format == 'dense':
            return np.asmatrix(_diag_k_R_nc_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases))
        return _diag_k_R_nc_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases).asformat(format)

    raise ValueError('matrix_diag_k_R_nc: only supports dtype in [complex64, complex128].')
