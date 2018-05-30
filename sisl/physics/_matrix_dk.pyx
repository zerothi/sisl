# Import libc functions
cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np

from _matrix_k_factor_dtype import *

_dot = np.dot

def matrix_dk(gauge, csr, const int idx, sc,
              np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    if gauge == 'R':
        return _matrix_dk_R(csr, idx, sc, k, dtype, format)
    raise ValueError('Currently only R gauge has been implemented in matrix_dk.')


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
def _matrix_dk_R(csr, const int idx, sc,
                 np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    """ Setup the k-point matrix """

    if dtype is None:
        dtype = np.complex128

    if is_gamma(k):
        phases = np.ones(sc.sc_off.shape[0], dtype=dtype)
    else:
        phases = np.exp(-1j * _dot(_dot(_dot(sc.rcell, k), sc.cell), sc.sc_off.T)).astype(dtype, copy=False)

    iRs = -1j * _dot(sc.sc_off, sc.cell).astype(dtype, copy=False) * phases.reshape(-1, 1)

    if dtype == np.complex128:
        
        if format == 'array':
            return _k_R_factor_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs)
        elif format == 'matrix' or format == 'dense':
            d1, d2, d3 = _k_R_factor_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs)
            return np.asmatrix(d1), np.asmatrix(d2), np.asmatrix(d3)
        
        # Default must be something else.
        d1, d2, d3 = _k_R_factor_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)
    
    elif dtype == np.complex64:
        if format == 'array':
            return _k_R_factor_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs)
        elif format == 'matrix' or format == 'dense':
            d1, d2, d3 = _k_R_factor_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs)
            return np.asmatrix(d1), np.asmatrix(d2), np.asmatrix(d3)
        d1, d2, d3 = _k_R_factor_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)

    raise ValueError('matrix_dk_R: currently only supports dtype in [complex64, complex128].')
