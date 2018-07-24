# Import libc functions
cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np

from _matrix_k_factor_dtype import *

_dot = np.dot

def matrix_ddk(gauge, M, const int idx, sc,
               np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    if gauge == 'R':
        return _matrix_ddk_R(M._csr, idx, sc, k, dtype, format)
    raise ValueError('Currently only R gauge has been implemented in matrix_ddk.')


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
def _matrix_ddk_R(csr, const int idx, sc,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    """ Setup the k-point matrix """

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

    phases.shape = (-1, 1)
    dd = [None, None, None, None, None, None]

    # This is the double differentiated matrix with respect to k
    #  - i R (- i R) == - R ** 2
    Rs = _dot(sc.sc_off, sc.cell).astype(dtype, copy=False)
    Rd = - Rs * Rs * phases.reshape(-1, 1)
    Ro = - np.roll(Rs, 1, axis=1) * phases.reshape(-1, 1) # z, x, y
    Ro *= np.roll(Rs, -1, axis=1) # y, z, x
    del Rs

    if dtype == np.complex128:
        
        if format == 'array':
            dd[:3] = _k_R_factor_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
        elif format == 'matrix' or format == 'dense':
            dd[:3] = _k_R_factor_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
            dd[0] = np.asmatrix(dd[0])
            dd[1] = np.asmatrix(dd[1])
            dd[2] = np.asmatrix(dd[2])
            dd[3] = np.asmatrix(dd[3])
            dd[4] = np.asmatrix(dd[4])
            dd[5] = np.asmatrix(dd[5])

        else:
            # Default must be something else.
            dd[:3] = _k_R_factor_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    elif dtype == np.float64:
        if format == 'array':
            dd[:3] = _k_R_factor_array_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_array_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
        elif format == 'matrix' or format == 'dense':
            dd[:3] = _k_R_factor_array_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_array_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
            dd[0] = np.asmatrix(dd[0])
            dd[1] = np.asmatrix(dd[1])
            dd[2] = np.asmatrix(dd[2])
            dd[3] = np.asmatrix(dd[3])
            dd[4] = np.asmatrix(dd[4])
            dd[5] = np.asmatrix(dd[5])
        else:
            dd[:3] = _k_R_factor_csr_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_csr_f64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    elif dtype == np.complex64:
        if format == 'array':
            dd[:3] = _k_R_factor_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
        elif format == 'matrix' or format == 'dense':
            dd[:3] = _k_R_factor_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
            dd[0] = np.asmatrix(dd[0])
            dd[1] = np.asmatrix(dd[1])
            dd[2] = np.asmatrix(dd[2])
            dd[3] = np.asmatrix(dd[3])
            dd[4] = np.asmatrix(dd[4])
            dd[5] = np.asmatrix(dd[5])
        else:
            dd[:3] = _k_R_factor_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    elif dtype == np.float32:
        if format == 'array':
            dd[:3] = _k_R_factor_array_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_array_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
        elif format == 'matrix' or format == 'dense':
            dd[:3] = _k_R_factor_array_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_array_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
            dd[0] = np.asmatrix(dd[0])
            dd[1] = np.asmatrix(dd[1])
            dd[2] = np.asmatrix(dd[2])
            dd[3] = np.asmatrix(dd[3])
            dd[4] = np.asmatrix(dd[4])
            dd[5] = np.asmatrix(dd[5])
        else:
            dd[:3] = _k_R_factor_csr_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Rd)
            dd[3:] = _k_R_factor_csr_f32(csr.ptr, csr.ncol, csr.col, csr._D, idx, Ro)
            dd[0] = dd[0].asformat(format)
            dd[1] = dd[1].asformat(format)
            dd[2] = dd[2].asformat(format)
            dd[3] = dd[3].asformat(format)
            dd[4] = dd[4].asformat(format)
            dd[5] = dd[5].asformat(format)

    else:
        raise ValueError('matrix_ddk_R: currently only supports dtype in [float32, float64, complex64, complex128].')

    return dd
