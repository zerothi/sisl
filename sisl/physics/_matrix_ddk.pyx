#!python
#cython: language_level=2
cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np

from ._phase import *
from ._matrix_phase3 import *

_dot = np.dot
_roll = np.roll

__all__ = ['matrix_ddk']


def matrix_ddk(gauge, M, const int idx, sc,
               np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype)
    # This is the double differentiated matrix with respect to k
    #  - i R (- i R) == - R ** 2
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

    return _matrix_ddk(M._csr, idx, Rd, Ro, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_ddk(csr, const int idx, Rd, Ro, dtype, format, p_opt):

    # Return list
    dd = [None, None, None, None, None, None]

    if dtype == np.complex128:

        if format in ['array', 'matrix', 'dense']:
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
        if format in ['array', 'matrix', 'dense']:
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
        if format in ['array', 'matrix', 'dense']:
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
        if format in ['array', 'matrix', 'dense']:
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
        raise ValueError('matrix_ddk: currently only supports dtype in [float32, float64, complex64, complex128].')

    return dd
