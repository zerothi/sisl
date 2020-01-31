#!python
#cython: language_level=2
cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np

from ._phase import *
from ._matrix_phase3 import *

_dot = np.dot

__all__ = ['matrix_dk']


def matrix_dk(gauge, M, const int idx, sc,
              np.ndarray[np.float64_t, ndim=1, mode='c'] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)

    # This is the differentiated matrix with respect to k
    #  - i R
    if gauge == 'R':
        iRs = phase_rsc(sc, k, dtype).reshape(-1, 1)
        iRs = (-1j * _dot(sc.sc_off, sc.cell) * iRs).astype(dtype, copy=False)
        p_opt = 1

    elif gauge == 'r':
        M.finalize()
        rij = M.Rij()._csr._D
        iRs = (-1j * rij * phase_rij(rij, sc, k, dtype).reshape(-1, 1)).astype(dtype, copy=False)
        del rij
        p_opt = 0

    return _matrix_dk(M._csr, idx, iRs, dtype, format, p_opt)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _matrix_dk(csr, const int idx, iRs, dtype, format, p_opt):

    if dtype == np.complex128:

        if format in ['array', 'matrix', 'dense']:
            return _phase3_array_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)

        # Default must be something else.
        d1, d2, d3 = _phase3_csr_c128(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)

    elif dtype == np.complex64:
        if format in ['array', 'matrix', 'dense']:
            return _phase3_array_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)
        d1, d2, d3 = _phase3_csr_c64(csr.ptr, csr.ncol, csr.col, csr._D, idx, iRs, p_opt)
        return d1.asformat(format), d2.asformat(format), d3.asformat(format)

    raise ValueError('matrix_dk: currently only supports dtype in [complex64, complex128].')
