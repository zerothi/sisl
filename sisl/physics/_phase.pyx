#!python
#cython: language_level=2
cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np
from numpy import pi, dot, ones, exp
from numpy import float32, float64, complex64, complex128
from numpy import ndarray
from numpy cimport float32_t, float64_t, complex64_t, complex128_t
from numpy cimport ndarray


__all__ = ['phase_dtype', 'phase_rsc', 'phase_rij']


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


def phase_dtype(ndarray[float64_t, ndim=1, mode='c'] k, M_dtype, R_dtype, force_complex=False):
    if is_gamma(k) and not force_complex:
        if R_dtype is None:
            return M_dtype
        elif R_dtype == complex64 or R_dtype == complex128:
            return R_dtype
        elif M_dtype == complex64 or M_dtype == complex128:
            return M_dtype
    else:
        if R_dtype is None:
            if M_dtype == float32:
                return complex64
            elif M_dtype == float64:
                return complex128
            else:
                # M *must* be complex
                return M_dtype
        elif R_dtype == float32:
            return complex64
        elif R_dtype == float64:
            return complex128
    return R_dtype


def phase_rsc(sc, ndarray[float64_t, ndim=1, mode='c'] k, dtype):
    """ Calculate the phases for the supercell interactions using k """

    # Figure out if this is a Gamma point or not
    if is_gamma(k):
        phases = ones(sc.sc_off.shape[0], dtype=dtype)
    else:
        # This is equivalent to (k.rcell).(sc_off.cell)^T
        # since rcell.cell^T == I * 2 * pi
        phases = exp(-1j * dot(sc.sc_off, k * 2 * pi)).astype(dtype, copy=False)

    return phases


def phase_rij(rij, sc, ndarray[float64_t, ndim=1, mode='c'] k, dtype):
    """ Calculate the phases for the distance matrix using k """

    # Figure out if this is a Gamma point or not
    if is_gamma(k):
        phases = ones(rij.shape[0], dtype=dtype)
    else:
        phases = exp(-1j * dot(rij, dot(k, sc.rcell))).astype(dtype, copy=False)

    return phases
