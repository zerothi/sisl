# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython
from libc.math cimport fabs, fabsf

from numpy import complex64, complex128, dot, exp, float32, float64, ndarray, ones, pi

from sisl._core._dtypes cimport floats_st


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline bint is_gamma(const floats_st[::1] k) noexcept nogil:
    if floats_st is cython.float:
        if fabsf(k[0]) > 0.0000001:
            return 0
        if fabsf(k[1]) > 0.0000001:
            return 0
        if fabsf(k[2]) > 0.0000001:
            return 0

    else:
        if fabs(k[0]) > 0.0000001:
            return 0
        if fabs(k[1]) > 0.0000001:
            return 0
        if fabs(k[2]) > 0.0000001:
            return 0
    return 1


def phase_dtype(const floats_st[::1] k, M_dtype, R_dtype, force_complex: bool=False):
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


def phase_rsc(sc, const floats_st[::1] k, dtype):
    """ Calculate the phases for the supercell interactions using k """

    # Figure out if this is a Gamma point or not
    if is_gamma(k):
        phases = ones(sc.sc_off.shape[0], dtype=dtype)
    else:
        # This is equivalent to (k.rcell).(sc_off.cell)^T
        # since rcell.cell^T == I * 2 * pi
        phases = exp((2j * pi) * dot(sc.sc_off, k)).astype(dtype, copy=False)

    return phases


def phase_rij(rij, sc, const floats_st[::1] k, dtype):
    """ Calculate the phases for the distance matrix using k """

    # Figure out if this is a Gamma point or not
    if is_gamma(k):
        phases = ones(rij.shape[0], dtype=dtype)
    else:
        phases = exp(1j * dot(rij, dot(k, sc.rcell))).astype(dtype, copy=False)

    return phases
