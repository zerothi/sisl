# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython
from libc.math cimport atan2, atan2f, sqrt, sqrtf

import numpy as np

from numpy cimport dtype, ndarray

from sisl._core._dtypes cimport floats_st, type2dtype


@cython.boundscheck(False)
@cython.wraparound(False)
def cross3(const floats_st[::1] u, const floats_st[::1] v):
    cdef object dtyp = type2dtype[floats_st](1)
    cdef ndarray[floats_st, mode='c'] Y = np.empty([3], dtype=dtyp)
    cdef floats_st[::1] y = Y
    y[0] = u[1] * v[2] - u[2] * v[1]
    y[1] = u[2] * v[0] - u[0] * v[2]
    y[2] = u[0] * v[1] - u[1] * v[0]
    return Y


@cython.boundscheck(False)
@cython.wraparound(False)
def dot3(const floats_st[::1] u, const floats_st[::1] v):
    cdef floats_st r
    r = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
def product3(const floats_st[::1] v):
    cdef floats_st r
    r = v[0] * v[1] * v[2]
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def is_ascending(const floats_st[::1] v):
    cdef Py_ssize_t i
    for i in range(1, v.shape[0]):
        if v[i-1] > v[i]:
            return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def xyz_to_spherical_cos_phi(floats_st[::1] x,
                             floats_st[::1] y,
                             floats_st[::1] z):
    """ In x, y, z coordinates shifted to origo

    Returns x = R, y = theta, z = cos_phi
    """
    cdef Py_ssize_t i
    cdef floats_st R

    if floats_st is cython.float:
        for i in range(x.shape[0]):
            # theta (radians)
            R = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])
            y[i] = atan2f(y[i], x[i])
            # Radius
            x[i] = R
            # cos(phi)
            if R > 0.:
                z[i] = z[i] / R
            else:
                z[i] = 0.
    else:
        for i in range(x.shape[0]):
            # theta (radians)
            R = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])
            y[i] = atan2(y[i], x[i])
            # Radius
            x[i] = R
            # cos(phi)
            if R > 0.:
                z[i] = z[i] / R
            else:
                z[i] = 0.
