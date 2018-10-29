#!python
#cython: language_level=2
cimport cython
from libc.math cimport atan2, sqrt

import numpy as np
# This enables Cython enhanced compatibilities
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def cross3(np.ndarray[np.float64_t, ndim=1, mode='c'] u, np.ndarray[np.float64_t, ndim=1, mode='c'] v):
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] y = np.empty([3], dtype=np.float64)
    y[0] = u[1] * v[2] - u[2] * v[1]
    y[1] = u[2] * v[0] - u[0] * v[2]
    y[2] = u[0] * v[1] - u[1] * v[0]
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
def dot3(np.ndarray[np.float64_t, ndim=1, mode='c'] u, np.ndarray[np.float64_t, ndim=1, mode='c'] v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@cython.boundscheck(False)
@cython.wraparound(False)
def product3(np.ndarray[np.float64_t, ndim=1, mode='c'] v):
    return v[0] * v[1] * v[2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def is_ascending(np.ndarray[np.float64_t, ndim=1, mode='c'] v):
    cdef double[::1] V = v
    cdef int i
    for i in range(1, V.shape[0]):
        if V[i-1] > V[i]:
            return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def xyz_to_spherical_cos_phi(np.ndarray[np.float64_t, ndim=1, mode='c'] x,
                             np.ndarray[np.float64_t, ndim=1, mode='c'] y,
                             np.ndarray[np.float64_t, ndim=1, mode='c'] z):
    """ In x, y, z coordinates shifted to origo

    Returns x = R, y = theta, z = cos_phi
    """
    cdef double[::1] X = x
    cdef double[::1] Y = y
    cdef double[::1] Z = z
    cdef int i
    cdef double R
    for i in range(X.shape[0]):
        # theta (radians)
        R = sqrt(X[i] * X[i] + Y[i] * Y[i] + Z[i] * Z[i])
        Y[i] = atan2(Y[i], X[i])
        # Radius
        X[i] = R
        # cos(phi)
        if R > 0.:
            Z[i] = Z[i] / R
        else:
            Z[i] = 0.
