# Enables usage of Cython decorators
cimport cython

import numpy as np
# This enables Cython enhanced compatibilities
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cross3(np.ndarray[np.float64_t, ndim=1] u, np.ndarray[np.float64_t, ndim=1] v):
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty([u.shape[0]], dtype=np.float64)
    y[0] = u[1] * v[2] - u[2] * v[1]
    y[1] = u[2] * v[0] - u[0] * v[2]
    y[2] = u[0] * v[1] - u[1] * v[0]
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dot3(np.ndarray[np.float64_t, ndim=1] u, np.ndarray[np.float64_t, ndim=1] v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef product3(np.ndarray[np.float64_t, ndim=1] v):
    return v[0] * v[1] * v[2]
