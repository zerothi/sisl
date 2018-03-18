# Enables usage of Cython decorators
cimport cython
from libc.math cimport sqrt, fabs

import numpy as np
# This enables Cython enhanced compatibilities
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def cross3(np.ndarray[np.float64_t, ndim=1] u, np.ndarray[np.float64_t, ndim=1] v):
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty([3], dtype=np.float64)
    y[0] = u[1] * v[2] - u[2] * v[1]
    y[1] = u[2] * v[0] - u[0] * v[2]
    y[2] = u[0] * v[1] - u[1] * v[0]
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
def dot3(np.ndarray[np.float64_t, ndim=1] u, np.ndarray[np.float64_t, ndim=1] v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@cython.boundscheck(False)
@cython.wraparound(False)
def product3(np.ndarray[np.float64_t, ndim=1] v):
    return v[0] * v[1] * v[2]


@cython.boundscheck(False)
@cython.wraparound(False)
def indices(np.ndarray[np.int32_t, ndim=1] search, np.ndarray[np.int32_t, ndim=1] value, int offset):
    """ Return indices of all `value` in the search array. If not found the index will be ``-1``

    Parameters
    ----------
    search : np.ndarray(np.int32)
        array to search in
    value : np.ndarray(np.int32)
        values to find the indices of in `search`
    offset : int
        index offset
    """
    cdef int n_search = search.shape[0]
    cdef int n_value = value.shape[0]

    # Ensure contiguous arrays
    cdef int[::1] SEARCH = search
    cdef int[::1] VALUE = value

    cdef np.ndarray[np.int32_t, ndim=1] idx = np.empty([n_value], dtype=np.int32)
    cdef int[::1] IDX = idx

    _indices(n_search, SEARCH, n_value, VALUE, offset, IDX)

    return idx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _indices(const int n_search, const int[::1] search,
              const int n_value, const int[::1] value,
              int offset, int[::1] idx) nogil:
    cdef int i, j

    # Fast return
    if n_value == 0:
        pass
    elif n_search == 0:
        for j in range(n_value):
            idx[j] = -1
        pass

    elif n_value > n_search:
        for j in range(n_value):
            idx[j] = -1
            for i in range(n_search):
                if value[j] == search[i]:
                    idx[j] = offset + i
                    break

    else:
        # We need to initialize
        for j in range(n_value):
            idx[j] = -1
        for i in range(n_search):
            for j in range(n_value):
                if value[j] == search[i]:
                    idx[j] = offset + i
                    break


@cython.boundscheck(False)
@cython.wraparound(False)
def indices_max_radius(np.ndarray[np.float64_t, ndim=2] dxyz, const double max_R):
    cdef int n = dxyz.shape[0]
    cdef double[:, ::1] dXYZ = dxyz
    cdef np.ndarray[np.int32_t, ndim=1] idx = np.empty([n], dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] dist = np.empty([n], dtype=np.float64)
    cdef int[::1] IDX = idx
    cdef double[::1] DIST = dist

    n = _indices_max_radius(dXYZ, max_R, DIST, IDX)

    if n == 0:
        return np.empty([0], dtype=np.int32), np.empty([0], dtype=np.float64)
    return idx[:n], dist[:n]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int _indices_max_radius(const double[:, ::1] dxyz, const double max_R,
                              double[::1] dist, int[::1] idx) nogil:
    cdef int n_xyz = dxyz.shape[0]
    cdef double R2 = max_R * max_R
    cdef double d
    cdef int i, n

    # Reset number of elements
    n = 0

    for i in range(n_xyz):
        if fabs(dxyz[i, 0]) > max_R or \
           fabs(dxyz[i, 1]) > max_R or \
           fabs(dxyz[i, 2]) > max_R:
            pass

        else:
            d = dxyz[i, 0] * dxyz[i, 0] + dxyz[i, 1] * dxyz[i, 1] + dxyz[i, 2] * dxyz[i, 2]
            if d <= R2:
                dist[n] = sqrt(d)
                idx[n] = i
                n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
def indices_below(np.ndarray[np.float64_t, ndim=1] dist, const double R):
    cdef double[::1] DIST = dist
    cdef np.ndarray[np.int32_t, ndim=1] idx = np.empty([dist.shape[0]], dtype=np.int32)
    cdef int[::1] IDX = idx
    cdef int n

    n = _indices_below(DIST, R, IDX)

    if n == 0:
        return np.empty([0], dtype=np.int32)
    return idx[:n]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int _indices_below(const double[::1] dist, const double R, int[::1] idx) nogil:
    cdef int n_dist = dist.shape[0]
    cdef int i, n
    n = 0
    for i in range(n_dist):
        if dist[i] <= R:
            idx[n] = i
            n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
def indices_between(np.ndarray[np.float64_t, ndim=1] dist, const double R1, const double R2):
    cdef double[::1] DIST = dist
    cdef np.ndarray[np.int32_t, ndim=1] idx = np.empty([dist.shape[0]], dtype=np.int32)
    cdef int[::1] IDX = idx
    cdef int n

    n = _indices_between(DIST, R1, R2, IDX)

    if n == 0:
        return np.empty([0], dtype=np.int32)
    return idx[:n]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int _indices_between(const double[::1] dist, const double R1, const double R2, int[::1] idx) nogil:
    cdef int n_dist = dist.shape[0]
    cdef int i, n
    n = 0
    for i in range(n_dist):
        if R1 < dist[i] and dist[i] <= R2:
            idx[n] = i
            n += 1
    return n
