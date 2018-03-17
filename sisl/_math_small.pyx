# Enables usage of Cython decorators
cimport cython

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
    cdef int [::1] SEARCH = search
    cdef int [::1] VALUE = value
    
    cdef np.ndarray[np.int32_t, ndim=1] indices = np.empty([n_value], dtype=np.int32)
    cdef int [::1] IDX = indices

    _indices(n_search, SEARCH, n_value, VALUE, offset, IDX)
    
    return indices


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _indices(const int n_search, const int[:] search,
              const int n_value, const int[:] value,
              int offset, int[:] idx):
    cdef int i, j

    # Fast return
    if n_value == 0:
        return
    elif n_search == 0:
        for j in range(n_value):
            idx[j] = -1
        return

    if n_value > n_search:
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
