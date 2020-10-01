cimport cython
from libc.math cimport sqrt, fabs

import numpy as np
# This enables Cython enhanced compatibilities
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def indices_only(np.ndarray[np.int32_t, ndim=1, mode='c'] search, np.ndarray[np.int32_t, ndim=1, mode='c'] value):
    """ Return indices of all `value` in the search array.

    Parameters
    ----------
    search : np.ndarray(np.int32)
        array to search in
    value : np.ndarray(np.int32)
        values to find the indices of in `search`
    """
    # Ensure contiguous arrays
    cdef int[::1] SEARCH = search
    cdef int[::1] VALUE = value
    cdef Py_ssize_t n_search = SEARCH.shape[0]
    cdef Py_ssize_t n_value = VALUE.shape[0]

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] idx = np.empty([max(n_value, n_search)], dtype=np.int32)
    cdef int[::1] IDX = idx

    cdef Py_ssize_t n = _indices_only(n_search, SEARCH, n_value, VALUE, IDX)

    return idx[:n]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_only(const int n_search, const int[::1] search,
                       const int n_value, const int[::1] value,
                       int[::1] idx) nogil:
    cdef Py_ssize_t i, j, n

    # Fast return
    if n_value == 0:
        return 0
    elif n_search == 0:
        return 0

    elif n_value > n_search:
        n = 0
        for j in range(n_value):
            for i in range(n_search):
                if value[j] == search[i]:
                    idx[n] = i
                    n += 1
                    break

    else:
        n = 0
        for i in range(n_search):
            for j in range(n_value):
                if value[j] == search[i]:
                    idx[n] = i
                    n += 1
                    break
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
def indices(np.ndarray[np.int32_t, ndim=1, mode='c'] search, np.ndarray[np.int32_t, ndim=1, mode='c'] value, int offset, both_sorted=False):
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
    # Ensure contiguous arrays
    cdef int[::1] SEARCH = search
    cdef int[::1] VALUE = value
    cdef int n_search = SEARCH.shape[0]
    cdef int n_value = VALUE.shape[0]

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] idx = np.empty([n_value], dtype=np.int32)
    cdef int[::1] IDX = idx

    if both_sorted:
        _indices_sorted_arrays(n_search, SEARCH, n_value, VALUE, offset, IDX)
    else:
        _indices(n_search, SEARCH, n_value, VALUE, offset, IDX)

    return idx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _indices(const int n_search, const int[::1] search,
                   const int n_value, const int[::1] value,
                   const int offset, int[::1] idx) nogil:
    cdef Py_ssize_t i, j

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
@cython.initializedcheck(False)
cdef void _indices_sorted_arrays(const int n_search, const int[::1] search,
                                 const int n_value, const int[::1] value,
                                 const int offset, int[::1] idx) nogil:
    cdef Py_ssize_t i, j
    cdef int cvalue, csearch

    # Fast return
    if n_value == 0:
        pass
    elif n_search == 0:
        for j in range(n_value):
            idx[j] = -1
        return

    i = 0
    j = 0
    while (i < n_search) and (j < n_value):
        csearch = search[i]
        cvalue = value[j]
        if csearch == cvalue:
            idx[j] = i + offset
            j += 1
        elif csearch < cvalue:
            i += 1
        elif csearch > cvalue:
            idx[j] = -1
            j += 1
    for j in range(j, n_value):
        idx[j] = -1


@cython.boundscheck(False)
@cython.wraparound(False)
def indices_in_sphere(np.ndarray[np.float64_t, ndim=2, mode='c'] dxyz, const double R):
    """ Indices for all coordinates that are within a sphere of radius `R`

    Parameters
    ----------
    dxyz : ndarray(np.float64)
       coordinates centered around the sphere
    R : float
       radius of sphere to check

    Returns
    -------
    index : np.ndarray(np.int32)
       indices of all dxyz coordinates that are within the sphere of radius `R`
    """
    cdef double[:, ::1] dXYZ = dxyz
    cdef Py_ssize_t n = dXYZ.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] idx = np.empty([n], dtype=np.int32)
    cdef int[::1] IDX = idx

    n = _indices_in_sphere(dXYZ, R, IDX)

    if n == 0:
        return np.empty([0], dtype=np.int32)
    return idx[:n].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_in_sphere(const double[:, ::1] dxyz, const double R, int[::1] idx) nogil:
    cdef Py_ssize_t N = dxyz.shape[0]
    cdef double R2 = R * R
    cdef Py_ssize_t i, n

    # Reset number of elements
    n = 0

    for i in range(N):
        if all_fabs_le(dxyz, i, R):
            if dxyz[i, 0] * dxyz[i, 0] + dxyz[i, 1] * dxyz[i, 1] + dxyz[i, 2] * dxyz[i, 2] <= R2:
                idx[n] = i
                n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
def indices_in_sphere_with_dist(np.ndarray[np.float64_t, ndim=2, mode='c'] dxyz, const double R):
    """ Indices and the distances for all coordinates that are within a sphere of radius `R`

    Parameters
    ----------
    dxyz : ndarray(np.float64)
       coordinates centered around the sphere
    R : float
       radius of sphere to check

    Returns
    -------
    index : np.ndarray(np.int32)
       indices of all dxyz coordinates that are within the sphere of radius `R`
    dist : np.ndarray(np.float64)
       distances for the coordinates within the sphere of radius `R` (corresponds to `index`)
    """
    cdef double[:, ::1] dXYZ = dxyz
    cdef Py_ssize_t n = dXYZ.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] idx = np.empty([n], dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] dist = np.empty([n], dtype=np.float64)
    cdef int[::1] IDX = idx
    cdef double[::1] DIST = dist

    n = _indices_in_sphere_with_dist(dXYZ, R, DIST, IDX)

    if n == 0:
        return np.empty([0], dtype=np.int32), np.empty([0], dtype=np.float64)
    return idx[:n].copy(), dist[:n].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_in_sphere_with_dist(const double[:, ::1] dxyz, const double R,
                                             double[::1] dist, int[::1] idx) nogil:
    cdef Py_ssize_t N = dxyz.shape[0]
    cdef double R2 = R * R
    cdef double d
    cdef Py_ssize_t i, n

    # Reset number of elements
    n = 0

    for i in range(N):
        if all_fabs_le(dxyz, i, R):
            d = dxyz[i, 0] * dxyz[i, 0] + dxyz[i, 1] * dxyz[i, 1] + dxyz[i, 2] * dxyz[i, 2]
            if d <= R2:
                dist[n] = sqrt(d)
                idx[n] = i
                n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
def indices_le(np.ndarray a, const double V):
    """ Indices for all values in `a` that are ``<= V``

    Parameters
    ----------
    a : np.ndarray(np.float64)
       array to check if 2D, all last dimension values must be ``<= V``
    V : float
       value that is checked against

    Returns
    -------
    index : np.ndarray(np.int32)
       indices for the values in `a` which are less than or equal to `V`
    """
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] idx = np.empty([a.shape[0]], dtype=np.int32)
    cdef int[::1] IDX = idx

    cdef Py_ssize_t ndim = a.ndim
    cdef double[::1] A1
    cdef double[:, ::1] A2
    cdef Py_ssize_t n

    if a.dtype != np.float64:
        raise ValueError('indices_le requires input array to be of float64 type')

    if ndim == 1:
        A1 = a
        n = _indices_le1(A1, V, IDX)

    elif ndim == 2:
        A2 = a
        n = _indices_le2(A2, V, IDX)

    if n == 0:
        return np.empty([0], dtype=np.int32)
    return idx[:n].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_le1(const double[::1] a, const double V, int[::1] idx) nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if a[i] <= V:
            idx[n] = i
            n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline int all_le(const double[:, ::1] a, const Py_ssize_t i, const double V) nogil:
    cdef Py_ssize_t j
    for j in range(a.shape[1]):
        if a[i, j] > V:
            return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_le2(const double[:, ::1] a, const double V, int[::1] idx) nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if all_le(a, i, V):
            idx[n] = i
            n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
def indices_fabs_le(np.ndarray a, const double V):
    """ Indices for all values in `a` that are ``| | <= V``

    Parameters
    ----------
    a : np.ndarray(np.float64)
       array to check if 2D, all last dimension values must be ``| | <= V``
    V : float
       value that is checked against

    Returns
    -------
    index : np.ndarray(np.int32)
       indices for the values in ``|a|`` which are less than or equal to `V`
    """
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] idx = np.empty([a.shape[0]], dtype=np.int32)
    cdef int[::1] IDX = idx

    cdef Py_ssize_t ndim = a.ndim
    cdef double[::1] A1
    cdef double[:, ::1] A2
    cdef Py_ssize_t n

    if a.dtype != np.float64:
        raise ValueError('indices_fabs_le requires input array to be of float64 type')

    if ndim == 1:
        A1 = a
        n = _indices_fabs_le1(A1, V, IDX)

    elif ndim == 2:
        A2 = a
        n = _indices_fabs_le2(A2, V, IDX)

    if n == 0:
        return np.empty([0], dtype=np.int32)
    return idx[:n].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_fabs_le1(const double[::1] a, const double V, int[::1] idx) nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if fabs(a[i]) <= V:
            idx[n] = i
            n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline int all_fabs_le(const double[:, ::1] a, const Py_ssize_t i, const double V) nogil:
    cdef Py_ssize_t j
    for j in range(a.shape[1]):
        if fabs(a[i, j]) > V:
            return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int _indices_fabs_le2(const double[:, ::1] a, const double V, int[::1] idx) nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if all_fabs_le(a, i, V):
            idx[n] = i
            n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
def indices_gt_le(np.ndarray a, const double V1, const double V2):
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] idx = np.empty([a.shape[0]], dtype=np.int32)
    cdef int[::1] IDX = idx

    cdef Py_ssize_t ndim = a.ndim
    cdef double[::1] A1
    cdef double[:, ::1] A2
    cdef Py_ssize_t n

    if a.dtype != np.float64:
        raise ValueError('indices_gt_le requires input array to be of float64 type')

    if ndim == 1:
        A1 = a
        n = _indices_gt_le1(A1, V1, V2, IDX)

    elif ndim == 2:
        A2 = a
        n = _indices_gt_le2(A2, V1, V2, IDX)

    if n == 0:
        return np.empty([0], dtype=np.int32)
    return idx[:n].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_gt_le1(const double[::1] a, const double V1, const double V2, int[::1] idx) nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if V1 < a[i]:
            if a[i] <= V2:
                idx[n] = i
                n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline int all_gt_le(const double[:, ::1] a, const Py_ssize_t i, const double V1, const double V2) nogil:
    cdef Py_ssize_t j
    for j in range(a.shape[1]):
        if a[i, j] <= V1:
            return 0
        elif V2 < a[i, j]:
            return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_gt_le2(const double[:, ::1] a, const double V1, const double V2, int[::1] idx) nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if all_gt_le(a, i, V1, V2):
            idx[n] = i
            n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline int in_1d(const int[::1] array, const int v) nogil:
    cdef Py_ssize_t N = array.shape[0]
    cdef Py_ssize_t i
    for i in range(N):
        if array[i] == v:
            return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def index_sorted(np.ndarray[np.int32_t, ndim=1, mode='c'] a, const int v):
    """ Return index for the value v in a sorted array, otherwise return -1

    Parameters
    ----------
    a : int[::1]
        sorted array to check
    v : int
        value to find

    Returns
    -------
    int : -1 if not found, otherwise the first index in `a` that is equal to `v`
    """
    # Ensure contiguous arrays
    cdef int[::1] A = a
    return <int> _index_sorted(A, v)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _index_sorted(const int[::1] a, const int v) nogil:
    """ Return index for the value v in a sorted array, otherwise return -1

    This implements a binary search method

    Parameters
    ----------
    a : int[::1]
        sorted array to check
    v : int
        value to find

    Returns
    -------
    int : 0 if not unique, otherwise 1.
    """
    cdef Py_ssize_t i, L, R

    # Simple binary search
    L = 0
    R = a.shape[0]
    if R == 0:
        return -1
    elif v < a[L]:
        return -1

    while L < R:
        i = (L + R) // 2
        if a[i] < v:
            L = i + 1
        elif a[i] == v:
            return i
        else:
            R = i
    if a[R] == v:
        return R
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
def sorted_unique(np.ndarray[np.int32_t, ndim=1, mode='c'] a):
    """ Return True/False if all elements of the sorted array `a` are unique

    Parameters
    ----------
    a : np.ndarray(np.int32)
        sorted array to check

    Returns
    -------
    int : 0 if not unique, otherwise 1.
    """
    # Ensure contiguous arrays
    cdef int[::1] A = a
    cdef Py_ssize_t n = A.shape[0]

    return _sorted_unique(n, A)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int _sorted_unique(const Py_ssize_t n_a, const int[::1] a) nogil:
    cdef Py_ssize_t i

    # Fast return
    if n_a <= 1:
        return 1

    for i in range(n_a - 1):
        if a[i] == a[i+1]:
            return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
def list_index_le(np.ndarray[np.int32_t, ndim=1, mode='c'] a, np.ndarray[np.int32_t, ndim=1, mode='c'] b):
    """ Find indices for each ``a`` such that the returned ``a[i] <= b[ret[i]]`` where `b` is assumed sorted

    This corresponds to:

    >>> a.shape = (-1, 1)
    >>> ret = np.argmax(a <= b, axis=1)

    Parameters
    ----------
    a : np.ndarray(np.int32)
        values to check indicies of
    b : np.ndarray(np.int32)
        sorted array to check against

    Returns
    -------
    np.ndarray(np.int32): same length as `a` with indicies
    """
    # Ensure contiguous arrays
    cdef int[::1] A = a
    cdef int[::1] B = b
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] c = np.empty([A.shape[0]], dtype=np.int32)
    cdef int[::1] C = c

    _list_index_le(A, B, C)
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _list_index_le(const int[::1] a, const int[::1] b, int[::1] c) nogil:
    cdef Py_ssize_t na = a.shape[0]
    cdef Py_ssize_t nb = b.shape[0]
    cdef Py_ssize_t ia, ib
    cdef int ai, alast
    cdef Py_ssize_t start = 0

    if na > 0:
        alast = a[0]

    for ia in range(na):
        ai = a[ia]
        if ai < alast:
            start = 0
        alast = ai
        for ib in range(start, nb):
            if ai <= b[ib]:
                c[ia] = ib
                start = ib
                break
