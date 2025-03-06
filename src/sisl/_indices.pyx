# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython
from libc.math cimport fabs, fabsf, sqrt, sqrtf

import numpy as np

cimport numpy as cnp
from numpy cimport dtype, ndarray

from sisl._core._dtypes cimport floats_st, ints_st, type2dtype


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def indices_only(ints_st[::1] element, ints_st[::1] test_element):
    """ Return indices of all `test_element` in the element array.

    Parameters
    ----------
    element :
        array to search in
    test_element :
        values to find the indices of in `element`
    """
    # Ensure contiguous arrays
    cdef Py_ssize_t n_element = element.shape[0]
    cdef Py_ssize_t n_test_element = test_element.shape[0]

    cdef object dtype = type2dtype[ints_st](1)
    cdef ndarray[ints_st, mode='c'] IDX = np.empty([max(n_test_element, n_element)], dtype=dtype)
    cdef ints_st[::1] idx = IDX

    cdef Py_ssize_t i, j, n

    n = 0
    with nogil:

        # Fast return
        if n_test_element == 0:
            pass

        elif n_element == 0:
            pass

        elif n_test_element > n_element:
            for j in range(n_test_element):
                for i in range(n_element):
                    if test_element[j] == element[i]:
                        idx[n] = <ints_st> i
                        n += 1
                        break

        else:
            for i in range(n_element):
                for j in range(n_test_element):
                    if test_element[j] == element[i]:
                        idx[n] = <ints_st> i
                        n += 1
                        break

    return IDX[:n].copy()



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def indices(ints_st[::1] element, ints_st[::1] test_element, ints_st offset=0,
            both_sorted: bool = False):
    """ Return indices of all `test_element` in the search array. If not found the index will be ``-1``

    Parameters
    ----------
    element :
        array to search in
    test_element :
        values to find the indices of in `element`
    offset :
        index offset
    """
    # Ensure contiguous arrays
    cdef Py_ssize_t n_element = element.shape[0]
    cdef Py_ssize_t n_test_element = test_element.shape[0]

    cdef object dtype = type2dtype[ints_st](1)
    cdef ndarray[ints_st, mode='c'] IDX = np.empty([n_test_element], dtype=dtype)
    cdef ints_st[::1] idx = IDX
    cdef Py_ssize_t i, j
    cdef ints_st ctest_element, celement

    if offset < 0:
        raise ValueError(f"indices requires offset argument >=0, got {offset}")

    if n_test_element == 0:
        # fast return
        pass

    elif n_element == 0:

        for j in range(n_test_element):
            idx[j] = <ints_st> -1

    elif both_sorted:

        i = j = 0
        while (i < n_element) and (j < n_test_element):
            celement = element[i]
            ctest_element = test_element[j]
            if celement == ctest_element:
                idx[j] = <ints_st> (i + offset)
                j += 1
            elif celement < ctest_element:
                i += 1
            elif celement > ctest_element:
                idx[j] = <ints_st> -1
                j += 1
        for i in range(j, n_test_element):
            idx[i] = <ints_st> -1

    else:
        if n_test_element > n_element:
            for j in range(n_test_element):
                idx[j] = <ints_st> -1
                for i in range(n_element):
                    if test_element[j] == element[i]:
                        idx[j] = <ints_st> (offset + i)
                        break

        else:
            # We need to initialize
            for j in range(n_test_element):
                idx[j] = <ints_st> -1
            for i in range(n_element):
                for j in range(n_test_element):
                    if test_element[j] == element[i]:
                        idx[j] = <ints_st> (offset + i)
                        break

    return IDX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def indices_in_cylinder(floats_st[:, ::1] dxyz, const floats_st R, const floats_st h):
    """ Indices for all coordinates that are within a cylinde radius `R` and height `h`

    Parameters
    ----------
    dxyz :
       coordinates centered around the cylinder
    R :
       radius of cylinder to check
    h :
       height of cylinder to check

    Returns
    -------
    index :
       indices of all dxyz coordinates that are within the cylinder
    """
    cdef Py_ssize_t n = dxyz.shape[0]
    cdef Py_ssize_t nxyz = dxyz.shape[1] - 1

    cdef ndarray[int32_t] IDX = np.empty([n], dtype=np.int32)
    cdef int[::1] idx = IDX

    cdef floats_st R2 = R * R
    cdef floats_st L2
    cdef Py_ssize_t i, j, m
    cdef bint skip

    # Reset number of elements
    m = 0

    with nogil:
        for i in range(n):
            skip = 0
            for j in range(nxyz):
                skip |= dxyz[i, j] > R
            if skip or dxyz[i, nxyz] > h: continue

            L2 = 0.
            for j in range(nxyz):
                L2 += dxyz[i, j] * dxyz[i, j]
            if L2 > R2: continue
            idx[m] = <int> i
            m += 1

    if m == 0:
        return np.empty([0], dtype=np.int32)
    return IDX[:m].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def indices_in_sphere(floats_st[:, ::1] dxyz, const floats_st R):
    """ Indices for all coordinates that are within a sphere of radius `R`

    Parameters
    ----------
    dxyz :
       coordinates centered around the sphere
    R :
       radius of sphere to check

    Returns
    -------
    index:
       indices of all dxyz coordinates that are within the sphere of radius `R`
    """
    cdef Py_ssize_t n = dxyz.shape[0]
    cdef ndarray[int32_t, mode='c'] IDX = np.empty([n], dtype=np.int32)
    cdef int[::1] idx = IDX

    cdef floats_st R2 = R * R
    cdef Py_ssize_t i, m

    # Reset number of elements
    m = 0

    with nogil:
        for i in range(n):
            if all_fabs_le(dxyz, i, R):
                if fabs2(dxyz, i) <= R2:
                    idx[m] = <int> i
                    m += 1
    if m == 0:
        return np.empty([0], dtype=np.int32)
    return IDX[:m].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def indices_in_sphere_with_dist(floats_st[:, ::1] dxyz, const floats_st R):
    """ Indices and the distances for all coordinates that are within a sphere of radius `R`

    Parameters
    ----------
    dxyz :
       coordinates centered around the sphere
    R : float
       radius of sphere to check

    Returns
    -------
    index :
       indices of all dxyz coordinates that are within the sphere of radius `R`
    dist :
       distances for the coordinates within the sphere of radius `R` (corresponds to `index`)
    """
    cdef Py_ssize_t n = dxyz.shape[0]
    cdef ndarray[int32_t, mode='c'] IDX = np.empty([n], dtype=np.int32)
    cdef object dtype = type2dtype[floats_st](1)
    cdef ndarray[floats_st, mode='c'] DIST = np.empty([n], dtype=dtype)
    cdef int[::1] idx = IDX
    cdef floats_st[::1] dist = DIST

    cdef floats_st R2 = R * R
    cdef floats_st d
    cdef Py_ssize_t i, m

    with nogil:

        # Reset number of elements
        m = 0

        if floats_st is cython.float:
            for i in range(n):
                if all_fabs_le(dxyz, i, R):
                    d = fabs2(dxyz, i)
                    if d <= R2:
                        dist[m] = sqrtf(d)
                        idx[m] = <int> i
                        m += 1

        else:
            for i in range(n):
                if all_fabs_le(dxyz, i, R):
                    d = fabs2(dxyz, i)
                    if d <= R2:
                        dist[m] = sqrt(d)
                        idx[m] = <int> i
                        m += 1

    if m == 0:
        return np.empty([0], dtype=np.int32), np.empty([0], dtype=dtype)
    return IDX[:m].copy(), DIST[:m].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def indices_le(ndarray a, const floats_st V):
    """ Indices for all values in `a` that are ``<= V``

    Parameters
    ----------
    a :
       array to check if 2D, all last dimension values must be ``<= V``
    V : float
       value that is checked against

    Returns
    -------
    index :
       indices for the values in `a` which are less than or equal to `V`
    """
    cdef ndarray[int32_t, mode='c'] IDX = np.empty([a.shape[0]], dtype=np.int32)
    cdef int[::1] idx = IDX

    cdef Py_ssize_t ndim = a.ndim
    cdef floats_st[::1] A1
    cdef floats_st[:, ::1] A2
    cdef Py_ssize_t n

    if ndim == 1:
        A1 = a
        n = _indices_le1(A1, V, idx)

    elif ndim == 2:
        A2 = a
        n = _indices_le2(A2, V, idx)

    else:
        raise NotImplementedError("indices_le not implemented for ndim>2")

    if n == 0:
        return np.empty([0], dtype=np.int32)
    return IDX[:n].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_le1(const floats_st[::1] a, const floats_st V, int[::1] idx) noexcept nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if a[i] <= V:
            idx[n] = <int> i
            n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline bint all_le(const floats_st[:, ::1] a, const Py_ssize_t i, const floats_st V) noexcept nogil:
    cdef Py_ssize_t j
    for j in range(a.shape[1]):
        if a[i, j] > V:
            return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_le2(const floats_st[:, ::1] a, const floats_st V, int[::1] idx) noexcept nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if all_le(a, i, V):
            idx[n] = <int> i
            n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def indices_fabs_le(ndarray a, const floats_st V):
    """ Indices for all values in `a` that are ``| | <= V``

    Parameters
    ----------
    a :
       array to check if 2D, all last dimension values must be ``| | <= V``
    V :
       value that is checked against

    Returns
    -------
    index :
       indices for the values in ``|a|`` which are less than or equal to `V`
    """
    cdef ndarray[int32_t, mode='c'] IDX = np.empty([a.shape[0]], dtype=np.int32)
    cdef int[::1] idx = IDX

    cdef Py_ssize_t ndim = a.ndim
    cdef floats_st[::1] A1
    cdef floats_st[:, ::1] A2
    cdef Py_ssize_t n

    if ndim == 1:
        A1 = a
        n = _indices_fabs_le1(A1, V, idx)

    elif ndim == 2:
        A2 = a
        n = _indices_fabs_le2(A2, V, idx)

    else:
        raise NotImplementedError("indices_fabs_le not implemented for ndim>2")

    if n == 0:
        return np.empty([0], dtype=np.int32)
    return IDX[:n].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline floats_st fabs2(const floats_st[:, ::1] a, const Py_ssize_t i) noexcept nogil:
    cdef Py_ssize_t j
    cdef floats_st abs2 = 0.

    for j in range(a.shape[1]):
        abs2 += a[i, j]*a[i, j]
    return abs2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_fabs_le1(const floats_st[::1] a, const floats_st V, int[::1] idx) noexcept nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    if floats_st is cython.float:
        for i in range(N):
            if fabsf(a[i]) <= V:
                idx[n] = <int> i
                n += 1
    else:
        for i in range(N):
            if fabs(a[i]) <= V:
                idx[n] = <int> i
                n += 1
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline bint all_fabs_le(const floats_st[:, ::1] a, const Py_ssize_t i, const floats_st V) noexcept nogil:
    cdef Py_ssize_t j

    if floats_st is cython.float:
        for j in range(a.shape[1]):
            if fabsf(a[i, j]) > V:
                return 0

    else:
        for j in range(a.shape[1]):
            if fabs(a[i, j]) > V:
                return 0

    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_fabs_le2(const floats_st[:, ::1] a, const floats_st V, int[::1] idx) noexcept nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if all_fabs_le(a, i, V):
            idx[n] = <int> i
            n += 1

    return n



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def indices_gt_le(ndarray a, const floats_st V1, const floats_st V2):
    cdef ndarray[int32_t, mode='c'] IDX = np.empty([a.shape[0]], dtype=np.int32)
    cdef int[::1] idx = IDX

    cdef Py_ssize_t ndim = a.ndim
    cdef floats_st[::1] A1
    cdef floats_st[:, ::1] A2
    cdef Py_ssize_t n

    if ndim == 1:
        A1 = a
        n = _indices_gt_le1(A1, V1, V2, idx)

    elif ndim == 2:
        A2 = a
        n = _indices_gt_le2(A2, V1, V2, idx)

    else:
        raise NotImplementedError("indices_gt_le not implemented for ndim>2")

    if n == 0:
        return np.empty([0], dtype=np.int32)

    return IDX[:n].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_gt_le1(const floats_st[::1] a, const floats_st V1, const floats_st
                                V2, int[::1] idx) noexcept nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if V1 < a[i]:
            if a[i] <= V2:
                idx[n] = <int> i
                n += 1

    return n



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef Py_ssize_t _indices_gt_le2(const floats_st[:, ::1] a, const floats_st V1, const floats_st
                              V2, int[::1] idx) noexcept nogil:
    cdef Py_ssize_t N = a.shape[0]
    cdef Py_ssize_t i, n
    n = 0
    for i in range(N):
        if all_gt_le(a, i, V1, V2):
            idx[n] = <int> i
            n += 1

    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline bint all_gt_le(const floats_st[:, ::1] a, const Py_ssize_t i, const floats_st V1,
                          const floats_st V2) noexcept nogil:
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
cdef inline bint in_1d(const ints_st[::1] array, const ints_st v) noexcept nogil:
    cdef Py_ssize_t N = array.shape[0]
    cdef Py_ssize_t i
    for i in range(N):
        if array[i] == v:
            return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def index_sorted(ints_st[::1] a, const ints_st v):
    """ Return index for the value v in a sorted array, otherwise return -1

    Parameters
    ----------
    a :
        sorted array to check
    v :
        value to find

    Returns
    -------
    int : -1 if not found, otherwise the first index in `a` that is equal to `v`
    """
    # Ensure contiguous arrays
    return _index_sorted(a, v)


# This small code needs all variants
# The variants are declared in the _indices.pxd file

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef Py_ssize_t _index_sorted(const ints_st[::1] a, const _ints_index_sorted_st v) noexcept nogil:
    """ Return index for the value v in a sorted array, otherwise return -1

    This implements a binary search method

    Parameters
    ----------
    a :
        sorted array to check
    v :
        value to find

    Returns
    -------
    int : 0 if not unique, otherwise 1.
    """
    cdef Py_ssize_t MIN1 = -1
    cdef Py_ssize_t i, L, R

    # Simple binary search
    R = a.shape[0] - 1
    if R == -1:
        return MIN1
    elif a[R] < v:
        return MIN1

    L = 0
    while L <= R:
        i = (L + R) / 2
        if a[i] < v:
            L = i + 1
        elif v < a[i]:
            R = i - 1
        else:
            return i
    return MIN1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def is_sorted_unique(ints_st[::1] a):
    """ Return True/False if all elements of the sorted array `a` are unique

    Parameters
    ----------
    a :
        sorted array to check

    Returns
    -------
    int : 0 if not unique, otherwise 1.
    """
    # Ensure contiguous arrays
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i, ret = 1

    if n > 1:
        # only check for larger than 1 arrays
        with nogil:
            for i in range(n - 1):
                if a[i] == a[i+1]:
                    ret = 0
                    break
    return ret



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def list_index_le(ints_st[::1] a, ints_st[::1] b):
    """ Find indices for each ``a`` such that the returned ``a[i] <= b[ret[i]]`` where `b` is assumed sorted

    This corresponds to:

    >>> a.shape = (-1, 1)
    >>> ret = np.argmax(a <= b, axis=1)

    Parameters
    ----------
    a :
        values to check indicies of
    b :
        sorted array to check against

    Returns
    -------
    indices with same length as `a`
    """
    # Ensure contiguous arrays
    cdef Py_ssize_t na = a.shape[0]
    cdef Py_ssize_t nb = b.shape[0]
    cdef object dtype = type2dtype[ints_st](1)
    cdef ndarray[ints_st] C = np.empty([na], dtype=dtype)
    cdef ints_st[::1] c = C

    cdef Py_ssize_t ia, ib
    cdef ints_st ai, alast
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
                c[ia] = <ints_st> ib
                start = ib
                break

    return C
