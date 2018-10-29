#!python
#cython: language_level=2
cimport cython
from libc.math cimport fabs

import numpy as np
# This enables Cython enhanced compatibilities
cimport numpy as np

from _indices cimport in_1d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline int _sum(const int[::1] array) nogil:
    cdef int total, i

    total = 0
    for i in range(array.shape[0]):
        total += array[i]
    return total


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def fold_csr_matrix(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] COL):
    """ Fold all columns into a square matrix """
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    # Number of rows
    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] FOLD_ptr = np.empty([nr + 1], dtype=np.int32)
    cdef int[::1] fold_ptr = FOLD_ptr
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] FOLD_ncol = np.empty([nr], dtype=np.int32)
    cdef int[::1] fold_ncol = FOLD_ncol
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] FOLD_col = np.empty([_sum(ncol)], dtype=np.int32)
    cdef int[::1] fold_col = FOLD_col
    # local variables
    cdef int r, ind, nz, c

    nz = 0
    fold_ptr[0] = 0
    # Loop on all rows
    for r in range(nr):

        # Initialize the pointer arrays
        if ncol[r] > 0:
            fold_ncol[r] = 1
            fold_col[fold_ptr[r]] = col[ptr[r]] % nr
        else:
            fold_ncol[r] = 0

        for ind in range(ptr[r] + 1, ptr[r] + ncol[r]):
            c = col[ind] % nr
            if not in_1d(fold_col[fold_ptr[r]:fold_ptr[r] + fold_ncol[r]], c):
                fold_col[fold_ptr[r] + fold_ncol[r]] = c
                fold_ncol[r] += 1

        # Sort indices (we should implement our own sorting algorithm)
        tmp = np.sort(fold_col[fold_ptr[r]:fold_ptr[r] + fold_ncol[r]])
        for ind in range(fold_ncol[r]):
            fold_col[fold_ptr[r] + ind] = tmp[ind]

        fold_ptr[r + 1] = fold_ptr[r] + fold_ncol[r]
        nz += fold_ncol[r]

    if nz > fold_col.shape[0]:
        raise ValueError('something went wrong')

    # Return objects
    return FOLD_ptr, FOLD_ncol, FOLD_col[:nz].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def fold_csr_matrix_nc(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                       np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                       np.ndarray[np.int32_t, ndim=1, mode='c'] COL):
    """ Fold all columns into a square matrix """
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    # Number of rows
    cdef int nr = ncol.shape[0]

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] FOLD_ptr = np.empty([nr * 2 + 1], dtype=np.int32)
    cdef int[::1] fold_ptr = FOLD_ptr
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] FOLD_ncol = np.empty([nr * 2], dtype=np.int32)
    cdef int[::1] fold_ncol = FOLD_ncol
    # We have to multiply by 4, 2 times the number of rows, and each row couples to 2 more elements
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] FOLD_col = np.empty([_sum(ncol) * 4], dtype=np.int32)
    cdef int[::1] fold_col = FOLD_col
    # local variables
    cdef int r, rr, ind, nz, c

    nz = 0
    fold_ptr[0] = 0
    # Loop on all rows
    for r in range(nr):
        rr = r * 2

        # Initialize the pointer arrays
        if ncol[r] > 0:
            c = (col[ptr[r]] % nr) * 2
            fold_ncol[rr] = 2
            fold_col[fold_ptr[rr]] = c
            fold_col[fold_ptr[rr] + 1] = c + 1
        else:
            fold_ncol[rr] = 0

        for ind in range(ptr[r] + 1, ptr[r] + ncol[r]):
            c = (col[ind] % nr) * 2
            if not in_1d(fold_col[fold_ptr[rr]:fold_ptr[rr] + fold_ncol[rr]], c):
                fold_col[fold_ptr[rr] + fold_ncol[rr]] = c
                fold_col[fold_ptr[rr] + fold_ncol[rr] + 1] = c + 1
                fold_ncol[rr] += 2

        # Duplicate pointers and counters for next row (off-diagonal)
        fold_ptr[rr + 1] = fold_ptr[rr] + fold_ncol[rr]
        fold_ncol[rr + 1] = fold_ncol[rr]

        # Sort indices (we should implement our own sorting algorithm)
        tmp = np.sort(fold_col[fold_ptr[rr]:fold_ptr[rr] + fold_ncol[rr]])
        for ind in range(fold_ncol[rr]):
            c = tmp[ind]
            fold_col[fold_ptr[rr] + ind] = c
            # Copy to next row as well
            fold_col[fold_ptr[rr+1] + ind] = c

        # Increment the next row
        fold_ptr[rr + 2] = fold_ptr[rr + 1] + fold_ncol[rr + 1]
        nz += fold_ncol[rr] * 2

    if nz > fold_col.shape[0]:
        raise ValueError('something went wrong NC')

    # Return objects
    return FOLD_ptr, FOLD_ncol, FOLD_col[:nz].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def fold_csr_diagonal_nc(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                         np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                         np.ndarray[np.int32_t, ndim=1, mode='c'] COL):
    """ Fold all columns into a square matrix """
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    # Number of rows
    cdef int nr = ncol.shape[0]

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] FOLD_ptr = np.empty([nr * 2 + 1], dtype=np.int32)
    cdef int[::1] fold_ptr = FOLD_ptr
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] FOLD_ncol = np.empty([nr * 2], dtype=np.int32)
    cdef int[::1] fold_ncol = FOLD_ncol
    # We have to multiply by 2, 2 times the number of rows
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] FOLD_col = np.empty([_sum(ncol) * 2], dtype=np.int32)
    cdef int[::1] fold_col = FOLD_col
    # local variables
    cdef int r, rr, ind, nz, c

    nz = 0
    fold_ptr[0] = 0
    # Loop on all rows
    for r in range(nr):
        rr = r * 2

        # Initialize the pointer arrays
        if ncol[r] > 0:
            c = (col[ptr[r]] % nr) * 2
            fold_ncol[rr] = 1
            fold_col[fold_ptr[rr]] = c
        else:
            fold_ncol[rr] = 0

        for ind in range(ptr[r] + 1, ptr[r] + ncol[r]):
            c = (col[ind] % nr) * 2
            if not in_1d(fold_col[fold_ptr[rr]:fold_ptr[rr] + fold_ncol[rr]], c):
                fold_col[fold_ptr[rr] + fold_ncol[rr]] = c
                fold_ncol[rr] += 1

        # Duplicate pointers and counters for next row (off-diagonal)
        fold_ptr[rr + 1] = fold_ptr[rr] + fold_ncol[rr]
        fold_ncol[rr + 1] = fold_ncol[rr]

        # Sort indices (we should implement our own sorting algorithm)
        tmp = np.sort(fold_col[fold_ptr[rr]:fold_ptr[rr] + fold_ncol[rr]])
        for ind in range(fold_ncol[rr]):
            c = tmp[ind]
            fold_col[fold_ptr[rr] + ind] = c
            # Copy to next row as well
            fold_col[fold_ptr[rr+1] + ind] = c + 1

        # Increment the next row
        fold_ptr[rr + 2] = fold_ptr[rr + 1] + fold_ncol[rr + 1]
        nz += fold_ncol[rr] * 2

    if nz > fold_col.shape[0]:
        raise ValueError('something went wrong overlap NC')

    # Return objects
    return FOLD_ptr, FOLD_ncol, FOLD_col[:nz].copy()
