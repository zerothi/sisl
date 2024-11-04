# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np

cimport numpy as cnp
from numpy cimport dtype, ndarray

from sisl._core._dtypes cimport inline_sum, ints_st, numerics_st, ssize_st, type2dtype
from sisl._indices cimport in_1d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void ncol2ptr_nc(const ints_st nr, const ints_st[::1] ncol, ints_st[::1] ptr, const ints_st per_elem) noexcept nogil:
    cdef ssize_st r, rr

    # this is NC/SOC
    ptr[0] = 0
    ptr[1] = ncol[0] * per_elem
    for r in range(1, nr):
        rr = r * 2
        # do both
        ptr[rr] = ptr[rr - 1] + ncol[r-1] * per_elem
        ptr[rr+1] = ptr[rr] + ncol[r] * per_elem

    ptr[nr * 2] = ptr[nr * 2 - 1] + ncol[nr - 1] * per_elem


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def fold_csr_matrix(ints_st[::1] ptr,
                    ints_st[::1] ncol,
                    ints_st[::1] col):
    """ Fold all columns into a square matrix """

    # Number of rows
    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[ints_st](1)
    cdef ndarray[ints_st, mode='c'] FOLD_ptr = np.empty([nr + 1], dtype=dtype)
    cdef ndarray[ints_st, mode='c'] FOLD_ncol = np.empty([nr], dtype=dtype)
    cdef ndarray[ints_st, mode='c'] FOLD_col = np.empty([inline_sum(ncol)], dtype=dtype)

    cdef ints_st[::1] fold_ptr = FOLD_ptr
    cdef ints_st[::1] fold_ncol = FOLD_ncol
    cdef ints_st[::1] fold_col = FOLD_col

    # local variables
    cdef ints_st r, c, nz, ind
    cdef ints_st[::1] tmp

    nz = 0
    fold_ptr[0] = 0

    # Loop on all rows
    for r in range(nr):

        # Initialize the pointer arrays
        # Even though large supercells has *many* double entries (after folding)
        # this turns out to be faster than incrementally searching
        # the array.
        # This kind-of-makes sense.
        # We can do:
        #  1.
        #    a) build a full list of folded items
        #    b) find unique (and sorted) elements
        # or
        #  2.
        #    a) incrementally add a value, only
        #       if it does not exist.
        # 1. creates a bigger temporary array, but only
        #    adds unique values 1 time through numpy fast algorithm
        # 2. searchs an array (of seemingly small arrays) ncol times
        #    which can be quite heavy.
        tmp = col[ptr[r]:ptr[r] + ncol[r]].copy()
        for ind in range(ncol[r]):
            tmp[ind] %= nr

        tmp = np.unique(tmp)
        fold_ncol[r] = tmp.shape[0]
        for ind in range(tmp.shape[0]):
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
def fold_csr_matrix_nc(ints_st[::1] ptr,
                       ints_st[::1] ncol,
                       ints_st[::1] col):
    """ Fold all columns into a square matrix """
    # Number of rows
    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[ints_st](1)
    cdef ndarray[ints_st, mode='c'] FOLD_ptr = np.empty([nr * 2 + 1], dtype=dtype)
    cdef ndarray[ints_st, mode='c'] FOLD_ncol = np.empty([nr * 2], dtype=dtype)
    # We have to multiply by 4, 2 times for the extra rows, and another
    # 2 for the possible double couplings
    cdef ndarray[ints_st, mode='c'] FOLD_col = np.empty([inline_sum(ncol) * 4], dtype=dtype)

    cdef ints_st[::1] fold_ptr = FOLD_ptr
    cdef ints_st[::1] fold_ncol = FOLD_ncol
    cdef ints_st[::1] fold_col = FOLD_col

    # local variables
    cdef ints_st r, rr, ind, nz, c
    cdef ints_st[::1] tmp

    nz = 0
    fold_ptr[0] = 0

    # Loop on all rows
    for r in range(nr):
        rr = r * 2

        tmp = col[ptr[r]:ptr[r] + ncol[r]].copy()
        for ind in range(ncol[r]):
            tmp[ind] = (tmp[ind] % nr) * 2

        tmp = np.unique(tmp)

        # Duplicate pointers and counters for next row (off-diagonal)
        fold_ncol[rr] = tmp.shape[0] * 2
        fold_ncol[rr + 1] = fold_ncol[rr]
        fold_ptr[rr + 1] = fold_ptr[rr] + fold_ncol[rr]
        fold_ptr[rr + 2] = fold_ptr[rr + 1] + fold_ncol[rr]

        for ind in range(tmp.shape[0]):
            fold_col[fold_ptr[rr] + ind * 2] = tmp[ind]
            fold_col[fold_ptr[rr] + ind * 2 + 1] = tmp[ind] + 1
            fold_col[fold_ptr[rr+1] + ind * 2] = tmp[ind]
            fold_col[fold_ptr[rr+1] + ind * 2 + 1] = tmp[ind] + 1

        nz += fold_ncol[rr] * 2

    if nz > fold_col.shape[0]:
        raise ValueError('something went wrong NC')

    # Return objects
    return FOLD_ptr, FOLD_ncol, FOLD_col[:nz].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def fold_csr_matrix_nc_diag(ints_st[::1] ptr,
                            ints_st[::1] ncol,
                            ints_st[::1] col):
    """ Fold all columns into a square matrix """
    # Number of rows
    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[ints_st](1)
    cdef ndarray[ints_st, mode='c'] FOLD_ptr = np.empty([nr * 2 + 1], dtype=dtype)
    cdef ndarray[ints_st, mode='c'] FOLD_ncol = np.empty([nr * 2], dtype=dtype)
    # We have to multiply by 2 times for the extra rows
    cdef ndarray[ints_st, mode='c'] FOLD_col = np.empty([inline_sum(ncol) * 2], dtype=dtype)

    cdef ints_st[::1] fold_ptr = FOLD_ptr
    cdef ints_st[::1] fold_ncol = FOLD_ncol
    cdef ints_st[::1] fold_col = FOLD_col

    # local variables
    cdef ints_st r, rr, ind, nz, c
    cdef ints_st[::1] tmp

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


def sparse_dense(M):
    cdef cnp.ndarray dense = np.zeros(M.shape, dtype=M.dtype)
    _sparse_dense(M.ptr, M.ncol, M.col, M._D, dense)
    return dense


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _sparse_dense(ints_st[::1] ptr,
                   ints_st[::1] ncol,
                   ints_st[::1] col,
                   numerics_st[:, ::1] data,
                   numerics_st[:, :, ::1] dense):

    cdef ints_st r, ind, ix, s2

    s2 = dense.shape[2]
    for r in range(ncol.shape[0]):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            for ix in range(s2):
                dense[r, col[ind], ix] += data[ind, ix]
