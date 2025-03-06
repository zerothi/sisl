# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np

cimport numpy as cnp
from numpy cimport dtype, ndarray

from sisl._core._dtypes cimport inline_sum, int_sp_st, numerics_st, type2dtype
from sisl._indices cimport in_1d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void ncol2ptr(const int_sp_st nr, const int_sp_st[::1] ncol, int_sp_st[::1] ptr,
                   const int_sp_st per_row, const int_sp_st per_elem) noexcept nogil:
    cdef Py_ssize_t r, rr, ir

    # this is NC/SOC
    ptr[0] = 0
    for ir in range(1, per_row):
        ptr[ir] = ptr[ir-1] + ncol[0] * per_elem
    for r in range(1, nr):
        rr = r * per_row
        ptr[rr] = ptr[rr-1] + ncol[r-1] * per_elem
        for ir in range(1, per_row):
            ptr[rr+ir] = ptr[rr+ir-1] + ncol[r] * per_elem

    ptr[nr * per_row] = ptr[nr * per_row - 1] + ncol[nr - 1] * per_elem


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def fold_csr_matrix(int_sp_st[::1] ptr,
                    int_sp_st[::1] ncol,
                    int_sp_st[::1] col,
                    int_sp_st per_row = 1,
                    ):
    """ Fold all columns into a square matrix """

    # Number of rows
    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[int_sp_st](1)
    cdef ndarray[int_sp_st, mode='c'] FOLD_ptr = np.empty([nr*per_row+ 1], dtype=dtype)
    cdef ndarray[int_sp_st, mode='c'] FOLD_ncol = np.empty([nr*per_row], dtype=dtype)
    cdef ndarray[int_sp_st, mode='c'] FOLD_col = np.empty([inline_sum(ncol)*per_row*per_row], dtype=dtype)

    cdef int_sp_st[::1] fold_ptr = FOLD_ptr
    cdef int_sp_st[::1] fold_ncol = FOLD_ncol
    cdef int_sp_st[::1] fold_col = FOLD_col

    # local variables
    cdef int_sp_st r, rr, ir, c, ic, nz, ind
    cdef int_sp_st[::1] tmp

    nz = 0
    fold_ptr[0] = 0

    # Loop on all rows
    for r in range(nr):
        rr = r * per_row

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
            # correct the column indices (this is related to the additional rows)
            tmp[ind] = (tmp[ind] % nr) * per_row

        tmp = np.unique(tmp)

        # Create first one, then we simply copy it
        # number of elements for this row
        fold_ncol[rr] = tmp.shape[0] * per_row

        # create the next columns
        for ind in range(tmp.shape[0]):
            for ic in range(per_row):
                fold_col[fold_ptr[rr]+ind*per_row+ic] = tmp[ind]+ic

        for ir in range(1, per_row):
            # number of elements for this row
            fold_ncol[rr+ir] = fold_ncol[rr]
            fold_ptr[rr+ir] = fold_ptr[rr+ir-1] + fold_ncol[rr+ir]

            # create the next columns
            for ind in range(tmp.shape[0]*per_row):
                fold_col[fold_ptr[rr+ir]+ind] = fold_col[fold_ptr[rr]+ind]

        # Update next pointer
        fold_ptr[rr+per_row] = fold_ptr[rr+per_row-1] + fold_ncol[rr+per_row-1]

        # update counter
        nz += fold_ncol[rr] * per_row

    if nz > fold_col.shape[0]:
        raise ValueError('something went wrong')

    # Return objects
    return FOLD_ptr, FOLD_ncol, FOLD_col[:nz].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def fold_csr_matrix_diag(int_sp_st[::1] ptr,
                         int_sp_st[::1] ncol,
                         int_sp_st[::1] col,
                         int_sp_st per_row,
                    ):
    """ Fold all columns into a square matrix """

    # Number of rows
    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[int_sp_st](1)
    cdef ndarray[int_sp_st, mode='c'] FOLD_ptr = np.empty([nr*per_row+ 1], dtype=dtype)
    cdef ndarray[int_sp_st, mode='c'] FOLD_ncol = np.empty([nr*per_row], dtype=dtype)
    cdef ndarray[int_sp_st, mode='c'] FOLD_col = np.empty([inline_sum(ncol)*per_row], dtype=dtype)

    cdef int_sp_st[::1] fold_ptr = FOLD_ptr
    cdef int_sp_st[::1] fold_ncol = FOLD_ncol
    cdef int_sp_st[::1] fold_col = FOLD_col

    # local variables
    cdef int_sp_st r, rr, ir, c, ic, nz, ind
    cdef int_sp_st[::1] tmp

    nz = 0
    fold_ptr[0] = 0

    # Loop on all rows
    for r in range(nr):
        rr = r * per_row

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
            # correct the column indices (this is related to the additional rows)
            tmp[ind] = (tmp[ind] % nr) * per_row

        tmp = np.unique(tmp)

        for ir in range(per_row):
            # number of elements for this row
            fold_ncol[rr+ir] = tmp.shape[0]

            # create the next columns
            for ind in range(tmp.shape[0]):
                fold_col[fold_ptr[rr+ir] + ind] = tmp[ind] + ir

            # create next pointer
            fold_ptr[rr+ir+1] = fold_ptr[rr+ir] + fold_ncol[rr+ir]

        # update counter
        nz += fold_ncol[rr] * per_row

    if nz > fold_col.shape[0]:
        raise ValueError('something went wrong')

    # Return objects
    return FOLD_ptr, FOLD_ncol, FOLD_col[:nz].copy()


def sparse_dense(M):
    cdef cnp.ndarray dense = np.zeros(M.shape, dtype=M.dtype)
    _sparse_dense(M.ptr, M.ncol, M.col, M._D, dense)
    return dense


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _sparse_dense(int_sp_st[::1] ptr,
                  int_sp_st[::1] ncol,
                  int_sp_st[::1] col,
                  numerics_st[:, ::1] data,
                  numerics_st[:, :, ::1] dense):

    cdef int_sp_st r, ind, ix, s2

    s2 = dense.shape[2]
    for r in range(ncol.shape[0]):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            for ix in range(s2):
                dense[r, col[ind], ix] += data[ind, ix]
