# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython

import numpy as np

cimport numpy as cnp

from scipy.sparse import csr_matrix

from sisl._core._dtypes cimport (
    complexs_st,
    floatcomplexs_st,
    floats_st,
    inline_sum,
    int_sp_st,
    type2dtype,
)
from sisl._core._sparse cimport ncol2ptr
from sisl._indices cimport _index_sorted

from ._matrix_utils cimport (
    f_matrix_box_nambu,
    f_matrix_box_nc,
    f_matrix_box_so,
    matrix_add_array_nambu,
    matrix_add_array_nc,
    matrix_add_csr_nambu,
    matrix_add_csr_nc,
    matrix_box_nambu_cmplx,
    matrix_box_nambu_real,
    matrix_box_nc_cmplx,
    matrix_box_nc_real,
    matrix_box_so_cmplx,
    matrix_box_so_real,
)

__all__ = [
    "phase_sc_csr",
    "phase_sc_array",
    "phase_sc_csr_nc",
    "phase_sc_array_nc",
    "phase_sc_csr_diag",
    "phase_sc_array_diag",
    "phase_sc_csr_so",
    "phase_sc_array_so",
    "phase_sc_csr_nambu",
    "phase_sc_array_nambu",
]


ctypedef fused phases_st:
    float
    double
    float complex
    double complex


def phase_sc_csr(int_sp_st[::1] ptr,
                  int_sp_st[::1] ncol,
                  int_sp_st[::1] col,
                  const int_sp_st nc,
                  floatcomplexs_st[:, ::1] D,
                  const int_sp_st idx,
                  const phases_st[::1] phases,
                  const int_sp_st p_opt):

    # Now copy the sparse matrix form
    cdef int_sp_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[int_sp_st](1)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_PTR = np.empty([nr + 1], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_NCOL = np.empty([nr], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_COL = np.empty([inline_sum(ncol)], dtype=idtype)

    cdef int_sp_st[::1] v_ptr = V_PTR
    cdef int_sp_st[::1] v_ncol = V_NCOL
    cdef int_sp_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[phases_st](1)
    cdef cnp.ndarray[phases_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef phases_st[::1] v = V

    cdef int_sp_st r, c, ind, cind
    cdef phases_st ph

    # Copy ncol
    v_ncol[:] = ncol[:]

    # This abstraction allows to handle non-finalized CSR matrices
    cind = 0

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                v_ptr[r] = cind
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    v[cind] = <phases_st> D[ind, idx]
                    v_col[cind] = col[ind]
                    cind = cind + 1

        elif p_opt == 0:
            for r in range(nr):
                v_ptr[r] = cind
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    ph = phases[ind]
                    v[cind] = <phases_st> (D[ind, idx] * ph)
                    v_col[cind] = col[ind]
                    cind = cind + 1

        else:
            for r in range(nr):
                v_ptr[r] = cind
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    ph = phases[col[ind] / nr]
                    v[cind] = <phases_st> (D[ind, idx] * ph)
                    v_col[cind] = col[ind]
                    cind = cind + 1

    v_ptr[nr] = cind

    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nc))


def phase_sc_array(int_sp_st[::1] ptr,
                    int_sp_st[::1] ncol,
                    int_sp_st[::1] col,
                    const int_sp_st nc,
                    floatcomplexs_st[:, ::1] D,
                    const int_sp_st idx,
                    const phases_st[::1] phases,
                    const int_sp_st p_opt):

    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[phases_st](1)
    cdef cnp.ndarray[phases_st, ndim=2, mode='c'] V = np.zeros([nr, nc], dtype=dtype)
    cdef phases_st[:, ::1] v = V

    cdef int_sp_st r, c, ind
    cdef phases_st ph

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    v[r, col[ind]] = <phases_st> D[ind, idx]

        elif p_opt == 0:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    ph = phases[ind]
                    v[r, col[ind]] = <phases_st> (D[ind, idx] * ph)

        else:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    ph = phases[col[ind] / nr]
                    v[r, col[ind]] = <phases_st> (D[ind, idx] * ph)

    return V


def phase_sc_csr_nc(int_sp_st[::1] ptr,
                     int_sp_st[::1] ncol,
                     int_sp_st[::1] col,
                     const int_sp_st nc,
                     floatcomplexs_st[:, ::1] D,
                     const complexs_st[::1] phases,
                     const int_sp_st p_opt):

    # Now copy the sparse matrix form
    cdef int_sp_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[int_sp_st](1)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_PTR = np.empty([nr*2 + 1], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_NCOL = np.empty([nr*2], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_COL = np.empty([inline_sum(ncol)*4], dtype=idtype)

    cdef int_sp_st[::1] v_ptr = V_PTR
    cdef int_sp_st[::1] v_ncol = V_NCOL
    cdef int_sp_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    cdef int_sp_st r, rr, cind, c, ind
    cdef complexs_st ph
    cdef f_matrix_box_nc func
    cdef floatcomplexs_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if floatcomplexs_st in complexs_st:
        func = matrix_box_nc_cmplx
    else:
        func = matrix_box_nc_real

    # We have to do it manually due to the double elements per matrix element
    ncol2ptr(nr, ncol, v_ptr, 2, 2)

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 2
                v_ncol[rr] = ncol[r] * 2
                v_ncol[rr+1] = ncol[r] * 2

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, cind, v, M)
                    v_col[v_ptr[rr] + cind] = c
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v_col[v_ptr[rr+1] + cind] = c
                    v_col[v_ptr[rr+1] + cind+1] = c + 1

                    cind = cind + 2

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                v_ncol[rr] = ncol[r] * 2
                v_ncol[rr+1] = ncol[r] * 2

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, cind, v, M)
                    v_col[v_ptr[rr] + cind] = c
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v_col[v_ptr[rr+1] + cind] = c
                    v_col[v_ptr[rr+1] + cind+1] = c + 1

                    cind = cind + 2

        else:
            for r in range(nr):
                rr = r * 2
                v_ncol[rr] = ncol[r] * 2
                v_ncol[rr+1] = ncol[r] * 2

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[col[ind] / nr]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, cind, v, M)
                    v_col[v_ptr[rr] + cind] = c
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v_col[v_ptr[rr+1] + cind] = c
                    v_col[v_ptr[rr+1] + cind+1] = c + 1

                    cind = cind + 2

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 2, nc * 2))


def phase_sc_array_nc(int_sp_st[::1] ptr,
                       int_sp_st[::1] ncol,
                       int_sp_st[::1] col,
                       const int_sp_st nc,
                       floatcomplexs_st[:, ::1] D,
                       const complexs_st[::1] phases,
                       const int_sp_st p_opt):

    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr*2, nc*2], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    cdef complexs_st ph
    cdef int_sp_st r, rr, c, ind
    cdef floatcomplexs_st *d
    cdef f_matrix_box_nc func
    cdef complexs_st *M = [0, 0, 0, 0]

    if floatcomplexs_st in complexs_st:
        func = matrix_box_nc_cmplx
    else:
        func = matrix_box_nc_real

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, v, M)

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, v, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[col[ind] / nr]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, v, M)

    return V

def phase_sc_csr_diag(int_sp_st[::1] ptr,
                      int_sp_st[::1] ncol,
                      int_sp_st[::1] col,
                      const int_sp_st nc,
                      floatcomplexs_st[:, ::1] D,
                      const int_sp_st idx,
                      const complexs_st[::1] phases,
                      const int_sp_st p_opt,
                      const int_sp_st per_row):

    # Now copy the sparse matrix form
    cdef int_sp_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[int_sp_st](1)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_PTR = np.empty([nr*per_row + 1], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_NCOL = np.empty([nr*per_row], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_COL = np.empty([inline_sum(ncol)*per_row], dtype=idtype)

    cdef int_sp_st[::1] v_ptr = V_PTR
    cdef int_sp_st[::1] v_ncol = V_NCOL
    cdef int_sp_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    cdef int_sp_st r, rr, ir, cind, c, ind, ic
    cdef complexs_st ph

    # We have to do it manually due to the double elements per row, but only
    # one per column
    ncol2ptr(nr, ncol, v_ptr, per_row, 1)

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                rr = r * per_row
                for ir in range(per_row):
                    v_ncol[rr+ir] = ncol[r]

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * per_row

                    for ic in range(per_row):
                        v[v_ptr[rr+ic] + cind] = <complexs_st> D[ind, idx]
                        v_col[v_ptr[rr+ic] + cind] = c + ic

                    cind = cind + 1

        elif p_opt == 0:
            for r in range(nr):
                rr = r * per_row
                for ir in range(per_row):
                    v_ncol[rr+ir] = ncol[r]

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * per_row
                    ph = phases[ind]

                    for ic in range(per_row):
                        v[v_ptr[rr+ic] + cind] = <complexs_st> (D[ind, idx] * ph)
                        v_col[v_ptr[rr+ic] + cind] = c + ic

                    cind = cind + 1

        else:
            for r in range(nr):
                rr = r * per_row
                for ir in range(per_row):
                    v_ncol[rr+ir] = ncol[r]

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * per_row
                    ph = phases[col[ind] / nr]

                    for ic in range(per_row):
                        v[v_ptr[rr+ic] + cind] = <complexs_st> (D[ind, idx] * ph)
                        v_col[v_ptr[rr+ic] + cind] = c + ic

                    cind = cind + 1

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * per_row, nc * per_row))


def phase_sc_array_diag(int_sp_st[::1] ptr,
                         int_sp_st[::1] ncol,
                         int_sp_st[::1] col,
                         const int_sp_st nc,
                         floatcomplexs_st[:, ::1] D,
                         const int_sp_st idx,
                         const complexs_st[::1] phases,
                         const int_sp_st p_opt,
                         const int_sp_st per_row):

    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr*per_row, nc*per_row], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    cdef complexs_st d
    cdef int_sp_st r, rr, c, ind, ic

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                rr = r * per_row
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * per_row
                    d = <complexs_st> D[ind, idx]
                    for ic in range(per_row):
                        v[rr+ic, c+ic] = d

        elif p_opt == 0:
            for r in range(nr):
                rr = r * per_row
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * per_row
                    d = <complexs_st> (D[ind, idx] * phases[ind])
                    for ic in range(per_row):
                        v[rr+ic, c+ic] = d

        else:
            for r in range(nr):
                rr = r * per_row
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * per_row
                    d = <complexs_st> (D[ind, idx] * phases[col[ind] / nr])
                    for ic in range(per_row):
                        v[rr+ic, c+ic] = d

    return V


def phase_sc_csr_so(int_sp_st[::1] ptr,
                     int_sp_st[::1] ncol,
                     int_sp_st[::1] col,
                     const int_sp_st nc,
                     floatcomplexs_st[:, ::1] D,
                     const complexs_st[::1] phases,
                     const int_sp_st p_opt):

    # Now copy the sparse matrix form
    cdef int_sp_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[int_sp_st](1)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_PTR = np.empty([nr*2 + 1], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_NCOL = np.empty([nr*2], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_COL = np.empty([inline_sum(ncol)*4], dtype=idtype)

    cdef int_sp_st[::1] v_ptr = V_PTR
    cdef int_sp_st[::1] v_ncol = V_NCOL
    cdef int_sp_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    cdef int_sp_st r, rr, cind, c, ind
    cdef complexs_st ph
    cdef f_matrix_box_so func
    cdef floatcomplexs_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if floatcomplexs_st in complexs_st:
        func = matrix_box_so_cmplx
    else:
        func = matrix_box_so_real

    # We have to do it manually due to the double elements per matrix element
    ncol2ptr(nr, ncol, v_ptr, 2, 2)

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 2
                v_ncol[rr] = ncol[r] * 2
                v_ncol[rr+1] = ncol[r] * 2

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, cind, v, M)
                    v_col[v_ptr[rr] + cind] = c
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v_col[v_ptr[rr+1] + cind] = c
                    v_col[v_ptr[rr+1] + cind+1] = c + 1

                    cind = cind + 2

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                v_ncol[rr] = ncol[r] * 2
                v_ncol[rr+1] = ncol[r] * 2

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, cind, v, M)
                    v_col[v_ptr[rr] + cind] = c
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v_col[v_ptr[rr+1] + cind] = c
                    v_col[v_ptr[rr+1] + cind+1] = c + 1

                    cind = cind + 2

        else:
            for r in range(nr):
                rr = r * 2
                v_ncol[rr] = ncol[r] * 2
                v_ncol[rr+1] = ncol[r] * 2

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[col[ind] / nr]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, cind, v, M)
                    v_col[v_ptr[rr] + cind] = c
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v_col[v_ptr[rr+1] + cind] = c
                    v_col[v_ptr[rr+1] + cind+1] = c + 1

                    cind = cind + 2

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 2, nc * 2))


def phase_sc_array_so(int_sp_st[::1] ptr,
                       int_sp_st[::1] ncol,
                       int_sp_st[::1] col,
                       const int_sp_st nc,
                       floatcomplexs_st[:, ::1] D,
                       const complexs_st[::1] phases,
                       const int_sp_st p_opt):

    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr*2, nc*2], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    cdef complexs_st ph
    cdef int_sp_st r, rr, c, ind
    cdef f_matrix_box_so func
    cdef floatcomplexs_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if floatcomplexs_st in complexs_st:
        func = matrix_box_so_cmplx
    else:
        func = matrix_box_so_real

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, v, M)

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, v, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[col[ind] / nr]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, v, M)

    return V


def phase_sc_csr_nambu(int_sp_st[::1] ptr,
                        int_sp_st[::1] ncol,
                        int_sp_st[::1] col,
                        const int_sp_st nc,
                        floatcomplexs_st[:, ::1] D,
                        const complexs_st[::1] phases,
                        const int_sp_st p_opt):

    # Now copy the sparse matrix form
    cdef int_sp_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[int_sp_st](1)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_PTR = np.empty([nr*4 + 1], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_NCOL = np.empty([nr*4], dtype=idtype)
    cdef cnp.ndarray[int_sp_st, mode='c'] V_COL = np.empty([inline_sum(ncol)*16], dtype=idtype)

    cdef int_sp_st[::1] v_ptr = V_PTR
    cdef int_sp_st[::1] v_ncol = V_NCOL
    cdef int_sp_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    cdef int_sp_st r, rr, cind, c, ind, ic
    cdef complexs_st ph
    cdef f_matrix_box_nambu func
    cdef floatcomplexs_st *d
    cdef complexs_st *M = [0] * 16

    if floatcomplexs_st in complexs_st:
        func = matrix_box_nambu_cmplx
    else:
        func = matrix_box_nambu_real

    # We have to do it manually due to the quadrouble elements per matrix element
    ncol2ptr(nr, ncol, v_ptr, 4, 4)

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 4
                v_ncol[rr] = ncol[r] * 4
                v_ncol[rr+1] = ncol[r] * 4
                v_ncol[rr+2] = ncol[r] * 4
                v_ncol[rr+3] = ncol[r] * 4

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 4

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nambu(v_ptr, rr, cind, v, M)

                    for ic in range(4):
                        v_col[v_ptr[rr+ic] + cind] = c + 0
                        v_col[v_ptr[rr+ic] + cind+1] = c + 1
                        v_col[v_ptr[rr+ic] + cind+2] = c + 2
                        v_col[v_ptr[rr+ic] + cind+3] = c + 3

                    cind = cind + 4

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 4
                v_ncol[rr] = ncol[r] * 4
                v_ncol[rr+1] = ncol[r] * 4
                v_ncol[rr+2] = ncol[r] * 4
                v_ncol[rr+3] = ncol[r] * 4

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 4
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nambu(v_ptr, rr, cind, v, M)

                    for ic in range(4):
                        v_col[v_ptr[rr+ic] + cind] = c + 0
                        v_col[v_ptr[rr+ic] + cind+1] = c + 1
                        v_col[v_ptr[rr+ic] + cind+2] = c + 2
                        v_col[v_ptr[rr+ic] + cind+3] = c + 3

                    cind = cind + 4

        else:
            for r in range(nr):
                rr = r * 4
                v_ncol[rr] = ncol[r] * 4
                v_ncol[rr+1] = ncol[r] * 4
                v_ncol[rr+2] = ncol[r] * 4
                v_ncol[rr+3] = ncol[r] * 4

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 4
                    ph = phases[col[ind] / nr]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nambu(v_ptr, rr, cind, v, M)

                    for ic in range(4):
                        v_col[v_ptr[rr+ic] + cind] = c + 0
                        v_col[v_ptr[rr+ic] + cind+1] = c + 1
                        v_col[v_ptr[rr+ic] + cind+2] = c + 2
                        v_col[v_ptr[rr+ic] + cind+3] = c + 3

                    cind = cind + 4

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 4, nc * 4))


def phase_sc_array_nambu(int_sp_st[::1] ptr,
                          int_sp_st[::1] ncol,
                          int_sp_st[::1] col,
                          const int_sp_st nc,
                          floatcomplexs_st[:, ::1] D,
                          const complexs_st[::1] phases,
                          const int_sp_st p_opt):

    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr*4, nc*4], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    cdef complexs_st ph
    cdef int_sp_st r, rr, c, ind
    cdef f_matrix_box_nambu func
    cdef floatcomplexs_st *d
    cdef complexs_st *M = [0] * 16

    if floatcomplexs_st in complexs_st:
        func = matrix_box_nambu_cmplx
    else:
        func = matrix_box_nambu_real

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 4

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nambu(rr, c, v, M)

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 4
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nambu(rr, c, v, M)

        else:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 4
                    ph = phases[col[ind] / nr]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nambu(rr, c, v, M)

    return V
