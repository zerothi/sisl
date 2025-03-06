# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython

import numpy as np

cimport numpy as cnp

from scipy.sparse import csr_matrix

from sisl._indices cimport _index_sorted

from sisl._core._sparse import fold_csr_matrix

from sisl._core._dtypes cimport (
    complexs_st,
    floatcomplexs_st,
    floats_st,
    int_sp_st,
    type2dtype,
)

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
    "phase3_csr",
    "phase3_array",
    "phase3_csr_nc",
    "phase3_array_nc",
    "phase3_csr_so",
    "phase3_array_so",
    "phase3_csr_nambu",
    "phase3_array_nambu",
]


ctypedef fused phases_st:
    float
    double
    float complex
    double complex


def phase3_csr(const int_sp_st[::1] ptr,
               const int_sp_st[::1] ncol,
               const int_sp_st[::1] col,
               floatcomplexs_st[:, ::1] D,
               const int_sp_st idx,
               const phases_st[:, ::1] phases,
               const int_sp_st p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col)
    cdef int_sp_st[::1] v_ptr = V_PTR
    cdef int_sp_st[::1] v_ncol = V_NCOL
    cdef int_sp_st[::1] v_col = V_COL

    # This may fail, when floatcomplexs_st is complex, but phases_st is float
    cdef object dtype = type2dtype[phases_st](1)
    cdef cnp.ndarray[phases_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[phases_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[phases_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef phases_st[::1] vx = Vx
    cdef phases_st[::1] vy = Vy
    cdef phases_st[::1] vz = Vz

    # Local columns
    cdef int_sp_st nr = ncol.shape[0]
    cdef int_sp_st r, ind, s, s_idx, c

    cdef floatcomplexs_st d

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                    d = D[ind, idx]
                    vx[v_ptr[r] + s_idx] += <phases_st> (d * phases[ind, 0])
                    vy[v_ptr[r] + s_idx] += <phases_st> (d * phases[ind, 1])
                    vz[v_ptr[r] + s_idx] += <phases_st> (d * phases[ind, 2])

        else:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    s = col[ind] / nr
                    s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                    d = D[ind, idx]
                    vx[v_ptr[r] + s_idx] += <phases_st> (d * phases[s, 0])
                    vy[v_ptr[r] + s_idx] += <phases_st> (d * phases[s, 1])
                    vz[v_ptr[r] + s_idx] += <phases_st> (d * phases[s, 2])

    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))




def phase3_array(const int_sp_st[::1] ptr,
                 const int_sp_st[::1] ncol,
                 const int_sp_st[::1] col,
                 floatcomplexs_st[:, ::1] D,
                 const int_sp_st idx,
                 const phases_st[:, ::1] phases,
                 const int_sp_st p_opt):

    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[phases_st](1)
    cdef cnp.ndarray[phases_st, ndim=2, mode='c'] Vx = np.zeros([nr, nr], dtype=dtype)
    cdef cnp.ndarray[phases_st, ndim=2, mode='c'] Vy = np.zeros([nr, nr], dtype=dtype)
    cdef cnp.ndarray[phases_st, ndim=2, mode='c'] Vz = np.zeros([nr, nr], dtype=dtype)
    cdef phases_st[:, ::1] vx = Vx
    cdef phases_st[:, ::1] vy = Vy
    cdef phases_st[:, ::1] vz = Vz

    # Local columns
    cdef int_sp_st r, ind, s, c
    cdef floatcomplexs_st d

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    d = D[ind, idx]
                    vx[r, c] += <phases_st> (d * phases[ind, 0])
                    vy[r, c] += <phases_st> (d * phases[ind, 1])
                    vz[r, c] += <phases_st> (d * phases[ind, 2])

        else:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    s = col[ind] / nr
                    d = D[ind, idx]
                    vx[r, c] += <phases_st> (d * phases[s, 0])
                    vy[r, c] += <phases_st> (d * phases[s, 1])
                    vz[r, c] += <phases_st> (d * phases[s, 2])

    return Vx, Vy, Vz


###
# Non-collinear code
###

def phase3_csr_nc(const int_sp_st[::1] ptr,
                  const int_sp_st[::1] ncol,
                  const int_sp_st[::1] col,
                  floatcomplexs_st[:, ::1] D,
                  const complexs_st[:, ::1] phases,
                  const int_sp_st p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col, 2)
    cdef int_sp_st[::1] v_ptr = V_PTR
    cdef int_sp_st[::1] v_ncol = V_NCOL
    cdef int_sp_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] vx = Vx
    cdef complexs_st[::1] vy = Vy
    cdef complexs_st[::1] vz = Vz
    cdef complexs_st ph, v12

    # Local columns (not in NC form)
    cdef int_sp_st nr = ncol.shape[0]
    cdef int_sp_st r, rr, ind, s, c
    cdef int_sp_st s_idx
    cdef floatcomplexs_st *d
    cdef f_matrix_box_nc func
    cdef complexs_st *M = [0, 0, 0, 0]

    if floatcomplexs_st in complexs_st:
        func = matrix_box_nc_cmplx
    else:
        func = matrix_box_nc_real

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                    d = &D[ind, 0]

                    ph = phases[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vz, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                    d = &D[ind, 0]

                    ph = phases[s, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vz, M)

    nr = nr * 2
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


def phase3_array_nc(const int_sp_st[::1] ptr,
                    const int_sp_st[::1] ncol,
                    const int_sp_st[::1] col,
                    floatcomplexs_st[:, ::1] D,
                    const complexs_st[:, ::1] phases,
                    const int_sp_st p_opt):

    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vx = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vy = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vz = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef complexs_st[:, ::1] vx = Vx
    cdef complexs_st[:, ::1] vy = Vy
    cdef complexs_st[:, ::1] vz = Vz

    cdef complexs_st ph
    cdef int_sp_st r, rr, ind, s, c
    cdef int_sp_st s_idx
    cdef floatcomplexs_st *d
    cdef f_matrix_box_nc func
    cdef complexs_st *M = [0, 0, 0, 0]

    if floatcomplexs_st in complexs_st:
        func = matrix_box_nc_cmplx
    else:
        func = matrix_box_nc_real

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    d = &D[ind, 0]

                    ph = phases[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vz, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    d = &D[ind, 0]

                    ph = phases[s, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vz, M)

    return Vx, Vy, Vz



###
# Spin-orbit coupling matrices
###

def phase3_csr_so(const int_sp_st[::1] ptr,
                  const int_sp_st[::1] ncol,
                  const int_sp_st[::1] col,
                  floatcomplexs_st[:, ::1] D,
                  const complexs_st[:, ::1] phases,
                  const int_sp_st p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col, 2)
    cdef int_sp_st[::1] v_ptr = V_PTR
    cdef int_sp_st[::1] v_ncol = V_NCOL
    cdef int_sp_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] vx = Vx
    cdef complexs_st[::1] vy = Vy
    cdef complexs_st[::1] vz = Vz
    cdef complexs_st ph

    # Local columns (not in NC form)
    cdef int_sp_st nr = ncol.shape[0]
    cdef int_sp_st r, rr, ind, s, c
    cdef int_sp_st s_idx
    cdef f_matrix_box_so func
    cdef floatcomplexs_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if floatcomplexs_st in complexs_st:
        func = matrix_box_so_cmplx
    else:
        func = matrix_box_so_real

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                    d = &D[ind, 0]

                    ph = phases[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vz, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                    d = &D[ind, 0]

                    ph = phases[s, 0]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    matrix_add_csr_nc(v_ptr, rr, s_idx, vz, M)

    nr = nr * 2
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


def phase3_array_so(const int_sp_st[::1] ptr,
                    const int_sp_st[::1] ncol,
                    const int_sp_st[::1] col,
                    floatcomplexs_st[:, ::1] D,
                    const complexs_st[:, ::1] phases,
                    const int_sp_st p_opt):

    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vx = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vy = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vz = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef complexs_st[:, ::1] vx = Vx
    cdef complexs_st[:, ::1] vy = Vy
    cdef complexs_st[:, ::1] vz = Vz

    cdef complexs_st ph
    cdef int_sp_st r, rr, ind, s, c
    cdef int_sp_st s_idx
    cdef f_matrix_box_so func
    cdef floatcomplexs_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if floatcomplexs_st in complexs_st:
        func = matrix_box_so_cmplx
    else:
        func = matrix_box_so_real

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    d = &D[ind, 0]

                    ph = phases[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vz, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    d = &D[ind, 0]

                    ph = phases[s, 0]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    matrix_add_array_nc(rr, c, vz, M)

    return Vx, Vy, Vz


def phase3_csr_nambu(const int_sp_st[::1] ptr,
                     const int_sp_st[::1] ncol,
                     const int_sp_st[::1] col,
                     floatcomplexs_st[:, ::1] D,
                     const complexs_st[:, ::1] phases,
                     const int_sp_st p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col, 4)
    cdef int_sp_st[::1] v_ptr = V_PTR
    cdef int_sp_st[::1] v_ncol = V_NCOL
    cdef int_sp_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] vx = Vx
    cdef complexs_st[::1] vy = Vy
    cdef complexs_st[::1] vz = Vz
    cdef complexs_st ph

    # Local columns (not in NC form)
    cdef int_sp_st nr = ncol.shape[0]
    cdef int_sp_st r, rr, ind, s, c
    cdef int_sp_st s_idx
    cdef f_matrix_box_nambu func
    cdef floatcomplexs_st *d
    cdef complexs_st *M = [0] * 16

    if floatcomplexs_st in complexs_st:
        func = matrix_box_nambu_cmplx
    else:
        func = matrix_box_nambu_real

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4
                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                    d = &D[ind, 0]

                    ph = phases[ind, 0]
                    func(d, ph, M)
                    matrix_add_csr_nambu(v_ptr, rr, s_idx, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    matrix_add_csr_nambu(v_ptr, rr, s_idx, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    matrix_add_csr_nambu(v_ptr, rr, s_idx, vz, M)

        else:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4
                    s = col[ind] / nr

                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                    d = &D[ind, 0]

                    ph = phases[s, 0]
                    func(d, ph, M)
                    matrix_add_csr_nambu(v_ptr, rr, s_idx, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    matrix_add_csr_nambu(v_ptr, rr, s_idx, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    matrix_add_csr_nambu(v_ptr, rr, s_idx, vz, M)

    nr = nr * 4
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


def phase3_array_nambu(const int_sp_st[::1] ptr,
                       const int_sp_st[::1] ncol,
                       const int_sp_st[::1] col,
                       floatcomplexs_st[:, ::1] D,
                       const complexs_st[:, ::1] phases,
                       const int_sp_st p_opt):

    cdef int_sp_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vx = np.zeros([nr * 4, nr * 4], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vy = np.zeros([nr * 4, nr * 4], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vz = np.zeros([nr * 4, nr * 4], dtype=dtype)
    cdef complexs_st[:, ::1] vx = Vx
    cdef complexs_st[:, ::1] vy = Vy
    cdef complexs_st[:, ::1] vz = Vz

    cdef complexs_st ph
    cdef int_sp_st r, rr, ind, s, c
    cdef int_sp_st s_idx
    cdef f_matrix_box_nambu func
    cdef floatcomplexs_st *d
    cdef complexs_st *M = [0] * 16

    if floatcomplexs_st in complexs_st:
        func = matrix_box_nambu_cmplx
    else:
        func = matrix_box_nambu_real

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4

                    d = &D[ind, 0]

                    ph = phases[ind, 0]
                    func(d, ph, M)
                    matrix_add_array_nambu(rr, c, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    matrix_add_array_nambu(rr, c, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    matrix_add_array_nambu(rr, c, vz, M)

        else:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4
                    s = col[ind] / nr

                    d = &D[ind, 0]

                    ph = phases[s, 0]
                    func(d, ph, M)
                    matrix_add_array_nambu(rr, c, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    matrix_add_array_nambu(rr, c, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    matrix_add_array_nambu(rr, c, vz, M)

    return Vx, Vy, Vz
