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
    ints_st,
    numerics_st,
    ssize_st,
    type2dtype,
)

from ._matrix_utils cimport (
    _f_matrix_box_nambu,
    _f_matrix_box_nc,
    _f_matrix_box_so,
    _matrix_add_array_nambu,
    _matrix_add_array_nc,
    _matrix_add_csr_nambu,
    _matrix_add_csr_nc,
    _matrix_box_nambu_cmplx,
    _matrix_box_nambu_real,
    _matrix_box_nc_cmplx,
    _matrix_box_nc_real,
    _matrix_box_so_cmplx,
    _matrix_box_so_real,
)

__all__ = [
    "_phase3_csr",
    "_phase3_array",
    "_phase3_csr_nc",
    "_phase3_array_nc",
    "_phase3_csr_so",
    "_phase3_array_so",
    "_phase3_csr_nambu",
    "_phase3_array_nambu",
]


def _phase3_csr(ints_st[::1] ptr,
                ints_st[::1] ncol,
                ints_st[::1] col,
                numerics_st[:, ::1] D,
                const int idx,
                floatcomplexs_st[:, ::1] phases,
                const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    # This may fail, when numerics_st is complex, but floatcomplexs_st is float
    cdef object dtype = type2dtype[floatcomplexs_st](1)
    cdef cnp.ndarray[floatcomplexs_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[floatcomplexs_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[floatcomplexs_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)

    # Local columns
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, ind, s, s_idx, c

    cdef numerics_st d

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                    d = D[ind, idx]
                    Vx[v_ptr[r] + s_idx] += <floatcomplexs_st> (d * phases[ind, 0])
                    Vy[v_ptr[r] + s_idx] += <floatcomplexs_st> (d * phases[ind, 1])
                    Vz[v_ptr[r] + s_idx] += <floatcomplexs_st> (d * phases[ind, 2])

        else:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    s = col[ind] / nr
                    s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                    d = D[ind, idx]
                    Vx[v_ptr[r] + s_idx] += <floatcomplexs_st> (d * phases[s, 0])
                    Vy[v_ptr[r] + s_idx] += <floatcomplexs_st> (d * phases[s, 1])
                    Vz[v_ptr[r] + s_idx] += <floatcomplexs_st> (d * phases[s, 2])

    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))




def _phase3_array(ints_st[::1] ptr,
                  ints_st[::1] ncol,
                  ints_st[::1] col,
                  numerics_st[:, ::1] D,
                  const int idx,
                  floatcomplexs_st[:, ::1] phases,
                  const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[floatcomplexs_st](1)
    cdef cnp.ndarray[floatcomplexs_st, ndim=2, mode='c'] Vx = np.zeros([nr, nr], dtype=dtype)
    cdef cnp.ndarray[floatcomplexs_st, ndim=2, mode='c'] Vy = np.zeros([nr, nr], dtype=dtype)
    cdef cnp.ndarray[floatcomplexs_st, ndim=2, mode='c'] Vz = np.zeros([nr, nr], dtype=dtype)

    # Local columns
    cdef ints_st r, ind, s, c
    cdef numerics_st d

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    d = D[ind, idx]
                    Vx[r, c] += <floatcomplexs_st> (d * phases[ind, 0])
                    Vy[r, c] += <floatcomplexs_st> (d * phases[ind, 1])
                    Vz[r, c] += <floatcomplexs_st> (d * phases[ind, 2])

        else:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    s = col[ind] / nr
                    d = D[ind, idx]
                    Vx[r, c] += <floatcomplexs_st> (d * phases[s, 0])
                    Vy[r, c] += <floatcomplexs_st> (d * phases[s, 1])
                    Vz[r, c] += <floatcomplexs_st> (d * phases[s, 2])

    return Vx, Vy, Vz


###
# Non-collinear code
###

def _phase3_csr_nc(ints_st[::1] ptr,
                   ints_st[::1] ncol,
                   ints_st[::1] col,
                   numerics_st[:, ::1] D,
                   complexs_st[:, ::1] phases,
                   const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col, 2)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] vx = Vx
    cdef complexs_st[::1] vy = Vy
    cdef complexs_st[::1] vz = Vz
    cdef complexs_st ph, v12

    # Local columns (not in NC form)
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx
    cdef numerics_st *d
    cdef _f_matrix_box_nc func
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_nc_cmplx
    else:
        func = _matrix_box_nc_real

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
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vz, M)

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
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vz, M)

    nr = nr * 2
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


def _phase3_array_nc(ints_st[::1] ptr,
                     ints_st[::1] ncol,
                     ints_st[::1] col,
                     numerics_st[:, ::1] D,
                     complexs_st[:, ::1] phases,
                     const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vx = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vy = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vz = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef complexs_st[:, ::1] vx = Vx
    cdef complexs_st[:, ::1] vy = Vy
    cdef complexs_st[:, ::1] vz = Vz

    cdef complexs_st ph
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx
    cdef numerics_st *d
    cdef _f_matrix_box_nc func
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_nc_cmplx
    else:
        func = _matrix_box_nc_real

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    d = &D[ind, 0]

                    ph = phases[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vz, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    d = &D[ind, 0]

                    ph = phases[s, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vz, M)

    return Vx, Vy, Vz



###
# Spin-orbit coupling matrices
###

def _phase3_csr_so(ints_st[::1] ptr,
                   ints_st[::1] ncol,
                   ints_st[::1] col,
                   numerics_st[:, ::1] D,
                   complexs_st[:, ::1] phases,
                   const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col, 2)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] vx = Vx
    cdef complexs_st[::1] vy = Vy
    cdef complexs_st[::1] vz = Vz
    cdef complexs_st ph

    # Local columns (not in NC form)
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx
    cdef _f_matrix_box_so func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_so_cmplx
    else:
        func = _matrix_box_so_real

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
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vz, M)

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
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, vz, M)

    nr = nr * 2
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


def _phase3_array_so(ints_st[::1] ptr,
                     ints_st[::1] ncol,
                     ints_st[::1] col,
                     numerics_st[:, ::1] D,
                     complexs_st[:, ::1] phases,
                     const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vx = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vy = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vz = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef complexs_st[:, ::1] vx = Vx
    cdef complexs_st[:, ::1] vy = Vy
    cdef complexs_st[:, ::1] vz = Vz

    cdef complexs_st ph
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx
    cdef _f_matrix_box_so func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_so_cmplx
    else:
        func = _matrix_box_so_real

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    d = &D[ind, 0]

                    ph = phases[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vz, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    d = &D[ind, 0]

                    ph = phases[s, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, vz, M)

    return Vx, Vy, Vz


def _phase3_csr_nambu(ints_st[::1] ptr,
                      ints_st[::1] ncol,
                      ints_st[::1] col,
                      numerics_st[:, ::1] D,
                      complexs_st[:, ::1] phases,
                      const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col, 4)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] vx = Vx
    cdef complexs_st[::1] vy = Vy
    cdef complexs_st[::1] vz = Vz
    cdef complexs_st ph

    # Local columns (not in NC form)
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx
    cdef _f_matrix_box_nambu func
    cdef numerics_st *d
    cdef complexs_st *M = [0] * 16

    if numerics_st in complexs_st:
        func = _matrix_box_nambu_cmplx
    else:
        func = _matrix_box_nambu_real

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
                    _matrix_add_csr_nambu(v_ptr, rr, s_idx, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    _matrix_add_csr_nambu(v_ptr, rr, s_idx, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    _matrix_add_csr_nambu(v_ptr, rr, s_idx, vz, M)

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
                    _matrix_add_csr_nambu(v_ptr, rr, s_idx, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    _matrix_add_csr_nambu(v_ptr, rr, s_idx, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    _matrix_add_csr_nambu(v_ptr, rr, s_idx, vz, M)

    nr = nr * 4
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


def _phase3_array_nambu(ints_st[::1] ptr,
                        ints_st[::1] ncol,
                        ints_st[::1] col,
                        numerics_st[:, ::1] D,
                        complexs_st[:, ::1] phases,
                        const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vx = np.zeros([nr * 4, nr * 4], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vy = np.zeros([nr * 4, nr * 4], dtype=dtype)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] Vz = np.zeros([nr * 4, nr * 4], dtype=dtype)
    cdef complexs_st[:, ::1] vx = Vx
    cdef complexs_st[:, ::1] vy = Vy
    cdef complexs_st[:, ::1] vz = Vz

    cdef complexs_st ph
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx
    cdef _f_matrix_box_nambu func
    cdef numerics_st *d
    cdef complexs_st *M = [0] * 16

    if numerics_st in complexs_st:
        func = _matrix_box_nambu_cmplx
    else:
        func = _matrix_box_nambu_real

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4

                    d = &D[ind, 0]

                    ph = phases[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nambu(rr, c, vx, M)

                    ph = phases[ind, 1]
                    func(d, ph, M)
                    _matrix_add_array_nambu(rr, c, vy, M)

                    ph = phases[ind, 2]
                    func(d, ph, M)
                    _matrix_add_array_nambu(rr, c, vz, M)

        else:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4
                    s = col[ind] / nr

                    d = &D[ind, 0]

                    ph = phases[s, 0]
                    func(d, ph, M)
                    _matrix_add_array_nambu(rr, c, vx, M)

                    ph = phases[s, 1]
                    func(d, ph, M)
                    _matrix_add_array_nambu(rr, c, vy, M)

                    ph = phases[s, 2]
                    func(d, ph, M)
                    _matrix_add_array_nambu(rr, c, vz, M)

    return Vx, Vy, Vz
