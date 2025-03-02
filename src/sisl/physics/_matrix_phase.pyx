# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython

import numpy as np

cimport numpy as cnp

from scipy.sparse import csr_matrix

from sisl._indices cimport _index_sorted

from sisl._core._sparse import fold_csr_matrix, fold_csr_matrix_diag

from sisl._core._dtypes cimport (
    complexs_st,
    floatcomplexs_st,
    floats_st,
    ints_st,
    numerics_st,
    reals_st,
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
    "_phase_csr",
    "_phase_array",
    "_phase_csr_nc",
    "_phase_array_nc",
    "_phase_csr_diag",
    "_phase_array_diag",
    "_phase_csr_so",
    "_phase_array_so",
    "_phase_csr_nambu",
    "_phase_array_nambu",
]

"""
In this Cython code we use `p_opt` to signal whether the resulting
matrices will use the phases variable.

There are 3 cases:

p_opt == -1:
    no phases are added, the `phases` array will not be accessed
p_opt == 0:
    the phases are *per* spares index, i.e. the array is as big
    as the sparse data.
p_opt == 1:
    the phases are in reduced format where each column block
    uses the same phase. A column block is defined as `col[ind] / nr` which
    results in a unique index.
"""


def _phase_csr(ints_st[::1] ptr,
               ints_st[::1] ncol,
               ints_st[::1] col,
               numerics_st[:, ::1] D,
               const ints_st idx,
               floatcomplexs_st[::1] phases,
               const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL
    cdef ints_st[::1] tmp

    # This may fail, when numerics_st is complex, but floatcomplexs_st is float
    cdef object dtype = type2dtype[floatcomplexs_st](1)
    cdef cnp.ndarray[floatcomplexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef floatcomplexs_st[::1] v = V

    # Local columns
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, ind, s, s_idx, c

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr

                    tmp = v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]]
                    s_idx = _index_sorted(tmp, c)
                    v[v_ptr[r] + s_idx] += <floatcomplexs_st> (D[ind, idx])

        elif p_opt == 0:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr

                    tmp = v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]]
                    s_idx = _index_sorted(tmp, c)
                    v[v_ptr[r] + s_idx] += <floatcomplexs_st> (D[ind, idx] * phases[ind])

        else:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    s = col[ind] / nr

                    tmp = v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]]
                    s_idx = _index_sorted(tmp, c)
                    v[v_ptr[r] + s_idx] += <floatcomplexs_st> (D[ind, idx] * phases[s])

    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nr))


def _phase_array(ints_st[::1] ptr,
                 ints_st[::1] ncol,
                 ints_st[::1] col,
                 numerics_st[:, ::1] D,
                 const int idx,
                 floatcomplexs_st[::1] phases,
                 const int p_opt):

    cdef ints_st[::1] tmp
    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[floatcomplexs_st](1)
    cdef cnp.ndarray[floatcomplexs_st, ndim=2, mode='c'] V = np.zeros([nr, nr], dtype=dtype)
    cdef floatcomplexs_st[:, ::1] v = V

    # Local columns
    cdef ints_st r, ind, s, c

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    v[r, c] += <floatcomplexs_st> (D[ind, idx])

        elif p_opt == 0:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    v[r, c] += <floatcomplexs_st> (D[ind, idx] * phases[ind])

        else:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] % nr
                    s = col[ind] / nr
                    v[r, c] += <floatcomplexs_st> (D[ind, idx] * phases[s])

    return V


def _phase_csr_diag(ints_st[::1] ptr,
                    ints_st[::1] ncol,
                    ints_st[::1] col,
                    numerics_st[:, ::1] D,
                    const int idx,
                    complexs_st[::1] phases,
                    const int p_opt,
                    const int per_row):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix_diag(ptr, ncol, col, per_row)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL
    cdef ints_st[::1] tmp

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    # Local columns
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, rr, ind, s, s_idx, c, ic

    cdef complexs_st d

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                rr = r * per_row
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * per_row

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = <complexs_st> D[ind, idx]
                    for ic in range(per_row):
                        v[v_ptr[rr+ic] + s_idx] += d

        elif p_opt == 0:
            for r in range(nr):
                rr = r * per_row
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * per_row

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = phases[ind] * D[ind, idx]
                    for ic in range(per_row):
                        v[v_ptr[rr+ic] + s_idx] += d

        else:
            for r in range(nr):
                rr = r * per_row
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * per_row
                    s = col[ind] / nr

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = phases[s] * D[ind, idx]
                    for ic in range(per_row):
                        v[v_ptr[rr+ic] + s_idx] += d

    nr = nr * per_row
    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nr))


def _phase_array_diag(ints_st[::1] ptr,
                      ints_st[::1] ncol,
                      ints_st[::1] col,
                      numerics_st[:, ::1] D,
                      const int idx,
                      complexs_st[::1] phases,
                      const int p_opt,
                      const int per_row):

    cdef ints_st[::1] tmp
    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr * per_row, nr * per_row], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    # Local columns
    cdef ints_st r, rr, ind, s, c, ic

    cdef complexs_st d

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                rr = r * per_row
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * per_row
                    d = D[ind, idx]
                    for ic in range(per_row):
                        v[rr + ic, c + ic] += d

        elif p_opt == 0:
            for r in range(nr):
                rr = r * per_row
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * per_row
                    d = phases[ind] * D[ind, idx]
                    for ic in range(per_row):
                        v[rr + ic, c + ic] += d

        else:
            for r in range(nr):
                rr = r * per_row
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * per_row
                    s = col[ind] / nr
                    d = phases[s] * D[ind, idx]
                    for ic in range(per_row):
                        v[rr + ic, c + ic] += d

    return V


def _phase_csr_nc(ints_st[::1] ptr,
                  ints_st[::1] ncol,
                  ints_st[::1] col,
                  numerics_st[:, ::1] D,
                  complexs_st[::1] phases,
                  const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col, 2)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL
    cdef ints_st[::1] tmp

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    # Local columns
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, rr, ind, s, s_idx, c

    cdef complexs_st ph
    cdef _f_matrix_box_nc func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_nc_cmplx
    else:
        func = _matrix_box_nc_real

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, v, M)

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    ph = phases[ind]

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, v, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr
                    ph = phases[s]

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, v, M)

    nr = nr * 2
    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nr))


def _phase_array_nc(ints_st[::1] ptr,
                    ints_st[::1] ncol,
                    ints_st[::1] col,
                    numerics_st[:, ::1] D,
                    complexs_st[::1] phases,
                    const int p_opt):

    cdef ints_st[::1] tmp
    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    # Local columns
    cdef ints_st r, rr, ind, s, c

    cdef complexs_st ph
    cdef _f_matrix_box_nc func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_nc_cmplx
    else:
        func = _matrix_box_nc_real

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, v, M)

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, v, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr
                    ph = phases[s]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, v, M)

    return V


def _phase_csr_so(ints_st[::1] ptr,
                  ints_st[::1] ncol,
                  ints_st[::1] col,
                  numerics_st[:, ::1] D,
                  complexs_st[::1] phases,
                  const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col, 2)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL
    cdef ints_st[::1] tmp

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    # Local columns
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, rr, ind, s, s_idx, c

    cdef complexs_st ph
    cdef _f_matrix_box_so func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_so_cmplx
    else:
        func = _matrix_box_so_real

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, v, M)

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    ph = phases[ind]

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, v, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr
                    ph = phases[s]

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_csr_nc(v_ptr, rr, s_idx, v, M)

    nr = nr * 2
    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nr))


def _phase_array_so(ints_st[::1] ptr,
                    ints_st[::1] ncol,
                    ints_st[::1] col,
                    numerics_st[:, ::1] D,
                    complexs_st[::1] phases,
                    const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr * 2, nr * 2], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    # Local columns
    cdef ints_st r, rr, s, c, ind

    cdef complexs_st ph
    cdef _f_matrix_box_so func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_so_cmplx
    else:
        func = _matrix_box_so_real

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, v, M)

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, v, M)

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr
                    ph = phases[s]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nc(rr, c, v, M)

    return V


def _phase_csr_nambu(ints_st[::1] ptr,
                     ints_st[::1] ncol,
                     ints_st[::1] col,
                     numerics_st[:, ::1] D,
                     complexs_st[::1] phases,
                     const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(ptr, ncol, col, 4)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL
    cdef ints_st[::1] tmp

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    # Local columns
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, rr, ind, s, s_idx, c

    cdef complexs_st ph
    cdef _f_matrix_box_nambu func
    cdef numerics_st *d
    cdef complexs_st *M = [0] * 16

    if numerics_st in complexs_st:
        func = _matrix_box_nambu_cmplx
    else:
        func = _matrix_box_nambu_real

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_csr_nambu(v_ptr, rr, s_idx, v, M)

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4
                    ph = phases[ind]

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_csr_nambu(v_ptr, rr, s_idx, v, M)

        else:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4
                    s = col[ind] / nr
                    ph = phases[s]

                    tmp = v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]]
                    s_idx = _index_sorted(tmp, c)

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_csr_nambu(v_ptr, rr, s_idx, v, M)

    nr = nr * 4
    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nr))


def _phase_array_nambu(ints_st[::1] ptr,
                       ints_st[::1] ncol,
                       ints_st[::1] col,
                       numerics_st[:, ::1] D,
                       complexs_st[::1] phases,
                       const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr * 4, nr * 4], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    # Local columns
    cdef ints_st r, rr, s, c, ind

    cdef complexs_st ph
    cdef _f_matrix_box_nambu func
    cdef numerics_st *d
    cdef complexs_st *M = [0] * 16

    if numerics_st in complexs_st:
        func = _matrix_box_nambu_cmplx
    else:
        func = _matrix_box_nambu_real

    with nogil:
        if p_opt == -1:
            ph = 1. + 0j
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nambu(rr, c, v, M)

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nambu(rr, c, v, M)

        else:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 4
                    s = col[ind] / nr
                    ph = phases[s]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    _matrix_add_array_nambu(rr, c, v, M)

    return V
