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
    ints_st,
    numerics_st,
    ssize_st,
    type2dtype,
)
from sisl._core._sparse cimport ncol2ptr
from sisl._indices cimport _index_sorted

from ._matrix_utils cimport (
    _f_matrix_box_nambu,
    _f_matrix_box_nc,
    _f_matrix_box_so,
    _matrix_box_nambu_cmplx,
    _matrix_box_nambu_real,
    _matrix_box_nc_cmplx,
    _matrix_box_nc_real,
    _matrix_box_so_cmplx,
    _matrix_box_so_real,
)

__all__ = [
    "_phase_sc_csr",
    "_phase_sc_array",
    "_phase_sc_csr_nc",
    "_phase_sc_array_nc",
    "_phase_sc_csr_diag",
    "_phase_sc_array_diag",
    "_phase_sc_csr_so",
    "_phase_sc_array_so",
    "_phase_sc_csr_nambu",
    "_phase_sc_array_nambu",
]


def _phase_sc_csr(ints_st[::1] ptr,
                  ints_st[::1] ncol,
                  ints_st[::1] col,
                  const ints_st nc,
                  numerics_st[:, ::1] D,
                  const int idx,
                  floatcomplexs_st[::1] phases,
                  const int p_opt):

    # Now copy the sparse matrix form
    cdef ints_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[ints_st](1)
    cdef cnp.ndarray[ints_st, mode='c'] V_PTR = np.empty([nr + 1], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_NCOL = np.empty([nr], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_COL = np.empty([inline_sum(ncol)], dtype=idtype)

    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[floatcomplexs_st](1)
    cdef cnp.ndarray[floatcomplexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef floatcomplexs_st[::1] v = V

    cdef ints_st r, c, nz, ind, cind
    cdef floatcomplexs_st ph

    # Copy ncol
    v_ncol[:] = ncol[:]

    # This abstraction allows to handle non-finalized CSR matrices
    cind = 0

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                v_ptr[r] = cind
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    v[cind] = <floatcomplexs_st> D[ind, idx]
                    v_col[cind] = col[ind]
                    cind = cind + 1

        elif p_opt == 0:
            for r in range(nr):
                v_ptr[r] = cind
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    ph = phases[ind]
                    v[cind] = <floatcomplexs_st> (D[ind, idx] * ph)
                    v_col[cind] = col[ind]
                    cind = cind + 1

        else:
            for r in range(nr):
                v_ptr[r] = cind
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    ph = phases[col[ind] / nr]
                    v[cind] = <floatcomplexs_st> (D[ind, idx] * ph)
                    v_col[cind] = col[ind]
                    cind = cind + 1

    v_ptr[nr] = cind

    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nc))


def _phase_sc_array(ints_st[::1] ptr,
                    ints_st[::1] ncol,
                    ints_st[::1] col,
                    const ints_st nc,
                    numerics_st[:, ::1] D,
                    const int idx,
                    floatcomplexs_st[::1] phases,
                    const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[floatcomplexs_st](1)
    cdef cnp.ndarray[floatcomplexs_st, ndim=2, mode='c'] V = np.zeros([nr, nc], dtype=dtype)
    cdef floatcomplexs_st[:, ::1] v = V

    cdef ints_st r, c, nz, ind
    cdef floatcomplexs_st ph

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    v[r, col[ind]] = <floatcomplexs_st> D[ind, idx]

        elif p_opt == 0:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    ph = phases[ind]
                    v[r, col[ind]] = <floatcomplexs_st> (D[ind, idx] * ph)

        else:
            for r in range(nr):
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    ph = phases[col[ind] / nr]
                    v[r, col[ind]] = <floatcomplexs_st> (D[ind, idx] * ph)

    return V


def _phase_sc_csr_nc(ints_st[::1] ptr,
                     ints_st[::1] ncol,
                     ints_st[::1] col,
                     const ints_st nc,
                     numerics_st[:, ::1] D,
                     complexs_st[::1] phases,
                     const int p_opt):

    # Now copy the sparse matrix form
    cdef ints_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[ints_st](1)
    cdef cnp.ndarray[ints_st, mode='c'] V_PTR = np.empty([nr*2 + 1], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_NCOL = np.empty([nr*2], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_COL = np.empty([inline_sum(ncol)*4], dtype=idtype)

    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    cdef ints_st r, rr, cind, c, nz, ind
    cdef complexs_st ph
    cdef _f_matrix_box_nc func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_nc_cmplx
    else:
        func = _matrix_box_nc_real

    # We have to do it manually due to the double elements per matrix element
    ncol2ptr(nr, ncol, v_ptr, 2, 2)

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                rr = r * 2
                v_ncol[rr] = ncol[r] * 2
                v_ncol[rr+1] = ncol[r] * 2

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2

                    v[v_ptr[rr] + cind] = <complexs_st> D[ind, 0]
                    v_col[v_ptr[rr] + cind] = c
                    ph = <complexs_st> (D[ind, 2] + 1j * D[ind, 3])
                    v[v_ptr[rr] + cind+1] = ph
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v[v_ptr[rr+1] + cind] = ph.conjugate()
                    v_col[v_ptr[rr+1] + cind] = c
                    v[v_ptr[rr+1] + cind+1] = <complexs_st> D[ind, 1]
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
                    v[v_ptr[rr] + cind] = M[0]
                    v_col[v_ptr[rr] + cind] = c
                    v[v_ptr[rr] + cind+1] = M[1]
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v[v_ptr[rr+1] + cind] = M[2]
                    v_col[v_ptr[rr+1] + cind] = c
                    v[v_ptr[rr+1] + cind+1] = M[3]
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

                    v[v_ptr[rr] + cind] = M[0]
                    v_col[v_ptr[rr] + cind] = c
                    v[v_ptr[rr] + cind+1] = M[1]
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v[v_ptr[rr+1] + cind] = M[2]
                    v_col[v_ptr[rr+1] + cind] = c
                    v[v_ptr[rr+1] + cind+1] = M[3]
                    v_col[v_ptr[rr+1] + cind+1] = c + 1

                    cind = cind + 2

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 2, nc * 2))


def _phase_sc_array_nc(ints_st[::1] ptr,
                       ints_st[::1] ncol,
                       ints_st[::1] col,
                       const ints_st nc,
                       numerics_st[:, ::1] D,
                       complexs_st[::1] phases,
                       const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr*2, nc*2], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    cdef complexs_st ph
    cdef ints_st r, rr, c, nz, ind
    cdef numerics_st *d
    cdef _f_matrix_box_nc func
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_nc_cmplx
    else:
        func = _matrix_box_nc_real

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    v[rr, c] = <complexs_st> D[ind, 0]
                    ph = <complexs_st> (D[ind, 2] + 1j * D[ind, 3])
                    v[rr, c+1] = ph
                    v[rr+1, c] = ph.conjugate()
                    v[rr+1, c+1] = <complexs_st> D[ind, 1]

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    v[rr, c] = M[0]
                    v[rr, c+1] = M[1]
                    v[rr+1, c] = M[2]
                    v[rr+1, c+1] = M[3]

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[col[ind] / nr]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    v[rr, c] = M[0]
                    v[rr, c+1] = M[1]
                    v[rr+1, c] = M[2]
                    v[rr+1, c+1] = M[3]

    return V

def _phase_sc_csr_diag(ints_st[::1] ptr,
                      ints_st[::1] ncol,
                      ints_st[::1] col,
                      const ints_st nc,
                      numerics_st[:, ::1] D,
                      const int idx,
                      complexs_st[::1] phases,
                      const int p_opt,
                      const int per_row):

    # Now copy the sparse matrix form
    cdef ints_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[ints_st](1)
    cdef cnp.ndarray[ints_st, mode='c'] V_PTR = np.empty([nr*per_row + 1], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_NCOL = np.empty([nr*per_row], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_COL = np.empty([inline_sum(ncol)*per_row], dtype=idtype)

    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    cdef ints_st r, rr, ir, cind, c, nz, ind, ic
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


def _phase_sc_array_diag(ints_st[::1] ptr,
                         ints_st[::1] ncol,
                         ints_st[::1] col,
                         const ints_st nc,
                         numerics_st[:, ::1] D,
                         const int idx,
                         complexs_st[::1] phases,
                         const int p_opt,
                         const int per_row):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr*per_row, nc*per_row], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    cdef complexs_st d
    cdef ints_st r, rr, c, nz, ind, ic

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


def _phase_sc_csr_so(ints_st[::1] ptr,
                     ints_st[::1] ncol,
                     ints_st[::1] col,
                     const ints_st nc,
                     numerics_st[:, ::1] D,
                     complexs_st[::1] phases,
                     const int p_opt):

    # Now copy the sparse matrix form
    cdef ints_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[ints_st](1)
    cdef cnp.ndarray[ints_st, mode='c'] V_PTR = np.empty([nr*2 + 1], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_NCOL = np.empty([nr*2], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_COL = np.empty([inline_sum(ncol)*4], dtype=idtype)

    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    cdef ints_st r, rr, cind, c, nz, ind
    cdef complexs_st ph
    cdef _f_matrix_box_so func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_so_cmplx
    else:
        func = _matrix_box_so_real

    # We have to do it manually due to the double elements per matrix element
    ncol2ptr(nr, ncol, v_ptr, 2, 2)

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                rr = r * 2
                v_ncol[rr] = ncol[r] * 2
                v_ncol[rr+1] = ncol[r] * 2

                cind = 0
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2

                    v[v_ptr[rr] + cind] = <complexs_st> (D[ind, 0] + 1j * D[ind, 4])
                    v_col[v_ptr[rr] + cind] = c
                    v[v_ptr[rr] + cind+1] = <complexs_st> (D[ind, 2] + 1j * D[ind, 3])
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v[v_ptr[rr+1] + cind] = <complexs_st> (D[ind, 6] + 1j * D[ind, 7])
                    v_col[v_ptr[rr+1] + cind] = c
                    v[v_ptr[rr+1] + cind+1] = <complexs_st> (D[ind, 1] + 1j * D[ind, 5])
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

                    v[v_ptr[rr] + cind] = M[0]
                    v_col[v_ptr[rr] + cind] = c
                    v[v_ptr[rr] + cind+1] = M[1]
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v[v_ptr[rr+1] + cind] = M[2]
                    v_col[v_ptr[rr+1] + cind] = c
                    v[v_ptr[rr+1] + cind+1] = M[3]
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

                    v[v_ptr[rr] + cind] = M[0]
                    v_col[v_ptr[rr] + cind] = c
                    v[v_ptr[rr] + cind+1] = M[1]
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v[v_ptr[rr+1] + cind] = M[2]
                    v_col[v_ptr[rr+1] + cind] = c
                    v[v_ptr[rr+1] + cind+1] = M[3]
                    v_col[v_ptr[rr+1] + cind+1] = c + 1

                    cind = cind + 2

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 2, nc * 2))


def _phase_sc_array_so(ints_st[::1] ptr,
                       ints_st[::1] ncol,
                       ints_st[::1] col,
                       const ints_st nc,
                       numerics_st[:, ::1] D,
                       complexs_st[::1] phases,
                       const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr*2, nc*2], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    cdef complexs_st ph
    cdef ints_st r, rr, c, nz, ind
    cdef _f_matrix_box_so func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_so_cmplx
    else:
        func = _matrix_box_so_real

    with nogil:
        if p_opt == -1:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2

                    v[rr, c] = <complexs_st> (D[ind, 0] + 1j * D[ind, 4])
                    v[rr, c+1] = <complexs_st> (D[ind, 2] + 1j * D[ind, 3])
                    v[rr+1, c] = <complexs_st> (D[ind, 6] + 1j * D[ind, 7])
                    v[rr+1, c+1] = <complexs_st> (D[ind, 1] + 1j * D[ind, 5])

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    v[rr, c] = M[0]
                    v[rr, c+1] = M[1]
                    v[rr+1, c] = M[2]
                    v[rr+1, c+1] = M[3]

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 2
                    ph = phases[col[ind] / nr]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    v[rr, c] = M[0]
                    v[rr, c+1] = M[1]
                    v[rr+1, c] = M[2]
                    v[rr+1, c+1] = M[3]

    return V


def _phase_sc_csr_nambu(ints_st[::1] ptr,
                        ints_st[::1] ncol,
                        ints_st[::1] col,
                        const ints_st nc,
                        numerics_st[:, ::1] D,
                        complexs_st[::1] phases,
                        const int p_opt):

    # Now copy the sparse matrix form
    cdef ints_st nr = ncol.shape[0]
    cdef object idtype = type2dtype[ints_st](1)
    cdef cnp.ndarray[ints_st, mode='c'] V_PTR = np.empty([nr*4 + 1], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_NCOL = np.empty([nr*4], dtype=idtype)
    cdef cnp.ndarray[ints_st, mode='c'] V_COL = np.empty([inline_sum(ncol)*16], dtype=idtype)

    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] V = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st[::1] v = V

    cdef ints_st r, rr, cind, c, nz, ind
    cdef complexs_st ph
    cdef _f_matrix_box_nambu func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0, 0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_nambu_cmplx
    else:
        func = _matrix_box_nambu_real

    # We have to do it manually due to the quadrouble elements per matrix element
    ncol2ptr(nr, ncol, v_ptr, 4, 4)

    with nogil:
        if p_opt == -1:
            pass

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

                    v[v_ptr[rr] + cind] = M[0]
                    v_col[v_ptr[rr] + cind] = c
                    v[v_ptr[rr] + cind+1] = M[1]
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v[v_ptr[rr+1] + cind] = M[2]
                    v_col[v_ptr[rr+1] + cind] = c
                    v[v_ptr[rr+1] + cind+1] = M[3]
                    v_col[v_ptr[rr+1] + cind+1] = c + 1
                    # Delta
                    v[v_ptr[rr] + cind+2] = M[4]
                    v_col[v_ptr[rr] + cind+2] = c + 2
                    v[v_ptr[rr] + cind+3] = M[5]
                    v_col[v_ptr[rr] + cind+3] = c + 3
                    v[v_ptr[rr+1] + cind+2] = M[6]
                    v_col[v_ptr[rr+1] + cind+2] = c + 2
                    v[v_ptr[rr+1] + cind+3] = M[7]
                    v_col[v_ptr[rr+1] + cind+3] = c + 3
                    # Delta^dagger
                    v[v_ptr[rr+2] + cind] = M[4].conjugate()
                    v_col[v_ptr[rr+2] + cind] = c
                    v[v_ptr[rr+2] + cind+1] = M[6].conjugate()
                    v_col[v_ptr[rr+2] + cind+1] = c + 1
                    v[v_ptr[rr+3] + cind] = M[5].conjugate()
                    v_col[v_ptr[rr+3] + cind] = c
                    v[v_ptr[rr+3] + cind+1] = M[7].conjugate()
                    v_col[v_ptr[rr+3] + cind+1] = c + 1
                    # -H^*
                    v[v_ptr[rr+2] + cind+2] = -M[0].conjugate()
                    v_col[v_ptr[rr+2] + cind+2] = c+2
                    v[v_ptr[rr+2] + cind+3] = -M[1].conjugate()
                    v_col[v_ptr[rr+2] + cind+3] = c + 3
                    v[v_ptr[rr+3] + cind+2] = -M[2].conjugate()
                    v_col[v_ptr[rr+3] + cind+2] = c +2
                    v[v_ptr[rr+3] + cind+3] = -M[3].conjugate()
                    v_col[v_ptr[rr+3] + cind+3] = c + 3

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

                    v[v_ptr[rr] + cind] = M[0]
                    v_col[v_ptr[rr] + cind] = c
                    v[v_ptr[rr] + cind+1] = M[1]
                    v_col[v_ptr[rr] + cind+1] = c + 1
                    v[v_ptr[rr+1] + cind] = M[2]
                    v_col[v_ptr[rr+1] + cind] = c
                    v[v_ptr[rr+1] + cind+1] = M[3]
                    v_col[v_ptr[rr+1] + cind+1] = c + 1
                    # Delta
                    v[v_ptr[rr] + cind+2] = M[4]
                    v_col[v_ptr[rr] + cind+2] = c + 2
                    v[v_ptr[rr] + cind+3] = M[5]
                    v_col[v_ptr[rr] + cind+3] = c + 3
                    v[v_ptr[rr+1] + cind+2] = M[6]
                    v_col[v_ptr[rr+1] + cind+2] = c + 2
                    v[v_ptr[rr+1] + cind+3] = M[7]
                    v_col[v_ptr[rr+1] + cind+3] = c + 3
                    # Delta^dagger
                    v[v_ptr[rr+2] + cind] = M[4].conjugate()
                    v_col[v_ptr[rr+2] + cind] = c
                    v[v_ptr[rr+2] + cind+1] = M[6].conjugate()
                    v_col[v_ptr[rr+2] + cind+1] = c + 1
                    v[v_ptr[rr+3] + cind] = M[5].conjugate()
                    v_col[v_ptr[rr+3] + cind] = c
                    v[v_ptr[rr+3] + cind+1] = M[7].conjugate()
                    v_col[v_ptr[rr+3] + cind+1] = c + 1
                    # -H^*
                    v[v_ptr[rr+2] + cind+2] = -M[0].conjugate()
                    v_col[v_ptr[rr+2] + cind+2] = c+2
                    v[v_ptr[rr+2] + cind+3] = -M[1].conjugate()
                    v_col[v_ptr[rr+2] + cind+3] = c + 3
                    v[v_ptr[rr+3] + cind+2] = -M[2].conjugate()
                    v_col[v_ptr[rr+3] + cind+2] = c +2
                    v[v_ptr[rr+3] + cind+3] = -M[3].conjugate()
                    v_col[v_ptr[rr+3] + cind+3] = c + 3

                    cind = cind + 4

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 4, nc * 4))


def _phase_sc_array_nambu(ints_st[::1] ptr,
                          ints_st[::1] ncol,
                          ints_st[::1] col,
                          const ints_st nc,
                          numerics_st[:, ::1] D,
                          complexs_st[::1] phases,
                          const int p_opt):

    cdef ints_st nr = ncol.shape[0]

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, ndim=2, mode='c'] V = np.zeros([nr*4, nc*4], dtype=dtype)
    cdef complexs_st[:, ::1] v = V

    cdef complexs_st ph
    cdef ints_st r, rr, c, nz, ind
    cdef _f_matrix_box_nambu func
    cdef numerics_st *d
    cdef complexs_st *M = [0, 0, 0, 0, 0, 0, 0, 0]

    if numerics_st in complexs_st:
        func = _matrix_box_nambu_cmplx
    else:
        func = _matrix_box_nambu_real

    with nogil:
        if p_opt == -1:
            pass

        elif p_opt == 0:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 4
                    ph = phases[ind]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    v[rr, c] = M[0]
                    v[rr, c+1] = M[1]
                    v[rr+1, c] = M[2]
                    v[rr+1, c+1] = M[3]
                    # Delta
                    v[rr, c+2] = M[4]
                    v[rr, c+3] = M[5]
                    v[rr+1, c+2] = M[6]
                    v[rr+1, c+3] = M[7]
                    # Delta^dagger
                    v[rr+2, c] = M[4].conjugate()
                    v[rr+2, c+1] = M[6].conjugate()
                    v[rr+3, c] = M[5].conjugate()
                    v[rr+3, c+1] = M[7].conjugate()
                    # -H^*
                    v[rr+2, c+2] = -M[0].conjugate()
                    v[rr+2, c+3] = -M[1].conjugate()
                    v[rr+3, c+2] = -M[2].conjugate()
                    v[rr+3, c+3] = -M[3].conjugate()

        else:
            for r in range(nr):
                rr = r * 4
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = col[ind] * 4
                    ph = phases[col[ind] / nr]

                    d = &D[ind, 0]
                    func(d, ph, M)
                    v[rr, c] = M[0]
                    v[rr, c+1] = M[1]
                    v[rr+1, c] = M[2]
                    v[rr+1, c+1] = M[3]
                    # Delta
                    v[rr, c+2] = M[4]
                    v[rr, c+3] = M[5]
                    v[rr+1, c+2] = M[6]
                    v[rr+1, c+3] = M[7]
                    # Delta^dagger
                    v[rr+2, c] = M[4].conjugate()
                    v[rr+2, c+1] = M[6].conjugate()
                    v[rr+3, c] = M[5].conjugate()
                    v[rr+3, c+1] = M[7].conjugate()
                    # -H^*
                    v[rr+2, c+2] = -M[0].conjugate()
                    v[rr+2, c+3] = -M[1].conjugate()
                    v[rr+3, c+2] = -M[2].conjugate()
                    v[rr+3, c+3] = -M[3].conjugate()

    return V
