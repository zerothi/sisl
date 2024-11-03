# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np

cimport numpy as cnp

from scipy.sparse import csr_matrix

from sisl._indices cimport _index_sorted

from sisl._core._sparse import fold_csr_matrix, fold_csr_matrix_nc

from sisl._core._dtypes cimport (
    complexs_st,
    floatcomplexs_st,
    floats_st,
    ints_st,
    numerics_st,
    ssize_st,
    type2dtype,
)

__all__ = [
    "_phase3_csr",
    "_phase3_array",
    "_phase3_csr_nc",
    "_phase3_array_nc",
    "_phase3_csr_so",
    "_phase3_array_so",
]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
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




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_csr_nc(ints_st[::1] ptr,
                   ints_st[::1] ncol,
                   ints_st[::1] col,
                   numerics_st[:, ::1] D,
                   complexs_st[:, ::1] phases,
                   const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix_nc(ptr, ncol, col)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st ph, v12

    # Local columns (not in NC form)
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                    ph = phases[ind, 0]
                    Vx[v_ptr[rr] + s_idx] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vx[v_ptr[rr] + s_idx+1] += ph * v12
                    Vx[v_ptr[rr+1] + s_idx] += ph * v12.conjugate()
                    Vx[v_ptr[rr+1] + s_idx+1] += ph * D[ind, 1]

                    ph = phases[ind, 1]
                    Vy[v_ptr[rr] + s_idx] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vy[v_ptr[rr] + s_idx+1] += ph * v12
                    Vy[v_ptr[rr+1] + s_idx] += ph * v12.conjugate()
                    Vy[v_ptr[rr+1] + s_idx+1] += ph * D[ind, 1]

                    ph = phases[ind, 2]
                    Vz[v_ptr[rr] + s_idx] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vz[v_ptr[rr] + s_idx+1] += ph * v12
                    Vz[v_ptr[rr+1] + s_idx] += ph * v12.conjugate()
                    Vz[v_ptr[rr+1] + s_idx+1] += ph * D[ind, 1]

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)


                    ph = phases[s, 0]
                    Vx[v_ptr[rr] + s_idx] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vx[v_ptr[rr] + s_idx+1] += ph * v12
                    Vx[v_ptr[rr+1] + s_idx] += ph * v12.conjugate()
                    Vx[v_ptr[rr+1] + s_idx+1] += ph * D[ind, 1]

                    ph = phases[s, 1]
                    Vy[v_ptr[rr] + s_idx] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vy[v_ptr[rr] + s_idx+1] += ph * v12
                    Vy[v_ptr[rr+1] + s_idx] += ph * v12.conjugate()
                    Vy[v_ptr[rr+1] + s_idx+1] += ph * D[ind, 1]

                    ph = phases[s, 2]
                    Vz[v_ptr[rr] + s_idx] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vz[v_ptr[rr] + s_idx+1] += ph * v12
                    Vz[v_ptr[rr+1] + s_idx] += ph * v12.conjugate()
                    Vz[v_ptr[rr+1] + s_idx+1] += ph * D[ind, 1]

    nr = nr * 2
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
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

    cdef complexs_st ph, vv
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    ph = phases[ind, 0]
                    Vx[rr, c] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vx[rr, c+1] += ph * v12
                    Vx[rr+1, c] += ph * v12.conjugate()
                    Vx[rr+1, c+1] += ph * D[ind, 1]

                    ph = phases[ind, 1]
                    Vy[rr, c] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vy[rr, c+1] += ph * v12
                    Vy[rr+1, c] += ph * v12.conjugate()
                    Vy[rr+1, c+1] += ph * D[ind, 1]

                    ph = phases[ind, 2]
                    Vz[rr, c] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vz[rr, c+1] += ph * v12
                    Vz[rr+1, c] += ph * v12.conjugate()
                    Vz[rr+1, c+1] += ph * D[ind, 1]

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    ph = phases[s, 0]
                    Vx[rr, c] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vx[rr, c+1] += ph * v12
                    Vx[rr+1, c] += ph * v12.conjugate()
                    Vx[rr+1, c+1] += ph * D[ind, 1]

                    ph = phases[s, 1]
                    Vy[rr, c] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vy[rr, c+1] += ph * v12
                    Vy[rr+1, c] += ph * v12.conjugate()
                    Vy[rr+1, c+1] += ph * D[ind, 1]

                    ph = phases[s, 2]
                    Vz[rr, c] += ph * D[ind, 0]
                    v12 = (D[ind, 2] + 1j * D[ind, 3])
                    Vz[rr, c+1] += ph * v12
                    Vz[rr+1, c] += ph * v12.conjugate()
                    Vz[rr+1, c+1] += ph * D[ind, 1]

    return Vx, Vy, Vz



###
# Spin-orbit coupling matrices
###

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_csr_so(ints_st[::1] ptr,
                   ints_st[::1] ncol,
                   ints_st[::1] col,
                   # complexs_st requires only 4 indices...
                   floats_st[:, ::1] D,
                   complexs_st[:, ::1] phases,
                   const int p_opt):

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix_nc(ptr, ncol, col)
    cdef ints_st[::1] v_ptr = V_PTR
    cdef ints_st[::1] v_ncol = V_NCOL
    cdef ints_st[::1] v_col = V_COL

    cdef object dtype = type2dtype[complexs_st](1)
    cdef cnp.ndarray[complexs_st, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef cnp.ndarray[complexs_st, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=dtype)
    cdef complexs_st ph, vv

    # Local columns (not in NC form)
    cdef ints_st nr = ncol.shape[0]
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                    ph = phases[ind, 0]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    Vx[v_ptr[rr] + s_idx] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    Vx[v_ptr[rr] + s_idx+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    Vx[v_ptr[rr+1] + s_idx] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    Vx[v_ptr[rr+1] + s_idx+1] += ph * vv

                    ph = phases[ind, 1]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    Vy[v_ptr[rr] + s_idx] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    Vy[v_ptr[rr] + s_idx+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    Vy[v_ptr[rr+1] + s_idx] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    Vy[v_ptr[rr+1] + s_idx+1] += ph * vv

                    ph = phases[ind, 2]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    Vz[v_ptr[rr] + s_idx] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    Vz[v_ptr[rr] + s_idx+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    Vz[v_ptr[rr+1] + s_idx] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    Vz[v_ptr[rr+1] + s_idx+1] += ph * vv

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                    ph = phases[s, 0]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    Vx[v_ptr[rr] + s_idx] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    Vx[v_ptr[rr] + s_idx+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    Vx[v_ptr[rr+1] + s_idx] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    Vx[v_ptr[rr+1] + s_idx+1] += ph * vv

                    ph = phases[s, 1]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    Vy[v_ptr[rr] + s_idx] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    Vy[v_ptr[rr] + s_idx+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    Vy[v_ptr[rr+1] + s_idx] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    Vy[v_ptr[rr+1] + s_idx+1] += ph * vv

                    ph = phases[s, 2]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    Vz[v_ptr[rr] + s_idx] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    Vz[v_ptr[rr] + s_idx+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    Vz[v_ptr[rr+1] + s_idx] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    Vz[v_ptr[rr+1] + s_idx+1] += ph * vv

    nr = nr * 2
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_array_so(ints_st[::1] ptr,
                     ints_st[::1] ncol,
                     ints_st[::1] col,
                     # complexs_st requires only 4 indices...
                     floats_st[:, ::1] D,
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

    cdef complexs_st ph, vv
    cdef ints_st r, rr, ind, s, c
    cdef ints_st s_idx

    with nogil:
        if p_opt == 0:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2

                    ph = phases[ind, 0]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    vx[rr, c] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    vx[rr, c+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    vx[rr+1, c] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    vx[rr+1, c+1] += ph * vv

                    ph = phases[ind, 1]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    vy[rr, c] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    vy[rr, c+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    vy[rr+1, c] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    vy[rr+1, c+1] += ph * vv

                    ph = phases[ind, 2]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    vz[rr, c] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    vz[rr, c+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    vz[rr+1, c] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    vz[rr+1, c+1] += ph * vv

        else:
            for r in range(nr):
                rr = r * 2
                for ind in range(ptr[r], ptr[r] + ncol[r]):
                    c = (col[ind] % nr) * 2
                    s = col[ind] / nr

                    ph = phases[s, 0]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    vx[rr, c] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    vx[rr, c+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    vx[rr+1, c] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    vx[rr+1, c+1] += ph * vv

                    ph = phases[s, 1]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    vy[rr, c] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    vy[rr, c+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    vy[rr+1, c] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    vy[rr+1, c+1] += ph * vv

                    ph = phases[s, 2]
                    vv = (D[ind, 0] + 1j * D[ind, 4])
                    vz[rr, c] += ph * vv
                    vv = (D[ind, 2] + 1j * D[ind, 3])
                    vz[rr, c+1] += ph * vv
                    vv = (D[ind, 6] + 1j * D[ind, 7])
                    vz[rr+1, c] += ph * vv
                    vv = (D[ind, 1] + 1j * D[ind, 5])
                    vz[rr+1, c+1] += ph * vv

    return Vx, Vy, Vz
