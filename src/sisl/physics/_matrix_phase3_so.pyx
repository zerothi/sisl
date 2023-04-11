# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from sisl._indices cimport _index_sorted
from sisl._sparse import fold_csr_matrix_nc

__all__ = ["_phase3_so_csr_c64", "_phase3_so_csr_c128",
           "_phase3_so_array_c64", "_phase3_so_array_c128"]

# The fused data-types forces the data input to be of "correct" values.
ctypedef fused numeric_complex:
    float
    double
    float complex
    double complex


def _phase3_so_csr_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                       np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                       np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                       numeric_complex[:, ::1] D,
                       np.ndarray[np.complex64_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[:, ::1] phases = PHASES
    # Local columns (not in NC form)
    cdef Py_ssize_t nr = ncol.shape[0]

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix_nc(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef float complex[::1] vx = Vx
    cdef float complex[::1] vy = Vy
    cdef float complex[::1] vz = Vz
    cdef float complex ph, vv
    cdef Py_ssize_t r, rr, ind, s, s_idx
    cdef int c

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                ph = phases[ind, 0]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vx[v_ptr[rr] + s_idx] = vx[v_ptr[rr] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vx[v_ptr[rr] + s_idx+1] = vx[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vx[v_ptr[rr+1] + s_idx] = vx[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vx[v_ptr[rr+1] + s_idx+1] = vx[v_ptr[rr+1] + s_idx+1] + ph * vv

                ph = phases[ind, 1]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vy[v_ptr[rr] + s_idx] = vy[v_ptr[rr] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vy[v_ptr[rr] + s_idx+1] = vy[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vy[v_ptr[rr+1] + s_idx] = vy[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vy[v_ptr[rr+1] + s_idx+1] = vy[v_ptr[rr+1] + s_idx+1] + ph * vv

                ph = phases[ind, 2]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vz[v_ptr[rr] + s_idx] = vz[v_ptr[rr] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vz[v_ptr[rr] + s_idx+1] = vz[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vz[v_ptr[rr+1] + s_idx] = vz[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vz[v_ptr[rr+1] + s_idx+1] = vz[v_ptr[rr+1] + s_idx+1] + ph * vv

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s = col[ind] / nr

                s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                ph = phases[s, 0]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vx[v_ptr[rr] + s_idx] = vx[v_ptr[rr] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vx[v_ptr[rr] + s_idx+1] = vx[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vx[v_ptr[rr+1] + s_idx] = vx[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vx[v_ptr[rr+1] + s_idx+1] = vx[v_ptr[rr+1] + s_idx+1] + ph * vv

                ph = phases[s, 1]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vy[v_ptr[rr] + s_idx] = vy[v_ptr[rr] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vy[v_ptr[rr] + s_idx+1] = vy[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vy[v_ptr[rr+1] + s_idx] = vy[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vy[v_ptr[rr+1] + s_idx+1] = vy[v_ptr[rr+1] + s_idx+1] + ph * vv

                ph = phases[s, 2]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vz[v_ptr[rr] + s_idx] = vz[v_ptr[rr] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vz[v_ptr[rr] + s_idx+1] = vz[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vz[v_ptr[rr+1] + s_idx] = vz[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vz[v_ptr[rr+1] + s_idx+1] = vz[v_ptr[rr+1] + s_idx+1] + ph * vv

    nr = nr * 2
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


def _phase3_so_csr_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                        np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                        np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                        numeric_complex[:, ::1] D,
                        np.ndarray[np.complex128_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[:, ::1] phases = PHASES
    # Local columns (not in NC form)
    cdef Py_ssize_t nr = ncol.shape[0]

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix_nc(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef double complex[::1] vx = Vx
    cdef double complex[::1] vy = Vy
    cdef double complex[::1] vz = Vz
    cdef double complex ph, vv
    cdef Py_ssize_t r, rr, ind, s, s_idx
    cdef int c

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                ph = phases[ind, 0]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vx[v_ptr[rr] + s_idx] = vx[v_ptr[rr] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vx[v_ptr[rr] + s_idx+1] = vx[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vx[v_ptr[rr+1] + s_idx] = vx[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vx[v_ptr[rr+1] + s_idx+1] = vx[v_ptr[rr+1] + s_idx+1] + ph * vv

                ph = phases[ind, 1]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vy[v_ptr[rr] + s_idx] = vy[v_ptr[rr] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vy[v_ptr[rr] + s_idx+1] = vy[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vy[v_ptr[rr+1] + s_idx] = vy[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vy[v_ptr[rr+1] + s_idx+1] = vy[v_ptr[rr+1] + s_idx+1] + ph * vv

                ph = phases[ind, 2]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vz[v_ptr[rr] + s_idx] = vz[v_ptr[rr] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vz[v_ptr[rr] + s_idx+1] = vz[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vz[v_ptr[rr+1] + s_idx] = vz[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vz[v_ptr[rr+1] + s_idx+1] = vz[v_ptr[rr+1] + s_idx+1] + ph * vv

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s = col[ind] / nr

                s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)

                ph = phases[s, 0]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vx[v_ptr[rr] + s_idx] = vx[v_ptr[rr] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vx[v_ptr[rr] + s_idx+1] = vx[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vx[v_ptr[rr+1] + s_idx] = vx[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vx[v_ptr[rr+1] + s_idx+1] = vx[v_ptr[rr+1] + s_idx+1] + ph * vv

                ph = phases[s, 1]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vy[v_ptr[rr] + s_idx] = vy[v_ptr[rr] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vy[v_ptr[rr] + s_idx+1] = vy[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vy[v_ptr[rr+1] + s_idx] = vy[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vy[v_ptr[rr+1] + s_idx+1] = vy[v_ptr[rr+1] + s_idx+1] + ph * vv

                ph = phases[s, 2]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vz[v_ptr[rr] + s_idx] = vz[v_ptr[rr] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vz[v_ptr[rr] + s_idx+1] = vz[v_ptr[rr] + s_idx+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vz[v_ptr[rr+1] + s_idx] = vz[v_ptr[rr+1] + s_idx] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vz[v_ptr[rr+1] + s_idx+1] = vz[v_ptr[rr+1] + s_idx+1] + ph * vv

    nr = nr * 2
    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


def _phase3_so_array_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                         np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                         np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                         numeric_complex[:, ::1] D,
                         np.ndarray[np.complex64_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[:, ::1] phases = PHASES
    cdef Py_ssize_t nr = ncol.shape[0]

    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] Vx = np.zeros([nr * 2, nr * 2], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] Vy = np.zeros([nr * 2, nr * 2], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] Vz = np.zeros([nr * 2, nr * 2], dtype=np.complex64)
    cdef float complex[:, ::1] vx = Vx
    cdef float complex[:, ::1] vy = Vy
    cdef float complex[:, ::1] vz = Vz
    cdef float complex ph, v12
    cdef Py_ssize_t r, rr, ind, s, c

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2

                ph = phases[ind, 0]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vx[rr, c] = vx[rr, c] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vx[rr, c+1] = vx[rr, c+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vx[rr+1, c] = vx[rr+1, c] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vx[rr+1, c+1] = vx[rr+1, c+1] + ph * vv

                ph = phases[ind, 1]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vy[rr, c] = vy[rr, c] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vy[rr, c+1] = vy[rr, c+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vy[rr+1, c] = vy[rr+1, c] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vy[rr+1, c+1] = vy[rr+1, c+1] + ph * vv

                ph = phases[ind, 2]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vz[rr, c] = vz[rr, c] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vz[rr, c+1] = vz[rr, c+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vz[rr+1, c] = vz[rr+1, c] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vz[rr+1, c+1] = vz[rr+1, c+1] + ph * vv

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s = col[ind] / nr

                ph = phases[s, 0]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vx[rr, c] = vx[rr, c] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vx[rr, c+1] = vx[rr, c+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vx[rr+1, c] = vx[rr+1, c] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vx[rr+1, c+1] = vx[rr+1, c+1] + ph * vv

                ph = phases[s, 1]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vy[rr, c] = vy[rr, c] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vy[rr, c+1] = vy[rr, c+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vy[rr+1, c] = vy[rr+1, c] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vy[rr+1, c+1] = vy[rr+1, c+1] + ph * vv

                ph = phases[s, 2]
                vv = <float complex> (D[ind, 0] + 1j * D[ind, 4])
                vz[rr, c] = vz[rr, c] + ph * vv
                vv = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                vz[rr, c+1] = vz[rr, c+1] + ph * vv
                vv = <float complex> (D[ind, 6] + 1j * D[ind, 7])
                vz[rr+1, c] = vz[rr+1, c] + ph * vv
                vv = <float complex> (D[ind, 1] + 1j * D[ind, 5])
                vz[rr+1, c+1] = vz[rr+1, c+1] + ph * vv

    return Vx, Vy, Vz


def _phase3_so_array_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                          np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                          np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                          numeric_complex[:, ::1] D,
                          np.ndarray[np.complex128_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[:, ::1] phases = PHASES
    cdef Py_ssize_t nr = ncol.shape[0]

    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] Vx = np.zeros([nr * 2, nr * 2], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] Vy = np.zeros([nr * 2, nr * 2], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] Vz = np.zeros([nr * 2, nr * 2], dtype=np.complex128)
    cdef double complex[:, ::1] vx = Vx
    cdef double complex[:, ::1] vy = Vy
    cdef double complex[:, ::1] vz = Vz
    cdef double complex ph, vv
    cdef Py_ssize_t r, rr, ind, s, c

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2

                ph = phases[ind, 0]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vx[rr, c] = vx[rr, c] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vx[rr, c+1] = vx[rr, c+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vx[rr+1, c] = vx[rr+1, c] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vx[rr+1, c+1] = vx[rr+1, c+1] + ph * vv

                ph = phases[ind, 1]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vy[rr, c] = vy[rr, c] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vy[rr, c+1] = vy[rr, c+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vy[rr+1, c] = vy[rr+1, c] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vy[rr+1, c+1] = vy[rr+1, c+1] + ph * vv

                ph = phases[ind, 2]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vz[rr, c] = vz[rr, c] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vz[rr, c+1] = vz[rr, c+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vz[rr+1, c] = vz[rr+1, c] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vz[rr+1, c+1] = vz[rr+1, c+1] + ph * vv

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s = col[ind] / nr

                ph = phases[s, 0]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vx[rr, c] = vx[rr, c] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vx[rr, c+1] = vx[rr, c+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vx[rr+1, c] = vx[rr+1, c] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vx[rr+1, c+1] = vx[rr+1, c+1] + ph * vv

                ph = phases[s, 1]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vy[rr, c] = vy[rr, c] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vy[rr, c+1] = vy[rr, c+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vy[rr+1, c] = vy[rr+1, c] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vy[rr+1, c+1] = vy[rr+1, c+1] + ph * vv

                ph = phases[s, 2]
                vv = <double complex> (D[ind, 0] + 1j * D[ind, 4])
                vz[rr, c] = vz[rr, c] + ph * vv
                vv = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                vz[rr, c+1] = vz[rr, c+1] + ph * vv
                vv = <double complex> (D[ind, 6] + 1j * D[ind, 7])
                vz[rr+1, c] = vz[rr+1, c] + ph * vv
                vv = <double complex> (D[ind, 1] + 1j * D[ind, 5])
                vz[rr+1, c+1] = vz[rr+1, c+1] + ph * vv

    return Vx, Vy, Vz
