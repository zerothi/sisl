#!python
#cython: language_level=2
cimport cython

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from sisl._indices cimport _index_sorted
from sisl._sparse import fold_csr_matrix

__all__ = ['_phase3_csr_f32', '_phase3_csr_f64',
           '_phase3_csr_c64', '_phase3_csr_c128',
           '_phase3_array_f32', '_phase3_array_f64',
           '_phase3_array_c64', '_phase3_array_c128']

# The fused data-types forces the data input to be of "correct" values.
ctypedef fused numeric_real:
    int
    long
    float
    double

ctypedef fused numeric_complex:
    int
    long
    float
    double
    float complex
    double complex


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_csr_f32(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                    numeric_real[:, ::1] D, const int idx,
                    np.ndarray[np.float32_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float[:, ::1] phases = PHASES

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef int nr = v_ncol.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=np.float32)
    cdef float[::1] vx = Vx
    cdef float[::1] vy = Vy
    cdef float[::1] vz = Vz
    cdef float d
    cdef int r, ind, c, s, s_idx

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                d = <float> D[ind, idx]
                vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + d * phases[ind, 0]
                vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + d * phases[ind, 1]
                vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + d * phases[ind, 2]

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s = col[ind] / nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                d = <float> D[ind, idx]
                vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + d * phases[s, 0]
                vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + d * phases[s, 1]
                vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + d * phases[s, 2]

    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_csr_f64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                    numeric_real[:, ::1] D, const int idx,
                    np.ndarray[np.float64_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double[:, ::1] phases = PHASES

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef int nr = v_ncol.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=np.float64)
    cdef double[::1] vx = Vx
    cdef double[::1] vy = Vy
    cdef double[::1] vz = Vz
    cdef double d
    cdef int r, ind, c, s, s_idx

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                d = <double> D[ind, idx]
                vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + d * phases[ind, 0]
                vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + d * phases[ind, 1]
                vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + d * phases[ind, 2]

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s = col[ind] / nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                d = <double> D[ind, idx]
                vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + d * phases[s, 0]
                vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + d * phases[s, 1]
                vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + d * phases[s, 2]

    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_csr_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                    numeric_complex[:, ::1] D, const int idx,
                    np.ndarray[np.complex64_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[:, ::1] phases = PHASES

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef int nr = v_ncol.shape[0]
    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef float complex[::1] vx = Vx
    cdef float complex[::1] vy = Vy
    cdef float complex[::1] vz = Vz
    cdef float complex d
    cdef int r, ind, c, s, s_idx

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                d = <float complex> D[ind, idx]
                vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + d * phases[ind, 0]
                vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + d * phases[ind, 1]
                vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + d * phases[ind, 2]

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s = col[ind] / nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                d = <float complex> D[ind, idx]
                vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + d * phases[s, 0]
                vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + d * phases[s, 1]
                vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + d * phases[s, 2]

    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_csr_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                     np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                     np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                     numeric_complex[:, ::1] D, const int idx,
                     np.ndarray[np.complex128_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[:, ::1] phases = PHASES

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef int nr = v_ncol.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] Vx = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] Vy = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] Vz = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef double complex[::1] vx = Vx
    cdef double complex[::1] vy = Vy
    cdef double complex[::1] vz = Vz
    cdef double complex d
    cdef int r, ind, c, s, s_idx

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                d = <double complex> D[ind, idx]
                vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + d * phases[ind, 0]
                vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + d * phases[ind, 1]
                vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + d * phases[ind, 2]

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s = col[ind] / nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                d = <double complex> D[ind, idx]
                vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + d * phases[s, 0]
                vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + d * phases[s, 1]
                vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + d * phases[s, 2]

    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_array_f32(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                      numeric_real[:, ::1] D, const int idx,
                      np.ndarray[np.float32_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float[:, ::1] phases = PHASES

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] Vx = np.zeros([nr, nr], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] Vy = np.zeros([nr, nr], dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] Vz = np.zeros([nr, nr], dtype=np.float32)
    cdef float[:, ::1] vx = Vx
    cdef float[:, ::1] vy = Vy
    cdef float[:, ::1] vz = Vz
    cdef float d
    cdef int r, ind, s, c

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                d = <float> D[ind, idx]
                vx[r, c] = vx[r, c] + d * phases[ind, 0]
                vy[r, c] = vy[r, c] + d * phases[ind, 1]
                vz[r, c] = vz[r, c] + d * phases[ind, 2]

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s = col[ind] / nr
                d = <float> D[ind, idx]
                vx[r, c] = vx[r, c] + d * phases[s, 0]
                vy[r, c] = vy[r, c] + d * phases[s, 1]
                vz[r, c] = vz[r, c] + d * phases[s, 2]

    return Vx, Vy, Vz


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_array_f64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                      numeric_real[:, ::1] D, const int idx,
                      np.ndarray[np.float64_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double[:, ::1] phases = PHASES

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Vx = np.zeros([nr, nr], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Vy = np.zeros([nr, nr], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Vz = np.zeros([nr, nr], dtype=np.float64)
    cdef double[:, ::1] vx = Vx
    cdef double[:, ::1] vy = Vy
    cdef double[:, ::1] vz = Vz
    cdef double d
    cdef int r, ind, s, c

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                d = <double> D[ind, idx]
                vx[r, c] = vx[r, c] + d * phases[ind, 0]
                vy[r, c] = vy[r, c] + d * phases[ind, 1]
                vz[r, c] = vz[r, c] + d * phases[ind, 2]

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s = col[ind] / nr
                d = <double> D[ind, idx]
                vx[r, c] = vx[r, c] + d * phases[s, 0]
                vy[r, c] = vy[r, c] + d * phases[s, 1]
                vz[r, c] = vz[r, c] + d * phases[s, 2]

    return Vx, Vy, Vz


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_array_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                      numeric_complex[:, ::1] D, const int idx,
                      np.ndarray[np.complex64_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[:, ::1] phases = PHASES

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] Vx = np.zeros([nr, nr], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] Vy = np.zeros([nr, nr], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] Vz = np.zeros([nr, nr], dtype=np.complex64)
    cdef float complex[:, ::1] vx = Vx
    cdef float complex[:, ::1] vy = Vy
    cdef float complex[:, ::1] vz = Vz
    cdef float complex d

    cdef int r, ind, s, c

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                d = <float complex> D[ind, idx]
                vx[r, c] = vx[r, c] + d * phases[ind, 0]
                vy[r, c] = vy[r, c] + d * phases[ind, 1]
                vz[r, c] = vz[r, c] + d * phases[ind, 2]

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s = col[ind] / nr
                d = <float complex> D[ind, idx]
                vx[r, c] = vx[r, c] + d * phases[s, 0]
                vy[r, c] = vy[r, c] + d * phases[s, 1]
                vz[r, c] = vz[r, c] + d * phases[s, 2]

    return Vx, Vy, Vz


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase3_array_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                       np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                       np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                       numeric_complex[:, ::1] D, const int idx,
                       np.ndarray[np.complex128_t, ndim=2, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[:, ::1] phases = PHASES

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] Vx = np.zeros([nr, nr], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] Vy = np.zeros([nr, nr], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] Vz = np.zeros([nr, nr], dtype=np.complex128)
    cdef double complex[:, ::1] vx = Vx
    cdef double complex[:, ::1] vy = Vy
    cdef double complex[:, ::1] vz = Vz
    cdef double complex d

    cdef int r, ind, s, c

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                d = <double complex> D[ind, idx]
                vx[r, c] = vx[r, c] + d * phases[ind, 0]
                vy[r, c] = vy[r, c] + d * phases[ind, 1]
                vz[r, c] = vz[r, c] + d * phases[ind, 2]

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s = col[ind] / nr
                d = <double complex> D[ind, idx]
                vx[r, c] = vx[r, c] + d * phases[s, 0]
                vy[r, c] = vy[r, c] + d * phases[s, 1]
                vz[r, c] = vz[r, c] + d * phases[s, 2]

    return Vx, Vy, Vz
