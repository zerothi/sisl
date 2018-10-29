#!python
#cython: language_level=2
cimport cython

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from sisl._indices cimport _index_sorted
from sisl._sparse import fold_csr_matrix

__all__ = ['_csr_f32', '_csr_f64', '_phase_csr_c64', '_phase_csr_c128',
           '_array_f32', '_array_f64', '_phase_array_c64', '_phase_array_c128']

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
def _csr_f32(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
             np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
             np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
             numeric_real[:, ::1] D, const int idx):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef int nr = v_ncol.shape[0]
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.float32)
    cdef float[::1] v = V
    cdef int r, ind, c, s_idx

    for r in range(nr):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            c = col[ind] % nr
            s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
            v[v_ptr[r] + s_idx] += <float> D[ind, idx]

    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _csr_f64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
             np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
             np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
             numeric_real[:, ::1] D, const int idx):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef int nr = v_ncol.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.float64)
    cdef double[::1] v = V
    cdef int r, ind, c, s_idx

    for r in range(nr):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            c = col[ind] % nr
            s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
            v[v_ptr[r] + s_idx] += <double> D[ind, idx]

    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase_csr_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                   numeric_complex[:, ::1] D, const int idx,
                   np.ndarray[np.complex64_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[::1] phases = PHASES

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef int nr = v_ncol.shape[0]
    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef float complex[::1] v = V
    cdef int r, ind, c, s_idx

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                v[v_ptr[r] + s_idx] = v[v_ptr[r] + s_idx] + <float complex> (phases[ind] * D[ind, idx])
    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                v[v_ptr[r] + s_idx] = v[v_ptr[r] + s_idx] + <float complex> (phases[col[ind] / nr] * D[ind, idx])

    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase_csr_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                    numeric_complex[:, ::1] D, const int idx,
                    np.ndarray[np.complex128_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[::1] phases = PHASES

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_matrix(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef int nr = v_ncol.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef double complex[::1] v = V
    cdef int r, ind, c, s_idx

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                v[v_ptr[r] + s_idx] = v[v_ptr[r] + s_idx] + <double complex> (phases[ind] * D[ind, idx])
    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                s_idx = _index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
                v[v_ptr[r] + s_idx] = v[v_ptr[r] + s_idx] + <double complex> (phases[col[ind] / nr] * D[ind, idx])

    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _array_f32(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
               np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
               np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
               numeric_real[:, ::1] D, const int idx):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] V = np.zeros([nr, nr], dtype=np.float32)
    cdef float[:, ::1] v = V
    cdef int r, ind

    for r in range(nr):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            v[r, col[ind] % nr] += <float> D[ind, idx]

    return V


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _array_f64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
               np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
               np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
               numeric_real[:, ::1] D, const int idx):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] V = np.zeros([nr, nr], dtype=np.float64)
    cdef double[:, ::1] v = V
    cdef int r, ind

    for r in range(nr):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            v[r, col[ind] % nr] += <double> D[ind, idx]

    return V


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase_array_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                     np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                     np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                     numeric_complex[:, ::1] D, const int idx,
                     np.ndarray[np.complex64_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[::1] phases = PHASES

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] V = np.zeros([nr, nr], dtype=np.complex64)
    cdef float complex[:, ::1] v = V
    cdef int r, ind, c

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                v[r, c] = v[r, c] + <float complex> (phases[ind] * D[ind, idx])

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                v[r, c] = v[r, c] + <float complex> (phases[col[ind] / nr] * D[ind, idx])

    return V


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase_array_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                      numeric_complex[:, ::1] D, const int idx,
                      np.ndarray[np.complex128_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[::1] phases = PHASES

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] V = np.zeros([nr, nr], dtype=np.complex128)
    cdef double complex[:, ::1] v = V
    cdef int r, ind, c

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                v[r, c] = v[r, c] + <double complex> (phases[ind] * D[ind, idx])

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = col[ind] % nr
                v[r, c] = v[r, c] + <double complex> (phases[col[ind] / nr] * D[ind, idx])

    return V
