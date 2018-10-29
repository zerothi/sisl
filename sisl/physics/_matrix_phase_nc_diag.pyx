#!python
#cython: language_level=2
cimport cython

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from sisl._indices cimport _index_sorted
from sisl._sparse import fold_csr_diagonal_nc

__all__ = ['_phase_nc_diag_csr_c64', '_phase_nc_diag_csr_c128',
           '_phase_nc_diag_array_c64', '_phase_nc_diag_array_c128']

# The fused data-types forces the data input to be of "correct" values.
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
def _phase_nc_diag_csr_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
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

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_diagonal_nc(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef float complex[::1] v = V
    cdef float complex vv
    cdef int r, rr, ind, c, s_idx

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)
                vv = <float complex> (phases[ind] * D[ind, idx])
                v[v_ptr[rr] + s_idx] = v[v_ptr[rr] + s_idx] + vv
                v[v_ptr[rr+1] + s_idx] = v[v_ptr[rr+1] + s_idx] + vv

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)
                vv = <float complex> (phases[col[ind] / nr] * D[ind, idx])
                v[v_ptr[rr] + s_idx] = v[v_ptr[rr] + s_idx] + vv
                v[v_ptr[rr+1] + s_idx] = v[v_ptr[rr+1] + s_idx] + vv

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 2, nr * 2))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase_nc_diag_csr_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
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

    # Now create the folded sparse elements
    V_PTR, V_NCOL, V_COL = fold_csr_diagonal_nc(PTR, NCOL, COL)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef double complex[::1] v = V
    cdef double complex vv
    cdef int r, rr, ind, c, s_idx

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)
                vv = <double complex> (phases[ind] * D[ind, idx])
                v[v_ptr[rr] + s_idx] = v[v_ptr[rr] + s_idx] + vv
                v[v_ptr[rr+1] + s_idx] = v[v_ptr[rr+1] + s_idx] + vv

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                s_idx = _index_sorted(v_col[v_ptr[rr]:v_ptr[rr] + v_ncol[rr]], c)
                vv = <double complex> (phases[col[ind] / nr] * D[ind, idx])
                v[v_ptr[rr] + s_idx] = v[v_ptr[rr] + s_idx] + vv
                v[v_ptr[rr+1] + s_idx] = v[v_ptr[rr+1] + s_idx] + vv

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 2, nr * 2))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase_nc_diag_array_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
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
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] V = np.zeros([nr * 2, nr * 2], dtype=np.complex64)
    cdef float complex[:, ::1] v = V
    cdef float complex vv
    cdef int r, rr, ind, c

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                vv = <float complex> (phases[ind] * D[ind, idx])
                v[rr, c] = v[rr, c] + vv
                v[rr+1, c+1] = v[rr+1, c+1] + vv

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                vv = <float complex> (phases[col[ind] / nr] * D[ind, idx])
                v[rr, c] = v[rr, c] + vv
                v[rr+1, c+1] = v[rr+1, c+1] + vv

    return V


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _phase_nc_diag_array_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
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
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] V = np.zeros([nr * 2, nr * 2], dtype=np.complex128)
    cdef double complex[:, ::1] v = V
    cdef double complex vv
    cdef int r, rr, ind, c

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                vv = <double complex> (phases[ind] * D[ind, idx])
                v[rr, c] = v[rr, c] + vv
                v[rr+1, c+1] = v[rr+1, c+1] + vv

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                c = (col[ind] % nr) * 2
                vv = <double complex> (phases[col[ind] / nr] * D[ind, idx])
                v[rr, c] = v[rr, c] + vv
                v[rr+1, c+1] = v[rr+1, c+1] + vv

    return V
