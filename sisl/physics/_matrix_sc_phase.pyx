# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from sisl._sparse cimport inline_sum

__all__ = ['_sc_phase_csr_c64', '_sc_phase_csr_c128',
           '_sc_phase_array_c64', '_sc_phase_array_c128']

# The fused data-types forces the data input to be of "correct" values.
ctypedef fused numeric_real:
    float
    double

ctypedef fused numeric_complex:
    float
    double
    float complex
    double complex


def _sc_phase_csr_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                      np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                      numeric_complex[:, ::1] D,
                      const int nc, const int idx,
                      np.ndarray[np.complex64_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[::1] phases = PHASES

    # Now copy the sparse matrix form
    cdef Py_ssize_t nr = ncol.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_PTR = np.empty([nr + 1], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_NCOL = np.empty([nr], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_COL = np.empty([inline_sum(ncol)], dtype=np.int32)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef float complex[::1] v = V
    cdef Py_ssize_t r, ind, cind

    # Copy ncol
    v_ncol[:] = ncol[:]

    cind = 0
    if p_opt == 0:
        for r in range(nr):
            v_ptr[r] = cind
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                v[cind] = <float complex> (phases[ind] * D[ind, idx])
                v_col[cind] = col[ind]
                cind = cind + 1
    else:
        for r in range(nr):
            v_ptr[r] = cind
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                v[cind] = <float complex> (phases[col[ind] / nr] * D[ind, idx])
                v_col[cind] = col[ind]
                cind = cind + 1
    v_ptr[nr] = cind

    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nc))


def _sc_phase_csr_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                       np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                       np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                       numeric_complex[:, ::1] D,
                       const int nc, const int idx,
                       np.ndarray[np.complex128_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[::1] phases = PHASES

    # Now copy the sparse matrix form
    cdef Py_ssize_t nr = ncol.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_PTR = np.empty([nr + 1], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_NCOL = np.empty([nr], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_COL = np.empty([inline_sum(ncol)], dtype=np.int32)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef double complex[::1] v = V
    cdef Py_ssize_t r, ind, cind

    # Copy ncol
    v_ncol[:] = ncol[:]

    cind = 0
    if p_opt == 0:
        for r in range(nr):
            v_ptr[r] = cind
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                v[cind] = <double complex> (phases[ind] * D[ind, idx])
                v_col[cind] = col[ind]
                cind = cind + 1
    else:
        for r in range(nr):
            v_ptr[r] = cind
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                v[cind] = <double complex> (phases[col[ind] / nr] * D[ind, idx])
                v_col[cind] = col[ind]
                cind = cind + 1
    v_ptr[nr] = cind

    return csr_matrix((V, V_COL, V_PTR), shape=(nr, nc))


def _sc_phase_array_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                        np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                        np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                        numeric_complex[:, ::1] D,
                        const int nc, const int idx,
                        np.ndarray[np.complex64_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[::1] phases = PHASES

    cdef Py_ssize_t nr = ncol.shape[0]
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] V = np.zeros([nr, nc], dtype=np.complex64)
    cdef float complex[:, ::1] v = V
    cdef Py_ssize_t r, ind

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                v[r, col[ind]] = <float complex> (phases[ind] * D[ind, idx])

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                v[r, col[ind]] = <float complex> (phases[col[ind] / nr] * D[ind, idx])

    return V


def _sc_phase_array_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                         np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                         np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                         numeric_complex[:, ::1] D,
                         const int nc, const int idx,
                         np.ndarray[np.complex128_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[::1] phases = PHASES

    cdef Py_ssize_t nr = ncol.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] V = np.zeros([nr, nc], dtype=np.complex128)
    cdef double complex[:, ::1] v = V
    cdef Py_ssize_t r, ind

    if p_opt == 0:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                v[r, col[ind]] = <double complex> (phases[ind] * D[ind, idx])

    else:
        for r in range(nr):
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                v[r, col[ind]] = <double complex> (phases[col[ind] / nr] * D[ind, idx])

    return V
