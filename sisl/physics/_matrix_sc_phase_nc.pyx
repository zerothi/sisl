# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from sisl._sparse cimport inline_sum
from sisl.physics._matrix_utils cimport ncol2ptr_double


__all__ = ['_sc_phase_nc_csr_c64', '_sc_phase_nc_csr_c128',
           '_sc_phase_nc_array_c64', '_sc_phase_nc_array_c128']

# The fused data-types forces the data input to be of "correct" values.
ctypedef fused numeric_complex:
    float
    double
    float complex
    double complex


def _sc_phase_nc_csr_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                         np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                         np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                         numeric_complex[:, ::1] D,
                         const int nc,
                         np.ndarray[np.complex64_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[::1] phases = PHASES

    # Now copy the sparse matrix form
    cdef Py_ssize_t nr = ncol.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_PTR = np.empty([nr*2 + 1], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_NCOL = np.empty([nr*2], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_COL = np.empty([inline_sum(ncol)*4], dtype=np.int32)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef np.ndarray[np.complex64_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.complex64)
    cdef float complex[::1] v = V
    cdef float complex ph, v12
    cdef Py_ssize_t r, rr, ind, cind, c

    # We have to do it manually due to the double elements per matrix element
    ncol2ptr_double(nr, ncol, v_ptr)

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            v_ncol[rr] = ncol[r] * 2
            v_ncol[rr+1] = ncol[r] * 2

            cind = 0
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                ph = phases[ind]
                c = col[ind] * 2

                v[v_ptr[rr] + cind] = <float complex> (ph * D[ind, 0])
                v_col[v_ptr[rr] + cind] = c
                v12 = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                v[v_ptr[rr] + cind+1] = ph * v12
                v_col[v_ptr[rr] + cind+1] = c + 1
                v[v_ptr[rr+1] + cind] = ph * v12.conjugate()
                v_col[v_ptr[rr+1] + cind] = c
                v[v_ptr[rr+1] + cind+1] = <float complex> (ph * D[ind, 1])
                v_col[v_ptr[rr+1] + cind+1] = c + 1

                cind = cind + 2

    else:
        for r in range(nr):
            rr = r * 2
            v_ncol[rr] = ncol[r] * 2
            v_ncol[rr+1] = ncol[r] * 2

            cind = 0
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                ph = phases[col[ind] / nr]
                c = col[ind] * 2

                v[v_ptr[rr] + cind] = <float complex> (ph * D[ind, 0])
                v_col[v_ptr[rr] + cind] = c
                v12 = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                v[v_ptr[rr] + cind+1] = ph * v12
                v_col[v_ptr[rr] + cind+1] = c + 1
                v[v_ptr[rr+1] + cind] = ph * v12.conjugate()
                v_col[v_ptr[rr+1] + cind] = c
                v[v_ptr[rr+1] + cind+1] = <float complex> (ph * D[ind, 1])
                v_col[v_ptr[rr+1] + cind+1] = c + 1

                cind = cind + 2

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 2, nc * 2))


def _sc_phase_nc_csr_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                          np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                          np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                          numeric_complex[:, ::1] D,
                          const int nc,
                          np.ndarray[np.complex128_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[::1] phases = PHASES

    # Now copy the sparse matrix form
    cdef Py_ssize_t nr = ncol.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_PTR = np.empty([nr*2 + 1], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_NCOL = np.empty([nr*2], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] V_COL = np.empty([inline_sum(ncol)*4], dtype=np.int32)
    cdef int[::1] v_ptr = V_PTR
    cdef int[::1] v_ncol = V_NCOL
    cdef int[::1] v_col = V_COL

    cdef np.ndarray[np.complex128_t, ndim=1, mode='c'] V = np.zeros([v_col.shape[0]], dtype=np.complex128)
    cdef double complex[::1] v = V
    cdef double complex ph, v12
    cdef Py_ssize_t r, rr, ind, cind, c

    ncol2ptr_double(nr, ncol, v_ptr)

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            v_ncol[rr] = ncol[r] * 2
            v_ncol[rr+1] = ncol[r] * 2

            cind = 0
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                ph = phases[ind]
                c = col[ind] * 2

                v[v_ptr[rr] + cind] = v[v_ptr[rr] + cind] + <double complex> (ph * D[ind, 0])
                v_col[v_ptr[rr] + cind] = c
                v12 = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                v[v_ptr[rr] + cind+1] = ph * v12
                v_col[v_ptr[rr] + cind+1] = c + 1
                v[v_ptr[rr+1] + cind] = ph * v12.conjugate()
                v_col[v_ptr[rr+1] + cind] = c
                v[v_ptr[rr+1] + cind+1] = <double complex> (ph * D[ind, 1])
                v_col[v_ptr[rr+1] + cind+1] = c + 1

                cind = cind + 2

    else:
        for r in range(nr):
            rr = r * 2
            v_ncol[rr] = ncol[r] * 2
            v_ncol[rr+1] = ncol[r] * 2

            cind = 0
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                ph = phases[col[ind] / nr]
                c = col[ind] * 2

                v[v_ptr[rr] + cind] = v[v_ptr[rr] + cind] + <double complex> (ph * D[ind, 0])
                v_col[v_ptr[rr] + cind] = c
                v12 = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                v[v_ptr[rr] + cind+1] = ph * v12
                v_col[v_ptr[rr] + cind+1] = c + 1
                v[v_ptr[rr+1] + cind] = ph * v12.conjugate()
                v_col[v_ptr[rr+1] + cind] = c
                v[v_ptr[rr+1] + cind+1] = <double complex> (ph * D[ind, 1])
                v_col[v_ptr[rr+1] + cind+1] = c + 1
                cind = cind + 2

    return csr_matrix((V, V_COL, V_PTR), shape=(nr * 2, nc * 2))


def _sc_phase_nc_array_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                           np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                           np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                           numeric_complex[:, ::1] D,
                           const int nc,
                           np.ndarray[np.complex64_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[::1] phases = PHASES

    cdef Py_ssize_t nr = ncol.shape[0]
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] V = np.zeros([nr * 2, nc * 2], dtype=np.complex64)
    cdef float complex[:, ::1] v = V
    cdef float complex ph, v12
    cdef Py_ssize_t r, rr, ind, c

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                ph = phases[ind]
                c = col[ind] * 2
                v[rr, c] = <float complex> (ph * D[ind, 0])
                v12 = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                v[rr, c+1] = ph * v12
                v[rr+1, c] = ph * v12.conjugate()
                v[rr+1, c+1] = <float complex> (ph * D[ind, 1])

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                ph = phases[col[ind] / nr]
                c = col[ind] * 2
                v[rr, c] = <float complex> (ph * D[ind, 0])
                v12 = <float complex> (D[ind, 2] + 1j * D[ind, 3])
                v[rr, c+1] = ph * v12
                v[rr+1, c] = ph * v12.conjugate()
                v[rr+1, c+1] = <float complex> (ph * D[ind, 1])

    return V


def _sc_phase_nc_array_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                            np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                            np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                            numeric_complex[:, ::1] D,
                            const int nc,
                            np.ndarray[np.complex128_t, ndim=1, mode='c'] PHASES, const int p_opt):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[::1] phases = PHASES

    cdef Py_ssize_t nr = ncol.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] V = np.zeros([nr * 2, nc * 2], dtype=np.complex128)
    cdef double complex[:, ::1] v = V
    cdef double complex ph, v12
    cdef Py_ssize_t r, rr, ind, c

    if p_opt == 0:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                ph = phases[ind]
                c = col[ind] * 2
                v[rr, c] = <double complex> (ph * D[ind, 0])
                v12 = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                v[rr, c+1] = ph * v12
                v[rr+1, c] = ph * v12.conjugate()
                v[rr+1, c+1] = <double complex> (ph * D[ind, 1])

    else:
        for r in range(nr):
            rr = r * 2
            for ind in range(ptr[r], ptr[r] + ncol[r]):
                ph = phases[col[ind] / nr]
                c = col[ind] * 2
                v[rr, c] = <double complex> (ph * D[ind, 0])
                v12 = <double complex> (D[ind, 2] + 1j * D[ind, 3])
                v[rr, c+1] = ph * v12
                v[rr+1, c] = ph * v12.conjugate()
                v[rr+1, c+1] = <double complex> (ph * D[ind, 1])

    return V
