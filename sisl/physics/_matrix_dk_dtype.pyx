# Import libc functions
cimport cython

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from sisl._indices cimport index_sorted
from sisl._sparse import fold_csr_matrix

__all__ = ['_dk_R_csr_c64', '_dk_R_csr_c128',
           '_dk_R_array_c64', '_dk_R_array_c128'] 

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
def _dk_R_csr_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                  np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                  np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                  numeric_complex[:, ::1] D, const int idx,
                  np.ndarray[np.complex64_t, ndim=2, mode='c'] IRS,
                  np.ndarray[np.complex64_t, ndim=1, mode='c'] PHASES):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[:, ::1] iRs = IRS
    cdef float complex[::1] phases = PHASES

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
    cdef float complex phD
    cdef int r, ind, c, s, s_idx

    for r in range(nr):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            c = col[ind] % nr
            s = col[ind] / nr
            s_idx = index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
            phD = <float complex> (phases[s] * D[ind, idx])
            vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + phD * iRs[s, 0]
            vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + phD * iRs[s, 1]
            vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + phD * iRs[s, 2]

    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _dk_R_csr_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                   np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                   numeric_complex[:, ::1] D, const int idx,
                   np.ndarray[np.complex128_t, ndim=2, mode='c'] IRS,
                   np.ndarray[np.complex128_t, ndim=1, mode='c'] PHASES):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[:, ::1] iRs = IRS
    cdef double complex[::1] phases = PHASES

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
    cdef double complex phD
    cdef int r, ind, c, s, s_idx

    for r in range(nr):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            c = col[ind] % nr
            s = col[ind] / nr
            s_idx = index_sorted(v_col[v_ptr[r]:v_ptr[r] + v_ncol[r]], c)
            phD = <double complex> (phases[s] * D[ind, idx])
            vx[v_ptr[r] + s_idx] = vx[v_ptr[r] + s_idx] + phD * iRs[s, 0]
            vy[v_ptr[r] + s_idx] = vy[v_ptr[r] + s_idx] + phD * iRs[s, 1]
            vz[v_ptr[r] + s_idx] = vz[v_ptr[r] + s_idx] + phD * iRs[s, 2]

    return csr_matrix((Vx, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vy, V_COL, V_PTR), shape=(nr, nr)), csr_matrix((Vz, V_COL, V_PTR), shape=(nr, nr))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _dk_R_array_c64(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                    np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                    numeric_complex[:, ::1] D, const int idx,
                    np.ndarray[np.complex64_t, ndim=2, mode='c'] IRS,
                    np.ndarray[np.complex64_t, ndim=1, mode='c'] PHASES):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef float complex[:, ::1] iRs = IRS
    cdef float complex[::1] phases = PHASES

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] Vx = np.zeros([nr, nr], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] Vy = np.zeros([nr, nr], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] Vz = np.zeros([nr, nr], dtype=np.complex64)
    cdef float complex[:, ::1] vx = Vx
    cdef float complex[:, ::1] vy = Vy
    cdef float complex[:, ::1] vz = Vz
    cdef float complex phD

    cdef int r, ind, s, c

    for r in range(nr):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            c = col[ind] % nr
            s = col[ind] / nr
            phD = <float complex> (phases[s] * D[ind, idx])
            vx[r, c] = vx[r, c] + phD * iRs[s, 0]
            vy[r, c] = vy[r, c] + phD * iRs[s, 1]
            vz[r, c] = vz[r, c] + phD * iRs[s, 2]

    return Vx, Vy, Vz


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _dk_R_array_c128(np.ndarray[np.int32_t, ndim=1, mode='c'] PTR,
                     np.ndarray[np.int32_t, ndim=1, mode='c'] NCOL,
                     np.ndarray[np.int32_t, ndim=1, mode='c'] COL,
                     numeric_complex[:, ::1] D, const int idx,
                     np.ndarray[np.complex128_t, ndim=2, mode='c'] IRS,
                     np.ndarray[np.complex128_t, ndim=1, mode='c'] PHASES):

    # Convert to memory views
    cdef int[::1] ptr = PTR
    cdef int[::1] ncol = NCOL
    cdef int[::1] col = COL
    cdef double complex[:, ::1] iRs = IRS
    cdef double complex[::1] phases = PHASES

    cdef int nr = ncol.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] Vx = np.zeros([nr, nr], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] Vy = np.zeros([nr, nr], dtype=np.complex128)
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] Vz = np.zeros([nr, nr], dtype=np.complex128)
    cdef double complex[:, ::1] vx = Vx
    cdef double complex[:, ::1] vy = Vy
    cdef double complex[:, ::1] vz = Vz
    cdef double complex phD

    cdef int r, ind, s, c

    for r in range(nr):
        for ind in range(ptr[r], ptr[r] + ncol[r]):
            c = col[ind] % nr
            s = col[ind] / nr
            phD = <double complex> (phases[s] * D[ind, idx])
            vx[r, c] = vx[r, c] + phD * iRs[s, 0]
            vy[r, c] = vy[r, c] + phD * iRs[s, 1]
            vz[r, c] = vz[r, c] + phD * iRs[s, 2]

    return Vx, Vy, Vz

    
