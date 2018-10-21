# Import libc functions
cimport cython
from libc.math cimport cos, sin, pi

import numpy as np
cimport numpy as np

__all__ = ['bloch_unfold']


def bloch_unfold(np.ndarray[np.int32_t, ndim=1, mode='c'] B, 
                 np.ndarray[np.float64_t, ndim=2, mode='c'] k,
                 np.ndarray M):
    """ Exposed unfolding method using the TILING method
    
    Parameters
    ----------
    B : [x, y, z]
      the number of unfolds per direction
    k : [product(B), 3]
      k-points where M has been evaluated at
    M : [B[2], B[1], B[0], :, :]
       matrix at given k-points
    dtype : dtype
       resulting unfolded matrix dtype
    """
    # Reshape M and check for layout
    if not M.flags.c_contiguous:
        raise ValueError('bloch_unfold: requires M to be C-contiguous.')

    # Quick return for all B == 1
    if B[0] == B[1] == B[2] == 1:
        return M

    if M.dtype == np.complex64:
        return _unfold64(B, k, M)
    elif M.dtype == np.complex128:
        return _unfold128(B, k, M)
    raise ValueError('bloch_unfold: requires dtype to be either complex64 or complex128.')


@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _unfold_M64(const double w,
                      const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                      const double k0, const double k1, const double k2,
                      const Py_ssize_t N1, const Py_ssize_t N2,
                      const float complex[:, ::1] m,
                      float complex[:, ::1] M) nogil:

    cdef Py_ssize_t j0, j1, j2 # looping the output rows
    cdef Py_ssize_t i, j # looping m[j, i]
    cdef Py_ssize_t I, J # looping output M[J, I]

    # Faster memory views
    cdef float complex[:] MJ, mj

    # Phase handling variables
    cdef double rph
    cdef float complex aph0, aph1, aph2
    cdef float complex ph, ph0, ph1, ph2

    # Construct the phases to be added
    aph0 = aph1 = aph2 = 1.
    if B0 > 1:
        aph0 = <float complex> (cos(k0) + 1j * sin(k0))
    if B1 > 1:
        aph1 = <float complex> (cos(k1) + 1j * sin(k1))
    if B2 > 1:
        aph2 = <float complex> (cos(k2) + 1j * sin(k2))

    J = 0
    for j2 in range(B2):
        for j1 in range(B1):
            for j0 in range(B0):
                rph = - j0 * k0 - j1 * k1 - j2 * k2
                ph = <float complex> ( w * cos(rph) + 1j * (w * sin(rph)) )
                for j in range(N1):
                    # Every column starts from scratch
                    ph2 = ph

                    mj = m[j]
                    MJ = M[J]

                    I = 0
                    for _ in range(B2):
                        ph1 = ph2
                        for _ in range(B1):
                            ph0 = ph1
                            for _ in range(B0):
                                for i in range(N2):
                                    MJ[I] = MJ[I] + mj[i] * ph0
                                    I += 1
                                ph0 = ph0 * aph0
                            ph1 = ph1 * aph1
                        ph2 = ph2 * aph2
                    J += 1
                    

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _unfold64(const int[::1] B, const double[:, ::1] K,
              const float complex[:, :, ::1] m):

    # N should now equal K.shape[0]
    cdef Py_ssize_t B0 = B[0]
    cdef Py_ssize_t B1 = B[1]
    cdef Py_ssize_t B2 = B[2]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef Py_ssize_t N1 = m.shape[1]
    cdef Py_ssize_t N2 = m.shape[2]
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] M = np.zeros([N * N1, N * N2], dtype=np.complex64)
    # Get view
    cdef float complex[:, ::1] MM = M

    cdef double pi2 = pi * 2
    cdef double k0, k1, k2

    cdef double w = 1. / N
    cdef Py_ssize_t I

    # Now perform expansion
    for I in range(N):
        k0 = K[I, 0] * pi2
        k1 = K[I, 1] * pi2
        k2 = K[I, 2] * pi2
        _unfold_M64(w, B0, B1, B2, k0, k1, k2, N1, N2, m[I], MM)

    return M


@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _unfold_M128(const double w,
                       const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                       const double k0, const double k1, const double k2,
                       const Py_ssize_t N1, const Py_ssize_t N2,
                       const double complex[:, ::1] m,
                       double complex[:, ::1] M) nogil:

    cdef Py_ssize_t j0, j1, j2 # looping the output rows
    cdef Py_ssize_t i, j # looping m[j, i]
    cdef Py_ssize_t I, J # looping output M[J, I]

    # Faster memory views
    cdef double complex[:] MJ, mj

    # Phase handling variables
    cdef double rph
    cdef double complex aph0, aph1, aph2
    cdef double complex ph, ph0, ph1, ph2

    # Construct the phases to be added
    aph0 = aph1 = aph2 = 1.
    if B0 > 1:
        aph0 = cos(k0) + 1j * sin(k0)
    if B1 > 1:
        aph1 = cos(k1) + 1j * sin(k1)
    if B2 > 1:
        aph2 = cos(k2) + 1j * sin(k2)

    J = 0
    for j2 in range(B2):
        for j1 in range(B1):
            for j0 in range(B0):
                rph = - j0 * k0 - j1 * k1 - j2 * k2
                ph = w * cos(rph) + 1j * (w * sin(rph))
                for j in range(N1):
                    # Every column starts from scratch
                    ph2 = ph

                    mj = m[j]
                    MJ = M[J]

                    I = 0
                    for _ in range(B2):
                        ph1 = ph2
                        for _ in range(B1):
                            ph0 = ph1
                            for _ in range(B0):
                                for i in range(N2):
                                    MJ[I] = MJ[I] + mj[i] * ph0
                                    I += 1
                                ph0 = ph0 * aph0
                            ph1 = ph1 * aph1
                        ph2 = ph2 * aph2
                    J += 1
                    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def _unfold128(const int[::1] B, const double[:, ::1] K,
               const double complex[:, :, ::1] m):

    # N should now equal K.shape[0]
    cdef Py_ssize_t B0 = B[0]
    cdef Py_ssize_t B1 = B[1]
    cdef Py_ssize_t B2 = B[2]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef Py_ssize_t N1 = m.shape[1]
    cdef Py_ssize_t N2 = m.shape[2]
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] M = np.zeros([N * N1, N * N2], dtype=np.complex128)
    # Get view
    cdef double complex[:, ::1] MM = M

    cdef double pi2 = pi * 2
    cdef double k0, k1, k2

    cdef double w = 1. / N
    cdef Py_ssize_t I

    # Now perform expansion
    for I in range(N):
        k0 = K[I, 0] * pi2
        k1 = K[I, 1] * pi2
        k2 = K[I, 2] * pi2
        _unfold_M128(w, B0, B1, B2, k0, k1, k2, N1, N2, m[I], MM)

    return M

