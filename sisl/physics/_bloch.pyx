#!python
#cython: language_level=2
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
        return _unfold64(B, k * 2 * pi, M)
    elif M.dtype == np.complex128:
        return _unfold128(B, k * 2 * pi, M)
    raise ValueError('bloch_unfold: requires dtype to be either complex64 or complex128.')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _unfold64(const int[::1] B, const double[:, ::1] K2pi,
              const float complex[:, :, ::1] m):
    """ Main unfolding routine for a matrix `m`. """

    # N should now equal K.shape[0]
    cdef Py_ssize_t B0 = B[0]
    cdef Py_ssize_t B1 = B[1]
    cdef Py_ssize_t B2 = B[2]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef Py_ssize_t N1 = m.shape[1]
    cdef Py_ssize_t N2 = m.shape[2]
    cdef np.ndarray[np.complex64_t, ndim=2, mode='c'] M = np.zeros([N * N1, N * N2], dtype=np.complex64)
    cdef float complex[:, ::1] MM = M

    # Split calculations into single expansion (easy to abstract)
    # and full calculation (which is too heavy!)
    if B0 == B1 == 1:
        _unfold64_single(B2, K2pi[:, 2], N1, N2, m, MM)
    elif B0 == B2 == 1:
        _unfold64_single(B1, K2pi[:, 1], N1, N2, m, MM)
    elif B1 == B2 == 1:
        _unfold64_single(B0, K2pi[:, 0], N1, N2, m, MM)
    else:
        _unfold64_3(B0, B1, B2, K2pi, N1, N2, m, MM)

    return M


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _unfold64_matrix(const double w,
                           const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                           const double k0, const double k1, const double k2,
                           const Py_ssize_t N1, const Py_ssize_t N2,
                           const float complex[:, ::1] m,
                           float complex[:, ::1] M) nogil:
    """ Unfold matrix `m` into `M` """

    cdef Py_ssize_t j0, j1, j2 # looping the output rows
    cdef Py_ssize_t i, j # looping m[j, i]
    cdef Py_ssize_t I, J # looping output M[J, I]

    # Faster memory views
    cdef float complex[::1] MJ
    cdef const float complex[::1] mj

    # Phase handling variables (still in double precision because we accummulate)
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

                    # Retrieve sub-arrays that we are to write too
                    mj = m[j]
                    MJ = M[J]

                    I = 0
                    for _ in range(B2):
                        ph1 = ph2
                        for _ in range(B1):
                            ph0 = ph1
                            for _ in range(B0):
                                for i in range(N2):
                                    MJ[I] = MJ[I] + mj[i] * <float complex> ph0
                                    I += 1
                                ph0 = ph0 * aph0
                            ph1 = ph1 * aph1
                        ph2 = ph2 * aph2
                    J += 1


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _unfold64_3(const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                      const double[:, ::1] K2pi,
                      const Py_ssize_t N1, const Py_ssize_t N2,
                      const float complex[:, :, ::1] m,
                      float complex[:, ::1] M) nogil:

    # N should now equal K.shape[0]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef double k0, k1, k2
    cdef double w = 1. / N
    cdef Py_ssize_t T

    # Now perform expansion
    for T in range(N):
        k0 = K2pi[T, 0]
        k1 = K2pi[T, 1]
        k2 = K2pi[T, 2]
        _unfold64_matrix(w, B0, B1, B2, k0, k1, k2, N1, N2, m[T], M)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _unfold64_single(const Py_ssize_t N, const double[:] K2pi,
                           const Py_ssize_t N1, const Py_ssize_t N2,
                           const float complex[:, :, ::1] m,
                           float complex[:, ::1] M) nogil:

    cdef double k, w
    cdef Py_ssize_t T, NN2, c
    cdef Py_ssize_t i, j, I, J, Jj
    cdef double complex ph, phc, aph
    cdef const float complex[:, ::1] mT

    # The algorithm for constructing the unfolded matrix can be done in the
    # following way:
    # 1. Fill the diagonal which corresponds to zero phases, i.e. M[:N1, :N2] = sum(m, 0) / w
    # 2. Construct the rest of the column M[:N1, N2:].
    # 3. Loop neighbouring columns and perform these steps:
    #    a) calculate the first N1 rows
    #    b) copy from the previous column-block the first N-1 blocks into 1:N

    # Dimension of final matrix
    NN2 = N * N2
    w = 1. / N

    for T in range(N):
        mT = m[T]
        k = K2pi[T]

        # 1: construct M[0, :, 0, :]
        for j in range(N1):
            for i in range(N2):
                M[j, i] = M[j, i] + mT[j, i] * <float> w

        # Initial phases along the column
        aph = cos(k) + 1j * sin(k)
        ph = w * aph

        for c in range(1, N):
            J = c * N1
            I = c * N2

            # Conjugate to construct first columns
            phc = ph.conjugate()
            for j in range(N1):
                Jj = J + j
                for i in range(N2):

                    # 2: construct M[0, :, 1:, :]
                    M[j, I+i] = M[j, I+i] + mT[j, i] * <float complex> ph

                    # 3a: construct M[1:, :, 0, :]
                    M[Jj, i] = M[Jj, i] + mT[j, i] * <float complex> phc

            # Increment phases
            ph = ph * aph

    for c in range(1, N):
        J = c * N1

        # 3b: copy all the previously calculated segments
        for j in range(J, J + N1):
            I = j - N1
            for i in range(N2, NN2):
                M[j, i] = M[I, i-N2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def _unfold128(const int[::1] B, const double[:, ::1] K2pi,
               const double complex[:, :, ::1] m):
    """ Main unfolding routine for a matrix `m`. """

    # N should now equal K.shape[0]
    cdef Py_ssize_t B0 = B[0]
    cdef Py_ssize_t B1 = B[1]
    cdef Py_ssize_t B2 = B[2]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef Py_ssize_t N1 = m.shape[1]
    cdef Py_ssize_t N2 = m.shape[2]
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] M = np.zeros([N * N1, N * N2], dtype=np.complex128)
    cdef double complex[:, ::1] MM = M

    # Split calculations into single expansion (easy to abstract)
    # and full calculation (which is too heavy!)
    if B0 == B1 == 1:
        _unfold128_single(B2, K2pi[:, 2], N1, N2, m, MM)
    elif B0 == B2 == 1:
        _unfold128_single(B1, K2pi[:, 1], N1, N2, m, MM)
    elif B1 == B2 == 1:
        _unfold128_single(B0, K2pi[:, 0], N1, N2, m, MM)
    else:
        _unfold128_3(B0, B1, B2, K2pi, N1, N2, m, MM)

    return M


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _unfold128_matrix(const double w,
                            const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                            const double k0, const double k1, const double k2,
                            const Py_ssize_t N1, const Py_ssize_t N2,
                            const double complex[:, ::1] m,
                            double complex[:, ::1] M) nogil:
    """ Unfold matrix `m` into `M` """

    cdef Py_ssize_t j0, j1, j2 # looping the output rows
    cdef Py_ssize_t i, j # looping m[j, i]
    cdef Py_ssize_t I, J # looping output M[J, I]

    # Faster memory views
    cdef double complex[::1] MJ
    cdef const double complex[::1] mj

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

                    # Retrieve sub-arrays that we are to write too
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _unfold128_3(const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                       const double[:, ::1] K2pi,
                       const Py_ssize_t N1, const Py_ssize_t N2,
                       const double complex[:, :, ::1] m,
                       double complex[:, ::1] M) nogil:

    # N should now equal K.shape[0]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef double k0, k1, k2
    cdef double w = 1. / N
    cdef Py_ssize_t T

    # Now perform expansion
    for T in range(N):
        k0 = K2pi[T, 0]
        k1 = K2pi[T, 1]
        k2 = K2pi[T, 2]
        _unfold128_matrix(w, B0, B1, B2, k0, k1, k2, N1, N2, m[T], M)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _unfold128_single(const Py_ssize_t N, const double[:] K2pi,
                            const Py_ssize_t N1, const Py_ssize_t N2,
                            const double complex[:, :, ::1] m,
                            double complex[:, ::1] M) nogil:

    cdef double k, w
    cdef Py_ssize_t T, NN2, c
    cdef Py_ssize_t i, j, I, J, Jj
    cdef double complex ph, phc, aph
    cdef const double complex[:, ::1] mT

    # The algorithm for constructing the unfolded matrix can be done in the
    # following way:
    # 1. Fill the diagonal which corresponds to zero phases, i.e. M[:N1, :N2] = sum(m, 0) / w
    # 2. Construct the rest of the column M[:N1, N2:].
    # 3. Loop neighbouring columns and perform these steps:
    #    a) calculate the first N1 rows
    #    b) copy from the previous column-block the first N-1 blocks into 1:N

    # Dimension of final matrix
    NN2 = N * N2
    w = 1. / N

    for T in range(N):
        mT = m[T]
        k = K2pi[T]

        # 1: construct M[0, :, 0, :]
        for j in range(N1):
            for i in range(N2):
                M[j, i] = M[j, i] + mT[j, i] * w

        # Initial phases along the column
        aph = cos(k) + 1j * sin(k)
        ph = w * aph

        for c in range(1, N):
            J = c * N1
            I = c * N2

            # Conjugate to construct first columns
            phc = ph.conjugate()
            for j in range(N1):
                Jj = J + j
                for i in range(N2):

                    # 2: construct M[0, :, 1:, :]
                    M[j, I+i] = M[j, I+i] + mT[j, i] * ph

                    # 3a: construct M[1:, :, 0, :]
                    M[Jj, i] = M[Jj, i] + mT[j, i] * phc

            # Increment phases
            ph = ph * aph

    for c in range(1, N):
        J = c * N1

        # 3b: copy all the previously calculated segments
        for j in range(J, J + N1):
            I = j - N1
            for i in range(N2, NN2):
                M[j, i] = M[I, i-N2]
