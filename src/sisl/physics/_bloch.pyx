# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython
from libc.math cimport cos, pi, sin

import numpy as np

cimport numpy as np

__all__ = ['bloch_unfold']


@cython.boundscheck(True)
@cython.wraparound(True)
@cython.initializedcheck(True)
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
        # the array should still be (:, :, :)
        return M[0]

    # handle different data-types
    if M.dtype == np.complex64:
        return _unfold64(B, (k * 2 * pi).reshape(B[2], B[1], B[0], 3), M)
    elif M.dtype == np.complex128:
        return _unfold128(B, (k * 2 * pi).reshape(B[2], B[1], B[0], 3), M)
    raise ValueError('bloch_unfold: requires dtype to be either complex64 or complex128.')


def _unfold64(const int[::1] B, const double[:, :, :, ::1] k2pi,
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
    cdef float complex[:, :, :, ::1] M4 = M.reshape(N, N1, N, N2)
    cdef float complex[:, ::1] M2 = M

    # Split calculations into single expansion (easy to abstract)
    # and full calculation (which is too heavy!)
    if B0 == 1:
        if B1 == 1: # only B2 == 1
            _unfold64_1(B2, k2pi[:, 0, 0, 2], N1, N2, m, M4)
        elif B2 == 1: # only B1 == 1
            _unfold64_1(B1, k2pi[0, :, 0, 1], N1, N2, m, M4)
        else:# only B0 == 1
            _unfold64_2(B1, k2pi[0, :, 0, 1], B2, k2pi[:, 0, 0, 2],
                         N1, N2, m, M4)
    elif B1 == 1:
        if B2 == 1:
            _unfold64_1(B0, k2pi[0, 0, :, 0], N1, N2, m, M4)
        else:# only B1 == 1
            _unfold64_2(B0, k2pi[0, 0, :, 0], B2, k2pi[:, 0, 0, 2],
                         N1, N2, m, M4)
    elif B2 == 1: # only B2 == 1
        _unfold64_2(B0, k2pi[0, 0, :, 0], B1, k2pi[0, :, 0, 1],
                     N1, N2, m, M4)
    else:
        _unfold64_3(B0, B1, B2, k2pi, N1, N2, m, M2)

    return M


cdef void _unfold64_matrix(const double w,
                           const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                           const double k0, const double k1, const double k2,
                           const Py_ssize_t N1, const Py_ssize_t N2,
                           const float complex[:, ::1] m,
                           float complex[:, ::1] M) noexcept nogil:
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
        aph0 = cos(k0) - 1j * sin(k0)
    if B1 > 1:
        aph1 = cos(k1) - 1j * sin(k1)
    if B2 > 1:
        aph2 = cos(k2) - 1j * sin(k2)

    J = 0
    for j2 in range(B2):
        for j1 in range(B1):
            for j0 in range(B0):
                rph = - j0 * k0 - j1 * k1 - j2 * k2
                ph = w * cos(rph) - 1j * (w * sin(rph))
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
                                    MJ[I] += mj[i] * <float complex> ph0
                                    I += 1
                                ph0 = ph0 * aph0
                            ph1 = ph1 * aph1
                        ph2 = ph2 * aph2
                    J += 1


cdef void _unfold64_3(const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                      const double[:, :, :, ::1] k2pi,
                      const Py_ssize_t N1, const Py_ssize_t N2,
                      const float complex[:, :, ::1] m,
                      float complex[:, ::1] M) noexcept nogil:

    # N should now equal K.shape[0]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef double k0, k1, k2
    cdef double w = 1. / N
    cdef Py_ssize_t T, A, B, C

    # Now perform expansion
    T = 0
    for C in range(B2):
        for B in range(B1):
            for A in range(B0):
                k0 = k2pi[C, B, A, 0]
                k1 = k2pi[C, B, A, 1]
                k2 = k2pi[C, B, A, 2]
                _unfold64_matrix(w, B0, B1, B2, k0, k1, k2, N1, N2, m[T], M)
                T = T + 1


cdef void _unfold64_1(const Py_ssize_t NA, const double[:] kA2pi,
                      const Py_ssize_t N1, const Py_ssize_t N2,
                      const float complex[:, :, ::1] m,
                      float complex[:, :, :, ::1] M) noexcept nogil:

    cdef double k, w
    cdef Py_ssize_t TA, iA
    cdef Py_ssize_t i, j
    cdef double complex ph, cph, pha
    cdef const float complex[:, ::1] mT

    # The algorithm for constructing the unfolded matrix can be done in the
    # following way:
    # 1. Fill the diagonal which corresponds to zero phases, i.e. M[:N1, :N2] = sum(m, 0) / w
    # 2. Construct the rest of the column M[:N1, N2:].
    # 3. Loop neighbouring columns and perform these steps:
    #    a) calculate the first N1 rows
    #    b) copy from the previous column-block the first N-1 blocks into 1:N

    # Dimension of final matrix
    w = 1. / NA

    for TA in range(NA):
        mT = m[TA]
        k = kA2pi[TA]

        # 1: construct M[0, :, 0, :]
        for j in range(N1):
            for i in range(N2):
                M[0, j, 0, i] = M[0, j, 0, i] + mT[j, i] * <float> w

        # Initial phases along the column
        pha = cos(k) - 1j * sin(k)
        ph = w * pha

        for iA in range(1, NA):
            # Conjugate to construct first columns
            cph = ph.conjugate()
            for j in range(N1):
                for i in range(N2):
                    # 2: construct M[0, :, 1:, :]
                    M[0, j, iA, i] += mT[j, i] * <float complex> ph
                for i in range(N2):
                    # 3a: construct M[1:, :, 0, :]
                    M[iA, j, 0, i] += mT[j, i] * <float complex> cph

            # Increment phases
            ph = ph * pha

    for TA in range(1, NA):
        for j in range(N1):
            for iA in range(1, NA):
                for i in range(N2):
                    M[TA, j, iA, i] = M[TA-1, j, iA-1, i]


cdef void _unfold64_2(const Py_ssize_t NA, const double[:] kA2pi,
                      const Py_ssize_t NB, const double[:] kB2pi,
                      const Py_ssize_t N1, const Py_ssize_t N2,
                      const float complex[:, :, ::1] m,
                      float complex[:, :, :, ::1] M):

    cdef double w, kA, kB
    cdef Py_ssize_t TA, iA, TB, iB
    cdef Py_ssize_t i, j
    cdef double complex ph, cph
    cdef double complex pha, pha_step, phb, phb_step

    cdef const float complex[:, ::1] mT

    # The algorithm for constructing the unfolded matrix can be done in the
    # following way:
    # 1. Fill the diagonal which corresponds to zero phases, i.e. M[:N1, :N2] = sum(m, 0) / w
    # 2. Construct the rest of the column M[:N1, N2:].
    # 3. Loop neighboring columns and perform these steps:
    #    a) calculate the first N1 rows
    #    b) copy from the previous column-block the first N-1 blocks into 1:N

    # Dimension of final matrix
    w = 1. / (NA * NB)
    for TB in range(NB):

        # Initial phases along the column
        kB = kB2pi[TB]
        phb_step = cos(kB) - 1j * sin(kB)

        for TA in range(NA):

            # Initial phases along the column
            kA = kA2pi[TA]
            pha_step = cos(kA) - 1j * sin(kA)

            mT = m[TB*NA+TA]

            for j in range(N1):
                for i in range(N2):
                    #(0,0,0,0) (C-index)
                    M[0, j, 0, i] += mT[j, i] * <float> w

            # Initial phases along the column
            pha = w * pha_step
            for iA in range(1, NA):

                # Conjugate to construct first columns
                ph = pha
                cph = pha.conjugate()
                for j in range(N1):
                    for i in range(N2):
                        #(0,0,0,iA)
                        M[0, j, iA, i] += mT[j, i] * <float complex> ph
                    for i in range(N2):
                        #(0,iA,0,0)
                        M[iA, j, 0, i] += mT[j, i] * <float complex> cph

                # Increment phases
                pha = pha * pha_step

            phb = w * phb_step
            for iB in range(1, NB):

                # Conjugate to construct first columns
                ph = phb
                cph = phb.conjugate()
                for j in range(N1):
                    for i in range(N2):
                        #(0,0,iB,0)
                        M[0, j, iB*NA, i] += mT[j, i] * <float complex>  ph
                    for i in range(N2):
                        #(iB,0,0,0)
                        M[iB*NA, j, 0, i] += mT[j, i] * <float complex> cph

                pha = pha_step
                for iA in range(1, NA):

                    ph = pha * phb
                    cph = pha.conjugate() * phb
                    for j in range(N1):
                        for i in range(N2):
                            #(0,0,iB,iA)
                            M[0, j, iB*NA+iA, i] += mT[j, i] * <float complex> ph
                        for i in range(N2):
                            #(0,iA,iB,0)
                            M[iA, j, iB*NA, i] += mT[j, i] * <float complex> cph

                    ph = pha * phb.conjugate()
                    cph = (pha * phb).conjugate()
                    for j in range(N1):
                        for i in range(N2):
                            #(iB,0,0,iA)
                            M[iB*NA, j, iA, i] += mT[j, i] * <float complex> ph
                        for i in range(N2):
                            #(iB,iA,0,0)
                            M[iB*NA+iA, j, 0, i] += mT[j, i] * <float complex> cph

                    # Increment phases
                    pha = pha * pha_step

                # Increment phases
                phb = phb * phb_step

    for TA in range(1, NA):
        for j in range(N1):
            for iA in range(1, NA):
                for i in range(N2):
                    M[TA, j, iA, i] = M[TA-1, j, iA-1, i]

    for iB in range(1, NB):
        for TA in range(1, NA):
            for j in range(N1):
                for iA in range(1, NA):
                    for i in range(N2):
                        M[TA, j, iB*NA+iA, i] = M[TA-1, j, iB*NA+iA-1, i]
                    for i in range(N2):
                        M[iB*NA+TA, j, iA, i] = M[iB*NA+TA-1, j, iA-1, i]

    for TB in range(1, NB):
        for TA in range(NA):
            for j in range(N1):
                for iB in range(1, NB):
                    for iA in range(NA):
                        for i in range(N2):
                            M[TB*NA+TA, j, iB*NA+iA, i] = M[(TB-1)*NA+TA, j, (iB-1)*NA+iA, i]


def _unfold128(const int[::1] B, const double[:, :, :, ::1] k2pi,
               const double complex[:, :, ::1] m):
    """ Main unfolding routine for a matrix `m`. """

    # N should now equal K.shape[0]
    cdef Py_ssize_t B0 = B[0]
    cdef Py_ssize_t B1 = B[1]
    cdef Py_ssize_t B2 = B[2]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef Py_ssize_t N1 = m.shape[1]
    cdef Py_ssize_t N2 = m.shape[2]
    cdef np.ndarray[np.complex128_t, ndim=2, mode='c'] M = np.zeros([N*N1, N*N2], dtype=np.complex128)
    cdef double complex[:, :, :, ::1] M4 = M.reshape(N, N1, N, N2)
    cdef double complex[:, ::1] M2 = M

    # Split calculations into single expansion (easy to abstract)
    # and full calculation (which is too heavy!)
    if B0 == 1:
        if B1 == 1: # only B2 == 1
            _unfold128_1(B2, k2pi[:, 0, 0, 2], N1, N2, m, M4)
        elif B2 == 1: # only B1 == 1
            _unfold128_1(B1, k2pi[0, :, 0, 1], N1, N2, m, M4)
        else:# only B0 == 1
            _unfold128_2(B1, k2pi[0, :, 0, 1], B2, k2pi[:, 0, 0, 2],
                         N1, N2, m, M4)
    elif B1 == 1:
        if B2 == 1:
            _unfold128_1(B0, k2pi[0, 0, :, 0], N1, N2, m, M4)
        else:# only B1 == 1
            _unfold128_2(B0, k2pi[0, 0, :, 0], B2, k2pi[:, 0, 0, 2],
                         N1, N2, m, M4)
    elif B2 == 1: # only B2 == 1
        _unfold128_2(B0, k2pi[0, 0, :, 0], B1, k2pi[0, :, 0, 1],
                     N1, N2, m, M4)
    else:
        _unfold128_3(B0, B1, B2, k2pi, N1, N2, m, M2)

    return M


cdef void _unfold128_matrix(const double w,
                            const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                            const double k0, const double k1, const double k2,
                            const Py_ssize_t N1, const Py_ssize_t N2,
                            const double complex[:, ::1] m,
                            double complex[:, ::1] M) noexcept nogil:
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
        aph0 = cos(k0) - 1j * sin(k0)
    if B1 > 1:
        aph1 = cos(k1) - 1j * sin(k1)
    if B2 > 1:
        aph2 = cos(k2) - 1j * sin(k2)

    J = 0
    for j2 in range(B2):
        for j1 in range(B1):
            for j0 in range(B0):
                rph = - j0 * k0 - j1 * k1 - j2 * k2
                ph = w * cos(rph) - 1j * (w * sin(rph))
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
                                    MJ[I] += mj[i] * ph0
                                    I += 1
                                ph0 = ph0 * aph0
                            ph1 = ph1 * aph1
                        ph2 = ph2 * aph2
                    J += 1


cdef void _unfold128_3(const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                       const double[:, :, :, ::1] k2pi,
                       const Py_ssize_t N1, const Py_ssize_t N2,
                       const double complex[:, :, ::1] m,
                       double complex[:, ::1] M):

    # N should now equal K.shape[0]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef double k0, k1, k2
    cdef double w = 1. / N
    cdef Py_ssize_t T, A, B, C

    # Now perform expansion
    T = 0
    for C in range(B2):
        for B in range(B1):
            for A in range(B0):
                k0 = k2pi[C, B, A, 0]
                k1 = k2pi[C, B, A, 1]
                k2 = k2pi[C, B, A, 2]
                _unfold128_matrix(w, B0, B1, B2, k0, k1, k2, N1, N2, m[T], M)
                T = T + 1


cdef void _unfold128_1(const Py_ssize_t NA, const double[:] kA2pi,
                       const Py_ssize_t N1, const Py_ssize_t N2,
                       const double complex[:, :, ::1] m,
                       double complex[:, :, :, ::1] M) noexcept nogil:

    cdef double k, w
    cdef Py_ssize_t TA, iA
    cdef Py_ssize_t i, j
    cdef double complex ph, cph, pha
    cdef const double complex[:, ::1] mT

    # The algorithm for constructing the unfolded matrix can be done in the
    # following way:
    # 1. Fill the diagonal which corresponds to zero phases, i.e. M[:N1, :N2] = sum(m, 0) / w
    # 2. Construct the rest of the column M[:N1, N2:].
    # 3. Loop neighbouring columns and perform these steps:
    #    a) calculate the first N1 rows
    #    b) copy from the previous column-block the first N-1 blocks into 1:N

    w = 1. / NA

    for TA in range(NA):
        mT = m[TA]
        k = kA2pi[TA]

        # 1: construct M[0, :, 0, :]
        for j in range(N1):
            for i in range(N2):
                M[0, j, 0, i] += mT[j, i] * w

        # Initial phases along the column
        pha = cos(k) - 1j * sin(k)
        ph = w * pha

        for iA in range(1, NA):
            # Conjugate to construct first columns
            cph = ph.conjugate()
            for j in range(N1):
                for i in range(N2):
                    # 2: construct M[0, :, 1:, :]
                    M[0, j, iA, i] += mT[j, i] * ph
                for i in range(N2):
                    # 3a: construct M[1:, :, 0, :]
                    M[iA, j, 0, i] += mT[j, i] * cph

            # Increment phases
            ph = ph * pha

    for TA in range(1, NA):
        for j in range(N1):
            for iA in range(1, NA):
                for i in range(N2):
                    M[TA, j, iA, i] = M[TA-1, j, iA-1, i]


cdef void _unfold128_2(const Py_ssize_t NA, const double[:] kA2pi,
                       const Py_ssize_t NB, const double[:] kB2pi,
                       const Py_ssize_t N1, const Py_ssize_t N2,
                       const double complex[:, :, ::1] m,
                       double complex[:, :, :, ::1] M):

    cdef double w, kA, kB
    cdef Py_ssize_t TA, iA, TB, iB
    cdef Py_ssize_t i, j
    cdef double complex ph, cph
    cdef double complex pha, pha_step, phb, phb_step

    cdef const double complex[:, ::1] mT

    # The algorithm for constructing the unfolded matrix can be done in the
    # following way:
    # 1. Fill the diagonal which corresponds to zero phases, i.e. M[:N1, :N2] = sum(m, 0) / w
    # 2. Construct the rest of the column M[:N1, N2:].
    # 3. Loop neighbouring columns and perform these steps:
    #    a) calculate the first N1 rows
    #    b) copy from the previous column-block the first N-1 blocks into 1:N

    # Dimension of final matrix
    w = 1. / (NA * NB)
    for TB in range(NB):

        # Initial phases along the column
        kB = kB2pi[TB]
        phb_step = cos(kB) - 1j * sin(kB)

        for TA in range(NA):

            # Initial phases along the column
            kA = kA2pi[TA]
            pha_step = cos(kA) - 1j * sin(kA)

            mT = m[TB*NA+TA]

            for j in range(N1):
                for i in range(N2):
                    #(0,0,0,0) (C-index)
                    M[0, j, 0, i] += mT[j, i] * w

            # Initial phases along the column
            pha = w * pha_step
            for iA in range(1, NA):

                # Conjugate to construct first columns
                ph = pha
                cph = pha.conjugate()
                for j in range(N1):
                    for i in range(N2):
                        #(0,0,0,iA)
                        M[0, j, iA, i] += mT[j, i] * ph
                    for i in range(N2):
                        #(0,iA,0,0)
                        M[iA, j, 0, i] += mT[j, i] * cph

                # Increment phases
                pha = pha * pha_step

            phb = w * phb_step
            for iB in range(1, NB):

                # Conjugate to construct first columns
                ph = phb
                cph = phb.conjugate()
                for j in range(N1):
                    for i in range(N2):
                        #(0,0,iB,0)
                        M[0, j, iB*NA, i] += mT[j, i] * ph
                    for i in range(N2):
                        #(iB,0,0,0)
                        M[iB*NA, j, 0, i] += mT[j, i] * cph

                pha = pha_step
                for iA in range(1, NA):

                    ph = pha * phb
                    cph = pha.conjugate() * phb
                    for j in range(N1):
                        for i in range(N2):
                            #(0,0,iB,iA)
                            M[0, j, iB*NA+iA, i] += mT[j, i] * ph
                        for i in range(N2):
                            #(0,iA,iB,0)
                            M[iA, j, iB*NA, i] += mT[j, i] * cph

                    ph = pha * phb.conjugate()
                    cph = (pha * phb).conjugate()
                    for j in range(N1):
                        for i in range(N2):
                            #(iB,0,0,iA)
                            M[iB*NA, j, iA, i] += mT[j, i] * ph
                        for i in range(N2):
                            #(iB,iA,0,0)
                            M[iB*NA+iA, j, 0, i] += mT[j, i] * cph

                    # Increment phases
                    pha = pha * pha_step

                # Increment phases
                phb = phb * phb_step

    for TA in range(1, NA):
        for j in range(N1):
            for iA in range(1, NA):
                for i in range(N2):
                    M[TA, j, iA, i] = M[TA-1, j, iA-1, i]

    for iB in range(1, NB):
        for TA in range(1, NA):
            for j in range(N1):
                for iA in range(1, NA):
                    for i in range(N2):
                        M[TA, j, iB*NA+iA, i] = M[TA-1, j, iB*NA+iA-1, i]
                    for i in range(N2):
                        M[iB*NA+TA, j, iA, i] = M[iB*NA+TA-1, j, iA-1, i]

    for TB in range(1, NB):
        for TA in range(NA):
            for j in range(N1):
                for iB in range(1, NB):
                    for iA in range(NA):
                        for i in range(N2):
                            M[TB*NA+TA, j, iB*NA+iA, i] = M[(TB-1)*NA+TA, j, (iB-1)*NA+iA, i]
