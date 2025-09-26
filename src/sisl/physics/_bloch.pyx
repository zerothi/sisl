# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython
from libc.math cimport cos, pi, sin

from time import time

import numpy as np

cimport numpy as np

from sisl._core._dtypes cimport complexs_st, type2dtype

__all__ = ['bloch_unfold', 'bloch_unfold_finalize']


@cython.boundscheck(True)
@cython.initializedcheck(True)
def bloch_unfold(np.ndarray[np.int32_t, ndim=1, mode='c'] B,
                 np.ndarray[np.float64_t, ndim=2, mode='c'] k,
                 np.ndarray[complexs_st, ndim=3, mode='c'] m,
                 const bint finalize):
    """ Exposed unfolding method using the TILING method

    Parameters
    ----------
    B : [x, y, z]
      the number of unfolds per direction
    k : [product(B), 3]
      k-points where M has been evaluated at
    m : [B[2], B[1], B[0], :, :]
       matrix at given k-points
    finalize :
        whether the matrix should be finalized

    Returns
    -------
    the Bloch-expanded matrix from `m`.
    """
    # Quick return for all B == 1
    if B[0] == B[1] == B[2] == 1:
        # the array should still be (:, :, :)
        return m[0]

    # Reshape M and check for layout
    if not m.flags.c_contiguous:
        raise ValueError('bloch_unfold: requires m to be C-contiguous.')

    # handle different data-types
    if m.dtype not in (np.complex64, np.complex128):
        raise ValueError('bloch_unfold: requires dtype to be either complex64 or complex128.')

    cdef:
        int[::1] b = B
        Py_ssize_t n0 = m.shape[1]
        Py_ssize_t n1 = m.shape[2]

        # Allocate the output matrix
        complexs_st[:, ::1] M = _unfold_allocate(b, n0, n1, m)

    _unfold(b, (k * (2 * pi)).reshape(B[2], B[1], B[0], 3), m, M)
    if finalize:
        _unfold_copy(b, M)
    return M.base


@cython.boundscheck(True)
@cython.initializedcheck(True)
def bloch_unfold_finalize(np.ndarray[np.int32_t, ndim=1, mode='c'] B,
                          np.ndarray[complexs_st, ndim=2] M):
    """Copy out the Toeplitz solved matrix elements.

    Parameters
    ----------
    B : [x, y, z]
      the number of unfolds per direction
    M : [:, :]
       full matrix only with the Toeplitz matrix elements.

    Returns
    -------
    the Bloch-expanded matrix from `m`.
    """
    if B[0] == B[1] == B[2] == 1:
        return M

    # handle different data-types
    if M.dtype not in (np.complex64, np.complex128):
        raise ValueError('bloch_unfold_finalize: requires dtype to be either complex64 or complex128.')

    cdef:
        int[::1] b = B
        complexs_st[:, ::1] MM

    # Get the correct view
    # For both C and F contiguous we can use the same algorithm. A transpose
    # Still works for the Toeplitz structure.
    if M.flags.c_contiguous:
        MM = M
    else:
        MM = M.T

    _unfold_copy(b, MM)
    return M


cdef complexs_st[:, ::1] _unfold_allocate(const int[::1] B,
                                          const Py_ssize_t n0,
                                          const Py_ssize_t n1,
                                          # unused m argument (only for deciding dtype)
                                          np.ndarray[complexs_st, ndim=3, mode='c'] m
                                          ):
    cdef Py_ssize_t N = B[0] * B[1] * B[2]
    cdef object dtype = type2dtype[complexs_st](1)
    cdef np.ndarray[complexs_st, ndim=2, mode='c'] M = np.zeros([N * n0, N * n1],
                                                                dtype=dtype)
    cdef complexs_st[:, ::1] MM = M
    return MM


@cython.initializedcheck(False)
def _unfold(const int[::1] B, const double[:, :, :, ::1] k2pi,
            const complexs_st[:, :, ::1] m,
            complexs_st[:, ::1] M,
            ):
    """ Main unfolding routine for a matrix `m`. """

    # N should now equal K.shape[0]
    cdef Py_ssize_t B0 = B[0]
    cdef Py_ssize_t B1 = B[1]
    cdef Py_ssize_t B2 = B[2]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef Py_ssize_t n0 = m.shape[1]
    cdef Py_ssize_t n1 = m.shape[2]
    cdef complexs_st[:, :, :, ::1] M4 = <complexs_st[:N, :n0, :N, :n1]> &M[0, 0]

    # Apparently, if we don't have a temporary, it fails... :(
    # When slicing a mem-view, it seems to create a copy, i.e. k2pi0.base is k2pi
    # fails... :(
    cdef const double[:] k2pi0, k2pi1, k2pi2

    # Split calculations into single expansion (easy to abstract)
    # and full calculation (which is too heavy!)
    if B0 == 1:
        if B1 == 1: # only B2 == 1
            k2pi2 = k2pi[:, 0, 0, 2]
            _unfold_1(B2, k2pi2, n0, n1, m, M4)
        elif B2 == 1: # only B1 == 1
            k2pi1 = k2pi[0, :, 0, 1]
            _unfold_1(B1, k2pi1, n0, n1, m, M4)
        else:# only B0 == 1
            k2pi1 = k2pi[0, :, 0, 1]
            k2pi2 = k2pi[:, 0, 0, 2]
            _unfold_2(B1, k2pi1, B2, k2pi2, n0, n1, m, M4)

    elif B1 == 1:
        if B2 == 1:
            k2pi0 = k2pi[0, 0, :, 0]
            _unfold_1(B0, k2pi0, n0, n1, m, M4)
        else:# only B1 == 1
            k2pi0 = k2pi[0, 0, :, 0]
            k2pi2 = k2pi[:, 0, 0, 2]
            _unfold_2(B0, k2pi0, B2, k2pi2, n0, n1, m, M4)

    elif B2 == 1: # only B2 == 1
        k2pi0 = k2pi[0, 0, :, 0]
        k2pi1 = k2pi[0, :, 0, 1]
        _unfold_2(B0, k2pi0, B1, k2pi1, n0, n1, m, M4)

    else:
        _unfold_3(B0, B1, B2, k2pi, n0, n1, m, M)


cdef void _unfold_matrix(const double w,
                         const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                         const double k0, const double k1, const double k2,
                         const Py_ssize_t n0, const Py_ssize_t n1,
                         const complexs_st[:, ::1] m,
                         complexs_st[:, ::1] M) noexcept nogil:
    """ Unfold matrix `m` into `M` """

    cdef Py_ssize_t j0, j1, j2 # looping the output rows
    cdef Py_ssize_t i, j # looping m[j, i]
    cdef Py_ssize_t I, J # looping output M[J, I]

    # Faster memory views
    cdef complexs_st[::1] MJ
    cdef const complexs_st[::1] mj

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
                for j in range(n0):
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
                                for i in range(n1):
                                    MJ[I] += mj[i] * ph0
                                    I += 1
                                ph0 = ph0 * aph0
                            ph1 = ph1 * aph1
                        ph2 = ph2 * aph2
                    J += 1


cdef void _unfold_3(const Py_ssize_t B0, const Py_ssize_t B1, const Py_ssize_t B2,
                    const double[:, :, :, ::1] k2pi,
                    const Py_ssize_t n0, const Py_ssize_t n1,
                    const complexs_st[:, :, ::1] m,
                    complexs_st[:, ::1] M) noexcept nogil:

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
                _unfold_matrix(w, B0, B1, B2, k0, k1, k2, n0, n1, m[T], M)
                T = T + 1


cdef void _unfold_1(const Py_ssize_t NA, const double[:] kA2pi,
                    const Py_ssize_t n0, const Py_ssize_t n1,
                    const complexs_st[:, :, ::1] m,
                    complexs_st[:, :, :, ::1] M) noexcept nogil:

    cdef double k, w
    cdef Py_ssize_t TA, iA
    cdef Py_ssize_t i, j
    cdef double complex ph, cph, pha
    cdef const complexs_st[:, ::1] mT

    # The algorithm for constructing the unfolded matrix can be done in the
    # following way:
    # 1. Fill the diagonal which corresponds to zero phases, i.e. M[:n0, :n1] = sum(m, 0) / w
    # 2. Construct the rest of the column M[:n0, n1:].
    # 3. Loop neighboring columns and perform these steps:
    #    a) calculate the first n0 rows
    #    b) copy from the previous column-block the first N-1 blocks into 1:N

    # Dimension of final matrix
    w = 1. / NA

    for TA in range(NA):
        mT = m[TA]
        k = kA2pi[TA]

        # 1: construct M[0, :, 0, :]
        for j in range(n0):
            for i in range(n1):
                M[0, j, 0, i] += mT[j, i] * w

        # Initial phases along the column
        pha = cos(k) - 1j * sin(k)
        ph = w * pha

        for iA in range(1, NA):
            # Conjugate to construct first columns
            cph = ph.conjugate()
            for j in range(n0):
                for i in range(n1):
                    # 2: construct M[0, :, 1:, :]
                    M[0, j, iA, i] += mT[j, i] * ph
                for i in range(n1):
                    # 3a: construct M[1:, :, 0, :]
                    M[iA, j, 0, i] += mT[j, i] * cph

            # Increment phases
            ph = ph * pha


cdef void _unfold_2(const Py_ssize_t NA, const double[:] kA2pi,
                    const Py_ssize_t NB, const double[:] kB2pi,
                    const Py_ssize_t n0, const Py_ssize_t n1,
                    const complexs_st[:, :, ::1] m,
                    complexs_st[:, :, :, ::1] M) noexcept nogil:

    cdef double w, kA, kB
    cdef Py_ssize_t TA, iA, TB, iB
    cdef Py_ssize_t i, j
    cdef double complex ph, cph
    cdef double complex pha, pha_step, phb, phb_step

    cdef const complexs_st[:, ::1] mT

    # The algorithm for constructing the unfolded matrix can be done in the
    # following way:
    # 1. Fill the diagonal which corresponds to zero phases, i.e. M[:n0, :n1] = sum(m, 0) / w
    # 2. Construct the rest of the column M[:n0, n1:].
    # 3. Loop neighboring columns and perform these steps:
    #    a) calculate the first n0 rows
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

            for j in range(n0):
                for i in range(n1):
                    #(0,0,0,0) (C-index)
                    M[0, j, 0, i] += mT[j, i] * w

            # Initial phases along the column
            pha = w * pha_step
            for iA in range(1, NA):

                # Conjugate to construct first columns
                ph = pha
                cph = pha.conjugate()
                for j in range(n0):
                    for i in range(n1):
                        #(0,0,0,iA)
                        M[0, j, iA, i] += mT[j, i] * ph
                    for i in range(n1):
                        #(0,iA,0,0)
                        M[iA, j, 0, i] += mT[j, i] * cph

                # Increment phases
                pha = pha * pha_step

            phb = w * phb_step
            for iB in range(1, NB):

                # Conjugate to construct first columns
                ph = phb
                cph = phb.conjugate()
                for j in range(n0):
                    for i in range(n1):
                        #(0,0,iB,0)
                        M[0, j, iB*NA, i] += mT[j, i] * ph
                    for i in range(n1):
                        #(iB,0,0,0)
                        M[iB*NA, j, 0, i] += mT[j, i] * cph

                pha = pha_step
                for iA in range(1, NA):

                    ph = pha * phb
                    cph = pha.conjugate() * phb
                    for j in range(n0):
                        for i in range(n1):
                            #(0,0,iB,iA)
                            M[0, j, iB*NA+iA, i] += mT[j, i] * ph
                        for i in range(n1):
                            #(0,iA,iB,0)
                            M[iA, j, iB*NA, i] += mT[j, i] * cph

                    ph = pha * phb.conjugate()
                    cph = (pha * phb).conjugate()
                    for j in range(n0):
                        for i in range(n1):
                            #(iB,0,0,iA)
                            M[iB*NA, j, iA, i] += mT[j, i] * ph
                        for i in range(n1):
                            #(iB,iA,0,0)
                            M[iB*NA+iA, j, 0, i] += mT[j, i] * cph

                    # Increment phases
                    pha = pha * pha_step

                # Increment phases
                phb = phb * phb_step


cdef void _unfold_copy(const int[::1] B, complexs_st[:, ::1] M):
    """ Finalize an unfolding by copying data around for the Toeplitz matrix `m`. """

    # N should now equal K.shape[0]
    cdef Py_ssize_t B0 = B[0]
    cdef Py_ssize_t B1 = B[1]
    cdef Py_ssize_t B2 = B[2]
    cdef Py_ssize_t N = B0 * B1 * B2
    cdef Py_ssize_t n0 = M.shape[0] // N
    cdef Py_ssize_t n1 = M.shape[1] // N
    cdef complexs_st[:, :, :, ::1] M4 = <complexs_st[:N, :n0, :N, :n1]> &M[0, 0]

    # Split calculations into single expansion (easy to abstract)
    # and full calculation (which is too heavy!)
    if B0 == 1:
        if B1 == 1: # only B2 == 1
            _unfold_copy_1(B2, n0, n1, M4)
        elif B2 == 1: # only B1 == 1
            _unfold_copy_1(B1, n0, n1, M4)
        else:# only B0 == 1
            _unfold_copy_2(B1, B2, n0, n1, M4)

    elif B1 == 1:
        if B2 == 1:
            _unfold_copy_1(B0, n0, n1, M4)
        else:# only B1 == 1
            _unfold_copy_2(B0, B2, n0, n1, M4)


    elif B2 == 1: # only B2 == 1
        _unfold_copy_2(B0, B1, n0, n1, M4)

    else:
        _unfold_copy_3(B0, B1, B2, n0, n1, M4)


cdef void _unfold_copy_block(const Py_ssize_t N0, const Py_ssize_t N1,
                             const Py_ssize_t n0, const Py_ssize_t n1,
                             const Py_ssize_t NA,
                             complexs_st *M) noexcept nogil:
    """Unfold a sub-block (the inner-most) one into its own region.

    This will expand the equivalent of a 1D Bloch-unfolding
    but at the memory position of `*M`.

    Parameters
    ----------
    N0, N1 :
        dimensions of the full matrix
    n0, n1 :
        dimensions of the Bloch-expanded matrix (i.e. the small one)
    NA :
        the number of sub-blocks it contains. I.e. the full expanded
        1D will be ``(n0 * NA, n1 * NA)``.
    """
    cdef Py_ssize_t rA, cA, r, c
    cdef Py_ssize_t n0N1 = n0 * N1
    cdef Py_ssize_t from_r, to_r, from_c, to_c

    # The matrix is in C-order (row-major).

    for rA in range(1, NA):
        to_r = rA * n0N1
        from_r = to_r - n0N1
        for r in range(n0):
            r = r * N1
            for cA in range(1, NA):
                to_c = cA * n1
                from_c = to_c - n1
                for c in range(n1):
                    M[to_r+r + to_c+c] = M[from_r+r + from_c+c]


cdef void _unfold_copy_1(const Py_ssize_t NA,
                         const Py_ssize_t n0, const Py_ssize_t n1,
                         complexs_st[:, :, :, ::1] M) noexcept nogil:
    _unfold_copy_block(NA * n0, NA * n1, n0, n1, NA, &M[0, 0, 0, 0])


cdef void _unfold_copy_2(const Py_ssize_t NA, const Py_ssize_t NB,
                         const Py_ssize_t n0, const Py_ssize_t n1,
                         complexs_st[:, :, :, ::1] M) noexcept nogil:
    # Loop through the B-segments we wish to copy in, and let
    # copy_block handle the rest

    cdef Py_ssize_t iB
    cdef Py_ssize_t N0 = NA * NB * n0
    cdef Py_ssize_t N1 = NA * NB * n1

    # the [0; 0] blocks
    _unfold_copy_block(N0, N1, n0, n1, NA, &M[0, 0, 0, 0])
    for iB in range(1, NB):
        # the [iB; 0] blocks
        _unfold_copy_block(N0, N1, n0, n1, NA, &M[iB * NA, 0, 0, 0])
        # the [0; iB] blocks
        _unfold_copy_block(N0, N1, n0, n1, NA, &M[0, 0, iB * NA, 0])

    # Now we have copied all necessary A blocks. Now we can treat it
    # as though it's only 1D bloch.
    _unfold_copy_block(N0, N1, NA*n0, NA*n1, NB, &M[0, 0, 0, 0])


cdef void _unfold_copy_3(const Py_ssize_t NA, const Py_ssize_t NB, const Py_ssize_t NC,
                         const Py_ssize_t n0, const Py_ssize_t n1,
                         complexs_st[:, :, :, ::1] M) noexcept nogil:
    # Loop through the B-segments we wish to copy in, and let
    # copy_block handle the rest

    cdef Py_ssize_t iB, iC
    cdef Py_ssize_t NAB = NA * NB
    cdef Py_ssize_t N0 = NA * NB * NC * n0
    cdef Py_ssize_t N1 = NA * NB * NC * n1

    # Start by copying the initial BxA block.
    # This is equivalent to doing it for iC == 0

    # (same as unfold_copy_2 but with correct sizes)
    # the [0; 0] blocks
    _unfold_copy_block(N0, N1, n0, n1, NA, &M[0, 0, 0, 0])
    for iB in range(1, NB):
        # the [0, iB; 0, 0] blocks
        _unfold_copy_block(N0, N1, n0, n1, NA, &M[iB * NA, 0, 0, 0])
        # the [0, 0; 0, iB] blocks
        _unfold_copy_block(N0, N1, n0, n1, NA, &M[0, 0, iB * NA, 0])

    _unfold_copy_block(N0, N1, NA*n0, NA*n1, NB, &M[0, 0, 0, 0])

    for iC in range(1, NC):
        # process [iC, 0; 0, 0] blocks
        _unfold_copy_block(N0, N1, n0, n1, NA, &M[iC * NAB, 0, 0, 0])
        for iB in range(1, NB):
            # the [0, iB; 0, 0] blocks
            _unfold_copy_block(N0, N1, n0, n1, NA, &M[iC * NAB + iB * NA, 0, 0, 0])
            # the [0, 0; 0, iB] blocks
            _unfold_copy_block(N0, N1, n0, n1, NA, &M[iC * NAB, 0, iB * NA, 0])

        _unfold_copy_block(N0, N1, NA*n0, NA*n1, NB, &M[iC * NAB, 0, 0, 0])

        # process [0, 0; iC, 0] blocks
        _unfold_copy_block(N0, N1, n0, n1, NA, &M[0, 0, iC * NAB, 0])

        for iB in range(1, NB):
            # the [0, iB; 0, 0] blocks
            _unfold_copy_block(N0, N1, n0, n1, NA, &M[iB * NA, 0, iC * NAB, 0])
            # the [0, 0; 0, iB] blocks
            _unfold_copy_block(N0, N1, n0, n1, NA, &M[0, 0, iC * NAB + iB * NA, 0])

        _unfold_copy_block(N0, N1, NA*n0, NA*n1, NB, &M[0, 0, iC * NAB, 0])

    # Now we have filled all C blocks, copy the full thing
    _unfold_copy_block(N0, N1, NAB*n0, NAB*n1, NC, &M[0, 0, 0, 0])
