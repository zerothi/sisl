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
    cdef:
        Py_ssize_t N = B[0] * B[1] * B[2]
        object dtype = type2dtype[complexs_st](1)
        np.ndarray[complexs_st, ndim=2, mode='c'] M = np.zeros([N * n0, N * n1],
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
    cdef:
        Py_ssize_t B0 = B[0]
        Py_ssize_t B1 = B[1]
        Py_ssize_t B2 = B[2]
        Py_ssize_t N = B0 * B1 * B2
        Py_ssize_t n0 = m.shape[1]
        Py_ssize_t n1 = m.shape[2]
        complexs_st[:, :, :, ::1] M4 = <complexs_st[:N, :n0, :N, :n1]> &M[0, 0]
        # Apparently, if we don't have a temporary, it fails... :(
        # When slicing a mem-view, it seems to create a copy, i.e. k2pi0.base is k2pi
        # fails... :(
        const double[:] k2pi0, k2pi1, k2pi2

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
        k2pi0 = k2pi[0, 0, :, 0]
        k2pi1 = k2pi[0, :, 0, 1]
        k2pi2 = k2pi[:, 0, 0, 2]
        _unfold_3(B0, k2pi0, B1, k2pi1, B2, k2pi2, n0, n1, m, M4)


cdef void _unfold_block(
        const Py_ssize_t N0, const Py_ssize_t N1,
        const Py_ssize_t NA,
        const double[:] k2pi,
        const double complex wph,
        const Py_ssize_t n0, const Py_ssize_t n1,
        const complexs_st[:, :, ::1] m,
        complexs_st *M) noexcept nogil:

    cdef:
        double k
        Py_ssize_t TA, iA
        Py_ssize_t i, j
        Py_ssize_t to_c, to_r, off_r, off_c
        double complex ph, cph, pha, pha_step
        const complexs_st[:, ::1] mT

    # The algorithm for constructing the unfolded matrix can be done in the
    # following way:
    # 1. Fill the diagonal which corresponds to zero phases, i.e. M[:n0, :n1] = sum(m, 0) / w
    # 2. Construct the rest of the column M[:n0, n1:].
    # 3. Loop neighboring columns and perform these steps:
    #    a) calculate the first n0 rows
    #    b) copy from the previous column-block the first N-1 blocks into 1:N

    for TA in range(NA):
        mT = m[TA]

        # 1: construct M[0, :, 0, :]
        for j in range(n0):
            to_r = j*N1
            for i in range(n1):
                M[to_r + i] += mT[j, i] * wph

        # Initial phases along the column
        k = k2pi[TA]
        pha_step = cos(k) - 1j * sin(k)
        pha = pha_step

        for iA in range(1, NA):
            off_r = iA * n0 * N1
            off_c = iA * n1

            # Conjugate to construct first columns
            ph = wph * pha
            cph = wph * pha.conjugate()
            for j in range(n0):
                to_r = j*N1
                for i in range(n1):
                    # 2: construct M[0, :, 1:, :]
                    M[to_r + off_c + i] += mT[j, i] * ph
                for i in range(n1):
                    # 3a: construct M[1:, :, 0, :]
                    M[off_r + to_r + i] += mT[j, i] * cph

            # Increment phases
            pha = pha * pha_step


cdef void _unfold_1(const Py_ssize_t NA, const double[:] kA2pi,
                    const Py_ssize_t n0, const Py_ssize_t n1,
                    const complexs_st[:, :, ::1] m,
                    complexs_st[:, :, :, ::1] M) noexcept nogil:

    cdef:
        double complex w = 1. / NA
        Py_ssize_t N0 = NA * n0
        Py_ssize_t N1 = NA * n1

    _unfold_block(N0, N1, NA, kA2pi, w, n0, n1, m, &M[0, 0, 0, 0])


cdef void _unfold_2(const Py_ssize_t NA, const double[:] kA2pi,
                    const Py_ssize_t NB, const double[:] kB2pi,
                    const Py_ssize_t n0, const Py_ssize_t n1,
                    const complexs_st[:, :, ::1] m,
                    complexs_st[:, :, :, ::1] M) noexcept nogil:

    cdef:
        double w, kB
        double complex phb_step, phb
        const complexs_st[:, :, ::1] mT
        Py_ssize_t TB, iB
        Py_ssize_t N0 = NA * NB * n0
        Py_ssize_t N1 = NA * NB * n1

    w = 1. / (NA * NB)
    for TB in range(NB):
        # Get sub-matrix for the A-parts
        mT = m[TB * NA:]

        # First do iB == 0
        phb = w
        _unfold_block(N0, N1, NA, kA2pi, phb, n0, n1, mT, &M[0, 0, 0, 0])

        # Calculate the B-phases
        kB = kB2pi[TB]
        # Step-size in the Bloch-expansion phases.
        phb_step = cos(kB) - 1j * sin(kB)

        phb = w * phb_step
        for iB in range(1, NB):

            #(0,0,iB,0)
            _unfold_block(N0, N1, NA, kA2pi, phb, n0, n1, mT, &M[0, 0, iB*NA, 0])
            #(iB,0,0,0)
            _unfold_block(N0, N1, NA, kA2pi, phb.conjugate(), n0, n1, mT, &M[iB*NA, 0, 0, 0])
            phb = phb * phb_step


cdef void _unfold_3(const Py_ssize_t NA, const double[:] kA2pi,
                    const Py_ssize_t NB, const double[:] kB2pi,
                    const Py_ssize_t NC, const double[:] kC2pi,
                    const Py_ssize_t n0, const Py_ssize_t n1,
                    const complexs_st[:, :, ::1] m,
                    complexs_st[:, :, :, ::1] M) noexcept nogil:

    cdef:
        double w, kB, kC
        double complex phc_step, phc, cphc
        double complex phb_step, phb
        const complexs_st[:, :, ::1] mT
        Py_ssize_t TB, iB, TC, iC
        Py_ssize_t NAB = NA * NB
        Py_ssize_t N0 = NA * NB * NC * n0
        Py_ssize_t N1 = NA * NB * NC * n1

    w = 1. / (NA * NB * NC)
    for TC in range(NC):

        # First do iC == 0
        for TB in range(NB):
            # Get sub-matrix for the A-parts
            mT = m[TC * NAB + TB * NA:]

            # First do iB == 0
            phb = w
            _unfold_block(N0, N1, NA, kA2pi, phb, n0, n1, mT, &M[0, 0, 0, 0])

            # Calculate the B-phases
            kB = kB2pi[TB]
            # Step-size in the Bloch-expansion phases.
            phb_step = cos(kB) - 1j * sin(kB)

            phb = w * phb_step
            for iB in range(1, NB):

                #(0,0,iB,0)
                _unfold_block(N0, N1, NA, kA2pi, phb, n0, n1, mT, &M[0, 0, iB*NA, 0])
                #(iB,0,0,0)
                _unfold_block(N0, N1, NA, kA2pi, phb.conjugate(), n0, n1, mT, &M[iB*NA, 0, 0, 0])
                phb = phb * phb_step


        # Calculate the C-phases
        kC = kC2pi[TC]
        # Step-size in the Bloch-expansion phases.
        phc_step = cos(kC) - 1j * sin(kC)

        phc = w * phc_step
        for iC in range(1, NC):
            cphc = phc.conjugate()

            for TB in range(NB):
                # Get sub-matrix for the A-parts
                mT = m[TC * NAB + TB * NA:]

                # Calculate the B-phases
                kB = kB2pi[TB]
                # Step-size in the Bloch-expansion phases.
                phb_step = cos(kB) - 1j * sin(kB)

                # First do iB == 0
                ##(0,0,iC,0)
                _unfold_block(N0, N1, NA, kA2pi, phc, n0, n1, mT, &M[0, 0, iC*NAB, 0])
                ##(iC,0,0,0)
                _unfold_block(N0, N1, NA, kA2pi, cphc, n0, n1, mT, &M[iC*NAB, 0, 0, 0])

                phb = phb_step
                for iB in range(1, NB):

                    ##(0,0,iC,0)
                    #(0,0,iB,0)
                    _unfold_block(N0, N1, NA, kA2pi, phc*phb, n0, n1, mT, &M[0, 0, iC*NAB+iB*NA, 0])
                    #(iB,0,0,0)
                    _unfold_block(N0, N1, NA, kA2pi, phc*phb.conjugate(), n0, n1, mT,
                                  &M[iB*NA, 0, iC*NAB, 0])

                    ##(iC,0,0,0)
                    #(0,0,iB,0)
                    _unfold_block(N0, N1, NA, kA2pi, cphc*phb, n0, n1, mT, &M[iC*NAB, 0, iB*NA, 0])
                    #(iB,0,0,0)
                    _unfold_block(N0, N1, NA, kA2pi, cphc*phb.conjugate(), n0, n1, mT,
                                  &M[iC*NAB+iB*NA, 0, 0, 0])

                    phb = phb * phb_step

            phc = phc * phc_step


cdef void _unfold_copy(const int[::1] B, complexs_st[:, ::1] M) noexcept:
    """ Finalize an unfolding by copying data around for the Toeplitz matrix `m`. """

    # N should now equal K.shape[0]
    cdef:
        Py_ssize_t B0 = B[0]
        Py_ssize_t B1 = B[1]
        Py_ssize_t B2 = B[2]
        Py_ssize_t N = B0 * B1 * B2
        Py_ssize_t n0 = M.shape[0] // N
        Py_ssize_t n1 = M.shape[1] // N
        complexs_st[:, :, :, ::1] M4 = <complexs_st[:N, :n0, :N, :n1]> &M[0, 0]

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
    cdef:
        Py_ssize_t rA, cA, r, c
        Py_ssize_t n0N1 = n0 * N1
        Py_ssize_t from_r, to_r, from_c, to_c

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
    cdef:
        Py_ssize_t iB
        Py_ssize_t N0 = NA * NB * n0
        Py_ssize_t N1 = NA * NB * n1

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
    cdef:
        Py_ssize_t iB, iC
        Py_ssize_t NAB = NA * NB
        Py_ssize_t N0 = NA * NB * NC * n0
        Py_ssize_t N1 = NA * NB * NC * n1

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
