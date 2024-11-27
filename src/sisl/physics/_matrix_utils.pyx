# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython

import numpy as np

cimport numpy as cnp

from sisl._core._dtypes cimport complexs_st, numerics_st, reals_st

"""
These routines converts an array of n-values into a spin-box matrix.

In all cases, the resulting linear returned matrix `M`
has 4 entries.

M[0] == spin[0, 0]
M[1] == spin[0, 1]
M[2] == spin[1, 0]
M[3] == spin[1, 1]

For nambu spin configurations, the spin box is 4x4, however
the spin box is:

           | M^ee           Delta |   | M^ee          Delta     |
 M_nambu = |                      | = |                         |
           | Delta^dagger   M^hh  |   | Delta^dagger  -(M^ee)^* |

So we only return M^ee and Delta.
The delta matrices are stored in the singlet (S) + triplet (Tuu, Tdd, T0) terms.
The delta expansion looks like this:

        |   Tuu    S + T0 |
Delta = |                 |
        | -S + T0   Tdd   |

M[4] == Delta[0, 0]
M[5] == Delta[0, 1]
M[6] == Delta[1, 0]
M[7] == Delta[1, 1]
"""


cdef inline void _matrix_box_nc_real(const reals_st *data,
                                     const complexs_st phase,
                                     complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> ((data[2] + 1j * data[3]) * phase)
    M[2] = <complexs_st> ((data[2] + 1j * data[3]).conjugate() * phase)
    M[3] = <complexs_st> (data[1] * phase)


cdef inline void _matrix_box_nc_cmplx(const _internal_complexs_st *data,
                                      const complexs_st phase,
                                      complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> (data[2] * phase)
    M[2] = <complexs_st> (data[2].conjugate() * phase)
    M[3] = <complexs_st> (data[1] * phase)


cdef inline void _matrix_box_so_real(const reals_st *data,
                                     const complexs_st phase,
                                     complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> ((data[0] + 1j * data[4]) * phase)
    M[1] = <complexs_st> ((data[2] + 1j * data[3]) * phase)
    M[2] = <complexs_st> ((data[6] + 1j * data[7]) * phase)
    M[3] = <complexs_st> ((data[1] + 1j * data[5]) * phase)


cdef inline void _matrix_box_so_cmplx(const _internal_complexs_st *data,
                                      const complexs_st phase,
                                      complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> (data[2] * phase)
    M[2] = <complexs_st> (data[3] * phase)
    M[3] = <complexs_st> (data[1] * phase)


cdef inline void _matrix_box_nambu_real(const reals_st *data,
                                        const complexs_st phase,
                                        complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> ((data[0] + 1j * data[4]) * phase)
    M[1] = <complexs_st> ((data[2] + 1j * data[3]) * phase)
    M[2] = <complexs_st> ((data[6] + 1j * data[7]) * phase)
    M[3] = <complexs_st> ((data[1] + 1j * data[5]) * phase)
    # delta matrix stored in [8-15]
    M[4] = <complexs_st> ((data[10] + 1j * data[11]) * phase)
    M[5] = <complexs_st> ((data[8] + data[14] + 1j * (data[9] + data[15])) * phase)
    M[6] = <complexs_st> ((-data[8] + data[14] + 1j * (-data[9] + data[15])) * phase)
    M[7] = <complexs_st> ((data[12] + 1j * data[13]) * phase)


cdef inline void _matrix_box_nambu_cmplx(const _internal_complexs_st *data,
                                         const complexs_st phase,
                                         complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> (data[2] * phase)
    M[2] = <complexs_st> (data[3] * phase)
    M[3] = <complexs_st> (data[1] * phase)
    M[4] = <complexs_st> (data[5] * phase)
    M[5] = <complexs_st> ((data[4] + data[7]) * phase)
    M[6] = <complexs_st> ((-data[4] + data[7]) * phase)
    M[7] = <complexs_st> (data[6] * phase)
