# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython

import numpy as np

cimport numpy as cnp

from sisl._core._dtypes cimport complexs_st, floatcomplexs_st, int_sp_st, reals_st

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


cdef inline void matrix_add_csr_nc(const int_sp_st[::1] v_ptr,
                                   const int_sp_st r,
                                   const int_sp_st r_idx,
                                   complexs_st[::1] v,
                                   const complexs_st *M) noexcept nogil:
    v[v_ptr[r] + r_idx] += M[0]
    v[v_ptr[r] + r_idx+1] += M[1]
    v[v_ptr[r+1] + r_idx] += M[2]
    v[v_ptr[r+1] + r_idx+1] += M[3]

cdef inline void matrix_add_array_nc(const int_sp_st r,
                                     const int_sp_st c,
                                     complexs_st[:, ::1] v,
                                     const complexs_st *M) noexcept nogil:
    v[r, c] += M[0]
    v[r, c+1] += M[1]
    v[r+1, c] += M[2]
    v[r+1,c+1] += M[3]

cdef inline void matrix_box_nc_real(const reals_st *data,
                                    const complexs_st phase,
                                    complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> ((data[2] + 1j * data[3]) * phase)
    M[2] = <complexs_st> ((data[2] - 1j * data[3]) * phase)
    M[3] = <complexs_st> (data[1] * phase)


cdef inline void matrix_box_nc_cmplx(const _internal_complexs_st *data,
                                     const complexs_st phase,
                                     complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> (data[2] * phase)
    M[2] = <complexs_st> (data[2].conjugate() * phase)
    M[3] = <complexs_st> (data[1] * phase)


cdef inline void matrix_box_so_real(const reals_st *data,
                                    const complexs_st phase,
                                    complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> ((data[0] + 1j * data[4]) * phase)
    M[1] = <complexs_st> ((data[2] + 1j * data[3]) * phase)
    M[2] = <complexs_st> ((data[6] + 1j * data[7]) * phase)
    M[3] = <complexs_st> ((data[1] + 1j * data[5]) * phase)


cdef inline void matrix_box_so_cmplx(const _internal_complexs_st *data,
                                     const complexs_st phase,
                                     complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> (data[2] * phase)
    M[2] = <complexs_st> (data[3] * phase)
    M[3] = <complexs_st> (data[1] * phase)


cdef inline void matrix_add_csr_nambu(const int_sp_st[::1] v_ptr,
                                      const int_sp_st r,
                                      const int_sp_st r_idx,
                                      complexs_st[::1] v,
                                      const complexs_st *M) noexcept nogil:
    # H e-e
    v[v_ptr[r] + r_idx] += M[0]
    v[v_ptr[r] + r_idx+1] += M[1]
    # Delta [e-h]
    v[v_ptr[r] + r_idx+2] += M[4]
    v[v_ptr[r] + r_idx+3] += M[5]
    # H e-e
    v[v_ptr[r+1] + r_idx] += M[2]
    v[v_ptr[r+1] + r_idx+1] += M[3]
    # Delta [e-h]
    v[v_ptr[r+1] + r_idx+2] += M[6]
    v[v_ptr[r+1] + r_idx+3] += M[7]
    # -Delta^* [h-e]
    v[v_ptr[r+2] + r_idx] += M[12]
    v[v_ptr[r+2] + r_idx+1] += M[13]
    # H h-h: -H^*
    v[v_ptr[r+2] + r_idx+2] += M[8]
    v[v_ptr[r+2] + r_idx+3] += M[9]
    # -Delta^* [h-e]
    v[v_ptr[r+3] + r_idx] += M[14]
    v[v_ptr[r+3] + r_idx+1] += M[15]
    # H h-h: -H^*
    v[v_ptr[r+3] + r_idx+2] += M[10]
    v[v_ptr[r+3] + r_idx+3] += M[11]

cdef inline void matrix_add_array_nambu(const int_sp_st r,
                                        const int_sp_st c,
                                        complexs_st[:, ::1] v,
                                        const complexs_st *M) noexcept nogil:
    # H e-e
    v[r, c] += M[0]
    v[r, c+1] += M[1]
    # Delta [e-h]
    v[r, c+2] += M[4]
    v[r, c+3] += M[5]
    # H e-e
    v[r+1, c] += M[2]
    v[r+1,c+1] += M[3]
    # Delta [e-h]
    v[r+1, c+2] += M[6]
    v[r+1, c+3] += M[7]
    # -Delta^* [h-e]
    v[r+2, c] += M[12]
    v[r+2, c+1] += M[13]
    # H h-h: -H^*
    v[r+2, c+2] += M[8]
    v[r+2, c+3] += M[9]
    # -Delta^* [h-e]
    v[r+3, c] += M[14]
    v[r+3, c+1] += M[15]
    # H h-h: -H^*
    v[r+3, c+2] += M[10]
    v[r+3, c+3] += M[11]


cdef inline void matrix_box_nambu_real(const reals_st *data,
                                       const complexs_st phase,
                                       complexs_st *M) noexcept nogil:
    # H e-e
    M[0] = <complexs_st> ((data[0] + 1j * data[4]) * phase)
    M[1] = <complexs_st> ((data[2] + 1j * data[3]) * phase)
    M[2] = <complexs_st> ((data[6] + 1j * data[7]) * phase)
    M[3] = <complexs_st> ((data[1] + 1j * data[5]) * phase)
    # delta matrix stored in [8:16]
    M[4] = <complexs_st> ((data[10] + 1j * data[11]) * phase)
    M[5] = <complexs_st> ((data[8] + data[14] + 1j * (data[9] + data[15])) * phase)
    M[6] = <complexs_st> ((-data[8] + data[14] + 1j * (-data[9] + data[15])) * phase)
    M[7] = <complexs_st> ((data[12] + 1j * data[13]) * phase)

    # Lower part (-M^*)
    # H h-h
    M[8] = <complexs_st> ((-data[0] + 1j * data[4]) * phase)
    M[9] = <complexs_st> ((-data[2] + 1j * data[3]) * phase)
    M[10] = <complexs_st> ((-data[6] + 1j * data[7]) * phase)
    M[11] = <complexs_st> ((-data[1] + 1j * data[5]) * phase)
    # delta matrix stored in [8:16]
    M[12] = <complexs_st> ((-data[10] + 1j * data[11]) * phase)
    M[13] = <complexs_st> ((-data[8] - data[14] + 1j * (data[9] + data[15])) * phase)
    M[14] = <complexs_st> ((data[8] - data[14] + 1j * (-data[9] + data[15])) * phase)
    M[15] = <complexs_st> ((-data[12] + 1j * data[13]) * phase)


cdef inline void matrix_box_nambu_cmplx(const _internal_complexs_st *data,
                                        const complexs_st phase,
                                        complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> (data[2] * phase)
    M[2] = <complexs_st> (data[3] * phase)
    M[3] = <complexs_st> (data[1] * phase)
    # delta matrix stored in [4:8]
    M[4] = <complexs_st> (data[5] * phase)
    M[5] = <complexs_st> ((data[4] + data[7]) * phase)
    M[6] = <complexs_st> ((-data[4] + data[7]) * phase)
    M[7] = <complexs_st> (data[6] * phase)

    # Lower part (-M^*)
    M[8] = <complexs_st> (-data[0].conjugate() * phase)
    M[9] = <complexs_st> (-data[2].conjugate() * phase)
    M[10] = <complexs_st> (-data[3].conjugate() * phase)
    M[11] = <complexs_st> (-data[1].conjugate() * phase)
    # delta matrix stored in [4:8]
    M[12] = <complexs_st> (-data[5].conjugate() * phase)
    M[13] = <complexs_st> (-(data[4] + data[7]).conjugate() * phase)
    M[14] = <complexs_st> ((data[4] - data[7]).conjugate() * phase)
    M[15] = <complexs_st> (-data[6].conjugate() * phase)
