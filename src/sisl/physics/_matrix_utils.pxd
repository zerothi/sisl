# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np

cimport numpy as cnp

from sisl._core._dtypes cimport complexs_st, floatcomplexs_st, int_sp_st, reals_st

ctypedef fused _internal_complexs_st:
    float complex
    double complex

ctypedef void(*f_matrix_box_nc)(const floatcomplexs_st *data,
                                const complexs_st phase,
                                complexs_st *M) noexcept nogil

cdef void matrix_box_nc_real(const reals_st *data,
                             const complexs_st phase,
                             complexs_st *M) noexcept nogil

cdef void matrix_box_nc_cmplx(const _internal_complexs_st *data,
                              const complexs_st phase,
                              complexs_st *M) noexcept nogil

ctypedef void(*f_matrix_box_so)(const floatcomplexs_st *data,
                                const complexs_st phase,
                                complexs_st *M) noexcept nogil

cdef void matrix_box_so_real(const reals_st *data,
                             const complexs_st phase,
                             complexs_st *M) noexcept nogil

cdef void matrix_box_so_cmplx(const _internal_complexs_st *data,
                              const complexs_st phase,
                              complexs_st *M) noexcept nogil

ctypedef void(*f_matrix_box_nambu)(const floatcomplexs_st *data,
                                   const complexs_st phase,
                                   complexs_st *M) noexcept nogil

cdef void matrix_box_nambu_real(const reals_st *data,
                                const complexs_st phase,
                                complexs_st *M) noexcept nogil

cdef void matrix_box_nambu_cmplx(const _internal_complexs_st *data,
                                 const complexs_st phase,
                                 complexs_st *M) noexcept nogil


# Finally, the interfaces for calling the addition routines
cdef void matrix_add_csr_nc(const int_sp_st[::1] v_ptr,
                            const int_sp_st r,
                            const int_sp_st r_idx,
                            complexs_st[::1] v,
                            const complexs_st *M) noexcept nogil

cdef void matrix_add_array_nc(const int_sp_st r,
                              const int_sp_st c,
                              complexs_st[:, ::1] v,
                              const complexs_st *M) noexcept nogil

cdef void matrix_add_csr_nambu(const int_sp_st[::1] v_ptr,
                               const int_sp_st r,
                               const int_sp_st r_idx,
                               complexs_st[::1] v,
                               const complexs_st *M) noexcept nogil

cdef void matrix_add_array_nambu(const int_sp_st r,
                                 const int_sp_st c,
                                 complexs_st[:, ::1] v,
                                 const complexs_st *M) noexcept nogil
