# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np

cimport numpy as cnp

from sisl._core._dtypes cimport complexs_st, numerics_st, reals_st

ctypedef fused _internal_complexs_st:
    float complex
    double complex

ctypedef void(*_f_matrix_box_so)(const numerics_st *data,
                                 const complexs_st phase,
                                 complexs_st *M) noexcept nogil

cdef void _matrix_box_nc(const numerics_st *data,
                         const complexs_st phase,
                         complexs_st *M) noexcept nogil

cdef void _matrix_box_so_real(const reals_st *data,
                              const complexs_st phase,
                              complexs_st *M) noexcept nogil

cdef void _matrix_box_so_cmplx(const _internal_complexs_st *data,
                               const complexs_st phase,
                               complexs_st *M) noexcept nogil
