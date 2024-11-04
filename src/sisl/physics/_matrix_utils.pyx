# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
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
"""


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline void _matrix_box_nc(const numerics_st *data,
                                const complexs_st phase,
                                complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> ((data[2] + 1j * data[3]) * phase)
    M[2] = <complexs_st> ((data[2] + 1j * data[3]).conjugate() * phase)
    M[3] = <complexs_st> (data[1] * phase)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline void _matrix_box_so_real(const reals_st *data,
                                     const complexs_st phase,
                                     complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> ((data[0] + 1j * data[4]) * phase)
    M[1] = <complexs_st> ((data[2] + 1j * data[3]) * phase)
    M[2] = <complexs_st> ((data[6] + 1j * data[7]) * phase)
    M[3] = <complexs_st> ((data[1] + 1j * data[5]) * phase)


# necessary to double the interfaces
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline void _matrix_box_so_cmplx(const _internal_complexs_st *data,
                                      const complexs_st phase,
                                      complexs_st *M) noexcept nogil:
    M[0] = <complexs_st> (data[0] * phase)
    M[1] = <complexs_st> (data[2] * phase)
    M[2] = <complexs_st> (data[3] * phase)
    M[3] = <complexs_st> (data[1] * phase)
