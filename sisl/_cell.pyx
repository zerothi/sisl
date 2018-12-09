#!python
#cython: language_level=2
cimport cython

import numpy as np
# This enables Cython enhanced compatibilities
cimport numpy as np
cimport numpy.math as npmath


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cell_invert(np.ndarray[np.float64_t, ndim=2] cell):
    cdef np.ndarray[np.float64_t, ndim=2] icell = np.empty([3, 3], dtype=np.float64)
    icell[0, 0] = cell[1, 1] * cell[2, 2] - cell[1, 2] * cell[2, 1]
    icell[0, 1] = cell[1, 2] * cell[2, 0] - cell[1, 0] * cell[2, 2]
    icell[0, 2] = cell[1, 0] * cell[2, 1] - cell[1, 1] * cell[2, 0]
    icell[1, 0] = cell[2, 1] * cell[0, 2] - cell[2, 2] * cell[0, 1]
    icell[1, 1] = cell[2, 2] * cell[0, 0] - cell[2, 0] * cell[0, 2]
    icell[1, 2] = cell[2, 0] * cell[0, 1] - cell[2, 1] * cell[0, 0]
    icell[2, 0] = cell[0, 1] * cell[1, 2] - cell[0, 2] * cell[1, 1]
    icell[2, 1] = cell[0, 2] * cell[1, 0] - cell[0, 0] * cell[1, 2]
    icell[2, 2] = cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0]
    cdef int i
    cdef int j
    cdef double f
    for i in range(3):
        f = 1. / (icell[i, 0] * cell[i, 0] + icell[i, 1] * cell[i, 1] + icell[i, 2] * cell[i, 2])
        for j in range(3):
            icell[i, j] = icell[i, j] * f
    return icell


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cell_reciprocal(np.ndarray[np.float64_t, ndim=2] cell):
    cdef np.ndarray[np.float64_t, ndim=2] rcell = np.empty([3, 3], dtype=np.float64)
    rcell[0, 0] = cell[1, 1] * cell[2, 2] - cell[1, 2] * cell[2, 1]
    rcell[0, 1] = cell[1, 2] * cell[2, 0] - cell[1, 0] * cell[2, 2]
    rcell[0, 2] = cell[1, 0] * cell[2, 1] - cell[1, 1] * cell[2, 0]
    rcell[1, 0] = cell[2, 1] * cell[0, 2] - cell[2, 2] * cell[0, 1]
    rcell[1, 1] = cell[2, 2] * cell[0, 0] - cell[2, 0] * cell[0, 2]
    rcell[1, 2] = cell[2, 0] * cell[0, 1] - cell[2, 1] * cell[0, 0]
    rcell[2, 0] = cell[0, 1] * cell[1, 2] - cell[0, 2] * cell[1, 1]
    rcell[2, 1] = cell[0, 2] * cell[1, 0] - cell[0, 0] * cell[1, 2]
    rcell[2, 2] = cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0]
    cdef int i
    cdef int j
    cdef double twopi = 2 * npmath.PI
    cdef double f
    for i in range(3):
        f = twopi / (rcell[i, 0] * cell[i, 0] + rcell[i, 1] * cell[i, 1] + rcell[i, 2] * cell[i, 2])
        for j in range(3):
            rcell[i, j] = rcell[i, j] * f
    return rcell
