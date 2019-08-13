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
    cdef double f
    f = 1. / (icell[0, 0] * cell[0, 0] + icell[0, 1] * cell[0, 1] + icell[0, 2] * cell[0, 2])
    icell[0, 0] = icell[0, 0] * f
    icell[0, 1] = icell[0, 1] * f
    icell[0, 2] = icell[0, 2] * f
    f = 1. / (icell[1, 0] * cell[1, 0] + icell[1, 1] * cell[1, 1] + icell[1, 2] * cell[1, 2])
    icell[1, 0] = icell[1, 0] * f
    icell[1, 1] = icell[1, 1] * f
    icell[1, 2] = icell[1, 2] * f
    f = 1. / (icell[2, 0] * cell[2, 0] + icell[2, 1] * cell[2, 1] + icell[2, 2] * cell[2, 2])
    icell[2, 0] = icell[2, 0] * f
    icell[2, 1] = icell[2, 1] * f
    icell[2, 2] = icell[2, 2] * f
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
    cdef double twopi = 2 * npmath.PI
    cdef double f
    f = twopi / (rcell[0, 0] * cell[0, 0] + rcell[0, 1] * cell[0, 1] + rcell[0, 2] * cell[0, 2])
    rcell[0, j] = rcell[0, j] * f
    rcell[0, j] = rcell[0, j] * f
    rcell[0, j] = rcell[0, j] * f
    f = twopi / (rcell[1, 0] * cell[1, 0] + rcell[1, 1] * cell[1, 1] + rcell[1, 2] * cell[1, 2])
    rcell[1, j] = rcell[1, j] * f
    rcell[1, j] = rcell[1, j] * f
    rcell[1, j] = rcell[1, j] * f
    f = twopi / (rcell[2, 0] * cell[2, 0] + rcell[2, 1] * cell[2, 1] + rcell[2, 2] * cell[2, 2])
    rcell[2, j] = rcell[2, j] * f
    rcell[2, j] = rcell[2, j] * f
    rcell[2, j] = rcell[2, j] * f
    return rcell
