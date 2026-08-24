"""
Inline-sum (all useful shared codes could be placed here
"""
cimport cython

import numpy as np

cimport numpy as cnp
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

ctypedef long long _longlong
ctypedef float complex _complex64_t
ctypedef double complex _complex128_t

@cython.initializedcheck(False)
cdef inline object type2dtype(const _type2dtype_types_st v):
    if _type2dtype_types_st is int8_t:
        return np.int8
    elif _type2dtype_types_st is int16_t:
        return np.int16
    elif _type2dtype_types_st is int32_t:
        return np.int32
    elif _type2dtype_types_st is int64_t:
        return np.int64
    elif _type2dtype_types_st is int:
        return np.int32
    elif _type2dtype_types_st is _longlong:
        return np.int64
    elif _type2dtype_types_st is float:
        return np.float32
    elif _type2dtype_types_st is double:
        return np.float64
    elif _type2dtype_types_st is _complex64_t:
        return np.complex64
    elif _type2dtype_types_st is _complex128_t:
        return np.complex128

    # More special cases
    elif _type2dtype_types_st is uint8_t:
        return np.uint8
    elif _type2dtype_types_st is uint16_t:
        return np.uint16
    elif _type2dtype_types_st is uint32_t:
        return np.uint32
    elif _type2dtype_types_st is uint64_t:
        return np.uint64

    raise ValueError("Could not determine data-type (type2dtype)")


@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef inline Py_ssize_t inline_sum(const _inline_sum_st[::1] array) noexcept nogil:
    cdef Py_ssize_t total, i

    total = 0
    for i in range(array.shape[0]):
        total += array[i]

    return total
