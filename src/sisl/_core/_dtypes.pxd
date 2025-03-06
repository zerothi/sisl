"""
Shared header for fused dtypes
"""
cimport cython

import numpy as np

cimport numpy as cnp
from numpy cimport (
    complex64_t,
    complex128_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

# Generic typedefs for sisl internal naming convention
ctypedef size_t size_st


ctypedef fused ints_st:
    int
    long


ctypedef fused int_sp_st:
    int


ctypedef fused floats_st:
    float
    double


ctypedef fused complexs_st:
    float complex
    double complex


ctypedef fused floatcomplexs_st:
    float
    double
    float complex
    double complex


# We need this fused data-type to omit complex data-types
ctypedef fused reals_st:
    int
    long
    float
    double

ctypedef fused numerics_st:
    int
    long
    float
    double
    float complex
    double complex

ctypedef fused _type2dtype_types_st:
    short
    int
    long
    float
    double
    float complex
    double complex
    float32_t
    float64_t
    #complex64_t # not usable...
    #complex128_t
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t


cdef object type2dtype(const _type2dtype_types_st v)


ctypedef fused _inline_sum_st:
    short
    int
    long
    int16_t
    int32_t
    int64_t
    uint16_t
    uint32_t
    uint64_t

cdef Py_ssize_t inline_sum(const _inline_sum_st[::1] array) noexcept nogil
