# Define the interfaces for the functions exposed through cimport
from numpy cimport int16_t, int32_t, int64_t

from sisl._core._dtypes cimport ints_st


cdef bint in_1d(const ints_st[::1] array, const ints_st v) noexcept nogil

ctypedef fused _ints_index_sorted_st:
    short
    int
    long
    int16_t
    int32_t
    int64_t

cdef Py_ssize_t _index_sorted(const ints_st[::1] array, const _ints_index_sorted_st v) noexcept nogil
