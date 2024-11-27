# Define the interfaces for the functions exposed through cimport
from sisl._core._dtypes cimport ints_st


cdef void ncol2ptr(const ints_st nr, const ints_st[::1] ncol, ints_st[::1] ptr,
                   const ints_st per_row, const ints_st per_elem) noexcept nogil
