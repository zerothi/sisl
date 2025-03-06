# Define the interfaces for the functions exposed through cimport
from sisl._core._dtypes cimport int_sp_st


cdef void ncol2ptr(const int_sp_st nr, const int_sp_st[::1] ncol, int_sp_st[::1] ptr,
                   const int_sp_st per_row, const int_sp_st per_elem) noexcept nogil
