# Define the interfaces for the functions exposed through cimport
cdef void ncol2ptr_double(const int nr, const int[::1] ncol, int[::1] ptr) nogil
cdef void ncol2ptr_single(const int nr, const int[::1] ncol, int[::1] ptr) nogil
