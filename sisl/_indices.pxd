# Define the interfaces for the functions exposed through cimport
cdef int in_1d(const int[::1] array, const int v) nogil
cdef int index_sorted(const int[::1] array, const int v) nogil
