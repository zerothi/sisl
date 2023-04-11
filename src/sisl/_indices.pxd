# Define the interfaces for the functions exposed through cimport
cdef int in_1d(const int[::1] array, const int v) nogil
cdef Py_ssize_t _index_sorted(const int[::1] array, const int v) nogil
