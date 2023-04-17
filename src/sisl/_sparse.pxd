# Define the interfaces for the functions exposed through cimport
cdef Py_ssize_t inline_sum(const int[::1] array) nogil
