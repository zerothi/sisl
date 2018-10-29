#!python
#cython: language_level=2
# Define the interfaces for the functions exposed through cimport
cdef int in_1d(const int[::1] array, const int v) nogil
cdef int _index_sorted(const int[::1] array, const int v) nogil
