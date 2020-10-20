# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport cython

__all__ = ["ncol2ptr_double", "ncol2ptr_single"]


cdef void ncol2ptr_double(const int nr, const int[::1] ncol, int[::1] ptr) nogil:
    cdef Py_ssize_t r, rr

    # this is NC/SOC
    ptr[0] = 0
    ptr[1] = ncol[0] * 2
    for r in range(1, nr):
        rr = r * 2
        # do both
        ptr[rr] = ptr[rr - 1] + ncol[r-1] * 2
        ptr[rr+1] = ptr[rr] + ncol[r] * 2

    ptr[nr * 2] = ptr[nr * 2 - 1] + ncol[nr - 1] * 2


cdef void ncol2ptr_single(const int nr, const int[::1] ncol, int[::1] ptr) nogil:
    cdef Py_ssize_t r, rr

    # this is NC/SOC
    ptr[0] = 0
    ptr[1] = ncol[0]
    for r in range(1, nr):
        rr = r * 2
        # do both
        ptr[rr] = ptr[rr - 1] + ncol[r-1]
        ptr[rr+1] = ptr[rr] + ncol[r]

    ptr[nr * 2] = ptr[nr * 2 - 1] + ncol[nr - 1]
