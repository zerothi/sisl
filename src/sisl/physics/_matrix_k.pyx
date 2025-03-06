# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np
cimport numpy as cnp

from sisl._core._dtypes cimport floats_st, int_sp_st
from ._common import comply_gauge
from ._matrix_phase import *
from ._matrix_phase_sc import *
from ._phase import *
from ._phase cimport is_gamma

__all__ = [
    "matrix_k",
    "matrix_k_nc",
    "matrix_k_so",
    "matrix_k_diag",
    "matrix_k_nambu",
]


def phase_dk(gauge, M, sc, cnp.ndarray[floats_st] K, dtype):
    cdef floats_st[::1] k = K

    # dtype *must* be passed through phase_dtype
    gauge = comply_gauge(gauge)

    if is_gamma(k):
        # no - phases required
        p_opt = -1
        phases = np.ones(1, dtype=dtype)

    elif gauge == "atomic":
        M.finalize()
        phases = phase_rij(M.Rij()._csr._D, sc, k, dtype)
        p_opt = 0

    elif gauge == "lattice":
        phases = phase_rsc(sc, k, dtype)
        p_opt = 1

    else:
        raise ValueError("phase_k: gauge must be in [lattice, atomic]")

    return p_opt, phases

def matrix_k(gauge, M, const int_sp_st idx, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype)
    p_opt, phases = phase_dk(gauge, M, sc, k, dtype)

    cdef int_sp_st udx = idx
    # Check that the dimension *works*
    cdef int_sp_st shapem1 = M.shape[-1]
    if idx < 0:
        udx += shapem1
    if udx < 0 or shapem1 <= udx:
        d = shapem1
        raise ValueError(f"matrix_k: unknown index specification {idx} must be in 0:{d}")

    csr = M._csr

    if format.startswith("sc:") or format == "sc":
        if format == "sc":
            format = "csr"
        else:
            format = format[3:]
        nc = M.geometry.no_s

        if format in ("array", "matrix", "dense"):
            return phase_sc_array(csr.ptr, csr.ncol, csr.col, nc, csr._D, udx, phases, p_opt)

        return phase_sc_csr(csr.ptr, csr.ncol, csr.col, nc, csr._D, udx, phases, p_opt).asformat(format)


    if format in ("array", "matrix", "dense"):
        return phase_array(csr.ptr, csr.ncol, csr.col, csr._D, udx, phases, p_opt)

    return phase_csr(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)


def matrix_k_nc(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, phases = phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format.startswith("sc:") or format == "sc":
        if format == "sc":
            format = "csr"
        else:
            format = format[3:]
        nc = M.geometry.no_s

        if format in ("array", "matrix", "dense"):
            return phase_sc_array_nc(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt)

        return phase_sc_csr_nc(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt).asformat(format)

    if format in ("array", "matrix", "dense"):
        return phase_array_nc(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt)

    return phase_csr_nc(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt).asformat(format)


def matrix_k_diag(gauge, M, const int_sp_st idx, const int_sp_st per_row,
                  sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, phases = phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format.startswith("sc:") or format == "sc":
        if format == "sc":
            format = "csr"
        else:
            format = format[3:]
        nc = M.geometry.no_s

        if format in ("array", "matrix", "dense"):
            return phase_sc_array_diag(csr.ptr, csr.ncol, csr.col, nc, csr._D, idx,
            phases, p_opt, per_row)

        return phase_sc_csr_diag(csr.ptr, csr.ncol, csr.col, nc, csr._D, idx, phases,
        p_opt, per_row).asformat(format)

    if format in ("array", "matrix", "dense"):
        return phase_array_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt,
        per_row)

    return phase_csr_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt,
    per_row).asformat(format)


def matrix_k_so(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, phases = phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format.startswith("sc:") or format == "sc":
        if format == "sc":
            format = "csr"
        else:
            format = format[3:]
        nc = M.geometry.no_s

        if format in ("array", "matrix", "dense"):
            return phase_sc_array_so(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt)

        return phase_sc_csr_so(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt).asformat(format)

    if format in ("array", "matrix", "dense"):
        return phase_array_so(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt)

    return phase_csr_so(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt).asformat(format)


def matrix_k_nambu(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, phases = phase_dk(gauge, M, sc, k, dtype)

    csr = M._csr

    if format.startswith("sc:") or format == "sc":
        if format == "sc":
            format = "csr"
        else:
            format = format[3:]
        nc = M.geometry.no_s

        if format in ("array", "matrix", "dense"):
            return phase_sc_array_nambu(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt)

        return phase_sc_csr_nambu(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt).asformat(format)

    if format in ("array", "matrix", "dense"):
        return phase_array_nambu(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt)

    return phase_csr_nambu(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt).asformat(format)
