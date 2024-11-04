# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
cimport cython

import numpy as np
cimport numpy as cnp

from sisl._core._dtypes cimport floats_st, ints_st
from ._common import comply_gauge
from ._matrix_phase import *
from ._matrix_phase_sc import *
from ._phase import *
from ._phase cimport is_gamma

__all__ = ["matrix_k", "matrix_k_nc", "matrix_k_so", "matrix_k_nc_diag"]


def _phase_k(gauge, M, sc, cnp.ndarray[floats_st] K, dtype):
    cdef floats_st[::1] k = K

    # dtype *must* be passed through phase_dtype
    gauge = comply_gauge(gauge)

    if is_gamma(k):
        # no - phases required
        p_opt = -1
        phases = np.empty([0], dtype=dtype)

    elif gauge == "atom":
        M.finalize()
        phases = phase_rij(M.Rij()._csr._D, sc, k, dtype)
        p_opt = 0

    elif gauge == "cell":
        phases = phase_rsc(sc, k, dtype)
        p_opt = 1

    else:
        raise ValueError("phase_k: gauge must be in [cell, atom]")

    return p_opt, phases

def matrix_k(gauge, M, const ints_st idx, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype)
    p_opt, phases = _phase_k(gauge, M, sc, k, dtype)

    # Check that the dimension *works*
    if idx < 0:
        idx += M.shape[-1]
    if idx < 0 or M.shape[-1] <= idx:
        d = M.shape[-1]
        raise ValueError(f"matrix_k: unknown index specification {idx} must be in 0:{d}")

    csr = M._csr

    if format.startswith("sc:") or format == "sc":
        if format == "sc":
            format = "csr"
        else:
            format = format[3:]
        nc = M.geometry.no_s

        if format in ("array", "matrix", "dense"):
            return _phase_sc_array(csr.ptr, csr.ncol, csr.col, nc, csr._D, idx, phases, p_opt)

        return _phase_sc_csr(csr.ptr, csr.ncol, csr.col, nc, csr._D, idx, phases, p_opt).asformat(format)


    if format in ("array", "matrix", "dense"):
        return _phase_array(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)

    return _phase_csr(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)


def matrix_k_nc(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, phases = _phase_k(gauge, M, sc, k, dtype)

    csr = M._csr

    if format.startswith("sc:") or format == "sc":
        if format == "sc":
            format = "csr"
        else:
            format = format[3:]
        nc = M.geometry.no_s

        if format in ("array", "matrix", "dense"):
            return _phase_sc_array_nc(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt)

        return _phase_sc_csr_nc(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt).asformat(format)

    if format in ("array", "matrix", "dense"):
        return _phase_array_nc(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt)

    return _phase_csr_nc(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt).asformat(format)


def matrix_k_nc_diag(gauge, M, const ints_st idx, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, phases = _phase_k(gauge, M, sc, k, dtype)

    csr = M._csr

    if format.startswith("sc:") or format == "sc":
        if format == "sc":
            format = "csr"
        else:
            format = format[3:]
        nc = M.geometry.no_s

        if format in ("array", "matrix", "dense"):
            return _phase_sc_array_nc_diag(csr.ptr, csr.ncol, csr.col, nc, csr._D, idx, phases, p_opt)

        return _phase_sc_csr_nc_diag(csr.ptr, csr.ncol, csr.col, nc, csr._D, idx, phases, p_opt).asformat(format)

    if format in ("array", "matrix", "dense"):
        return _phase_array_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt)

    return _phase_csr_nc_diag(csr.ptr, csr.ncol, csr.col, csr._D, idx, phases, p_opt).asformat(format)


def matrix_k_so(gauge, M, sc, cnp.ndarray[floats_st] k, dtype, format):
    dtype = phase_dtype(k, M.dtype, dtype, True)
    p_opt, phases = _phase_k(gauge, M, sc, k, dtype)

    csr = M._csr

    if format.startswith("sc:") or format == "sc":
        if format == "sc":
            format = "csr"
        else:
            format = format[3:]
        nc = M.geometry.no_s

        if format in ("array", "matrix", "dense"):
            return _phase_sc_array_so(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt)

        return _phase_sc_csr_so(csr.ptr, csr.ncol, csr.col, nc, csr._D, phases, p_opt).asformat(format)

    if format in ("array", "matrix", "dense"):
        return _phase_array_so(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt)

    return _phase_csr_so(csr.ptr, csr.ncol, csr.col, csr._D, phases, p_opt).asformat(format)
