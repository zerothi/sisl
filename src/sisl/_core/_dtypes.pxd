"""
Shared header for fused dtypes
"""
cimport cython

import numpy as np

cimport numpy as cnp
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)


cdef extern from *:
    """
    #include <limits.h>
    #if INT_MAX != 2147483647
    #error "sisl requires a 32-bit int"
    #endif
    #if LLONG_MAX != 9223372036854775807LL
    #error "sisl requires a 64-bit long long"
    #endif
    """
    pass

# Generic typedefs for sisl internal naming convention
ctypedef size_t size_st


# Signed integers.
#
# The members are the *fixed width* types, deliberately not `int`/`long`:
# `long` is 64-bit on LP64 (Linux/macOS) but 32-bit on LLP64 (Windows/MSVC).
# A fused (int, long) therefore expands to two *identical* 32-bit
# specializations on Windows, and then nothing matches an int64 array at all
# (dispatch is on itemsize+kind, so the second specialization is simply dead
# code). int32_t and int64_t are distinct on every platform we build for, so
# the fused type has exactly one member per width, everywhere.
#
# There is no way to make the member list itself conditional: the fused `is`
# test is only valid inside a function body (where it is a compile-time
# branch per specialization), and the compile-time IF/DEF statements are
# deprecated and slated for removal. Choosing non-overlapping types is the
# way to avoid duplicate specializations.
ctypedef fused ints_st:
    int
    long long


# Index type of the scipy sparse matrices (int32 everywhere)
ctypedef fused int_sp_st:
    int


ctypedef fused floats_st:
    float
    double


ctypedef fused complexs_st:
    float complex
    double complex


ctypedef fused floatcomplexs_st:
    float
    double
    float complex
    double complex


# A second, *independent* float/complex fused type.
# Cython ties together all occurrences of the same fused ctypedef in a
# signature, so a routine that needs two unrelated float/complex types
# (e.g. matrix data and phases) must use two distinct names.
ctypedef fused _floatcomplexs2_st:
    float
    double
    float complex
    double complex


# The two casting helpers below exist so that mixed real/complex fused
# routines can be written *once*, without duplicating the loop bodies for
# the real and the complex variants.
#
# Writing ``out[0] = <real_type> complex_value`` directly is not portable:
# it only compiles when Cython maps complex numbers onto the native C99
# ``_Complex`` type. With the struct fallback (``CYTHON_CCOMPLEX == 0``,
# e.g. MSVC) or in C++ mode (``std::complex``) the generated cast is a hard
# compile error, and Cython instantiates *every* combination of the fused
# types, including the complex -> real ones that never occur at runtime.
#
# NOTE: `value` is deliberately *not* declared ``const``. Cython treats
# ``const T`` as a distinct type from ``T`` for a by-value parameter, and
# coerces a complex argument into it via a decompose/recompose round trip
# (``from_parts(CREAL(x), CIMAG(x))``). That round trip is emitted at the
# *call site*, so inlining cannot remove it, and gcc does not fold it away
# even at -ffast-math. A by-value scalar gains nothing from ``const``.
#
# The ``is`` / ``in`` tests on the fused types are *compile-time*
# conditions: Cython prunes the non-matching branches for every
# specialization, so the offending cast is never emitted and neither a
# runtime branch nor a call survives (both helpers are inlined).
@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef inline void cast_assign(floatcomplexs_st *out,
                             _floatcomplexs2_st value) noexcept nogil:
    """``out[0] = value``, dropping the imaginary part if `out` is real"""
    if floatcomplexs_st is _floatcomplexs2_st:
        out[0] = value
    elif floatcomplexs_st in complexs_st:
        out[0] = <floatcomplexs_st> value
    elif _floatcomplexs2_st in complexs_st:
        out[0] = <floatcomplexs_st> value.real
    else:
        out[0] = <floatcomplexs_st> value


@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef inline void cast_add(floatcomplexs_st *out,
                          _floatcomplexs2_st value) noexcept nogil:
    """``out[0] += value``, dropping the imaginary part if `out` is real

    Not written as an in-place ``+=`` at the call sites: for complex fused
    types Cython lowers in-place operators on memoryview elements to a raw
    C ``+=``, which does not compile for the struct/std::complex fallbacks.
    """
    if floatcomplexs_st is _floatcomplexs2_st:
        # identical types, no conversion (avoids a redundant complex
        # decompose/recompose round trip that gcc does not fully fold)
        out[0] = out[0] + value
    elif floatcomplexs_st in complexs_st:
        out[0] = out[0] + <floatcomplexs_st> value
    elif _floatcomplexs2_st in complexs_st:
        out[0] = out[0] + <floatcomplexs_st> value.real
    else:
        out[0] = out[0] + <floatcomplexs_st> value


# We need this fused data-type to omit complex data-types
ctypedef fused reals_st:
    int
    long long
    float
    double

ctypedef fused numerics_st:
    int
    long long
    float
    double
    float complex
    double complex

ctypedef fused _type2dtype_types_st:
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    int
    long long
    float
    double
    float complex
    double complex


cdef object type2dtype(const _type2dtype_types_st v)


ctypedef fused _inline_sum_st:
    int16_t
    int32_t
    int64_t
    uint16_t
    uint32_t
    uint64_t
    int
    long long


cdef Py_ssize_t inline_sum(const _inline_sum_st[::1] array) noexcept nogil
