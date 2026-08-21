portable complex arithmetic in the fused-type Cython kernels

The k-space matrix kernels are templated on independent fused types for the
matrix data and the phases, so Cython instantiates real/complex combinations
that never occur at runtime. Those instantiations emitted a cast from a
complex to a real value, and in-place ``+=`` on complex memoryview elements,
neither of which compiles unless the C compiler maps complex numbers onto the
native C99 ``_Complex`` type. Both are now routed through the new
``cast_add``/``cast_assign`` helpers in ``sisl/_core/_dtypes.pxd``, which
resolve the real/complex distinction at compile time, so the extensions also
build with the struct-based complex fallback (MSVC) and in C++ mode.
