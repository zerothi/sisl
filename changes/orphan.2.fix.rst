Fixed `SparseCSR.eliminate_zeros` skipping matrices whose only zero was in row 0

The early-return short-circuit summed row *indices* rather than counting zero
elements, so a matrix whose only explicit zero resided in row 0 was left
untouched. The zeros are now detected and removed correctly.
