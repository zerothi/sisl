Sped up sparse matrix element-wise operations, `Rij` and `eliminate_zeros`

`SparseAtom.Rij`/`SparseOrbital.Rij` (and the derived `rij`) now build the
distance vectors in a single vectorized call instead of looping over every
row in Python. Element-wise operations between two sparse matrices (e.g.
``H1 + H2``) no longer allocate a per-row index array, and
`SparseCSR.eliminate_zeros` now compacts all zero elements in one pass rather
than deleting row-by-row.
