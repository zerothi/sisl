# 20 · Sparse matrices

Owns: `src/sisl/_core/sparse.py`, `src/sisl/_core/_sparse*`, `src/sisl/_core/sparse_geometry.py`, `src/sisl/_core/_ufuncs_sparse*`, `src/sisl/_core/_dtypes.p*`, `src/sisl/_indices.p*`, `src/sisl/physics/sparse.py`, `src/sisl/physics/_matrix_*`, `src/sisl/physics/_phase.p*`

Paths are relative to `src/sisl/`.

## Class hierarchy

| Class | File | Rows | Purpose |
|---|---|---|---|
| `SparseCSR` | `_core/sparse.py` | any | A general CSR container with an extra dimension. It has no geometry. |
| `_SparseGeometry` | `_core/sparse_geometry.py` | `_size` | Holds a `SparseCSR` in `_csr` and a `Geometry`. |
| `SparseAtom` | `_core/sparse_geometry.py` | atoms | One row for each atom. |
| `SparseOrbital` | `_core/sparse_geometry.py` | orbitals | One row for each orbital. |
| `SparseOrbitalBZ` | `physics/sparse.py` | orbitals | Adds the overlap, `Pk`, `Sk`, `dPk`, `ddPk` and `eig*`. |
| `SparseOrbitalBZSpin` | `physics/sparse.py` | orbitals | Adds spin. `Hamiltonian` and `DensityMatrix` are subclasses. |

`_SparseGeometry.__getattr__` sends unknown attributes to the geometry.
`SparseCSR` supports numpy ufuncs through `__array_ufunc__`.

## Storage layout

`SparseCSR` has a shape of `(M, N, K)`. It is always 3-D.
K is the extra dimension (spin components, overlap).

| Array | Shape | Content |
|---|---|---|
| `ptr` | `M+1` | Start of each row in `col` and `_D`. |
| `ncol` | `M` | Number of used entries in each row. |
| `col` | allocated | Column indices. |
| `_D` | `(allocated, K)` | Values. The K values of one entry are adjacent in memory. |

`_nnz` is the number of used entries.

### Rows have holes

The used entries of row `r` are `ptr[r] : ptr[r] + ncol[r]`.
Thus `ncol[r] <= ptr[r+1] - ptr[r]`.
The other slots are holes. A hole has `col == -1` and `_D == 0`.

Reason: you can add an entry to a row without a shift of all the rows after it.

- A new row gets `nnzpr` slots.
- When a row is full, `_extend` inserts `max(10, needed)` slots. Then it shifts `ptr` after the row.
- `_extend` appends new columns. It does not sort them.
- `_D` starts with zeros. Thus you can set the K components one at a time.

### Index dtype

`ptr`, `ncol` and `col` always have the same dtype: `int32` or `int64`.
All setters use `_as_index`.
When a value does not fit in `int32`, all three arrays change to `int64`.
The Cython fused type `int_sp_st` (`_core/_dtypes.pxd`) matches this.

Use `_a.repeat`, not `numpy.repeat`, when the counts are `ncol` (or other index arrays).
`numpy.repeat` casts the counts to `intp` with the `safe` rule.
On a 32-bit platform (pyodide) `intp` is `int32`, thus `int64` counts fail.
`_a.repeat` casts the counts without a check. The cast cannot overflow:
each count is not larger than the result length, and the result length must fit in `intp`.
When `intp` has 64 bits, `_a.repeat` is `numpy.repeat`. Thus 64-bit platforms do not pay for the cast.
`test_repeat_intp` tests the 32-bit path on all platforms.

### Row index

The internal row methods (`_extend`, `_get`) accept only one scalar row.
Do not add support for array rows. The maintainer will add it later.

## Finalize

`finalize(sort=True)`:

1. Removes the holes.
2. Makes `ptr = [0, cumsum(ncol)]`.
3. Sorts the columns of each row.
4. Raises `SislError` for duplicate `(i, j)` entries.
5. Warns when a column is outside `shape[1]`.

"Finalized" means "without holes AND sorted".
`finalize(sort=False)` removes the holes but `finalized` stays `False`.

Only the grow branch of `_extend` sets `_finalized = False`.
Reason: a finalized matrix has no holes. Thus the first insertion after `finalize` always grows.
`_extend_empty`, `__delitem__` and `translate_columns` also set `_finalized = False`.

Fast paths use `ptr[-1] == nnz` as a test for "no holes".

A matrix made from a scipy matrix or from `(data, indices, indptr)` is finalized only if each row has 0 or 1 entries.
Reason: sisl does not trust the order of external input.

## Supercell columns

`N = size * n_s`. `size` is the number of rows (atoms or orbitals). `n_s` is the number of supercells.

A column index is `col = isc * size + uc`:

- `col % size` is the index in the unit cell.
- `col // size` is the supercell index. `sc_off[isc]` is its offset.

`M[i, j, (1, 0, 0)]` gives the column `sc_index((1, 0, 0)) * size + uc(j)`.

- `set_nsc` moves blocks of columns with `translate_columns`. It deletes supercells that are not in the new range.
- `transpose` maps supercell `isc` to the index of `-sc_off[isc]`.
- `SparseCSR` does not check N. Without a shape, `N = (max(col) // M + 1) * M`.

## Spin, overlap and dtype

K depends on the spin and on the dtype (`Spin.size(dtype)` in `physics/spin.py`):

| Spin | Real dtype | Complex dtype |
|---|---|---|
| unpolarized | 1 | 1 |
| polarized | 2 | 2 |
| non-colinear | 4 | 3 |
| spin-orbit | 8 | 4 |
| Nambu | 16 | 8 |

A complex dtype stores a real and imaginary pair in one complex component.

- A non-orthogonal basis adds one component for the overlap. `S_idx = K - 1`.
- An orthogonal basis has `S_idx = -100`.
- `_reset` binds `Pk`, `Sk` and similar methods from the spin and the orthogonality.
- `_def_dim` selects one component for the next `__getitem__` or `__setitem__`.
  `M.S[...]` sets it to `S_idx`. The next access resets it to `-1`.

## Cython and Python

The logic of `SparseCSR` is in Python and numpy (`_core/sparse.py`).
Cython supplies the kernels:

| Module | Content |
|---|---|
| `_core/_sparse.pyx` | `sparse_dense`, `fold_csr_array`, `fold_csr_array_diag`, `ncol2ptr`. |
| `_indices.pyx` | `indices`, `indices_only`: column look-up for `_get` and `_extend`. |
| `physics/_matrix_*.pyx` | k-space kernels for `Pk`, `dPk`, `ddPk`. |

- `_core/_sparse.pyx` has no `.py` companion.
- The k-space kernels loop over `ptr[r] : ptr[r] + ncol[r]`. They work on matrices that are not finalized.
  The exception is `gauge="atomic"`. It calls `finalize()` first.
- The fold functions make sorted, unique columns with `np.unique`. The binary search in `_matrix_phase.pyx` needs this.
- `_core/_sparse_distribute.py` (MPI) is imported lazily. Methods that change the matrix use `@distribute_changes`.
