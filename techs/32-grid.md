# 32 · Grid

Owns: `src/sisl/_core/grid.py`, `src/sisl/_core/_ufuncs_grid.py`, `src/sisl/_sparse_grid*`

Paths are relative to `src/sisl/`.

## Data layout

`Grid.grid` is a 3-D numpy array in C order. The default dtype is `float64`.

- Grid axis `i` is lattice vector `i`.
- `dcell = cell / shape[:, None]` (the voxel vectors, one in each row).
- `dvolume = lattice.volume / size`.
- A `float` as `shape` is a spacing: `shape[i] = rint(|a_i| / spacing)`.
- `Grid` is a `LatticeChild`. The `Geometry` is optional.
  `set_geometry(also_lattice=True)` replaces the lattice of the grid with the lattice of the geometry.

## Index and position

- `index(xyz)` is `floor(icell @ xyz * shape)`. It returns `int32`.
- `index2xyz(i)` is `i @ dcell`.
- Neither method subtracts `lattice.origin`. The grid frame is relative to the origin.
  The caller subtracts the origin (see `wavefunction` and `DensityMatrix.density`).
- `index_fold` wraps indices periodically. `index_truncate` removes indices outside the grid.
- `index(shape)` uses the smaller (by volume) of `shape.to.Cuboid` and `shape.to.Ellipsoid` as a bound.
  Then it filters the voxel corners with `shape.within_index`.

## Boundary conditions

The boundary conditions are on `Lattice` (issue #626). See [30 · Boundary conditions](30-geometry.md#boundary-conditions).
`Grid.PERIODIC` and similar names, the `bc=` argument and `set_bc` are deprecated.

### Poisson

`grid.to.pyamg` makes `pyamg.gallery.poisson(shape)`. It has Dirichlet conditions by default.
`pyamg_boundary_condition` then changes `A` and `b`:

| Condition | Change |
|---|---|
| Periodic | `A[lo, hi] = A[hi, lo] = -1` |
| Neumann | Row set to zero. `A[bc, bc] = 1`, `A[bc, nbr] = -1` |
| Dirichlet | `b = 0` |

`pyamg_fix` and `pyamg_source` fix the value of voxels.
The linear index is C order: `i * n1 * n2 + j * n2 + k`.

## Operations

The operations are registered with `register_sisl_dispatch` in `_core/_ufuncs_grid.py`.

- `interp` uses `scipy.ndimage.zoom`. The default is `mode="wrap"` (periodic).
- `apply` returns a `Grid` only when the result is a 3-D array.
- `sum` and `average` keep the axis with size 1. The cell becomes smaller. The atoms are scaled in fractional coordinates.
- `sub` translates the geometry when the indices are contiguous. Otherwise it scales it.
  `remove`, `sub_part` and `remove_part` use `sub`.
- `tile` requires a geometry that is commensurate with the lattice (tolerance `1e-5`).
- `swapaxes` swaps the lattice axes and copies the data in C order.

## Arithmetic

- `+`, `-`, `*`, `/` and the in-place forms accept a `Grid` or a scalar. There is no `__array_ufunc__`.
- `__eq__` compares only the shape. It does not compare values or the lattice.
  It is a "compatible for arithmetic" test.
- `abs` changes a complex grid to the matching float dtype.

## Fill a grid from physics

Both functions add to the grid in place.

- `wavefunction(v, grid, ...)` (`physics/electron.py`):
  - It raises an error for complex coefficients or complex k on a real grid.
  - It finds the atoms with `geometry.within_inf`.
  - For each atom, it uses a box of indices, spherical coordinates and `orbital.psi_spher`, with the phase `exp(i k · isc)`.
- `DensityMatrix.density(grid, method=...)` (`physics/densitymatrix.py`):
  - `"pre-compute"` (default, experimental): `Geometry._orbital_values` gives a `SparseGridOrbitalBZ`. Then `reduce_orbital_products` adds into `grid.grid`.
  - `"direct"`: a loop over atoms with `grid.index(atom.to.Sphere(...))`.

## Sparse grids

`_sparse_grid.py` and `_sparse_grid_ops.py` are internal. They can change.

- `Sparse4DGrid` is a 3-D grid with one sparse extra dimension. It is a `SparseCSR`.
  The rows are the raveled grid indices. The columns are the extra dimension.
- `SparseGridOrbitalBZ` uses the supercell orbitals as the extra dimension.
  It has `get_overlap_matrix`, `reduce_orbitals` and `reduce_orbital_products`.
- `_sparse_grid_ops.py` holds typed kernels in Cython pure-Python mode.
- Only `Geometry._orbital_values` uses it.

## Known defects

Confirmed by running the code (2026-10-14).

- `append(grid1, grid2, axis)` returns a grid of zeros. It does not copy the data.
