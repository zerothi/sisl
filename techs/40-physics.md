# 40 · Physics

Owns: `src/sisl/physics/*`, `src/sisl/constant.py`, `src/sisl/unit/*`, `src/sisl/_core/oplist.py`

Paths are relative to `src/sisl/`.

## Class layering

| Class | Base | File |
|---|---|---|
| `Hamiltonian` | `SparseOrbitalBZSpin` | `physics/hamiltonian.py` |
| `DensityMatrix`, `EnergyDensityMatrix` | `SparseOrbitalBZSpin` (shared code in `_densitymatrix`) | `physics/densitymatrix.py`, `physics/energydensitymatrix.py` |
| `DynamicalMatrix` | `SparseOrbitalBZ` | `physics/dynamicalmatrix.py` |
| `Overlap` | `SparseOrbitalBZ` | `physics/overlap.py` |

For the storage and the overlap index (`S_idx`), see [20 · Spin, overlap and dtype](20-sparse.md#spin-overlap-and-dtype).

## Method selection in `_reset`

`_reset()` binds `Pk`, `Sk`, `dPk` and `ddPk` on the instance.
The choice depends on the spin kind and the orthogonality.
Thus there are no subclasses for each spin kind.

- `SparseOrbitalBZSpin._reset` selects `_Pk_polarized`, `_Pk_non_colinear`, `_Pk_spin_orbit` or `_Pk_nambu`.
- `Hamiltonian._reset` sets `Hk = Pk`, `dHk = dPk`, `ddHk = ddPk`.
  `DynamicalMatrix` does the same for `Dk`.
- `def Hk(...)` in the class has only a doc-string. The instance attribute hides it.
  Keep the doc-string there. Sphinx reads it.

## k-space matrices

All k-space matrices go through `_Pk`. It calls the Cython `matrix_k` (`physics/_matrix_k.pyx`).
Non-colinear, spin-orbit and Nambu have their own kernels (`matrix_k_nc`, `matrix_k_so`, `matrix_k_nambu`).

### Gauge

`GaugeType = Literal["lattice", "atomic"]` (`typing/_physics.py`). The default is `"lattice"`.
`comply_gauge` (`physics/_common.py`) changes aliases to these two names.

| Gauge | Aliases | Phase |
|---|---|---|
| `lattice` | `R`, `cell` | `exp(2πi k · sc_off)` |
| `atomic` | `r`, `orbital(s)`, `atom(s)` | `exp(2πi k · R_ij)` from `M.Rij()`. Calls `M.finalize()` first. |

At Γ, sisl applies no phase.

### dtype

`phase_dtype` (`physics/_phase.pyx`) selects the dtype of the result:

- At Γ, the dtype of the matrix stays, unless the kernel forces complex.
- At other k, `float32` changes to `complex64` and `float64` changes to `complex128`.
- The spin kernels force a complex result.

### format

- `"csr"` is the default.
- `"array"`, `"matrix"` and `"dense"` give a numpy array.
- Other scipy formats go through `.asformat`.
- `"sc"` or `"sc:<format>"` gives the supercell matrix with shape `(no, no * n_s)`.

## Spin

`Spin` (`physics/spin.py`) has the kinds `UNPOLARIZED`, `POLARIZED`, `NONCOLINEAR`, `SPINORBIT`, `NAMBU`.

- The kinds are ordered. You can compare them with `<`.
- `size(dtype)` depends on the dtype. See [20 · Spin, overlap and dtype](20-sparse.md#spin-overlap-and-dtype).
- `spinor` is 1, 2, or 4 (Nambu).
- `io/siesta/_help.py:_mat_siesta2sisl` converts the Siesta spin layout to the sisl layout.

## Brillouin zones

`BrillouinZone`, `MonkhorstPack` and `BandStructure` are in `physics/brillouinzone.py`.

- `bz.apply` is a `ClassDispatcher`. It sends unknown attributes to `bz.parent`.
- The apply methods are registered in `physics/_brillouinzone_apply.py`:
  `iter` (default), `average`, `sum`, `array`/`ndarray`, `none`, `list`, `oplist`, `dataarray`/`xarray`.
  `grid` is only for `MonkhorstPack`.
- `physics/__init__.py` imports this module after the physics classes.
- Parallel execution uses `pathos` when it is installed. It maps the k-points with `pool.imap`.
  Set the pool with `bz.apply.renew(pool=...)`.
- `MonkhorstPack.grid()` handles the time-reversal symmetry reduction.

## States

`physics/state.py`:

- `ParentContainer` has `parent` and `info`. `info` holds `k`, `gauge`, `format` and `spin`.
- `Coefficient`, `State` and `StateC` (state with coefficients) are subclasses.
- `State.change_gauge` reads and updates `info["gauge"]`.

`physics/electron.py`:

- A mixin `_electron_State` with `State` or `StateC` gives `StateElectron` and `StateCElectron`.
- `EigenvalueElectron`, `EigenvectorElectron` and `EigenstateElectron` follow the same pattern.
- `Hamiltonian.eigenstate` returns `EigenstateElectron(v.T, e, ...)`. The states are rows.

### Functions and methods

The physics is in module functions that take arrays: `DOS`, `PDOS`, `COP`, `spin_moment`, `ahc`, `shc`, `berry_phase`, `wavefunction`.
The methods on the state classes only call these functions with their own data.
Some operations use `register_sisl_dispatch` instead (for example `velocity` in `physics/_ufuncs_electron.py`).

Put new physics in a module function. Add a thin method when a state class needs it.

## Bloch expansion

`Bloch` (`physics/bloch.py`) separates the k-points from the evaluation of the matrix:

1. `unfold_points(k)` gives the k-points of the small cell.
2. The caller evaluates the matrix at each k-point.
3. `unfold(M, k)` makes the large matrix (Cython `bloch_unfold`).

`__call__` does all three steps. Its `indices` argument makes only a sub-matrix. This saves memory.

## Units and constants

- The internal units are eV and Å.
- `PhysicalConstant` (`constant.py`) is a `float` with a unit. `C.hbar("eV ps")` converts it.
- `SISL_CODATA` selects the CODATA table (`unit/codata.py`).
- `SISL_UNIT_SIESTA` selects the Siesta unit table (`unit/siesta.py`).
- Put derived constants at module level (for example `_velocity_const` in `physics/electron.py`).
