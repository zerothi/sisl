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
  It takes `|k|` along one direction. This is `k -> -k` only when the k-points along the
  other directions are symmetric for `k -> -k`.
  Thus `MonkhorstPack` uses TRS only when the `grid()` of each direction is symmetric (a
  displacement of 0 or 1/2, or e.g. 1/8 for 4 k-points). Other grids have no `k`, `-k` pairs,
  so they cannot be reduced. Rejected: fold only one displaced direction. That is correct,
  but it does not remove k-points, and it moves them to `-k`.
- `MonkhorstPack` does not store the map from the reduced to the full grid.
  `MonkhorstPack.unpack_trs` makes this map again from `_diag`, `_displ`, `_size` and `_centered`.
  It finds the full grid index of each `k` and `-k` along each axis.
  A full point that is also a reduced point maps to that point, without conjugation.
  The multiplicity is the number of full points for each reduced point (`np.bincount`).
  It is not always 2, so do not use a constant.
  The weights of the full grid must sum to the reduced weights; if not, the method raises `SislError`.
  This check finds grids changed by `replace`.

### Adaptive Brillouin zones

`AdaptiveBrillouinZone` and `GaussKronrodGrid` are in `physics/_brillouinzone_adaptive.py`.

- `AdaptiveBrillouinZone` is an abstract subclass of `BrillouinZone`.
  It keeps `set_parent`, `tocartesian`, `volume` and the `isinstance` checks of the self-energies.
- An adaptive grid has no k-points before an integration. Thus `k`, `weight` and `__len__` raise `TypeError`.
  We rejected these alternatives:
  - `len == 0`: `apply.average` then returns an empty result without an error.
  - The length of the initial grid: `apply` then integrates on the coarsest grid without an error.
  - The grid of the last integration: each energy needs a different grid.
    A stored grid also makes the object depend on the call order, and a parallel pool cannot share it.
- `TypeError` is the Python protocol for "no length". The `BrillouinZone` apply modes call `len(bz)`, so they fail at once.
  `__bool__` returns `True`, because `bool()` otherwise calls `__len__`.
- `AdaptiveBrillouinZone.apply` is a separate `BrillouinZoneDispatcher`, not a child of `BrillouinZone.apply`.
  `MonkhorstPack` shares the dispatcher of `BrillouinZone`, so a registration on one is also on the other.
  - `average` (`AdaptiveAverageApply`) calls `integrate` with the parent method (or function) and `wrap`.
    Thus the self-energies (`RealSpaceSE`, `RealSpaceSI`) use an adaptive grid without changes, through `bz.apply.average`.
    The error is measured on the wrapped value, because that is the value that is integrated.
  - `wrap` gets `weight=None`. A node has more than one weight (Kronrod, and the error weights), and the
    Kronrod weight of a box is not final until the box is not halved.
  - `eta` shows a progress bar without a total, because the number of k-points is not known.
  - `pool` warns and runs serially. The boxes are refined one by one.
    The nodes of a box can run in parallel, but in 1D that is only 15 to 30 k-points.
  - All other `BrillouinZone` modes (also the default `iter`) are registered as `AdaptiveUndefinedApply`.
    They raise `TypeError` and refer to `last_bz().apply.<mode>`.
    `sum` is not a weighted sum, so its value depends on the number of k-points that the refinement selects.
    The other modes return a value for each k-point, so they need fixed k-points.
    We rejected to use `last_bz()` in these modes, for the same reasons as for `__len__` (call order, pools).
- The self-energies move the k-points for a k-point that is not zero (`bz._k += k`).
  An adaptive grid has no `_k`, so they raise `NotImplementedError` for an adaptive grid and a k-point that is not zero.
- `integrate(func)` and `last_bz()` are the abstract methods.
  `last_bz()` returns a plain `BrillouinZone` with the k-points and weights of the last `integrate` call.
  Use it with `apply` when you need more quantities on the same grid.
- `integrate` stores only the data that is necessary to create these k-points again (`_last`).
  `last_bz()` creates the `BrillouinZone` only on request. Most integrations do not need it.
  `GaussKronrodGrid` stores the lower bound and the width of each box, and the TRS weight scale.
  `_box_kw` makes the k-points and weights both in `integrate` and in `last_bz`.
  Thus `last_bz` gives the same values, and its weighted sum is equal to the integral.
- We replaced the old `integrate(..., ret_bz=True)`. It always created the k-points, also when nobody used them.
- `last_bz` has only the k-points of the final result.
  `GaussKronrodGrid` also evaluates the k-points of a box before it halves the box. These are not in `last_bz`.
- `_last` is state of the object. It changes with each `integrate` call.
  Call `last_bz` before the next `integrate` call on the same object.
- The weights sum to 1, as for `MonkhorstPack`. The result is the Brillouin zone average.
- `AdaptiveBrillouinZone.__init__` takes only `parent`.
  Other adaptive grids can need other arguments, so the directions, limits, tolerances and `trs` are in the sub-class.
  The base class has only the helpers that do not depend on these arguments:
  `_not_converged` (the warning) and `_str_attrs` (the attributes that `__str__` shows).
- Each sub-class must work with only `parent` (all other arguments have defaults).
  The common tests are parametrized over the sub-classes, and call `cls(parent)`.

`GaussKronrodGrid`:

- `nkpt` has one value for each lattice direction (scalar or `(3,)`), as `nkpt` of `MonkhorstPack`.
  `nkpt[a]` is the initial number of k-points along `a`, per reciprocal lattice vector.
  `nkpt[a] == 0` means that `a` is not integrated (`k_a = 0`).
- A scalar `nkpt` is set to 0 along the non-periodic directions (`parent.pbc`, as
  `BrillouinZone.volume`). Each integrated direction multiplies the k-points by 15 or more,
  so the default must not integrate a slab or molecule direction. Without a lattice to check,
  all 3 directions are integrated; with no periodic direction, it raises (give `nkpt` per direction).
  An explicit `(3,)` `nkpt` is used as given. Rejected: `nsc > 1`, because `Lattice(1)` has
  `nsc == 1` and `func(k)` need not come from a sparse matrix.
  `MonkhorstPack` keeps a scalar for all 3 directions (existing behaviour).
  We merged the old `axes` argument into it, because the two arguments had a 1-1 correspondence.
- The user gives k-points, not boxes, because k-points are the unit of `MonkhorstPack` and of `max_k`.
  The initial number of boxes is `_n = max(1, ceil(nkpt * width / 15))`, with `width` from `_domain()`.
  Thus `nkpt` is rounded up to a multiple of 15. With TRS, the half direction gets half the k-points,
  so `nkpt` is a density for the full direction (as for `MonkhorstPack`).
  `_axes` (the integrated directions) and `_n` (their boxes) are derived from `nkpt`.
- The error of an array result is its largest absolute element.
  The tolerance is `max(atol, rtol * max|I|)`.
- An integrand can return a sequence (`list`, `tuple`, `oplist`, not `ndarray`).
  `_sums_new` then makes sums for each element (also nested sequences), and `_sums_get` returns an `oplist`.
  This is the same as `bz.apply.average` with `oplist`: the elements can have different shapes.
  `_maxabs` takes the maximum over all elements. Thus the box error, the error of each direction
  and `max|I|` for `rtol` include all elements, and the most difficult element decides the refinement.
  We did not use a tolerance for each element.
  The box error is one number for the heap, and a tolerance for each element needs the final value of each element.
  Elements with very different magnitudes need `atol`, or the user must scale them.
- A sequence must have the same length at all k-points, else `_sums_add` raises `ValueError`.
- When `max_k` stops the integration, `integrate` warns and returns the last value.
- A NaN or inf in the integrand gives a NaN or inf error. The loop condition is
  `not (err <= tol and isfinite(err))`, because `err > tol` is false for NaN, and `inf <= inf`
  is true when the value (and thus `rtol * max|I|`) is inf. `integrate` then warns and stops
  at once: halving does not remove a NaN, and a NaN breaks the order of the box heap.
- A scalar `max_k` is the total number of integrand evaluations (`_max_k`).
  A `(3,)` sequence is a k-point density for each lattice direction (`_max_k_axes` keeps the integrated directions).
  An adaptive grid has no fixed number of k-points along a direction. Thus a density is the only well-defined limit for each direction.
  A sequence does not limit the total, but the densities limit it to approximately `prod(max_k)`.
- `trs` follows `MonkhorstPack`: the default comes from `parent.spin`, and `_trs` is the direction index or -1.
  The self-energies read `bz._trs >= 0`, so do not rename it.
- With TRS, `_domain()` gives `[0, 1/2]` along the first integrated direction, and the weights are scaled to sum to 1.
  This is the same contract as `MonkhorstPack`: the caller must symmetrize the result (for example `(G + G.T) / 2`).
  An odd integrand gives `2 * integral(0, 1/2)`, not 0. A test checks this.

- It uses the G7-K15 rule of QUADPACK (`qk15`). The tests check the polynomial degree of the constants.
  Other rule orders need the Kronrod nodes, and sisl does not compute them.
- A box uses the tensor product rule, with `15**d` points for `d` integrated directions. Use it in 1 or 2 dimensions.
- The error of a box is `|K - G|` with the full Gauss tensor rule. This is the error of G7.
  It is larger than the error of K15 (often by orders of magnitude), so the result is more accurate than the tolerance.
  We do not use the QUADPACK scaling `(200 |K - G| / resasc)**1.5`, because it can underestimate the error.
- The box is halved along the axis with the largest error `|K - Q_a|`.
  `Q_a` uses the Gauss weights along axis `a` and the Kronrod weights along the other axes.
  This needs no new integrand values.
- `_box` does not store the integrand values. `_WeightedSums` adds each value at once to the sums
  with the weights `wK`, `wK - wG` and `wK - wQ_a`, then the value is dropped.
  A box thus needs `6` values (`6 + d` for `d > 1`): the sums, a compensation, a line sum and 2 scratch buffers.
  The sums use BLAS `axpy` (one pass, in place) for single and double precision, and `numpy` for other data-types (e.g. `longdouble`).
  For a large value it is faster than the old `np.stack` and `tensordot`, for a scalar it costs approximately 8 µs for each k-point.
  Before, `np.stack` copied the `15**d` values (`2 * 15**d` values in memory, about 7000 for `d = 3`).
  We rejected these alternatives:
  - Pre-allocate the `15**d` values: this only removes the copy, the memory still scales with `15**d`.
  - Nested sums along each direction (as `tensordot`): `d` sums for each weight, and no better precision than compensation.
- The integrand can return the same array at each call (a buffer changed in place).
  The stored list had `15**d` references to the last value, and gave wrong results. A test checks this.
- The errors are sums with the weights `wK - wG` (not `K - G` of two sums).
  `K - G` cancels: its round-off is `eps * |K|`, and it is not the error when the box has converged.
- The Kronrod sum of a box is a compensated sum (Knuth TwoSum). A plain sum of `15**d` values lost
  10 `eps` in 3D, the old `tensordot` did not. The error sums are plain, they are estimates.
  Each line of 15 values (the last direction) is summed plainly, and then added with TwoSum.
  TwoSum for each value took 8 array passes, and made large values 2 times slower.
- The integral and the error are running sums. A split adds the two children and subtracts the parent.
  We do not sum all boxes again: it costs `N_boxes` values of `func` for each split.
  The running integral is a compensated sum (`_two_sum`), and the error a scalar Neumaier sum (`_fsum_add`).
  The terms are added one by one. A coarse box can overestimate a peak by orders of magnitude, so
  `(children - parent)` cancels, and a plain running sum lost `1e-13` (relative). A test checks this against `math.fsum`.
- Each box keeps its Kronrod value, because the value is subtracted when the box is halved.
  The memory is thus `N_boxes` values of `func`. We do not evaluate a box again to get this value.
- `_WeightedSums` and `_two_sum` change the data-type to complex when a later value is complex.
  A different shape or sequence length raises `ValueError`.
- With `_max_k_axes`, a box is halved along the axis with the largest error that keeps `15 / (h / 2) <= max_k[a]`.
  A box that cannot be halved along any axis moves to a list of frozen boxes. It stays in the integral and the error.
  When no box can be halved, `integrate` warns.
- The nodes of the 15-point rule are inside the box. Thus with TRS no k-point is on the `k = 0` or `k = 1/2` planes,
  and no weight correction for points on the boundary is necessary (unlike `MonkhorstPack`).

Rejected: Farey grids (Chen *et al.*, Phys. Rev. B 106, 155102 (2022)).

- We implemented and tested them, then removed them. Do not add them again without a new reason.
- The Voronoi weights of the paper make the trapezoid rule on a non-uniform grid (in 1D).
  This rule is not exact for `cos(2 pi k)`. Thus the error decreases as `N**-1.5`.
  A uniform grid of `N` points is exact for all Fourier modes below `N`. Its error decreases exponentially.
- The gaps of order `l` are approximately `1/(q l)` near `p/q`. The largest gaps (`1/l`) are next to Γ.
  For `1/(1.5 - cos 2 pi k)` at `l = 64`, the 2 intervals next to Γ gave all of the error.
- At `l = 128` (5022 k-points), the error was `1.2e-5`. The uniform sub-grid `j/l` (128 k-points) was exact.
- The gain in the paper is for supercell methods, where the cost of `p/q` increases with `q`.
  sisl evaluates each k-point directly, so all k-points have the same cost.
  Nested uniform grids (double `N`) are then better.

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
