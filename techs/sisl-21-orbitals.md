# sisl-21 · Orbitals

Owns: `src/sisl/_core/orbital.py`, `src/sisl/_core/_ufuncs_orbital.py`

Paths are relative to `src/sisl/`. All classes are in `_core/orbital.py`.

## Classes

| Class | Base | Content |
|---|---|---|
| `Orbital` | — | `R`, `q0`, `tag`. No radial function. `psi` raises `NotImplementedError`. |
| `SphericalOrbital` | `Orbital` | `l` and one radial function. `m` is an argument of `psi(r, m=0)`. |
| `AtomicOrbital` | `Orbital` | `n`, `l`, `m`, `zeta`, `P`. Holds a `SphericalOrbital` in `orb` (composition). |
| `HydrogenicOrbital` | `AtomicOrbital` | Analytic hydrogen radial function for charge `Z`. |
| `_ExponentialOrbital` | `Orbital` | Private base. `n`, `l`, `m`, `alpha`, `coeff`. No `zeta`, no `P`. |
| `GTOrbital` | `_ExponentialOrbital` | `r^l Σ c_i exp(-α_i r²)`. `α` in 1/Å². |
| `STOrbital` | `_ExponentialOrbital` | `r^(n-1) Σ c_i exp(-α_i r)`. `α` in 1/Å. |

- `AtomicOrbital` sends `radial`, `psi`, `spher` and `psi_spher` to `self.orb` with its own `m`.
- `SphericalOrbital.toAtomicOrbital` makes one `AtomicOrbital` for each `m`. Each gets `q0 / (2l + 1)`.
- An atom must hold `AtomicOrbital`s for grid projection.
  A `SphericalOrbital` on an atom gives only `m = 0`.

## Cutoff `R`

`Orbital.__init__` sets `R`:

- `R=None` becomes `-0.9999`.
- With a radial function: a negative `R` becomes `{"contains": -R}`.
  A `dict` goes to `radial_minimize_range`. It finds the smallest `R` that contains the fraction `contains` of `∫|f(r)|`.
- Without a radial function: a `dict` gives a warning. A negative `R` stays.

A negative `R` means "no extent in real space".

- `toGrid` raises an error.
- `wavefunction` skips the orbital.
- `scale` keeps it at `-1`.

`AtomicOrbital.R` is `orb.R`. `Atom.maxR()` is the maximum `R` of its orbitals.

## Radial functions

`SphericalOrbital.set_radial` accepts:

- nothing: a zero function;
- a callable `f(r)`;
- `(r, f)`: a cubic interpolating spline (`UnivariateSpline(k=3, s=0, ext=1)`). Outside the data it is zero.
  `interp=` selects another interpolator.

`radial(r)` is zero for `r > R`.
sisl does not normalize user radial functions or GTO/STO coefficients.

## Quantum numbers

`AtomicOrbital` accepts positional `n, l, m[, zeta[, P]]`, keywords, or a name such as `"2pxZ2P"`.

- Without `n`: from a name, `s, p, d, f, g` give `n = 1, 2, 3, 4, 5`. Otherwise `n = l + 1`.
- Defaults: `m = 0`, `zeta = 1`, `P = False`.
- Checks: `n >= 1`, `zeta >= 1`, `l < 8`, `|m| <= l`, `orb.l == l`.
- `l < 8` is the size of the `_rspher_harm_fact` table.
- `name()` makes the name string. Only one digit after `Z` is parsed.

## Spherical harmonics

sisl uses real spherical harmonics only (`_rspherical_harm`). The convention is the Siesta convention.

- `m = 0`: `N P_l^0`.
- `m > 0`: `N √2 P_l^m cos(mθ)`.
- `m < 0`: `-(-1)^m N' P_l^m sin(mθ)`.

The angle names are different from the common physics names:

- `θ` is the azimuthal angle in the xy plane, from x.
- `φ` is the polar angle from z.
- `cos_phi=True` gives `cos(φ)` directly. This saves a `cos` call.

Real orbital labels: p is `(y, z, x)` for `m = -1, 0, 1`. d is `(xy, yz, z2, xz, x2-y2)`.

## Equality, copy and pickle

- `equal` compares `tag` exactly. It compares `R` and `q0` with `atol = 1e-4`.
  `radial=True` and `psi=True` also compare sampled function values.
- `copy` is registered in `_core/_ufuncs_orbital.py`. `SphericalOrbital.copy` shares the radial callable.
- `SphericalOrbital` pickles the radial function as 1000 samples on `[0, R]`.
- `AtomicOrbital` pickles `name()` and the samples.

## Add an orbital type

1. Subclass `Orbital`, `AtomicOrbital` or `_ExponentialOrbital`.
2. Define `_radial` before `super().__init__`. Then `R` as a `dict` works.
3. Implement `psi_spher(r, theta, phi, cos_phi)`. The grid projection uses it.
4. Implement `__hash__`, `equal`, `__getstate__` and `__setstate__`.
5. Register `copy` (and `scale` if necessary) with `register_sisl_dispatch` in `_core/_ufuncs_orbital.py`.
6. Add tests for pickle, `copy` and `equal`.

## Known defects

These are confirmed by running the code (2026-10-14). Do not copy the patterns.

- `GTOrbital` and `STOrbital` cannot be pickled. `_ExponentialOrbital` has no `__getstate__`.
- `scale` is registered only for `Orbital`. It calls `cls(R, q0, tag)`. This fails for `AtomicOrbital`.
- `SphericalOrbital.__hash__` and `AtomicOrbital.__hash__` hash `super(Orbital, self)`. This is not a value hash.
