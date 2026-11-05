# sisl-40 · IO

Owns: `src/sisl/io/*`

Paths are relative to `src/sisl/`.
A Sile is a sisl file object. It reads and writes one file format.

## Registration

`add_sile(suffix, cls, case=True, gzip=False)` (`io/sile.py`) registers a Sile class for a suffix.

- The call is at the end of the module of the format. Import of the module registers the class.
- `gzip=True` also matches `<suffix>.gz`.
- `case=False` matches the suffix in all letter cases.

## Look-up

`get_sile_class` (`io/sile.py`) finds the class for a file name:

- It tries the extensions from the longest to the shortest. Thus `grid.nc` wins over `nc`.
- `file{spec}` selects from the candidates, for example `{contains=Siesta}`.
  Without a spec, it uses `SISL_IO_DEFAULT`.
- When many classes match, it opens the file with each class and tries the `read_*` methods.
  If the result is still not one class, it raises `ValueError`.

`get_sile` removes the `{spec}`. It also accepts `zipfile.Path`.

## Base classes

| Class | Use |
|---|---|
| `BaseSile` | The base of all Siles. |
| `Sile` | Text files. Opens `.gz` files with `gzip.open`. Skips comments in `readline`. |
| `SileCDF` | NetCDF files. `lvl` sets the compression level. |
| `SileBin` | Binary files. Adds `"b"` to the mode. |
| `SileSiesta`, `SileCDFSiesta`, `SileBinSiesta` | The same for Siesta files. |

- `BaseSile.__init_subclass__` makes a `<Name>Buffer` class for each Sile.
  `BaseSile.__new__` uses it when the input is an `IOBase` buffer.
- `Info` and `InfoAttr` give attributes from regular expressions. They update while the Sile reads lines.

## Method conventions

- `read_<thing>` and `write_<thing>` are the public methods.
- `read(**kw)` and `write(**kw)` call `read_<key>` and `write_<key>`.
- `_r_<thing>` is an internal reader.
- Override `_setup()` for set-up. Do not override `_base_setup`.
- `__getattr__` sends unknown attributes to `self.fh`.

### `sile_fh_open`

`@sile_fh_open(from_closed=False, reset=None)` opens the file around a `read_`/`write_` method.

- If the file is open, it does not open it again. Thus calls can nest.
- `from_closed=True` starts from the start of the file. It calls `reset(self)` first.

### `SileBinder`

`@SileBinder()` (`io/_multiple.py`) makes a reader sliceable: `sile.read_geometry[0:5]`.
A `skip_func` (for example `_r_geometry_skip`) skips frames without parsing them.

## Add a file format

Follow `io/xyz.py`:

1. Subclass `Sile`, `SileCDF` or `SileBin`.
2. Implement `read_<thing>` and `write_<thing>`. Decorate them with `@sile_fh_open()` or `@SileBinder()`.
3. Call `add_sile("<ext>", <cls>, ...)` at the end of the module.
4. Import the module in the `__init__.py` of the package.
5. Add tests. Put test files in `sisl-files`. See [dev-02 · External test files](dev-02-testing.md#external-test-files).

## Siesta and Fortran

The f2py module `sisl.io.siesta._siesta` holds the Fortran readers and writers
(`io/siesta/_src/*.f90`: HSX, TSHS, DM, GF, grid, TSDE, WFSX).

- CMake builds it only with `WITH_FORTRAN=ON`.
  It is on by default, except on Windows. The pyodide build sets it off.
- Without the module, `_siesta` is a `_MissingFortranModule` stub.
  The binary Siles stay registered. Use of one raises `MissingFortranSileError`.
- `SileBinSiesta` manages Fortran unit numbers through `_siesta.io_m`.
- `io/siesta/_help.py` converts the Siesta sparse layout and spin layout to sisl
  (`_csr_from_siesta`, `_csr_to_siesta`, `_mat_siesta2sisl`).

## DFTB Slater-Koster files

`skfSileDFTB.read_slater_koster` (`io/dftb/skf.py`) makes a `SlaterKoster` from `A-B.skf`.
The format is from the v1.0 format document of DFTB+. That document does not tell the orientation,
the sign, the interpolation or the Fortran syntax. These come from the code of DFTB+.

- The table has `nGridPoints - 1` rows. Row `k` is at `r = k gridDist`.
  A file can have more rows (`mio` `O-H.skf`: 518 rows for `nGridPoints = 500`).
  DFTB+ reads only `nGridPoints - 1` rows, thus sisl does the same.
- DFTB+ reads each line with the list-directed read of Fortran. Thus `,` is a separator,
  `n*value` is `n` times `value`, and line 1 can have more fields (`mio`: `0.02, 500,2`).
- The extended format (`@`, f shells) raises: `SlaterKoster` stops at `l=2`.
- In `A-B.skf`, the shell with the lower `l` is on `A`. The sign is that of Slater and Koster, as in sisl:
  `Hsp` has `s` of `A` at the origin and `p` of `B` along `+z`. Thus the reader changes no sign.
  The `mio` `H-O.skf` agrees: at 1.7 Bohr, `Hsp > 0` and `Ssp < 0`.
- A heteronuclear pair needs both files. DFTB+ takes the equal-`l` integrals of `A` to `B` from `A-B.skf`,
  and of `B` to `A` from `B-A.skf`, without a check. A `SlaterKoster` has one integral for both directions.
  Thus one call reads both files, takes the equal-`l` integrals from `A-B.skf`, and raises if `B-A.skf`
  does not agree (relative 1e-10). If `B-A.skf` is not there, the call warns (decided by the maintainer).
- The repulsive energy (polynomial and `Spline`) is not part of `H`. The reader does not read it.

### Interpolation

The reader uses the default interpolation of DFTB+ (`skEqGridNew`). `OldSKInterpolation` is not implemented.

- Inside the grid: the polynomial through 8 grid points. The last point is
  `clip(floor(r / gridDist) + 4, 8, n)`, with `n` rows.
  sisl uses the Lagrange form, DFTB+ the Neville form. The polynomial is the same.
- After the last point: a polynomial of 5th order that goes to 0, with its two derivatives, at 1 Bohr after the last point.
  It starts with the value and the two derivatives of the polynomial at the last point. `R` is `n gridDist + 1` Bohr.
- DFTB+ gets the two derivatives from a finite difference of 1e-5 Bohr. sisl uses the exact derivatives of
  the polynomial. The finite difference of the second derivative has a rounding error of approximately `4 eps / 1e-10`, relative.
  For all files of the `mio` set, the tails differ by less than 9.5e-7 of the last table value.

### Verification

- `test_skf_*` uses tables of known polynomials: units, columns, orientation, sign, tail.
- The `H` and `S` of water from DFTB+ (`files/tests/dftb/h2o`, `mio` set) have no coordinates.
  A fit of the positions to `S` gives `H` to 6.7e-16 Ha and `S` to 1.2e-15 (2026-11-05, outside the repository:
  the `mio` files are not in sisl-files).
  The fit hides a scale of all distances and an inversion (the sign of `sp`). Thus the tests check those.
- `SlaterKoster.from_hamiltonian` on that `H` gives the integrals of the read model at the bond lengths to 5e-15 eV.

Sources:
FILE[10]: slakoformat.pdf (Format of the v1.0 Slater-Koster files, DFTB+ developers)
URL[9]: https://github.com/dftbplus/dftbplus/blob/main/src/dftbp/dftb/slakoeqgrid.F90
URL[9]: https://github.com/dftbplus/dftbplus/blob/main/src/dftbp/dftb/sk.F90
URL[8]: https://github.com/dftbplus/dftbplus/blob/main/src/dftbp/type/oldskdata.F90
URL[8]: https://github.com/dftbplus/dftbplus/blob/main/src/dftbp/math/interpolation.F90
URL[7]: https://github.com/dftbplus/dftbplus/blob/main/src/dftbp/dftbplus/parser.F90
URL[6]: https://github.com/dftbplus/dftbplus/blob/main/src/dftbp/dftb/nonscc.F90
DOI[5]: 10.1103/PhysRev.94.1498
