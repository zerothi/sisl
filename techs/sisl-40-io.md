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
