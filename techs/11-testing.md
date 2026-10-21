# 11 · Testing

Owns: `*conftest.py`, `src/sisl/_environ.py`

sisl uses `pytest`. The configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.
Tests are next to the code, in `tests/` subfolders of `src/`.

## Run the tests

```bash
# All tests
pytest
# One file
pytest src/sisl/physics/tests/test_hamiltonian.py
# One file, tests that match a name
pytest src/sisl/_core/tests/test_geometry.py -k rotate
# Do not run slow tests
pytest -m "not slow"
```

## Markers

| Marker | Use |
|---|---|
| `slow` | A test that takes a long time. Mark it so that fast runs skip it. |
| `btd` | A test of `sisl_toolbox.btd`. |

## External test files

Many tests use files from the separate repository
[`zerothi/sisl-files`](https://github.com/zerothi/sisl-files).

- Clone it. Set `SISL_FILES_TESTS` to the clone.
- If `SISL_FILES_TESTS` is not set, tests that use the `sisl_files` fixture are `xfail`. They do not run.
- See `src/sisl/conftest.py`.

A green run without `SISL_FILES_TESTS` does not test the IO code. Do not report it as a full pass.

## Optional modules

Some tests need a module that is not always available,
for example `netCDF4` or the Fortran module `sisl.io.siesta._siesta`.
Use the decorator `sisl_requires` from `sisl.conftest`:

```python
from sisl.conftest import sisl_requires

@sisl_requires("sisl.io.siesta._siesta")
def test_fortran():
    ...

# skip all tests in a module
pytestmark = sisl_requires("netCDF4")
```

- `pytest.importorskip` at module level skips all tests in the module.
  `sisl_requires` skips only the decorated test or class.
- `sisl_requires` imports the module, and does not use only `importlib.util.find_spec`.
  An extension that is on disk but does not load (for example, a missing `libgfortran`)
  makes the import fail. Then `sisl.io.siesta.sile` uses the stub, and the test must skip.
- The result of the import is cached. Each module is imported a maximum of one time.
- Inside a test body, the fixture `sisl_importskip` does the same.
- A test that uses a Fortran Sile without `sisl_requires` becomes `xfail`
  when the Fortran sources are not built. See [50 · Siesta and Fortran](50-io.md#siesta-and-fortran).

## Temporary files

Use the fixture `sisl_tmp` for files that a test writes.
Each test gets its own folder: `<basetemp>/<package>/<test-name><N>`, for example
`sisl-io-siesta/test_geom_order0`.

- Do not share a folder between tests. Many readers also read other files in the same folder.
  For example, `fdfSileSiesta.read_basis` reads `siesta.nc` first, if it exists.
  A file from a different test then changes the result.
- On Windows, a file that is open cannot be removed.
  Thus a file handle that is not closed leaves the file in the folder.
- The teardown removes the folder of the test.
  If it cannot remove a file, it emits a `UserWarning` with the file name.
  Do not ignore this warning. It shows a file handle that is not closed.

## Parallelism

Set `SISL_NUM_PROCS=1` when you debug. This stops the parallel execution.
The tox configuration does this.

## What a new test must do

- A bug fix has a test that fails before the fix.
- A new feature has tests for the normal case and for the edge cases.
- Compare floating point results with a tolerance that matches the dtype.
- A performance-critical change has a regression test. See [12 · Performance](12-performance.md).
