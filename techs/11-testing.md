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

## Parallelism

Set `SISL_NUM_PROCS=1` when you debug. This stops the parallel execution.
The tox configuration does this.

## What a new test must do

- A bug fix has a test that fails before the fix.
- A new feature has tests for the normal case and for the edge cases.
- Compare floating point results with a tolerance that matches the dtype.
- A performance-critical change has a regression test. See [12 · Performance](12-performance.md).
