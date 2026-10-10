# Project configuration

## Project overview
- **Name**: sisl
- **Tech Stack**: Python, numpy, Cython, Fortran
- **Team Size**: 5 developers


## What this is

`sisl` is a Python library (with Cython + Fortran extensions) for post-processing DFT output and building tight-binding models for NEGF transport. It couples tightly to LCAO/Siesta codes and feeds tight-binding matrices to TBtrans. Source lives in `src/sisl` (the main package) and `src/sisl_toolbox` (user-contributed toolboxes).

## Source decisions

The heavy computations are offloaded to Cython extensions. The Fortran extensions are
*only* for interoperability with codes that writes files using Fortran codes.

sisl tries to use classes for implementing contained functionality, but should try and
avoid too many sub-classes as it obfuscates understanding.

It's usage is guided by ease of use for beginners in programming, but capable of doing
advanced computations by experienced users. It should strive to be easy to understand the
basics of, but as easily extendable for expert users.

## Build & install

The build system is **scikit-build-core + CMake + Cython + f2py** (see `CMakeLists.txt`, `pyproject.toml`).
It is **not** a pure-Python package — editing `.pyx`/`.f90` files requires a rebuild.

```bash
# Editable/dev install (recommended). Compiles Cython + Fortran extensions.
python -m pip install -e .

# Faster iterative rebuilds: skip build isolation and reuse a persistent build dir
python -m pip install -e . --no-build-isolation -Cbuild-dir=build-editable
```

Useful CMake toggles passed via `SKBUILD_CMAKE_ARGS`:
- `-DWITH_FORTRAN=OFF` (skip Fortran)
- `-DWITH_COVERAGE`
- `-DWITH_ANNOTATE` (Cython HTML annotations).

## Tests

Tests use `pytest`; `testpaths = src`, so tests live next to the code in `tests/` subdirectories throughout `src/sisl` and `src/sisl_toolbox`.

```bash
pytest  # full suite, additional args specified in pyproject.toml
pytest src/sisl/physics/tests/test_hamiltonian.py    # one file
pytest src/sisl/_core/tests/test_geometry.py -k rotate  # single test by pattern
pytest -m "not slow"                      # skip slow-marked tests
```

- **Many tests require external data files** from the separate [`zerothi/sisl-files`](https://github.com/zerothi/sisl-files) repo. Point `SISL_FILES_TESTS` at a clone of it, otherwise tests using the `sisl_files` fixture are `xfail`-ed (not run). See `src/sisl/conftest.py`.
- Set `SISL_NUM_PROCS=1` when debugging to disable parallelism (the tox config does this).

## Lint / formatting

Style is enforced by **pre-commit**. Set it up once with `pre-commit install`. Don't hand-format — let pre-commit fix style on commit.

## Changelog (required for user-facing PRs)

Add a news fragment in `changes/` named `<PR-number>.<type>.rst` (types: `feat`, `fix`, `change`, `highlight`, `doc`, `dev`). These are assembled by **towncrier** into release notes.
Toolbox changes go in `changes/toolbox/`. See `changes/README.rst`.

## Architecture

### Core data structures — `src/sisl/_core`
The foundational objects, re-exported at the top level (`sisl.Geometry`, etc.):
- `Geometry`, `Lattice` (cell/supercell), `Atom`/`Atoms`, `Orbital`/`AtomicOrbital`, `Grid` (real-space grids).
- Sparse matrix stack: `SparseCSR` (`_sparse.pyx`) → `SparseAtom` / `SparseOrbital` (`sparse_geometry.py`). Nearly every physical matrix in sisl is a `SparseOrbital` subclass. Cython `.pyx`/`.pxd` files here are performance-critical hot paths.

### Physics matrices — `src/sisl/physics`
`Hamiltonian`, `DensityMatrix`, `EnergyDensityMatrix`, `Overlap`, `DynamicalMatrix` all subclass the sparse-geometry types. Electronic/phonon analysis in `electron.py`, `phonon.py`, `state.py`; `brillouinzone.py`, `bloch.py`, `self_energy.py`. The `_matrix_k.pyx`, `_matrix_dk.pyx`, `_matrix_ddk.pyx`, `_bloch.pyx`, `_phase.pyx` Cython modules build k-space matrices — this is the compiled numerical core.

### I/O — `src/sisl/io` (the "Sile" system)
File formats are `Sile` classes registered by suffix. `get_sile("foo.fdf")` dispatches to the right class via rules registered with `add_sile(suffix, cls)` (see `io/sile.py`). Base classes: `Sile` (text), `SileCDF` (netCDF), `SileBin` (binary/Fortran). Format families live in subpackages: `siesta/`, `tbtrans/`, `vasp/`, `wannier90/`, `orca/`, etc. When adding a format, create the `*Sile*` class and register it.
One can use declarations for files that share extensions `get_sile("foo.fdf{siesta}")`
ensures it's a classname that has `siesta` in it.

### Dispatch patterns (pervasive — understand before extending classes)
- **`_dispatcher.py`**: `AbstractDispatch` / `ClassDispatcher` / `TypeDispatcher`. Powers namespaces like `geometry.new(...)` (construct from other types) and `geometry.to(...)` (convert to other types, e.g. ASE/pymatgen).
- **`_ufuncs.py`** (`register_sisl_dispatch`): methods are attached to core classes from separate `_ufuncs_*.py` files rather than defined inline in the class body. If a method seems "missing" from a class, grep the `_ufuncs_*.py` files. This keeps large classes split across files.
  `_ufuncs_*.py` provides functional programming style that dispatches depending on first
  argument class-types.

### Other notable modules
- `sisl_toolbox/` — higher-level toolboxes: `btd` (block-tri-diagonal Green functions / `DeviceGreen`), `transiesta`, `siesta`, `models`. Exposed via the `stoolbox` CLI.
- `viz/` — plotting framework (plotly/matplotlib/blender backends), lazily loaded.
- `mpi/` — distribution framework (mpi4py), lazily loaded.
- `_environ.py` — all runtime config is `SISL_*` env vars registered via `register_environ_variable` (e.g. `SISL_NUM_PROCS`, `SISL_FILES_TESTS`, `SISL_SHOW_PROGRESS`, `SISL_IO_DEFAULT`). Use `sisl_environ(...)` context manager to override temporarily.
- `geom/` — geometry generators (graphene, nanotubes, etc.); imported eagerly because its categories are needed for subclass handling.

### CLI entry points (`pyproject.toml [project.scripts]`)
- `sgeom` → geometry conversion/manipulation, `sgrid` → grid ops, `sdata` → generic Sile inspector (`sdata --cite` prints the BibTeX), `stoolbox` → toolbox commands.

## Conventions

- `sisl` is a physics code which requires strict precision in the algorithms.
  Ensure correctness of code suggestions.
  Once changes have been made, re-check for correctness and consistency.
- Every source file starts with the MPL-2.0 header (`.LICENSE_header`; `add_header.sh` applies it).
- `from __future__ import annotations` is used throughout; type hints live in `src/sisl/typing`.
- Cython sources (`.pyx`/`.pxd`) are compiled — after editing them you must reinstall/rebuild before changes take effect. The pure-Python `.py` file and its `.pyx` companion (e.g. `_sparse.py` vs `_sparse.pyx`) are distinct; check which one is imported.
- Claude commits should use a trailer as specified in the `.gitconfig` file.
  When committing with an agent, ensure to use:
  ```
  git commit ... --trailer ai="<agent-name>:<model>[:<effort>]" where effort is optional.
  ```
  For Claude code Opus 5.5 it would read `--trailer ai=claude-code:claude-opus-5-5`
