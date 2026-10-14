# 10 · Build

Owns: `CMakeLists.txt`, `*/CMakeLists.txt`, `pyproject.toml`, `tools/build_tools/*`

sisl is not a pure-Python package.
The build system is scikit-build-core, CMake, Cython and f2py.
See `pyproject.toml` and `CMakeLists.txt`.

## Install for development

```bash
# Editable install. Compiles the Cython and Fortran extensions.
python -m pip install -e .

# Faster rebuilds: no build isolation, and a persistent build directory.
python -m pip install -e . --no-build-isolation -Cbuild-dir=build-editable
```

A change to a `.pyx`, `.pxd`, `.f` or `.f90` file has no effect until you rebuild.
A change to a `.py` file has an immediate effect in an editable install.

## CMake options

Give options through `SKBUILD_CMAKE_ARGS`, separated by `;`.

| Option | Effect |
|---|---|
| `-DWITH_FORTRAN=OFF` | Do not build the Fortran extensions. The Siesta binary Siles then raise `MissingFortranSileError`. See [50 · Siesta and Fortran](50-io.md#siesta-and-fortran). |
| `-DWITH_COVERAGE=ON` | Build Cython with coverage support. |
| `-DWITH_ANNOTATE=ON` | Write Cython HTML annotations. Use them to find Python calls in hot loops. |
| `-DWITH_LINE_DIRECTIVES=ON` | Add line directives to the generated C code. |
| `-DWITH_GDB=ON` | Build for debugging with gdb. |
| `-DCMAKE_BUILD_TYPE=Debug` | Fast compilation. Do not use it for performance checks. |

## `.py` and `.pyx` companions

Some modules exist as a `.py` file and as a `.pyx` file with a similar name
(for example `_sparse.py` and `_sparse.pyx`).
They are different modules.
Before you change one, find which one the code imports.

## Stale Cython builds

The build can complete without errors and still use old code.

- Ninja does not track `.pxd` dependencies.
  When you change a `.pxd`, `touch` each `.pyx` that `cimport`s from it.
- Cython finds `.pxd` files through `sys.path`.
  An old installed sisl on `PYTHONPATH` can supply an old `.pxd`.
  This affects modules outside `sisl` that `cimport` from `sisl._core`, for example `sisl_toolbox`.
- Symptom: a fused function lacks a specialization. The error at run time is "No matching signature found".
- Check: look at `func.__signatures__` of the fused `def` function.

The CMake minimum version is in `CMakeLists.txt`. Use a CMake that is new enough.
