# dev-01 · Build

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
| `-DWITH_FORTRAN=OFF` | Do not build the Fortran extensions. The Siesta binary Siles then raise `MissingFortranSileError`. See [sisl-40 · Siesta and Fortran](sisl-40-io.md#siesta-and-fortran). |
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

## Editable installs and worktrees

An editable install binds `import sisl` to one source tree.
A git worktree does not change this binding. Check it before you evaluate or test:

```bash
python -c "import sisl; print(sisl.__file__)"
```

- If the path is not below your source tree, your evaluations and tests run different code.
- `pytest` in such a tree stops with an `ImportPathMismatchError` for `sisl.conftest`.
- A pass in such a tree does not test your change.

To test a worktree, build it in a separate virtual environment outside the repository:

```bash
python -m venv --system-site-packages <env>
<env>/bin/python -m pip install -e . --no-build-isolation -Cbuild-dir=build-editable
<env>/bin/python -c "import sisl; print(sisl.__file__)"
```

- `--system-site-packages` gives the build its dependencies, for example `numpy`.
- `-Cbuild-dir=build-editable` is relative to the source tree. Thus each tree has its own build folder.
  Do not let two source trees use one build folder.
- An entry for sisl in `PYTHONPATH` wins over the virtual environment. Check the path after the install.
- Do not install a worktree into an environment that other people or processes use.
  That changes the code that they run.
