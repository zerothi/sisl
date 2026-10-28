# dev-11 · Code tools

Owns: `tools/dev/*`

The scripts in `tools/dev/` give short and exact answers to frequent questions about the code.
They are for developers and agents.

| Script | Question |
|---|---|
| [`sym.py`](#sym) | Where is a function or a class, and what is its source? |
| [`envcheck.py`](#envcheck) | Is the sisl that runs the sisl of this source tree? |
| [`t.py`](#t) | Do the tests pass? |

## Why the scripts exist

An analysis of the agent sessions in this repository gave these results:

- Agents read the code with `grep -n` and `sed -n`.
  60% of the `grep -n` calls had a `sed -n` of a line range in the next two calls.
- The largest outputs were slices of 600 lines. The agent needed only one function in them.
- Agents wrote the same environment checks and `pytest` options again in each session.

Each line of output costs context. Thus each script prints only the answer.

## sym

```bash
python tools/dev/sym.py FILE                  # outline: line ranges of classes and functions
python tools/dev/sym.py FILE NAME...          # source of each symbol
python tools/dev/sym.py FILE NAME... --sig    # signature and doc-string only
python tools/dev/sym.py FILE NAME --outline   # outline of the members of NAME
```

- `NAME` is a qualified name (`Geometry.close`) or the last part of it (`close`).
  The last part can match more symbols. Then the script prints each of them.
- The outline shows the decorators, for example `@register_sisl_dispatch(Geometry, module="sisl")`.
  It does not show the `@cython.*` decorators.
- A symbol of more than 300 lines with members prints as an outline. Use `--full` for the source.
- An unknown `NAME` gives the close names and exit code 1.
- `--sig`, `--outline` and `--full` need a `NAME`.

The script reads `.py`, `.pyx` and `.pxd` files with one text parser.
Python's `ast` cannot read Cython. One parser for all three file types keeps the script short.
The parser uses the indentation, the brackets of the signature and the strings.
It ignores the text in strings and comments, also with escaped quotes.
A comment line between a decorator and its `def` does not stop the decorators.
It finds `def`, `class`, `cdef`, `cpdef` and `cdef class`, also with a C tuple return type.
It does not find `cdef` variables.
For all `.py` files below `src/`, it gives the same names and line ranges as `ast`.

## envcheck

```bash
python tools/dev/envcheck.py           # import path, PYTHONPATH, stale builds, test variables
python tools/dev/envcheck.py PATH...   # also: the module that runs for each source file
```

The source tree is the git top-level of the current directory.
Run the script with the Python that runs your tests, for example `<env>/bin/python`.
Each line starts with `ok`, `note`, `skip` or `FAIL`. One `FAIL` gives exit code 1.

| Check | `FAIL` when |
|---|---|
| `sisl` and `sisl_toolbox` import path | No folder of the package is below `src/` of the tree. See [dev-01 · Editable installs and worktrees](dev-01-build.md#editable-installs-and-worktrees). |
| `PYTHONPATH` | An entry other than `src/` has a `sisl` or `sisl_toolbox` folder. See [dev-01 · Stale Cython builds](dev-01-build.md#stale-cython-builds). |
| Stale builds | A compiled module is older than its `.pyx`, or older than a `.pxd` that it `cimport`s, also through other `.pxd` files. See [dev-01 · Stale Cython builds](dev-01-build.md#stale-cython-builds). |
| `SISL_FILES_TESTS` | The variable is set, but it is not a folder. If it is not set, a `note` tells that the IO tests xfail. |

- When `sisl` is from a different tree, the script skips the stale build check.
  The compiled modules of a different tree tell nothing about this tree.
- An editable install loads the `.py` files from `src/` and the compiled modules from `site-packages`.
  Thus the stale build check compares times, not folders.
- `sisl_toolbox` is a namespace package. Other packages can add folders to it.
  Thus one folder below `src/` is sufficient.
- For each `PATH`, the script names its module, the file that the module loads from, and whether `import sisl` loads it.
  It does the same for each companion: a `.py` or `.pyx` file with the same name without a leading `_`.
  See [dev-01 · `.py` and `.pyx` companions](dev-01-build.md#py-and-pyx-companions).

## t

```bash
python tools/dev/t.py src/sisl/_core/tests/test_geometry.py -k rotate
python tools/dev/t.py --durations=0 src/sisl/physics/tests/test_hamiltonian.py
```

The script runs `pytest -q -p no:cacheprovider --tb=short` with the arguments that you give.
It sets `SISL_NUM_PROCS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1.
It replaces values from your shell, because the CI uses 1 thread.
See [dev-02 · Measure test timings](dev-02-testing.md#measure-test-timings).
The exit code is the exit code of `pytest`.

The report keeps:

- The result line, for example `23 passed, 3 xfailed, 29 warnings in 0.72s`.
- For each failure and error: the frames in the tree, all `E` lines up to 20, and 5 lines of each captured output.
  Frames outside the tree are removed, for example frames in `numpy`.
- A maximum of 10 failures. The summary names the others.
- The short test summary. The `XFAIL` lines are grouped by reason, with a count.
  Thus `XFAIL [3] ... requires env(SISL_FILES_TESTS) ...` tells that these tests did not run.

The report removes the warnings summary. It removes the durations, unless you give `--durations`.
When `pytest` stops early (exit codes 2, 3 and 4), the script prints all of the output.
Examples are an error at collection and a usage error.
For the full output, run `pytest` directly.

The `-p no:cacheprovider` option stops `pytest` from writing `.pytest_cache` into the tree.

## Rules for the scripts

- Use only the Python standard library. The scripts must run in each environment that runs sisl.
- Print only the answer. Limit the output of each part, and tell what the script removed.
- Do not change files. The scripts only read and report.
- Do not read from the network. See [dev-10 · Security](dev-10-security.md).
