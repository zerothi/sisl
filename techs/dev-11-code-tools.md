# dev-11 · Code tools

Owns: `tools/dev/*`

The scripts in `tools/dev/` give short and exact answers to frequent questions about the code.
They are for developers and agents.

| Script | Question |
|---|---|
| [`sym.py`](#sym) | Where is a function or a class, and what is its source? |

## Why the scripts exist

An analysis of the agent sessions in this repository gave these results:

- Agents read the code with `grep -n` and `sed -n`. 60% of the `grep -n` calls were followed by a `sed -n` of a guessed line range.
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

The script reads `.py`, `.pyx` and `.pxd` files with one text parser.
Python's `ast` cannot read Cython. One parser for all three file types keeps the script short.
The parser uses the indentation, the brackets of the signature and the triple-quoted strings.
It finds `def`, `class`, `cdef`, `cpdef` and `cdef class`. It does not find `cdef` variables.
For all `.py` files below `src/`, it gives the same names and line ranges as `ast`.

## Rules for the scripts

- Use only the Python standard library. The scripts must run in each environment that runs sisl.
- Print only the answer. Limit the output of each part, and tell what the script removed.
- Do not change files. The scripts only read and report.
- Do not read from the network. See [dev-10 · Security](dev-10-security.md).
