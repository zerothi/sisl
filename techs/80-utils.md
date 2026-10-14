# 80 · Utilities

Owns: `src/sisl/utils/*`, `src/sisl/_lib/_argparse.py`

Paths are relative to `src/sisl/`.

## Package surface

- `utils/__init__.py` imports `*` from `_arrays`, `cmd`, `misc` and `ranges`. The `__all__` of each module controls the names.
- `mathematics` is available only as `sisl.utils.math`. `parse_rotation` is also in `sisl.utils`.
- `_search_mro` and `_sisl_cmd` are not exported.
- `sisl` does not export names from `utils`. `sisl.__getattr__` imports `sisl.utils` lazily.

## Where a helper goes

| Location | Content |
|---|---|
| `utils/` | Helpers that users can call. Few dependencies. Add the name to `__all__`. |
| `utils/mathematics.py` | Numerical helpers. It imports sisl core modules. Do not import it early. |
| `_help.py` | Internal dtype and iterable helpers (`get_dtype`, `isiterable`, `has_module`). |
| `_array.py` | Typed array constructors. |
| `_internal.py` | `set_module` only. |
| `_lib/` | Wrappers of third-party libraries. |

## ranges.py

Parses index ranges from strings. The CLI and some Siles use it.

- `a-b` and `a:b` are ranges. **The end is included.**
- **The 3-field form is `start:step:end`.** It is not the Python `start:stop:step`.
  `fileindex("f[1:2:7]")` gives `[1, 3, 5, 7]`.
- A leading `-` is a negative number. `-2-2` is `(-2, 2)`.
- A single negative value is relative to `end` (`-2` with `end=7` gives `5`). Negative values in a range are not changed.
- An empty side is `start` or `end`.
- `strmap` splits on commas. It keeps pieces with brackets together. Thus nesting works.
  `5[1-2]` gives `[5, [1, 2]]`. `sep="c"` uses `{}`.
- `lstranges` expands the ranges. `list2str` is the inverse (`"1-3, 5"`).
- `strseq` parses one sequence.
- `fileindex` (`file[1,3-6]`) is not used in sisl. Sile names use `{...}` through `misc.str_spec`.

## misc.py

| Name | Use |
|---|---|
| `direction` | `x`/`a`/`0` → 0, `y`/`b`/`1` → 1, `z`/`c`/`2` → 2. Ignores letter case. With `abc=` or `xyz=` it returns that row (`abc` first). |
| `angle` | Prefix `r`/`a` sets the input unit. Suffix `r`/`a` sets the output unit (radians or degrees). They override `in_rad` and `rad`. `pi` changes to `pi` or `180`. |
| `math_eval` | Evaluates arithmetic through an AST whitelist. It does not use `eval`. **`^` is XOR, not power.** |
| `PropertyDict` | A `dict` with attribute access. `NotNonePropertyDict` ignores `None` values. |
| `str_spec` | Splits `name{spec}` at the last `{`, only when the name ends with `}`. |
| `listify` | Use as `x \| listify`. It sets `__array_priority__` so that numpy does not take `\|`. |

`iter_shape`, `merge_instances`, `allow_kwargs`, `size_to_elements` and `import_attr` each have one user.
`iter_shape` yields tuples in C order (`itertools.product`). Use a tuple to index a numpy or netCDF array.
`size_to_num` reads an `int` or `float` as MB.
`lazy_import` has no user. Its doc-string says that it does not work.

## mathematics.py

- **Angle convention: `theta` is the azimuthal angle in the xy plane from x. `phi` is the polar angle from z.**
  This is the opposite of the ISO physics convention. `cart2spher`, `spher2cart` and the orbitals use it.
  See [31 · Spherical harmonics](31-orbitals.md#spherical-harmonics).
- `cart2spher` uses `eps = 1e-16` for `r = 0`. At the origin, `phi = 0`.
  `cos_phi=True` returns `cos(phi)` without `arccos`. With `maxR` it also returns indices (only for 2-D input).
- `spherical_harm` multiplies by `(-1)**m` (Condon-Shortley phase).
  It supports scipy `sph_harm_y` (scipy >= 1.15) and the old `sph_harm`. Their argument orders are different.
- `fnorm` and `fnorm2` are `sqrt(sum(x**2))`. They do not scale against overflow (`np.linalg.norm` does).
- `orthogonalize` treats vectors as parallel at a relative tolerance of `1e-7`.
- `intersect_and_diff_sets` uses a stable merge sort. It assumes unique input.
  It is in a hot path of `_core/sparse.py`.
- `parse_rotation` returns a `Quaternion`. It accepts:
  three angles; `(angle, "x+y")` (`+` adds axes); `("xy", angle)` (two letters give the cross product);
  `(angle, vector)`; a list of these (applied in order).

## _arrays.py

`batched_indices(ref, y, atol)` finds where `y` matches `ref` within `atol`.
It splits `y` into batches so that the broadcast array stays below `batch_size` MB.
It is numpy only. `physics/brillouinzone.py` uses it.

## Command line

Console scripts (`pyproject.toml`): `sgeom` (`Geometry`), `sgrid` (`Grid`), `sdata` (`utils/_sisl_cmd.py`).
`python -m sisl` runs `sisl_cmd`.

### Protocol

A class or Sile that supports the CLI implements:

- `ArgumentParser(p=None, ...)`, decorated with `@default_ArgumentParser(description=...)`.
  The decorator makes a parser when `p` is `None`.
- `_ArgumentParser_args_single()`, and optionally `ArgumentParser_out`.

The base in `io/sile.py` raises `NotImplementedError`.
Geometry Siles use `Geometry._ArgumentParser_args_single()`.

`collect_arguments` (`cmd.py`):

1. Takes the input file from the positional arguments.
2. Calls `get_sile(...).ArgumentParser`.
3. Reads `-o/--out` early. Thus the output Sile can add options. Errors in this step are ignored.

### Deferred actions

Options such as `--atom` must take effect before `--out` writes.

- `default_namespace` adds `_actions` and `_actions_run` to the namespace.
- `@collect_action` puts an `argparse.Action.__call__` in the queue.
- `@run_actions` runs the queue, then clears it.

Other details:

- `argv_negative_fix` adds a space before arguments that parse as ranges. Thus argparse does not read `-1` as an option.
- `_sisl_cmd.argparse_patch` replaces the sub-parser action so that it uses the parent namespace (Python issue 27859).
- `_lib/_argparse.SislHelpFormatter` uses `rich_argparse` when it is installed.

## _search_mro.py

`find_implementation` is a copy of the MRO resolution of `functools.singledispatch` (PSF license).
It selects the registered class that is the best match for a type.
Only `_dispatcher.py` uses it (`new` and `to`). See [30 · Conversion](30-geometry.md#conversion-new-and-to).

## Known defects

Confirmed by running the code (2026-10-14).

- `allow_kwargs` raises `KeyError` when the removed keyword is not given (`del kwargs[arg]`).
- `sisl_cmd(..., sile=None)` does not use `sile`.
