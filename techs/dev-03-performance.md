# dev-03 · Performance

Owns: `benchmarks/*`

Performance is critical in sisl.
Users work with matrices that have millions of non-zero elements.

## Rules

- A new implementation in a hot path needs a performance regression test.
- Measure before and after. Do not guess.
- Keep the measurement short and to the point.
  Report the size of the system and the times.
- Do not measure with a `Debug` build. See [dev-01 · Build](dev-01-build.md#cmake-options).

## Slow idioms

When you find a slow idiom, do not fix it silently.
Tell the maintainer. The maintainer decides what to do.

Typical slow idioms:

- A Python loop over atoms, orbitals or non-zero elements.
- A copy of a large array when a view is sufficient.
- A conversion between sparse formats in a loop.
- A dtype upcast that is not necessary (for example `float32` to `float64`).

## Port to Cython

You can suggest a port to Cython for slow code.
The maintainer must guide the port. Do not start it alone.

- Look at the Cython annotations first (`-DWITH_ANNOTATE=ON`).
- Use fused types for dtypes. Look at the existing `.pxd` files in `src/sisl/_core/`.
- After a change to a `.pxd`, read [dev-01 · Stale Cython builds](dev-01-build.md#stale-cython-builds).

## Benchmarks

`benchmarks/` holds scripts that measure typical work loads
(for example `graphene.py`, `bloch.py`, `sparse_matrices.py`).
`benchmarks/run.sh` and `benchmarks/stats.py` run them with a profiler.
Use them to compare a change with `main`.
