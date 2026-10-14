# 01 · Welcome

sisl is a Python library for electronic structure and transport calculations.
It reads and writes the files of many DFT codes.
It builds and manipulates tight-binding models and large sparse matrices.

## Who uses sisl

- Researchers in condensed matter physics.
- Most users are master students and new PhD students.
- Users install wheels on computational resources (clusters, workstations).

Thus the API must be coherent, fast and easy to extend.
Users write their own scripts on top of sisl. Do not break public API without a deprecation.

## Where things are

| Path | Content |
|---|---|
| `src/sisl/` | The core package. |
| `src/sisl/_core/` | `Lattice`, `Geometry`, `Atom`, `Orbital`, `Grid`, `SparseCSR` and the sparse geometry classes. |
| `src/sisl/physics/` | Physical quantities: `Hamiltonian`, `DensityMatrix`, `Spin`, `BrillouinZone`, states. |
| `src/sisl/io/` | Siles: readers and writers for file formats. |
| `src/sisl/geom/` | Functions that create standard geometries (`graphene`, `fcc`, ...). |
| `src/sisl/linalg/` | Linear algebra wrappers. |
| `src/sisl/mixing/` | Mixers for self-consistent loops. See [70 · Mixing](70-mixing.md). |
| `src/sisl/utils/` | Helpers, range parsing, CLI machinery. See [80 · Utilities](80-utils.md). |
| `src/sisl/typing/` | Type hints. |
| `src/sisl/viz/` | Plotting. Loaded lazily. See [60 · Visualization](60-viz.md). |
| `src/sisl_toolbox/` | A namespace package. It holds tools that are not core (BTD, NEGF, models, CLI). Other packages can add to it. See [100 · Toolbox](100-toolbox.md). |
| `docs/` | Public Sphinx documentation (RST). |
| `techs/` | This internal reference. |
| `changes/` | towncrier news fragments. See [13 · Conventions](13-conventions.md#changelog). |
| `benchmarks/` | Performance scripts. See [12 · Performance](12-performance.md). |

The subfolder names of `src/sisl/` tell their purpose.
Put new code in the subfolder that matches its purpose.

## What to read next

- [02 · Concepts](02-concepts.md): the principles that all other chapters assume.
- [10 · Build](10-build.md): sisl is not a pure-Python package.
- [14 · Verifying a change](14-verifying-a-change.md): before each commit.
