# overview · Welcome

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
| `src/sisl/mixing/` | Mixers for self-consistent loops. See [sisl-60 · Mixing](sisl-60-mixing.md). |
| `src/sisl/utils/` | Helpers, range parsing, CLI machinery. See [sisl-70 · Utilities](sisl-70-utils.md). |
| `src/sisl/typing/` | Type hints. |
| `src/sisl/viz/` | Plotting. Loaded lazily. See [sisl-50 · Visualization](sisl-50-viz.md). |
| `src/sisl_toolbox/` | A namespace package. It holds tools that are not core (BTD, NEGF, models, CLI). Other packages can add to it. See [toolbox · Welcome](toolbox-welcome.md). |
| `docs/` | Public Sphinx documentation (RST). |
| `techs/` | This internal reference. |
| `changes/` | towncrier news fragments. See [dev-04 · Conventions](dev-04-conventions.md#changelog). |
| `benchmarks/` | Performance scripts. See [dev-03 · Performance](dev-03-performance.md). |

The subfolder names of `src/sisl/` tell their purpose.
Put new code in the subfolder that matches its purpose.

## What to read next

- [overview-01 · Concepts](overview-01-concepts.md): the principles that all other chapters assume.
- [dev-01 · Build](dev-01-build.md): sisl is not a pure-Python package.
- [dev-05 · Verifying a change](dev-05-verifying-a-change.md): before each commit.
- [agent · Welcome](agent-welcome.md): if you are an agent. It tells which agent chapter to read for your task.
