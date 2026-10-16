# techs/ — technical documentation

This folder is the internal reference of sisl.
It records decisions, invariants and the reasons for them.
It is for developers and agents. It is not for end-users.

## Two kinds of documentation

| Folder | Audience | Format | Content |
|---|---|---|---|
| `techs/` | developers, agents | Markdown | Why the code is as it is. Internal API. Decisions. |
| `docs/` | end-users, agents | Sphinx RST | How to use sisl. Public API, examples, tutorials. |

Write in `docs/` only when a change affects what end-users can do or how they do it.
Write in `techs/` for all other prose. Write a decision in `techs/` when:

- you choose between alternatives, or
- you remove a code comment that explains a decision, or
- a change makes a statement in `techs/` wrong.
Doc-strings stay in the source files. They follow the `docs/` style.

## Writing rules

Write 95% in ASD-STE100 (Simplified Technical English):

- Use short sentences. Use a maximum of 20 words in a sentence.
- Write one idea in one sentence.
- Use the active voice.
- Use the imperative for instructions ("Run the tests.").
- Use one word for one meaning. Do not change words for style.
- Use the names from the code without change (`SparseCSR`, `nsc`, `finalize`).

The remaining 5% is for formulas, tables and code that STE cannot express.

## Chapters

| # | Chapter | Topic |
|---|---|---|
| 01 | [Welcome](01-welcome.md) | Orientation. Where things are. |
| 02 | [Concepts](02-concepts.md) | The principles that all other chapters assume. |
| 10 | [Build](10-build.md) | scikit-build-core, CMake, Cython, f2py. |
| 11 | [Testing](11-testing.md) | pytest, external test files, markers. |
| 12 | [Performance](12-performance.md) | Regression tests, benchmarks, Cython ports. |
| 13 | [Conventions](13-conventions.md) | Headers, typing, style, changelog. |
| 14 | [Verifying a change](14-verifying-a-change.md) | The order of checks before you commit. |
| 15 | [Public documentation](15-public-docs.md) | How to write in `docs/`: structure, style, doc-strings, RST. |
| 16 | [Agents](16-agents.md) | Rules for agents only: ask first, staging, commit trailer, reports. |
| 20 | [Sparse matrices](20-sparse.md) | `SparseCSR` and the sparse geometry classes. |
| 30 | [Geometry](30-geometry.md) | `Lattice`, `Geometry`, supercells, dispatch. |
| 31 | [Orbitals](31-orbitals.md) | Orbital classes, cutoff, radial functions, spherical harmonics. |
| 32 | [Grid](32-grid.md) | `Grid`, Poisson, filling from physics, sparse grids. |
| 33 | [Shapes](33-shape.md) | `Shape` contract, composites, tolerance. |
| 34 | [Geometry factories](34-geom.md) | `sisl.geom`: factories, composites, `NeighborFinder`, categories. |
| 40 | [Physics](40-physics.md) | Hamiltonians, spin, Brillouin zones, states. |
| 50 | [IO](50-io.md) | Siles and file formats. |
| 60 | [Visualization](60-viz.md) | `sisl.viz`: lazy loading, nodify plots, backends. |
| 70 | [Mixing](70-mixing.md) | `sisl.mixing`: SCF mixers, history, DIIS. |
| 80 | [Utilities](80-utils.md) | `sisl.utils`: ranges, math conventions, CLI machinery. |
| 100 | [Toolbox](100-toolbox.md) | `sisl_toolbox`: namespace package, CLI, subpackages. |
| 101 | [BTD](101-btd.md) | `sisl_toolbox.btd`: `DeviceGreen`, electrodes, partitioning. |

## How to add or change a chapter

- Numbers are in groups:
  - `0x`: orientation.
  - `1x`: code development (build, test, performance, conventions, verification, public documentation, agents).
  - `20`–`99`: one group of ten for each code area of `sisl` (sparse, geometry, physics, IO, viz, mixing, utils).
  - `100` and higher: namespace packages next to `sisl` (`100`: `sisl_toolbox`). The next package gets `110`.
    Chapters on subpackages increment from the package number (`101`: `sisl_toolbox.btd`).
- Put a new chapter in the group of its topic. A new code area of `sisl` gets the next free group of ten below `100`.
- One chapter owns one topic. Do not write the same decision in two chapters. Link to it.
- Use stable heading names. Other files link to headings with anchors.
- When you remove a code comment that explains a decision, move the text here in the same commit.
  See [02 · Code carries little prose](02-concepts.md#code-carries-little-prose).
- Update the table above when you add a chapter.

## Ownership

A chapter that documents code starts with an `Owns:` line. It lists glob patterns of the files it owns:

```markdown
# 20 · Sparse matrices

Owns: `src/sisl/_core/sparse.py`, `src/sisl/_core/_sparse*`
```

- `*` also matches `/`. Thus `src/sisl/io/*` matches all files below `src/sisl/io/`.
- When many patterns match a file, the most specific pattern wins: the longest text before the first wildcard.
  Thus `src/sisl/physics/sparse.py` (20) wins over `src/sisl/physics/*` (40).
- Files in `tests/` folders have no owner.
- When you add a chapter or move a file, update the `Owns:` lines.

`tools/techs_check.py` uses the `Owns:` lines. pre-commit runs it on each commit. It only warns.

- For each staged file, it names the owning chapter, if that chapter is not staged too.
  Read the chapter. Correct it if the change makes it wrong. If it is still correct, do nothing.
- It names staged files below `src/` that no chapter owns.
- It names `Owns:` patterns that match no tracked file (for example after a rename).

`python tools/techs_check.py --all` lists all files below `src/` that no chapter owns.
