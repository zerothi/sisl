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

The chapters are in segments. A file name is `<segment>-<NN>-<topic>.md`.
A segment can have an entry chapter `<segment>-welcome.md`.

### overview: the technical documentation

| Chapter | Topic |
|---|---|
| [overview · Welcome](overview-welcome.md) | Orientation. Where things are. |
| [overview-01 · Concepts](overview-01-concepts.md) | The principles that all other chapters assume. |

### agent: agents only

| Chapter | Topic |
|---|---|
| [agent · Welcome](agent-welcome.md) | Entry for agents. The chapter for each task. The agent rules win over tool defaults. |
| [agent-01 · Harness](agent-01-harness.md) | How to work in a session. Ask first, git, reports. |
| [agent-02 · Git workflow](agent-02-git.md) | The agent part of git: worktree folder, staging, trailer, history before a report, push. |
| [agent-03 · Parallel agents and builds](agent-03-parallel.md) | One worktree each, the sisl that runs, own build, shared state, test timings. |
| [agent-04 · Implement](agent-04-implement.md) | Rules to implement, fix and test code. Lean code. Review and merge request when the code is complete. |
| [agent-05 · Debugging](agent-05-debugging.md) | The agent part of debugging: scratch files, ask first, report the cause. |
| [agent-06 · Review](agent-06-review.md) | The agent part of review: report, do not fix; read before you run; report what was not done. |
| [agent-07 · Security](agent-07-security.md) | The agent part of security: URL lookups need a human, ask first for automatic code, check your own commits. |

### dev: how to build, test and keep the code

| Chapter | Topic |
|---|---|
| [dev-01 · Build](dev-01-build.md) | scikit-build-core, CMake, Cython, f2py. |
| [dev-02 · Testing](dev-02-testing.md) | pytest, external test files, markers. |
| [dev-03 · Performance](dev-03-performance.md) | Regression tests, benchmarks, Cython ports. |
| [dev-04 · Conventions](dev-04-conventions.md) | Headers, typing, style, changelog. |
| [dev-05 · Verifying a change](dev-05-verifying-a-change.md) | The order of checks before you commit. |
| [dev-06 · Public documentation](dev-06-public-docs.md) | How to write in `docs/`: structure, style, doc-strings, RST, logo. |
| [dev-07 · Git workflow](dev-07-git.md) | Worktrees, small commits, commit messages, merge request description, history. |
| [dev-08 · Debugging](dev-08-debugging.md) | Reproduce, make it repeatable, rule out the environment, find the cause, do not hide it. |
| [dev-09 · Review](dev-09-review.md) | Reference `main`, read before you run, what to check, unrelated code, trojan horses, report. |
| [dev-10 · Security](dev-10-security.md) | No URL lookups, no code from data, dependency sources, automatic code, secrets, trojan horse signs. |
| [dev-11 · Code tools](dev-11-code-tools.md) | `tools/dev/`: symbol outlines and source, environment check, short test reports. |

### sisl: the code in `src/sisl/`

| Chapter | Topic |
|---|---|
| [sisl-10 · Sparse matrices](sisl-10-sparse.md) | `SparseCSR` and the sparse geometry classes. |
| [sisl-20 · Geometry](sisl-20-geometry.md) | `Lattice`, `Geometry`, supercells, dispatch. |
| [sisl-21 · Orbitals](sisl-21-orbitals.md) | Orbital classes, cutoff, radial functions, spherical harmonics. |
| [sisl-22 · Grid](sisl-22-grid.md) | `Grid`, Poisson, filling from physics, sparse grids. |
| [sisl-23 · Shapes](sisl-23-shape.md) | `Shape` contract, composites, tolerance. |
| [sisl-24 · Geometry factories](sisl-24-geom.md) | `sisl.geom`: factories, composites, `NeighborFinder`, categories. |
| [sisl-30 · Physics](sisl-30-physics.md) | Hamiltonians, spin, Brillouin zones, states. |
| [sisl-40 · IO](sisl-40-io.md) | Siles and file formats. |
| [sisl-50 · Visualization](sisl-50-viz.md) | `sisl.viz`: lazy loading, nodify plots, backends. |
| [sisl-60 · Mixing](sisl-60-mixing.md) | `sisl.mixing`: SCF mixers, history, DIIS. |
| [sisl-70 · Utilities](sisl-70-utils.md) | `sisl.utils`: ranges, math conventions, CLI machinery. |

### release: versions, release, CI and deprecations

| Chapter | Topic |
|---|---|
| [release-01 · Versioning](release-01-versioning.md) | `setuptools_scm`, tags, `fallback_version`, `CITATION.cff`. |
| [release-02 · Release process](release-02-process.md) | The decisions behind `DEPLOY.md`: release commit, release notes, publish, stale parts. |
| [release-03 · CI](release-03-ci.md) | GitHub workflows, the wheel and publish sequence, rules for a change. |
| [release-04 · Deprecations](release-04-deprecations.md) | When a deprecated name goes away. Overdue deprecations. |

### toolbox: the code in `src/sisl_toolbox/`

| Chapter | Topic |
|---|---|
| [toolbox · Welcome](toolbox-welcome.md) | `sisl_toolbox`: namespace package, CLI, subpackages. |
| [toolbox-10 · BTD](toolbox-10-btd.md) | `sisl_toolbox.btd`: `DeviceGreen`, electrodes, partitioning. |

## How to add or change a chapter

- Name a chapter `<segment>-<NN>-<topic>.md`. Use two digits for `NN`.
  The title is `# <segment>-<NN> · <Title>`. A link text starts with the same `<segment>-<NN> ·`.
- The segments:
  - `overview`: orientation in the technical documentation.
  - `agent`: rules for agents only. Numbers increment: `01`, `02`, ...
  - `dev`: how to build, test, measure, verify and document. Numbers increment: `01`, `02`, ...
  - `sisl`: the code in `src/sisl/` only. One group of ten for each code area
    (`10` sparse, `20` geometry, `30` physics, ...). A sub-chapter increments from its area (`21`: orbitals).
    A new code area gets the next free group of ten.
  - `release`: versions, the release process, CI and the life of deprecations. Numbers increment: `01`, `02`, ...
  - `toolbox`: the code in `src/sisl_toolbox/` only. One group of ten for each subpackage (`10`: `btd`).
- A new segment needs the approval of the maintainer. A namespace package next to `sisl` gets its own segment.
- Put a new chapter in the segment of its topic.
- One chapter owns one topic. Do not write the same decision in two chapters. Link to it.
- Use stable heading names. Other files link to headings with anchors.
- When you remove a code comment that explains a decision, move the text here in the same commit.
  See [overview-01 · Code carries little prose](overview-01-concepts.md#code-carries-little-prose).
- Update the table above when you add a chapter.

## Ownership

A chapter that documents code starts with an `Owns:` line. It lists glob patterns of the files it owns:

```markdown
# sisl-10 · Sparse matrices

Owns: `src/sisl/_core/sparse.py`, `src/sisl/_core/_sparse*`
```

- `*` also matches `/`. Thus `src/sisl/io/*` matches all files below `src/sisl/io/`.
- When many patterns match a file, the most specific pattern wins: the longest text before the first wildcard.
  Thus `src/sisl/physics/sparse.py` (sisl-10) wins over `src/sisl/physics/*` (sisl-30).
- Files in `tests/` folders have no owner.
- When you add a chapter or move a file, update the `Owns:` lines.

`tools/techs_check.py` uses the `Owns:` lines. pre-commit runs it on each commit. It only warns.

- For each staged file, it names the owning chapter, if that chapter is not staged too.
  Read the chapter. Correct it if the change makes it wrong. If it is still correct, do nothing.
- It names staged files below `src/` that no chapter owns.
- It names `Owns:` patterns that match no tracked file (for example after a rename).

`python tools/techs_check.py --owner <path>...` names the owning chapter of each path.
Use it before you change a file.

`python tools/techs_check.py --all` lists all files below `src/` that no chapter owns.
