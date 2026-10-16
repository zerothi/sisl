# sisl project

sisl is a Python/numpy/Cython/Fortran library for condensed matter physics
(electronic structure, tight-binding, transport). It ships as wheels that
end-users (master and PhD students) install on computational resources, so
portability, a stable API and few dependencies matter.

- `src/sisl/` holds the core sources. Sub-folders tell their intent,
  e.g. `sisl/physics` holds physics specific implementations.
- `src/sisl_toolbox/` holds a toolbox (namespace package) which other
  packages may extend.

[`techs/`](techs/README.md) is the **technical documentation**: the internal
reference for exact, API-specific code decisions, and the only internal prose this
repo maintains by hand. Read [01 · Welcome](techs/01-welcome.md) for orientation
and [02 · Concepts](techs/02-concepts.md) for the ideas everything else assumes.
Write there when you choose between alternatives, or when you remove an explaining
comment. Write in 95% ASD-STE100 — see [techs/README.md](techs/README.md#writing-rules).

[`docs/`](docs/index.rst) is the PUBLIC guide (Sphinx RST), for end-users and
agents alike. Write there only when a change affects end-user capabilities or
usage patterns. Keep it lean, with explicit usage and clear purpose —
see [15 · Public documentation](techs/15-public-docs.md).


## Rules that cost the most when broken

- **Code carries little prose.** WHY goes in the owning `techs/` chapter, in the same
  commit. If a change contradicts `techs/`, fix the chapter in the same commit (the `techs-check`
  pre-commit hook names the owning chapters) —
  [02 · Concepts](techs/02-concepts.md#code-carries-little-prose).
- **Documentation is in-file.** Doc-strings are kept and follow `docs/` style —
  [02 · Concepts](techs/02-concepts.md#documentation-is-in-the-file).
- **Public API is stable.** Deprecate before removing or renaming
  (`sisl.messages.deprecate*`) — [13 · Conventions](techs/13-conventions.md#public-and-private-names).
- **Precision first, then performance.** Re-check correctness after each change —
  [02 · Concepts](techs/02-concepts.md#precision-is-a-requirement). Hot paths need
  regression tests — [12 · Performance](techs/12-performance.md).
- **Verify cheapest first**, and report what was not run (e.g. IO tests without
  `SISL_FILES_TESTS` are xfail-ed, not passed) —
  [14 · Verifying a change](techs/14-verifying-a-change.md).
- **Few strong classes.** Prefer a small generalization over many subclasses —
  [02 · Concepts](techs/02-concepts.md#few-strong-classes).
- **Ask first:** Cython ports, fixes to slow idioms, new required dependencies,
  public API changes — [16 · Agents](techs/16-agents.md#ask-first).
- **Stage explicitly.** Never `git add -A` / `git add .`; untracked directories are
  personal scratch — [16 · Agents](techs/16-agents.md#stage-explicitly).


## Quick commands

```bash
# First install (builds Cython + Fortran extensions)
python -m pip install -e .
# Faster rebuilds afterwards; required after editing .pyx/.pxd/.f90
python -m pip install -e . --no-build-isolation -Cbuild-dir=build-editable
# Tests (set SISL_FILES_TESTS for IO tests, SISL_NUM_PROCS=1 when debugging)
pytest src/sisl/_core/tests/test_geometry.py -k rotate
pytest -m "not slow"
```

Details: [10 · Build](techs/10-build.md), [11 · Testing](techs/11-testing.md),
[13 · Conventions](techs/13-conventions.md) (license header, typing, dependencies,
pre-commit, changelog fragments in `changes/`).


## Agent rules

[16 · Agents](techs/16-agents.md) holds the rules for agents only, with their reasons.
Read it before your first change. In short:

- **Ask first** and **stage explicitly** (above).
- **No URL lookups.** Never add code below `src/` that reads from a URL. Only a human
  can make that change: stop and tell the maintainer.
- **Worktrees** are ok, but always in `<repo-root>/wt/<branch-name>`:
  `git worktree add -b <branch-name> wt/<branch-name>`.
- **Commit trailer.** Mark agent commits only with the `ai` trailer (load it once with
  `git config include.path ../.gitconfig`), never with `Co-authored-by` or other trailers:
  `git commit ... --trailer ai=<agent>:<model>[:<effort>]`, e.g. `ai=claude-code:claude-opus-5-5`.
- **Find the module that runs.** `_x.py` and `_x.pyx` companions are distinct modules:
  check which one is imported.
- **Report what you did not run**, and why.

When you change a rule in 16, change this summary in the same commit.
