# sisl project

sisl is a Python/numpy/Cython/Fortran library for condensed matter physics
(electronic structure, tight-binding, transport). It ships as wheels that
end-users (master and PhD students) install on computational resources, so
portability, a stable API and few dependencies matter.

- `src/sisl/` holds the core sources. Sub-folders tell their intent,
  e.g. `sisl/physics` holds physics specific implementations.
- `src/sisl_toolbox/` holds a toolbox (namespace package) which other
  packages may extend.

## Agents: read first

These chapters are mandatory. They win over the defaults of your tool.

1. **Each session:** [agent · Welcome](techs/agent-welcome.md) — the chapter for each task —
   and [agent-01 · Harness](techs/agent-01-harness.md) — how to work here.
2. **Before you branch, make a worktree or commit:**
   [agent-02 · Git workflow](techs/agent-02-git.md).
3. **Before you build, test or time tests in a worktree, or next to other agents:**
   [agent-03 · Parallel agents and builds](techs/agent-03-parallel.md).
4. **Before you implement, fix, refactor or test code:**
   [agent-04 · Implement](techs/agent-04-implement.md).
5. **Before you debug a bug, a failing test or a wrong result:**
   [agent-05 · Debugging](techs/agent-05-debugging.md).
6. **Before you review a change, branch or pull request:**
   [agent-06 · Review](techs/agent-06-review.md).
7. **Before you add a dependency, touch CI, hooks or build files, or handle outside data,
   and in each review:** [agent-07 · Security](techs/agent-07-security.md).

A task can be of more than one kind: read each chapter that applies.
Most agent chapters extend a `dev-*` chapter with the rules for everyone. Read both.
The [summary below](#agent-rules) does not replace the chapters.

[`techs/`](techs/README.md) is the **technical documentation**: the internal
reference for exact, API-specific code decisions, and the only internal prose this
repo maintains by hand. Read [overview · Welcome](techs/overview-welcome.md) for orientation
and [overview-01 · Concepts](techs/overview-01-concepts.md) for the ideas everything else assumes.
Write there when you choose between alternatives, or when you remove an explaining
comment. Write in 95% ASD-STE100 — see [techs/README.md](techs/README.md#writing-rules).

[`docs/`](docs/index.rst) is the PUBLIC guide (Sphinx RST), for end-users and
agents alike. Write there only when a change affects end-user capabilities or
usage patterns. Keep it lean, with explicit usage and clear purpose —
see [dev-06 · Public documentation](techs/dev-06-public-docs.md).


## Rules that cost the most when broken

- **Code carries little prose.** WHY goes in the owning `techs/` chapter, in the same
  commit. If a change contradicts `techs/`, fix the chapter in the same commit (the `techs-check`
  pre-commit hook names the owning chapters) —
  [overview-01 · Concepts](techs/overview-01-concepts.md#code-carries-little-prose).
- **Documentation is in-file.** Doc-strings are kept and follow `docs/` style —
  [overview-01 · Concepts](techs/overview-01-concepts.md#documentation-is-in-the-file).
- **Public API is stable.** Deprecate before removing or renaming
  (`sisl.messages.deprecate*`) — [dev-04 · Conventions](techs/dev-04-conventions.md#public-and-private-names).
- **Precision first, then performance.** Re-check correctness after each change —
  [overview-01 · Concepts](techs/overview-01-concepts.md#precision-is-a-requirement). Hot paths need
  regression tests — [dev-03 · Performance](techs/dev-03-performance.md).
- **Verify cheapest first**, and report what was not run (e.g. IO tests without
  `SISL_FILES_TESTS` are xfail-ed, not passed) —
  [dev-05 · Verifying a change](techs/dev-05-verifying-a-change.md).
- **Few strong classes.** Prefer a small generalization over many subclasses —
  [overview-01 · Concepts](techs/overview-01-concepts.md#few-strong-classes).
- **Ask first:** Cython ports, fixes to slow idioms, new required dependencies,
  changes to CI or hooks, public API changes — [agent-01 · Harness](techs/agent-01-harness.md#ask-first).
- **Stage explicitly.** Never `git add -A` / `git add .`; untracked directories are
  personal scratch — [agent-02 · Git workflow](techs/agent-02-git.md#stage-explicitly).


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

Details: [dev-01 · Build](techs/dev-01-build.md), [dev-02 · Testing](techs/dev-02-testing.md),
[dev-04 · Conventions](techs/dev-04-conventions.md) (license header, typing, dependencies,
pre-commit, changelog fragments in `changes/`).


## Agent rules

The agent chapters hold the rules for agents, with their reasons.
The `dev-*` chapter that each one extends holds the rules for everyone. In short:

[agent-01 · Harness](techs/agent-01-harness.md) — each session:

- **Ask first** (above).
- **Report what you did not run**, and why.

[agent-02 · Git workflow](techs/agent-02-git.md), extends
[dev-07 · Git workflow](techs/dev-07-git.md) — branches, worktrees, commits:

- **Worktrees** are ok, but always in `<repo-root>/wt/<branch-name>`:
  `git worktree add -b <branch-name> wt/<branch-name>`.
- **Commit in small steps.** Many small commits, each one a working step that passes
  pre-commit and its targeted tests. Do not make one large commit at the end.
  Keep moves and renames apart from changes of behavior.
- **Messages:** `<area>: <imperative summary>`, at most 72 characters; the body tells why.
- **Stage explicitly** (above). Do not touch untracked files you did not make.
- **Commit trailer.** Mark agent commits only with one `Assisted-by` trailer, never with
  `Co-authored-by` or other trailers:
  `git commit ... --trailer "Assisted-by: <agent>:<model>[:<effort>]"`,
  e.g. `Assisted-by: claude-code:claude-opus-5-5`.
- **Keep the history.** Do not merge into `main`, force-push, or rewrite commits you reported.
- **Do not push** unless a human tells you to. Commit locally and report the branch.

[agent-03 · Parallel agents and builds](techs/agent-03-parallel.md), with
[dev-01 · Editable installs and worktrees](techs/dev-01-build.md#editable-installs-and-worktrees) — worktrees, builds, other agents:

- **One agent, one worktree.** Do not change files in the main checkout.
- **Find the sisl that runs.** An editable install binds `import sisl` to one tree:
  check `sisl.__file__` before you test. A pass of other code is not a pass.
- **Build in your own environment** (a virtual environment outside the repository).
  Never install a worktree into the shared environment.
- **No `git stash`** (shared by all worktrees). Scratch files go outside the repository.
- **Time tests with 2 agents or fewer.** When you measure test timings, or change tests because of them,
  run a maximum of 2 agents at the same time, or ask first. Power and efficiency cores skew the timings.
  Set the thread variables (`SISL_NUM_PROCS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`)
  to 1: CI systems have few threads.

[agent-04 · Implement](techs/agent-04-implement.md), with
[dev-02 · Testing](techs/dev-02-testing.md) — implement, fix, test:

- **Read the owning chapter** of each file before you change it:
  `python tools/techs_check.py --owner <path>...`.
- **Find the module that runs.** `_x.py` and `_x.pyx` companions are distinct modules:
  check which one is imported.
- **Keep the code safe** ([dev-10](techs/dev-10-security.md), [agent-07](techs/agent-07-security.md)).
- **Lean code.** Write only the code the task needs: no unneeded options, abstractions,
  checks or copies of sisl functions. Read your diff and remove what is not needed.
- **Fast tests, no redundant tests.** CI runs each test many times. Use the smallest system,
  extend or parametrize an existing test before adding one, and mark slow tests `slow`.
- **When the code is complete:** ask the maintainer if it is complete, then review the branch
  ([agent-06](techs/agent-06-review.md)), then ask to write the merge request description
  ([dev-07](techs/dev-07-git.md#merge-request-description)): overview line, problem,
  lean details; a bug fix names its test that fails before the fix.

[agent-05 · Debugging](techs/agent-05-debugging.md), extends
[dev-08 · Debugging](techs/dev-08-debugging.md) — find the cause:

- **Reproduce first** with the smallest system, and compare with a known value.
  Keep the script outside the repository.
- **Make it repeatable:** thread variables to 1, fixed seeds, run the test alone.
- **Rule out the environment:** the sisl that runs, `.py` vs `.pyx`, stale builds, `SISL_FILES_TESTS`.
- **Find the cause**, not the symptom; `git bisect` against `main`; read the owning chapter.
- **Do not hide it:** no larger tolerances, `xfail`, `skip`, broad `except` or weaker asserts.
- **Reproduction becomes the test** that fails before the fix, in the same commit as the fix.

[agent-06 · Review](techs/agent-06-review.md), extends
[dev-09 · Review](techs/dev-09-review.md) — review a change:

- **Compare with `main`** (`git diff main...HEAD`) unless told otherwise, and review
  each commit in `main..HEAD`, not only the final diff.
- **Read before you run.** Read changes to `conftest.py`, build files, pre-commit hooks
  and `tools/` before you test or build the branch.
- **No code unrelated to sisl.** Report scratch files, tool configuration, copied code,
  generated files and unexplained code.
- **No trojan horses.** Examine every commit, also merges (`git log -p --remerge-diff`), for the
  signs in [dev-10](techs/dev-10-security.md#signs-of-a-trojan-horse). Report each sign first;
  never rewrite history.
- **Report, do not fix**, unless the maintainer tells you to.

[agent-07 · Security](techs/agent-07-security.md), extends
[dev-10 · Security](techs/dev-10-security.md) — develop and review:

- **No URL lookups.** Never add code below `src/` that reads from a URL. Only a human
  can make that change: stop and tell the maintainer.
- **Do not run the content of a file:** no `eval`/`exec`/`pickle` on data that sisl reads.
- **Dependencies:** none required without approval; no package indexes, git URLs or unpinned hooks.
- **Ask first** before you change CI workflows, pre-commit hooks or build hooks.
- **No secrets and no local data:** credentials, absolute paths, user and host names.
- **Signs of a trojan horse:** network access, dynamic code, hidden content and Unicode, binaries,
  code that runs automatically, tree changes, code added then removed, new dependency sources.

When you change a rule in an `agent-*` chapter, or a `dev-*` rule in this summary,
change this summary in the same commit.
