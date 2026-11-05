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

The agent chapters are mandatory and win over the defaults of your tool.
Each session, read [agent · Welcome](techs/agent-welcome.md) and
[agent-01 · Harness](techs/agent-01-harness.md).
Before a task, read the agent chapter that the [agent rules](#agent-rules) name for that task,
and the `dev-*` chapter that it extends. The summary does not replace the chapters.

[`techs/`](techs/README.md) is the **technical documentation**: the internal reference for
code decisions, and the only internal prose maintained by hand. Start with
[overview · Welcome](techs/overview-welcome.md) and [overview-01 · Concepts](techs/overview-01-concepts.md).
Write there when you choose between alternatives, or remove an explaining comment.
Write in 95% ASD-STE100 ([writing rules](techs/README.md#writing-rules)), and list the
sources of a decision below it ([sources](techs/README.md#sources)).
List outside material that inspires a rule or the workflow in [overview-02](techs/overview-02-attribution.md).

[`docs/`](docs/index.rst) is the PUBLIC guide (Sphinx RST). Write there only when a change
affects end-user capabilities or usage. Keep it lean — [dev-06](techs/dev-06-public-docs.md).

## Rules that cost the most when broken

- **Code carries little prose.** WHY goes in the owning `techs/` chapter, in the same commit.
  If a change contradicts `techs/`, fix the chapter in the same commit (the `techs-check`
  pre-commit hook names it) — [overview-01](techs/overview-01-concepts.md#code-carries-little-prose).
- **Documentation is in-file.** Keep doc-strings, in `docs/` style —
  [overview-01](techs/overview-01-concepts.md#documentation-is-in-the-file).
- **Public API is stable.** Deprecate (`sisl.messages.deprecate*`) before removing or renaming —
  [dev-04](techs/dev-04-conventions.md#public-and-private-names).
- **Precision first, then performance.** Re-check correctness after each change; hot paths need
  regression tests — [overview-01](techs/overview-01-concepts.md#precision-is-a-requirement),
  [dev-03](techs/dev-03-performance.md).
- **Verify cheapest first**, and report what was not run (IO tests without `SISL_FILES_TESTS`
  are xfail-ed, not passed) — [dev-05](techs/dev-05-verifying-a-change.md).
- **Few strong classes.** Prefer a small generalization over many subclasses —
  [overview-01](techs/overview-01-concepts.md#few-strong-classes).

## Quick commands

```bash
# First install (builds Cython + Fortran extensions)
python -m pip install -e .
# Faster rebuilds afterwards; required after editing .pyx/.pxd/.f90
python -m pip install -e . --no-build-isolation -Cbuild-dir=build-editable
# Tests (set SISL_FILES_TESTS for IO tests, SISL_NUM_PROCS=1 when debugging)
pytest src/sisl/_core/tests/test_geometry.py -k rotate
pytest -m "not slow"
# Code tools: short test report (1 thread), environment check, symbol outline/source
python tools/dev/t.py src/sisl/_core/tests/test_geometry.py -k rotate
python tools/dev/envcheck.py [path...]
python tools/dev/sym.py <file> [Class.method] [--sig]
# GitHub conversations (no credentials; stored in gh/): sync once, then read items
python tools/gh.py sync
python tools/gh.py issue <N>   # or: pr <N>, discussion <N> (needs GITHUB_TOKEN)
```

Details: [dev-01 · Build](techs/dev-01-build.md), [dev-02 · Testing](techs/dev-02-testing.md),
[dev-11 · Code tools](techs/dev-11-code-tools.md), [dev-12 · GitHub archive](techs/dev-12-gh-archive.md),
[dev-04 · Conventions](techs/dev-04-conventions.md) (license header, typing, dependencies,
pre-commit, changelog fragments in `changes/`).

## Agent rules

**Each session** — [agent-01 · Harness](techs/agent-01-harness.md):

- **Ask first, then wait:** Cython ports, fixes to slow idioms, new required dependencies,
  changes to CI, pre-commit or build hooks, public API changes.
- **Issues and pull requests:** for each reference (`I123`, `#123`, `PR456`), search
  `gh/issues/N.md` or `gh/pulls/N.md` (closed ones in `closed/N.md`) of the main checkout.
- **Report** what you did, the branch and commits, and each check you did not run, and why.

**Branch, worktree, commit** — [agent-02](techs/agent-02-git.md), extends [dev-07](techs/dev-07-git.md):

- **Worktrees** only in `<repo-root>/wt/<branch-name>`: `git worktree add -b <branch-name> wt/<branch-name>`.
  You can always make one there; do not ask first.
- **Small commits**, each a working step that passes pre-commit and its targeted tests.
  Do not make one large commit at the end. Keep moves and renames apart from changes of behavior.
- **Messages:** `<area>: <imperative summary>`, at most 72 characters; the body tells why.
- **Stage explicitly.** Never `git add -A` / `git add .`: untracked files are personal scratch.
  Do not touch untracked files you did not make.
- **Trailer:** only one, `--trailer "Assisted-by: <agent>:<model>[:<effort>]"`,
  e.g. `Assisted-by: claude-code:claude-opus-5-5`. Never `Co-authored-by` or other trailers.
- **Prompts:** quote each user prompt that caused the commit, without change, below one `Prompt:` line;
  one empty line without `>` between prompts; secrets and local data become `<...>`.
- **Sources:** after each prompt, the online documents and user material read for it,
  by importance 10..1: `DOI[10]: <doi>`, `URL[7]: <url>`, `FILE[4]: <path>` (+ author, date, description).
- **History:** do not merge into `main`, force-push, or rewrite commits you reported.
  Merge only when a human tells you to, with `git merge --no-ff` (no trailer, no prompt),
  and only after a review without findings.
- **Do not push** unless a human tells you to. Commit locally and report the branch.

**Build, test, other agents** — [agent-03](techs/agent-03-parallel.md),
with [dev-01](techs/dev-01-build.md#editable-installs-and-worktrees):

- **One agent, one worktree.** Do not change files in the main checkout.
- **Find the sisl that runs.** An editable install binds `import sisl` to one tree:
  run `<env>/bin/python tools/dev/envcheck.py` before you test. A pass of other code is not a pass.
- **Build in your own environment** (a virtual environment outside the repository).
  Never install a worktree into the shared environment.
- **No `git stash`** (shared by all worktrees). Scratch files go outside the repository.
- **Time tests with 2 agents or fewer** (also to change tests because of timings), or ask first.
  Power and efficiency cores skew the timings. Set `SISL_NUM_PROCS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` to 1: CI systems have few threads.

**Implement, fix, refactor, test** — [agent-04](techs/agent-04-implement.md), with [dev-02](techs/dev-02-testing.md):

- **Read the owning chapter** first: `python tools/techs_check.py --owner <path>...`.
- **Read only the code you need:** `python tools/dev/sym.py <file> [Class.method] [--sig]`,
  not `cat` or guessed `sed -n` ranges.
- **Find the module that runs:** `_x.py` and `_x.pyx` are distinct; `python tools/dev/envcheck.py <path>`.
- **Lean code:** no unneeded options, abstractions, checks or copies of sisl functions.
  Read your diff and remove what is not needed.
- **Fast tests, no redundant tests:** CI runs each test many times. Smallest system; extend or parametrize an existing test; mark slow tests `slow`.
- **When complete:** ask in one message: 1. review the branch ([agent-06](techs/agent-06-review.md))?
  2. merge if the review passes? 3. remove the branch and the worktree after the merge?
  Never merge without a review; with findings, report and stop. Without a merge, ask to write the
  [merge request description](techs/dev-07-git.md#merge-request-description):
  overview line, problem, lean details; a bug fix names its test that fails before the fix.

**New code and features** — [agent-08](techs/agent-08-ambition.md), with agent-04:

- **Better, not bigger:** raise the outcome (altitude), the standard (depth) or the methods (range);
  do not add options or side tasks.
- **Plan first, then wait:** goal beyond the goal, the types that apply, what is missing, the outside judge
  (known value, invariant, convergence, reference output, failing test, benchmark).
  The maintainer selects the step; do not implement before the answer.
- **Use the expert's methods, not words.** Set a hard target, report true numbers, never invent results.
- **Ambition raises the standard, not the permissions:** the ask-first rules stay in force.

**Debug** — [agent-05](techs/agent-05-debugging.md), extends [dev-08](techs/dev-08-debugging.md):

- **A failing test**, in any run: find why, do not assume. If its physics is incorrect, ask.
  Do not change code to pass a test without a correct physics test.
- **Reproduce first** with the smallest system and a known value; script outside the repository.
- **Make it repeatable:** thread variables to 1, fixed seeds, the test alone.
- **Rule out the environment:** the sisl that runs, `.py` vs `.pyx`, stale builds, `SISL_FILES_TESTS`.
- **Find the cause**, not the symptom; `git bisect` against `main`; read the owning chapter.
- **Do not hide it:** no larger tolerances, `xfail`, `skip`, broad `except` or weaker asserts,
  unless the maintainer agrees.
- **The reproduction becomes the test** that fails before the fix, in the same commit as the fix.

**Review** — [agent-06](techs/agent-06-review.md), extends [dev-09](techs/dev-09-review.md):

- **Compare with `main`** (`git diff main...HEAD`) unless told otherwise, and review
  each commit in `main..HEAD`, not only the final diff.
- **Read before you run:** changes to `conftest.py`, build files, pre-commit hooks and `tools/`.
- **No code unrelated to sisl:** report scratch files, tool configuration, copied, generated or unexplained code.
- **No trojan horses:** examine every commit, also merges (`git log -p --remerge-diff`), for the
  [signs](techs/dev-10-security.md#signs-of-a-trojan-horse). Report each sign first.
- **Report, do not fix**, unless the maintainer tells you to. Never rewrite history.

**Dependencies, CI, hooks, build files, outside data, and each review** — [agent-07](techs/agent-07-security.md),
extends [dev-10](techs/dev-10-security.md):

- **No URL lookups** below `src/`. Only a human can add one: stop and tell the maintainer.
- **Do not run file content:** no `eval`/`exec`/`pickle` on data that sisl reads.
- **Dependencies:** none required without approval; no package indexes, git URLs or unpinned hooks.
- **No secrets and no local data:** credentials, absolute paths, user and host names.
- **Signs of a trojan horse:** network access, dynamic code, hidden content and Unicode, binaries,
  code that runs automatically, tree changes, code added then removed, new dependency sources.
  Check your own commits too, and explain each sign.

When you change a rule in an `agent-*` chapter, or a `dev-*` rule in this summary,
change this summary in the same commit.
