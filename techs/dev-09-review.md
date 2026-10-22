# dev-09 · Review

This chapter tells how to review a change, a branch or a pull request.

## Compare with main

Use `main` as the reference, unless the author or the maintainer names a different reference.

```bash
git log --format='%h %an %s' main..HEAD   # the commits under review
git diff --stat main...HEAD                # the change of the branch, from the merge base
```

- Use three dots in `git diff`. Then the diff starts at the merge base,
  and changes on `main` after the branch point are not shown.
- Review each commit in `main..HEAD`, not only the final diff.
  See [No trojan horses](#no-trojan-horses).

## Read before you run

Some files run code when you test or build the branch:

- `pytest` imports each `conftest.py`.
- `pip install` runs `pyproject.toml`, `CMakeLists.txt` and the files they include.
- `pre-commit` runs the hooks in `.pre-commit-config.yaml` and the scripts in `tools/`.

Read the changes to these files before you run a test, a build or a hook on the branch.
If a change is not clear, do not run it. Report it.

## What to check

Check the change against the rules of the chapters that own the changed files.
`python tools/techs_check.py` names them. In addition, check:

- **Precision.** The numbers are correct, and the dtype is kept.
  See [overview-01 · Precision is a requirement](overview-01-concepts.md#precision-is-a-requirement).
- **Public API.** A removed or changed public name has a deprecation.
  See [dev-04 · Public and private names](dev-04-conventions.md#public-and-private-names).
- **Tests.** A fix has a test that fails before the fix. The tests are fast and not redundant.
  See [dev-02 · What a new test must do](dev-02-testing.md#what-a-new-test-must-do)
  and [dev-02 · Keep tests fast](dev-02-testing.md#keep-tests-fast).
- **Performance.** A change in a hot path has a regression test and a measurement.
  See [dev-03 · Performance](dev-03-performance.md).
- **Prose.** The WHY is in `techs/`, not in long code comments. Doc-strings are kept.
  See [overview-01 · Code carries little prose](overview-01-concepts.md#code-carries-little-prose).
- **Documentation.** A user-facing change has a news fragment and, if necessary, a change in `docs/`.
  See [dev-04 · Changelog](dev-04-conventions.md#changelog) and [dev-06 · Public documentation](dev-06-public-docs.md).
- **Dependencies.** A new required dependency has the approval of the maintainer.
  See [dev-04 · Dependencies](dev-04-conventions.md#dependencies).
- **Security.** The change follows [dev-10 · Security](dev-10-security.md).

## No code unrelated to sisl

Each added file and each added line must serve sisl or `sisl_toolbox`.
Report all other content as a finding. Examples:

- Personal scratch: notebooks, ad hoc scripts, profiles, plots, output files.
- Configuration of a tool or an editor, for example `.claude/`, `.vscode/` or `.idea/`.
- Code or data from a different project, copied into the tree.
- Generated files that the build makes, for example `.c` files from Cython or compiled libraries.
- Debug output, dead code, and code that the commit message does not explain.

## No trojan horses

Git keeps each commit. A later commit that removes code does not remove it from the history.
`git checkout`, `git bisect` and a release from an old tag can run it.
Thus examine each commit in `main..HEAD`, also merge commits.

```bash
# each commit, with the changes of merge commits that no parent has
git log -p --remerge-diff main..HEAD
# added, removed, renamed files, and changes of the file mode
git diff --summary main...HEAD
git log --summary --format='%h %s' main..HEAD
# hidden and bidirectional Unicode characters
git log -p --remerge-diff main..HEAD \
  | LC_ALL=C.UTF-8 grep -nP '[\x{200B}-\x{200F}\x{202A}-\x{202E}\x{2060}-\x{2069}\x{FEFF}]'
```

Look for each sign in [dev-10 · Signs of a trojan horse](dev-10-security.md#signs-of-a-trojan-horse).
Check also the other rules of [dev-10 · Security](dev-10-security.md).

A sign is not always an attack. Report each sign, its commit and why the code needs it.
The maintainer decides if the history must change.

## Report the review

- Give each finding its commit, its file and line, and a short failure scenario.
- Put the most severe finding first. A trojan horse sign is always first.
- Name each check that you did not do, and say why.
