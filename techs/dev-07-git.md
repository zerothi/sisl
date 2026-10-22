# dev-07 · Git workflow

This chapter gives the rules for branches, worktrees, commits and the history.

## Worktrees

Put each git worktree in `<repo-root>/wt/<branch-name>`:

```bash
# from <repo-root>
git worktree add -b <branch-name> wt/<branch-name>
```

Use the same name for the branch and the folder.
Then each worktree and its branch are in one place.
`.gitignore` contains `/wt/`, thus `git status` does not show the worktrees.

- Give the branch a short name in lowercase with hyphens. The name tells the work, for example `fix-sisl-tmp-isolation`.
- Start the branch from `main`, unless the maintainer names a different start point.
- A worktree does not have its own build.
  See [dev-01 · Editable installs and worktrees](dev-01-build.md#editable-installs-and-worktrees).

## Commit in small steps

Make many small commits. Do not make one large commit at the end of the work.

- One commit holds one step. Examples: a move or rename, a refactor without a change of behavior,
  a fix and its test, a new feature and its tests.
- Commit each step when it works. Do not wait until the full task is done.
- Each commit must pass pre-commit and the targeted tests of the code it changes.
  `git bisect` needs a history where each commit works.
- Put a fix and the test that shows the fix in the same commit.
- Put the change to the owning `techs/` chapter in the same commit as the code.
  See [overview-01 · Code carries little prose](overview-01-concepts.md#code-carries-little-prose).
- Do not mix a change of behavior with a move, a rename or a format change.
  Use `git mv` for a move, so that the history follows the file.

Small commits are easy to review, to revert and to bisect.
The maintainer can take some commits and not others.

## Commit messages

- Write the subject as `<area>: <what the commit does>`, for example `tests: isolate sisl_tmp per test`.
  Use the imperative. Use a maximum of 72 characters.
- The area is a folder or a topic: `techs`, `tests`, `docs`, `io`, `physics`, `build`.
- In the body, tell why. Do not repeat the diff.
  Name the `techs/` chapter if the decision is there.

## Keep the history

- The maintainer merges into `main`.
- Do not rewrite commits that other people have seen. Add a new commit.
- Do not use `git push --force` or `git rebase` on a branch that others use.
