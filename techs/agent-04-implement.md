# agent-04 · Implement

This chapter is for agents only. Human developers can skip it.
It gives the rules for an agent that implements, fixes, refactors or tests code.
Read [agent · Welcome](agent-welcome.md) and [agent-01 · Harness](agent-01-harness.md) first.

## Read the owning chapter

Before you change a file, read the `techs/` chapter that owns it:

```bash
python tools/techs_check.py --owner <path>...
```

See [README · Ownership](README.md#ownership).
The chapter tells the decisions that your change must keep.

## Find the module that runs

Before you change a `.py` or a `.pyx` file, find the module that the code imports.
See [dev-01 · `.py` and `.pyx` companions](dev-01-build.md#py-and-pyx-companions).

## Write lean code

Write only the code that the task needs. Each line must have a function.

- Do not add options, parameters or abstractions that the task does not need.
- Do not add a helper that has only one caller, unless it makes the code clearer.
- Do not add checks for cases that cannot occur, or checks that a caller already does.
- Use the functions that sisl has. Do not write a copy of them.
- Prefer a small generalization to a new class.
  See [overview-01 · Few strong classes](overview-01-concepts.md#few-strong-classes).
- Before you commit, read your diff. Remove dead code, debug output and lines that the task does not need.

Reason: each line costs time to read, test and keep. Agents tend to write too much code.

## Keep the code safe

Follow [agent-07 · Security](agent-07-security.md) for each change.
For example, do not add code that reads from a URL, and do not commit secrets.

## Write fast tests, and no redundant tests

Follow [dev-02 · Keep tests fast](dev-02-testing.md#keep-tests-fast)
and [dev-02 · What a new test must do](dev-02-testing.md#what-a-new-test-must-do).

## Test the code of your worktree

Check that `import sisl` loads the code of your worktree, before you evaluate or test.
See [agent-03 · Test the code of your worktree](agent-03-parallel.md#test-the-code-of-your-worktree).
To measure test timings, follow [agent-03 · Time tests](agent-03-parallel.md#time-tests-with-a-maximum-of-2-agents).

## Verify the change

Follow the order of checks in [dev-05 · Verifying a change](dev-05-verifying-a-change.md).
Then report as [agent-01 · Report what you did not run](agent-01-harness.md#report-what-you-did-not-run) tells.

## When the code is complete

Do these steps in this order. Wait for the answer of the maintainer at each question.

1. **Ask if the code is complete.** The maintainer decides when the implementation stops.
2. **Review the change.** Review the branch as [agent-06 · Review](agent-06-review.md) tells.
   Report the findings. Fix them only when the maintainer tells you to.
3. **Ask to write the merge request description.** Write it as
   [dev-07 · Merge request description](dev-07-git.md#merge-request-description) tells.
   Do not push and do not open the merge request. See [agent-02 · Do not push](agent-02-git.md#do-not-push).

Reason: a review of the full change finds problems that the small steps do not show.
