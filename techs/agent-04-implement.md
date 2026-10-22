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
