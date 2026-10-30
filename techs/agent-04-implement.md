# agent-04 · Implement

For agents that implement, fix, refactor or test code.
Follow [agent-07 · Security](agent-07-security.md) for each change.

## Read the owning chapter

Before you change a file, read the `techs/` chapter that owns it ([README · Ownership](README.md#ownership)).
It tells the decisions that your change must keep.

```bash
python tools/techs_check.py --owner <path>...
```

## Read only the code that you need

Read the outline of a file first. Then read only the symbols that you need ([dev-11 · sym](dev-11-code-tools.md#sym)).
Do not use `cat`, or `sed -n` with a guessed line range, to find one function.
Each line that you read costs context, and large reads push out the rules and the task.

```bash
python tools/dev/sym.py <file>                 # outline with line ranges
python tools/dev/sym.py <file> Class.method    # source of one symbol
python tools/dev/sym.py <file> name --sig      # signature and doc-string
```

## Find the module that runs

Before you change a `.py` or a `.pyx` file, find the module that the code imports:
`python tools/dev/envcheck.py <path>`.
See [dev-01 · `.py` and `.pyx` companions](dev-01-build.md#py-and-pyx-companions).

## Write lean code

Write only the code that the task needs. Agents tend to write too much code,
and each line costs time to read, test and keep.

- Do not add options, parameters or abstractions that the task does not need.
- Do not add a helper with only one caller, unless it makes the code clearer.
- Do not add checks for cases that cannot occur, or checks that a caller already does.
- Use the functions that sisl has. Do not copy them.
- Prefer a small generalization to a new class ([overview-01 · Few strong classes](overview-01-concepts.md#few-strong-classes)).
- Before you commit, read your diff. Remove dead code, debug output and lines that the task does not need.

## Tests and verification

Before you evaluate or test, check that `import sisl` loads the code of your worktree
([agent-03 · Test the code of your worktree](agent-03-parallel.md#test-the-code-of-your-worktree)).
To measure test timings, follow [agent-03 · Time tests](agent-03-parallel.md#time-tests-with-a-maximum-of-2-agents).
Follow [dev-02 · Keep tests fast](dev-02-testing.md#keep-tests-fast),
[dev-02 · What a new test must do](dev-02-testing.md#what-a-new-test-must-do)
and the order of checks in [dev-05 · Verifying a change](dev-05-verifying-a-change.md).
Then report as [agent-01](agent-01-harness.md#report-what-you-did-not-run) tells.

## When the code is complete

Do these steps in this order. Wait for the answer of the maintainer at each step.

1. **Ask if the code is complete.** The maintainer decides when the implementation stops.
2. **Review the branch** as [agent-06 · Review](agent-06-review.md) tells. Report the findings.
   Fix them only when the maintainer tells you to.
   A review of the full change finds problems that the small steps do not show.
3. **Ask to write the merge request description**
   ([dev-07](dev-07-git.md#merge-request-description)).
   Do not push and do not open the merge request ([agent-02](agent-02-git.md#do-not-push)).
