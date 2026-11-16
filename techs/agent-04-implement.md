# agent-04 · Implement

For agents that implement, fix, refactor or test code.
Follow [agent-07 · Security](agent-07-security.md) for each change.

## Aim higher for new code

Before you make new code, a new feature or a plan for one, follow [agent-08 · Ambition](agent-08-ambition.md).
Find the goal beyond the goal and an outside judge. Make the result better, not bigger.
A small fix or a refactor does not need it.

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

## Own the code that you make

You own the code that you implement, and you are its first user.
Before you tell that the code is complete, make its objects and extend them with the public API,
as a user does. If this is difficult, the API is not complete. Change the API, not the test.

The API must be simple to use, and also flexible:

- **Simple.** The usual case needs only a small number of arguments, and the defaults give a correct result.
- **Flexible.** A user can extend the code to a new case without a change to the code.
  Get this from general inputs, for example array-like values or a callable,
  not from more options ([Write lean code](#write-lean-code)).
- **Public only.** Tests and examples use only the public API.
  If they must set a private attribute, the API is not complete.

A change to the public API still needs approval ([agent-01 · Ask first](agent-01-harness.md#ask-first)).

## Tests and verification

Before you evaluate or test, check that `import sisl` loads the code of your worktree
([agent-03 · Test the code of your worktree](agent-03-parallel.md#test-the-code-of-your-worktree)).
To measure test timings, follow [agent-03 · Time tests](agent-03-parallel.md#time-tests-with-a-maximum-of-2-agents).
Follow [dev-02 · Keep tests fast](dev-02-testing.md#keep-tests-fast),
[dev-02 · What a new test must do](dev-02-testing.md#what-a-new-test-must-do)
and the order of checks in [dev-05 · Verifying a change](dev-05-verifying-a-change.md).
Then report as [agent-01](agent-01-harness.md#report-what-you-did-not-run) tells.

## When the code is complete

When you think that the code is complete, ask the maintainer these questions in one message:

1. **Review?** Review the branch as [agent-06 · Review](agent-06-review.md) tells.
2. **Merge if the review passes?** Merge into `main` as [agent-02 · Merge](agent-02-git.md#merge) tells.
3. **Remove the branch and the worktree after the merge?**

The maintainer decides when the implementation stops. Wait for the answer.
Then do the selected steps in the order of the list:

- A review of the full change finds problems that the small steps do not show.
  Never merge new code without a review.
- When the review has findings, report them and stop. Do not merge.
  Fix them only when the maintainer tells you to. Then review again.
- When the maintainer does not select the merge, ask to write the merge request description
  ([dev-07](dev-07-git.md#merge-request-description)).
  Do not push and do not open the merge request ([agent-02](agent-02-git.md#do-not-push)).
