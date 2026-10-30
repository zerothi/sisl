# agent-01 · Harness

For agents. Read at the start of each session.
[agent · Welcome](agent-welcome.md) tells which other agent chapters to read for your task.

## Ask first

Ask the maintainer before you do one of these changes.
Tell what you want to change and why. Then wait for the answer.

- **Port code to Cython.** A port adds build cost and makes the code harder to read.
  See [dev-03 · Performance](dev-03-performance.md).
- **Fix a slow idiom.** Code can be slow for a reason, for example precision.
  See [overview-01 · Precision is a requirement](overview-01-concepts.md#precision-is-a-requirement).
- **Add a required dependency.** See [dev-04 · Dependencies](dev-04-conventions.md#dependencies).
- **Change CI workflows, pre-commit hooks or build hooks.** See [dev-10 · Code that runs automatically](dev-10-security.md#code-that-runs-automatically).
- **Change the public API.** See [dev-04 · Public and private names](dev-04-conventions.md#public-and-private-names).

## Report what you did not run

At the end of a task, report what you did, the branch and the commits.
Name each check that you did not run, and say why.
See [dev-05 · Before you commit](dev-05-verifying-a-change.md#before-you-commit) for examples.
