# agent-03 · Parallel agents and builds

For agents that work at the same time, and for builds in a worktree.
Extends [dev-01 · Editable installs and worktrees](dev-01-build.md#editable-installs-and-worktrees).

## One agent, one worktree

Each agent works in its own worktree ([dev-07 · Worktrees](dev-07-git.md#worktrees)).
The main checkout belongs to the maintainer. Do not change files there, unless the maintainer tells you to.

## Test the code of your worktree

- Before you evaluate or test, run `<env>/bin/python tools/dev/envcheck.py` in your worktree.
  A `FAIL` can mean that your tests do not test your code. Correct it, or report it.
  See [dev-11 · envcheck](dev-11-code-tools.md#envcheck).
- Never install your worktree into the shared environment of the maintainer and the other agents.
- If you cannot build, tell the maintainer, and report that the tests did not run.

## Shared state

- Do not use `git stash`. All worktrees share it. Use a temporary commit on your branch.
- Put scratch files outside the repository, for example in the temporary folder of your tool.
- The CPU is shared. Builds and test runs of many agents make each other slow.

## Time tests with a maximum of 2 agents

When you measure test timings, or change tests because of them (for example a `slow` mark),
run a maximum of 2 agents at the same time. If you need more, ask the maintainer first.
More load makes the timings incorrect.
Measure as [dev-02 · Measure test timings](dev-02-testing.md#measure-test-timings) tells.
