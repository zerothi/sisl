# agent-03 · Parallel agents and builds

This chapter is for agents only. Human developers can skip it.
It gives the rules for agents that work at the same time, and for builds in a worktree.
Read [agent · Welcome](agent-welcome.md) and [agent-01 · Harness](agent-01-harness.md) first.

## One agent, one worktree

- Each agent works in its own worktree. See [dev-07 · Worktrees](dev-07-git.md#worktrees).
- Two agents do not work in the same worktree.
- The main checkout is the checkout of the maintainer.
  Do not change files there, unless the maintainer tells you to.

## Test the code of your worktree

Follow [dev-01 · Editable installs and worktrees](dev-01-build.md#editable-installs-and-worktrees).
In addition:

- Before you evaluate or test, run `<env>/bin/python tools/dev/envcheck.py` in your worktree.
  A `FAIL` can mean that your tests do not test your code. Correct it, or report it.
  See [dev-11 · envcheck](dev-11-code-tools.md#envcheck).
- The shared environment is the environment of the maintainer and the other agents.
  Never install your worktree into it.
- A build takes minutes and much CPU. If you cannot build, tell the maintainer.
  Then report that the tests did not run.

## Shared state

Some state is shared by all worktrees and all agents.

- `git stash` is shared by all worktrees. Do not use it. Use a temporary commit on your branch.
- Put scratch files outside the repository, for example in the temporary folder of your tool.
  Do not put them in the main checkout or in a different worktree.
- The CPU is shared. Builds and test runs of many agents make each other slow.

## Time tests with a maximum of 2 agents

When you measure test timings, run a maximum of 2 agents at the same time.
This rule also applies when you change tests because of their timings, for example to add a `slow` mark.
If you need more agents, ask the maintainer first.

More agents put a load on the computer, and then the timings are not correct.
Measure as [dev-02 · Measure test timings](dev-02-testing.md#measure-test-timings) tells.
