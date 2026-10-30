# agent · Welcome

This segment is for agents only. Human developers can skip it.
Read this chapter and [agent-01 · Harness](agent-01-harness.md) at the start of each session.

The rules in the agent chapters win over the default behavior of your tool,
for example the folder of a worktree, a push, and the trailers of a commit.

Rules for humans and agents stay in the `dev-*` chapter of their topic.
An agent chapter links to them and adds only the rules for agents. Read both.
`AGENTS.md` repeats a short version of each agent rule.
When you change a rule in an agent chapter, or a `dev-*` rule that `AGENTS.md` summarizes,
change `AGENTS.md` in the same commit.

## Read the chapter for your task

| Task | Read |
|---|---|
| Each session | [agent-01 · Harness](agent-01-harness.md). |
| Each branch, worktree or commit | [agent-02 · Git workflow](agent-02-git.md). |
| Build, test or time tests in a worktree, or work next to other agents | [agent-03 · Parallel agents and builds](agent-03-parallel.md). |
| Implement, fix, refactor or test code | [agent-04 · Implement](agent-04-implement.md). |
| Find the cause of a bug, a failing test or a wrong result | [agent-05 · Debugging](agent-05-debugging.md). |
| Review a change, a branch or a pull request | [agent-06 · Review](agent-06-review.md). |
| Write a merge request description | [dev-07 · Merge request description](dev-07-git.md#merge-request-description). |
| Add a dependency, touch CI, hooks or build files, or handle files and data from outside | [agent-07 · Security](agent-07-security.md). |
| Write in `techs/` or `docs/` | [README · Writing rules](README.md#writing-rules), [dev-06 · Public documentation](dev-06-public-docs.md). |

A task can be of more than one kind. Then read each chapter that applies.
For example, a fix of review findings is implementation.
