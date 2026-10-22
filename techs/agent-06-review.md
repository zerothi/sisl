# agent-06 · Review

This chapter is for agents only. Human developers can skip it.
It gives the rules for an agent that reviews a change, a branch or a pull request.
Read [agent · Welcome](agent-welcome.md) and [agent-01 · Harness](agent-01-harness.md) first.

Follow [dev-09 · Review](dev-09-review.md). This chapter adds the rules for agents.

## Report, do not fix

A review finds problems and reports them. Do not change code during a review.
Change code only when the maintainer tells you to. Then [agent-04 · Implement](agent-04-implement.md) applies.

Do not rewrite the history to remove a finding. The maintainer decides.

## Read before you run

Read the changes as [dev-09 · Read before you run](dev-09-review.md#read-before-you-run) tells,
before you run a test, a build or a hook of the branch.
If a change is not clear, do not run it. Report it.

## Report the review

Report as [dev-09 · Report the review](dev-09-review.md#report-the-review) tells.
Name each check that you did not do, and say why.
See [agent-01 · Report what you did not run](agent-01-harness.md#report-what-you-did-not-run).
