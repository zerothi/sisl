# agent-06 · Review

For agents that review a change, a branch or a pull request.
Extends [dev-09 · Review](dev-09-review.md).

## Report, do not fix

Do not change code during a review, and do not rewrite the history to remove a finding.
Change code only when the maintainer tells you to. Then [agent-04 · Implement](agent-04-implement.md) applies.

## Read before you run

Read the changes as [dev-09 · Read before you run](dev-09-review.md#read-before-you-run) tells,
before you run a test, a build or a hook of the branch.
If a change is not clear, do not run it. Report it.

## Compare the requests with the change

Read the user prompts and the sources of the commits before you read the diff:
`python tools/dev/prompts.py main..HEAD` ([dev-11 · prompts](dev-11-code-tools.md#prompts)).
Then compare them with the change. Report:

- A change that no prompt asks for, for example a side task or an option.
- A prompt that the change does not do.
- A decision that cites no source, or a source that the change does not use.
- A commit of an agent without a `Prompt:` line ([agent-02](agent-02-git.md#user-prompts-in-the-message)).

## Report the review

Report as [dev-09 · Report the review](dev-09-review.md#report-the-review) tells.
Name each check that you did not do, and say why.
