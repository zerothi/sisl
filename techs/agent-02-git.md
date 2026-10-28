# agent-02 · Git workflow

This chapter is for agents only. Human developers can skip it.
It gives the rules for how an agent uses git: branches, worktrees, commits and the history.
Read [agent · Welcome](agent-welcome.md) and [agent-01 · Harness](agent-01-harness.md) first.

Follow [dev-07 · Git workflow](dev-07-git.md): worktrees, small commits, commit messages and the history.
This chapter adds the rules for agents.

## Worktrees

Put each worktree in `<repo-root>/wt/<branch-name>`, as [dev-07 · Worktrees](dev-07-git.md#worktrees) tells.
Do not put a worktree in a different folder, for example a folder of your tool.
Then the maintainer can find each worktree and its branch in one place.

## Commit in small steps

Follow [dev-07 · Commit in small steps](dev-07-git.md#commit-in-small-steps).
Do not collect the work of a full task in one commit.

## Stage explicitly

- Stage each file by its name: `git add <path>`.
- Do not use `git add -A` or `git add .`.
- Do not change, move or delete untracked files that you did not make.

The maintainer keeps personal scratch files and folders as untracked files in the checkout.
A broad `git add` puts them in the commit.

## Commit trailer

Mark each agent commit with one `Assisted-by` trailer. Do not use other trailers in the message.
Do not use `Co-authored-by`.

```bash
git commit ... --trailer "Assisted-by: <agent>:<model>[:<effort>]"
# example
git commit ... --trailer "Assisted-by: claude-code:claude-opus-5-5"
```

One trailer format makes agent commits easy to find in the history:
`git log --grep='^Assisted-by:'`.

## User prompts in the message

Put the user prompts that caused the commit in the message body, before the trailer.
Copy each prompt without change, as a quote below a `Prompt:` line.

```text
techs: add rule for user prompts in commits

Prompt:
> add agent rule.
>
> Always store the user-prompts in the commit messages to
> enable transparency.

Assisted-by: claude-code:claude-opus-5-5
```

If more prompts caused the commit, put each prompt in the message, in the order of the session.
Replace secrets, user names, host names and absolute paths with `<...>`,
as [agent-07 · Security](agent-07-security.md) tells.

The prompts make the work of an agent transparent.
A reviewer can compare the request with the change.

## Keep the history

Follow [dev-07 · Keep the history](dev-07-git.md#keep-the-history). In addition:

- Do not merge into `main`, unless a human tells you to merge. Then follow [Merge](#merge).
- You can amend, reorder or squash your own commits only before you report them.
  After you report a commit, do not rewrite it. Add a new commit.
- Do not use `git reset --hard` on a branch that you did not make.

## Merge

Merge only when a human tells you to merge.

- Use `git merge --no-ff <branch>`. The merge commit must stay in the history.
- The merge message does not need the `Assisted-by` trailer. `git merge` has no `--trailer` option.
- The merge message does not need the user prompts.

A merge commit keeps the commits of a branch together. The maintainer can find, review or revert
the full branch with one commit.
The prompt for a merge is usually short and gives no context. The commits of the branch keep their
trailers and prompts.

## Do not push

Do not push to a remote. Push only when a human tells you to push.
Approval for one push is not approval for the next push.

Commit your work locally. In your report, tell the branch name and the commits.
The maintainer reviews the commits and decides when to push.
