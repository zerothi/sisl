# agent-02 · Git workflow

For agents. Extends [dev-07 · Git workflow](dev-07-git.md): worktrees, small commits, messages and the history.

## Worktrees

Put each worktree in `<repo-root>/wt/<branch-name>`, as [dev-07 · Worktrees](dev-07-git.md#worktrees) tells.
Do not use a different folder, for example a folder of your tool.
Then the maintainer finds each worktree and its branch in one place.

## Commit in small steps

Follow [dev-07 · Commit in small steps](dev-07-git.md#commit-in-small-steps).
Do not collect the work of a full task in one commit.

## Stage explicitly

Stage each file by its name: `git add <path>`. Do not use `git add -A` or `git add .`.
Do not change, move or delete untracked files that you did not make.
The maintainer keeps personal scratch files and folders as untracked files in the checkout.
A broad `git add` puts them in the commit.

## Commit trailer

Mark each agent commit with one `Assisted-by` trailer. Do not use other trailers, also not `Co-authored-by`.
Then `git log --grep='^Assisted-by:'` finds all agent commits.

```bash
git commit ... --trailer "Assisted-by: <agent>:<model>[:<effort>]"
# example
git commit ... --trailer "Assisted-by: claude-code:claude-opus-5-5"
```

## User prompts in the message

Copy each user prompt that caused the commit into the message body without change, before the trailer,
as a quote below one `Prompt:` line, in the order of the session.
Between two prompts, put one empty line without `>`. In one prompt, an empty line is `>`.
Replace secrets, user names, host names and absolute paths with `<...>` ([agent-07](agent-07-security.md)).
A reviewer can then compare the request with the change.

```text
techs: add rule for user prompts in commits

Prompt:
> add agent rule.
>
> Always store the user-prompts in the commit messages.

> commit

Assisted-by: claude-code:claude-opus-5-5
```

## Sources in the message

After each prompt, list the online documents and the user material that you read for it, one line each.
Sort by importance for the decision, from 10 (the basis) to 1 (supplementary).

```text
Prompt:
> Please read doi:10.1234/abc and implement the method

DOI[10]: 10.1234/abc
FILE[4]: <path>
 author: <author>
 date: <document date>
 description: <small paragraph of file content/topic>
URL[3]: https://www.example.org/page
```

- `DOI` if the document has a DOI. `https://doi.org/<DOI>` must open the document.
- `URL` for an online document without a DOI.
- `FILE` for user material without a DOI. Use a path relative to the repository, or only the file name.

## Keep the history

Follow [dev-07 · Keep the history](dev-07-git.md#keep-the-history). In addition:

- Do not merge into `main` unless a human tells you to.
- Amend, reorder or squash your own commits only before you report them. After that, add a new commit.
- Do not use `git reset --hard` on a branch that you did not make.

## Merge

Merge only when a human tells you to, with `git merge --no-ff <branch>`.
The merge commit must stay in the history.
It keeps the branch together: the maintainer can review or revert it as one commit.
The merge message needs no `Assisted-by` trailer (`git merge` has no `--trailer` option) and no user prompts.
The commits of the branch keep them.

## Do not push

Push only when a human tells you to. Approval for one push is not approval for the next push.
Commit locally, and report the branch name and the commits.
