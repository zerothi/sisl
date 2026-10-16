# 16 · Agents

This chapter is for agents only. Human developers can skip it.
It gives the rules for how an agent works in this repository.

Rules for humans and agents stay in the chapter of their topic.
This chapter does not repeat them. It links to them.
`AGENTS.md` repeats a short version of each rule here, so that agents find them fast.
When you change a rule here, change `AGENTS.md` in the same commit.

## Ask first

Ask the maintainer before you do one of these changes:

- **Port code to Cython.** A port adds build cost and makes the code harder to read.
  The maintainer decides if the speed-up is worth it. See [12 · Performance](12-performance.md).
- **Fix a slow idiom.** Code that looks slow can be slow for a reason, for example precision.
  See [02 · Precision is a requirement](02-concepts.md#precision-is-a-requirement).
- **Add a required dependency.** See [13 · Dependencies](13-conventions.md#dependencies).
- **Change the public API.** See [13 · Public and private names](13-conventions.md#public-and-private-names).

Tell what you want to change and why. Then wait for the answer.

## No URL lookups

Do not add code below `src/` that reads from a URL.
This rule applies to all files below `src/`, also to the tests there.
Examples are `urllib`, `requests`, `httpx`, `pooch`, or a call to `curl` or `wget`.

Only a human can make this change.
If a task needs a URL lookup, stop and tell the maintainer.

- sisl runs on computational resources. Many compute nodes have no internet access.
- A result must not change when a remote file changes.
- Data from a URL is not trusted.

## Stage explicitly

- Stage each file by its name: `git add <path>`.
- Do not use `git add -A` or `git add .`.

The maintainer keeps personal scratch files and folders as untracked files in the checkout.
A broad `git add` puts them in the commit.

## Commit trailer

Mark each agent commit with the `ai` trailer. Do not use other trailers in the message.
Do not use `Co-authored-by`.

Load the trailer definition one time:

```bash
git config include.path ../.gitconfig
```

Then commit:

```bash
git commit ... --trailer ai=<agent>:<model>[:<effort>]
# example
git commit ... --trailer ai=claude-code:claude-opus-5-5
```

`.gitconfig` writes the trailer as `Assisted-by:`.
One trailer format makes agent commits easy to find in the history.

## Find the module that runs

A file `_x.py` and a file `_x.pyx` with the same name are two different modules.
Before you change one, find the module that the code imports.
A change to the other file has no effect.

## Report what you did not run

Follow the order of checks in [14 · Verifying a change](14-verifying-a-change.md).
In your report, name each check that you did not run, and say why.

- An IO test without `SISL_FILES_TESTS` is xfail-ed. It did not pass.
  See [11 · Testing](11-testing.md#external-test-files).
- A `.pyx`, `.pxd` or Fortran change without a rebuild was not tested.
