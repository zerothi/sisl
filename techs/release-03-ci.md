# release-03 · CI

Owns: `.github/*`

The CI runs on GitHub Actions. The workflows are in `.github/workflows/`.
A change to them can publish packages or use secrets. Thus only the maintainer approves such a change.

## Workflows

| Workflow | Runs on | Does |
|---|---|---|
| `test.yaml` | push to `main` and pull request that change `.py` or `.pyx` files, schedule, manual | Calls `linter.yml` first. Then tests with coverage on many platforms. Sets `SISL_NUM_PROCS` and `OMP_NUM_THREADS` to 1. |
| `linter.yml` | call from `test.yaml` | Lint checks. A failure stops the tests. |
| `wheels.yaml` | tag `v*`, manual | Builds the sdist and the wheels. Publishes them for a tag. |
| `documentation_ghpages.yaml` | schedule, manual | Builds the Sphinx documentation and deploys it to GitHub Pages. |
| `codeql.yml` | pull request to `main`, manual | Static security analysis. |

`dependabot.yaml` proposes updates of GitHub Actions and pip dependencies each week.

## Wheels

A push of a tag `v*` starts this sequence in `wheels.yaml`:

1. Build the sdist and the wheels for each platform.
2. Publish them on TestPyPI.
3. Install sisl from TestPyPI on many platforms, and test the installation.
4. Publish them on PyPI.

The uploads use the secrets `TESTPYPI_TOKEN` and `PYPI_TOKEN`.
Step 3 stops a broken wheel before it gets to PyPI.

## Rules for a change

- Pin each action to a version, as the other actions are.
- Do not give a job more `permissions` than it needs.
- Do not print a secret, and do not give it to a step that does not need it.
- Keep `SISL_NUM_PROCS` and `OMP_NUM_THREADS` of `test.yaml` at 1. The CI computers have few threads.
