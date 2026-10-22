# release-02 · Release process

Owns: `DEPLOY.md`, `tools/changelog.py`, `tools/tag.sh`, `docs/release.rst`, `docs/release/*`

`DEPLOY.md` holds the steps of a release. This chapter holds the decisions behind them.
Only the maintainer makes a release.

## A release commit has no code changes

The release commit changes only release files:

- `CITATION.cff`: version and date. See [release-01 · Citation](release-01-versioning.md#citation).
- `fallback_version` in `pyproject.toml`. See [release-01 · The version comes from git](release-01-versioning.md#the-version-comes-from-git).
- The release notes in `docs/release/` and the entry in `docs/release.rst`.
- The news fragments in `changes/` that towncrier removes.

Commit all code changes before. Then the tag points to code that the CI tested.

## Release notes

- During development, each user-facing change adds a news fragment.
  See [dev-04 · Changelog](dev-04-conventions.md#changelog).
- At the release, `towncrier build --version <version>` assembles the fragments into
  `docs/release/<version>-notes.rst` and removes them. See `[tool.towncrier]` in `pyproject.toml`.
- `tools/changelog.py` adds the list of pull requests and authors from GitHub.
  It needs a GitHub token. It is a release tool. Do not call it from `src/`.

## Publish

- Push the tag `v<version>`. The CI builds the wheels and publishes them.
  See [release-03 · Wheels](release-03-ci.md#wheels).
- conda packages come from the conda-forge `sisl-feedstock`.
  Update the version and the `sha256` in its `recipe/meta.yaml` with a merge request.

## Known stale parts

- `tools/tag.sh` changes `setup.py`. sisl has no `setup.py` now. Do not use the script.
- `DEPLOY.md` tells to change the version in `pyproject.toml`. Only `fallback_version` is there.
  The version comes from the tag.
