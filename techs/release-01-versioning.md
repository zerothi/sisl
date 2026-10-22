# release-01 · Versioning

Owns: `CITATION.cff`

## The version comes from git

`setuptools_scm` makes the version from the last git tag. See `[tool.setuptools_scm]` in `pyproject.toml`.

- A release is the tag `v<MAJOR>.<MINOR>.<MICRO>` on a commit of `main`, for example `v0.20.0`.
- Between releases, the version is a development version, for example `0.20.1.dev3+g1234abc`.
- The build writes `src/sisl/_version.py`. Git ignores it. Do not edit it.
- `fallback_version` in `pyproject.toml` is used only when git cannot give a version,
  for example in a source copy without `.git`. Set it to the last release.

Do not write the version in a source file. One source of the version prevents two versions that do not agree.

A change of the public API needs a deprecation first. See [release-04 · Deprecations](release-04-deprecations.md).

## Citation

`CITATION.cff` gives `version` and `date-released` for Zenodo.
Change both in the release commit. See [release-02 · Release process](release-02-process.md).
