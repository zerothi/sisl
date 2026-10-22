# release-04 · Deprecations

A public name changes or goes away only after a deprecation.
How to deprecate is in [dev-04 · Public and private names](dev-04-conventions.md#public-and-private-names).
This chapter tells when a deprecated name goes away.

## The life of a deprecation

1. A release adds the deprecation. `from_version` is that release.
   The old name still works, and it emits a deprecation warning.
2. `remove_version` is a later minor release. Users can change their scripts until then.
3. At or before the release `remove_version`, remove the old name and its deprecation.
   This is a change for users. Add a `change` news fragment. See [dev-04 · Changelog](dev-04-conventions.md#changelog).

## Before each minor release

Find the deprecations that are due:

```bash
grep -rn "remove_version=" src/
```

Remove each deprecation with a `remove_version` that is equal to or lower than the new release.
Do this in a commit before the release commit. See [release-02 · A release commit has no code changes](release-02-process.md#a-release-commit-has-no-code-changes).

## Known overdue deprecations

At the release of `0.20.0`, one deprecation still has `remove_version="0.18"`:
the arguments `angle` and `v` in `src/sisl/_core/_ufuncs_geometry.py` (`from_version="0.16.3"`).
