# dev-05 · Verifying a change

Owns: `tools/techs_check.py`

Use the fastest check that can prove the change.
Then go to the next, slower check.
Do not run the full build and test suite to find out if a small change works.

## The order of checks

1. **Evaluate.** Run the changed function in a short Python call.
   Use a small system (for example `sisl.geom.graphene()`).
   Compare the result with a known value.
2. **Targeted tests.** Run the tests of the changed module.
   `pytest src/sisl/<module>/tests/test_<x>.py -k <name>`
3. **Rebuild.** Do this only if you changed a `.pyx`, `.pxd` or Fortran file.
   Then do steps 1 and 2 again. See [dev-01 · Build](dev-01-build.md).
4. **Wider tests.** Run the tests of the package and its users.
   Set `SISL_FILES_TESTS` if the change can affect IO. See [dev-02 · Testing](dev-02-testing.md#external-test-files).
5. **Performance.** If the change is in a hot path, compare the time with `main`.
   See [dev-03 · Performance](dev-03-performance.md).

## Before you commit

- The new and changed tests pass.
- You checked the numbers, not only that the code runs.
- You checked correctness and consistency again after the last edit.
- pre-commit passes.
- The `techs/` chapters that own the changed files are still correct.
  The `techs-check` hook names them. See [README · Ownership](README.md#ownership).
- A user-facing change has a news fragment. See [dev-04 · Conventions](dev-04-conventions.md#changelog).
- Report what you did not run, and why. For example:
  - An IO test without `SISL_FILES_TESTS` is xfail-ed. It did not pass.
    See [dev-02 · External test files](dev-02-testing.md#external-test-files).
  - A `.pyx`, `.pxd` or Fortran change without a rebuild was not tested.
  - A test in a worktree that imports a different sisl did not test the change.
    See [dev-01 · Editable installs and worktrees](dev-01-build.md#editable-installs-and-worktrees).
