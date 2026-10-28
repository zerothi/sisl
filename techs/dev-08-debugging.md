# dev-08 · Debugging

This chapter tells how to find the cause of a bug, a failing test or a wrong result.

## Reproduce first

Do not change code before you can reproduce the problem.

- Make the smallest reproduction: a short Python call with a small system,
  for example `sisl.geom.graphene()`. See [dev-05 · The order of checks](dev-05-verifying-a-change.md#the-order-of-checks).
- Compare the result with a known value, not only with "no error".
- Do not commit the reproduction script. It becomes a test. See [From reproduction to test](#from-reproduction-to-test).

## Make the result repeatable

- Set `SISL_NUM_PROCS=1`. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1.
- Give a fixed seed to each random generator.
- A problem that comes and goes is often an order or a shared state between tests.
  Run the failing test alone, and then with the tests before it.
  See [dev-02 · Temporary files](dev-02-testing.md#temporary-files).

## Rule out the environment

Before you suspect the logic, make sure that the code you read is the code that runs:

- `import sisl` loads the expected source tree.
  See [dev-01 · Editable installs and worktrees](dev-01-build.md#editable-installs-and-worktrees).
- The `.py` or the `.pyx` module runs. See [dev-01 · `.py` and `.pyx` companions](dev-01-build.md#py-and-pyx-companions).
- The build is not stale after a `.pyx`, `.pxd` or Fortran change.
  See [dev-01 · Stale Cython builds](dev-01-build.md#stale-cython-builds).
- An IO problem can need `SISL_FILES_TESTS`. See [dev-02 · External test files](dev-02-testing.md#external-test-files).

`python tools/dev/envcheck.py <path>...` checks these four points.
See [dev-11 · envcheck](dev-11-code-tools.md#envcheck).

## Find the cause

- Find the cause, not only the place where the error shows.
- Compare with `main`. If `main` works, use `git bisect` to find the commit that broke it.
- Read the owning `techs/` chapter. The behavior can be a decision, not a bug.
- If the cause is a decision in `techs/`, or a fix needs a change of the public API,
  discuss it with the maintainer before you change it.

## Do not hide the problem

- Do not make a tolerance larger to make a test pass. A tolerance follows the dtype.
  See [overview-01 · Precision is a requirement](overview-01-concepts.md#precision-is-a-requirement).
- Do not add `xfail`, `skip` or a `try`/`except` that hides the error, unless the maintainer agrees.
- Do not remove or weaken an assertion.

## From reproduction to test

- Make the reproduction into a test that fails before the fix.
  See [dev-02 · What a new test must do](dev-02-testing.md#what-a-new-test-must-do).
- Put the fix and the test in the same commit. See [dev-07 · Commit in small steps](dev-07-git.md#commit-in-small-steps).
