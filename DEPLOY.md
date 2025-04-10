
Deployment of sisl
==================

This document describes the deployment details to perform
a version release (or development release).


Version release
---------------

A version release cycle *must* not contain any changes to the
code other than the below specified changes.
Any pending commits should be committed before proceeding with the
below sequence.

The release cycle should be performed like this:

1. Update released versions in `CITATION.cff` and `pyproject.toml`.

2. Insert correct dates in `CITATION.cff` (for Zenodo).

3. Create release notes and changelogs:

   1. Create the release-notes for the documentation:

      ```shell
      towncrier build --version 0.14.1 --yes
      ```

      This will create a file here: `docs/release/0.14.1-notes.rst`.

      Amend to `docs/release.rst` something like this:

         0.14.1 <release/0.14.1-notes.rst>

   2. Create simpler release notes (for Github):

      ```shell
      # append to towncrier release notes:
      python tools/changelog.py --format rst $GH_TOKEN v0.14.0..v0.14.1 >> docs/release/0.14.1-notes.rst
      python tools/changelog.py --format md $GH_TOKEN v0.14.0..v0.14.1 > changelog.md
      ```

4. Commit changes.

5. Tag the commit with:

       git tag -a "vVERSION" -m "Releasing vVERSION"

6. Create tarballs and wheels and upload them

   These steps should be done via the github actions step, so generally
   not required.

       python3 -m pip install --upgrade build
       python3 -m build
       python3 -m pip install --upgrade twine
       # requires .pypirc with testpypi section
       python3 -m twine upload --repository testpypi dist/*

       # test installation, preferably in a venv
       python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sisl

       # once checked, upload to pypi
       python3 -m twine upload dist/sisl-0.12.0.tar.gz

7. Create conda uploads.

   The conda uploads are based on conda-forge and an associated
   sisl-feedstock is used. To update it, follow these steps:

   1. branch off https://github.com/conda-forge/sisl-feedstock
   2. Edit `recipe/meta.yaml` by updating version and sha256
   3. Propose merge-request.
   4. Check CI succeeds.
   5. Accept merge and the new version will be uploaded.

8. Update pyodide version

   Until web assembly (wasm) wheels are supported by PyPi, they
   are managed directly in the pyodide repository. The update steps
   are very similar to conda, except all packages are managed
   in a single repository. The meta.yaml is at `packages/sisl/meta.yaml`.
   Follow these steps:

   1. branch off https://github.com/pyodide/pyodide
   2. Edit `packages/sisl/meta.yaml` by updating version, source url and sha256
   3. Propose merge-request.
   4. Check CI succeeds. If it doesn't you can test locally by following
      instructions [here](https://pyodide.org/en/stable/development/new-packages.html#building-a-python-package-in-tree)
   5. Wait for the pyodide team to accept your request.
