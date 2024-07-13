
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

1. Update released versions in `CHANGELOG.md` and `CITATION.cff`

2. Insert correct dates in `CITATION.cff` (for Zenodo)

3. Go to `tools` and run changelog.py v0.14.0..v0.14.1
   and generate both the RST and MD documentation.
   The rst should go directly into the `docs/changelog/`
   folder and do

   git add docs/changelog/VVV.rst
   <add it to the docs/changelog/index.rst>

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
   2. Edit recipe/meta.yaml by updating version and sha256
   3. Propose merge-request.
   4. Check CI succeeds.
   5. Accept merge and the new version will be uploaded.

8. Update pyodide version

   Until web assembly (wasm) wheels are supported by PyPi, they
   are managed directly in the pyodide repository. The update steps
   are very similar to conda, except all packages are managed
   in a single repository. The meta.yaml is at packages/sisl/meta.yaml.
   Follow these steps:

   1. branch off https://github.com/pyodide/pyodide
   2. Edit packages/sisl/meta.yaml by updating version, source url and sha256
   3. Propose merge-request.
   4. Check CI succeeds. If it doesn't you can test locally by following
      instructions [here](https://pyodide.org/en/stable/development/new-packages.html#building-a-python-package-in-tree)
   5. Wait for the pyodide team to accept your request.
