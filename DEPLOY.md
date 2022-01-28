
Deployment of sisl
==================

This document describes the deployment details to perform
an version release (or development release).


Version release
---------------

A version release cycle *must* not contain any changes to the
code other than the below specified changes.
Any pending commits should be committed before proceeding with the
below sequence.

The release cycle should be performed like this:

1. Update released versions in `CHANGELOG.md` and `CITATION.cff`

2. Commit changes.

3. Tag the commit with:

		git tag -a "vVERSION" -m "Releasing vVERSION"

4. Create tarballs and wheels and upload them

		python3 -m pip install --upgrade build
		python3 -m build
		python3 -m pip install --upgrade twine
		# requires .pypirc with testpypi section
		python3 -m twine upload --repository testpypi dist/*

		# test installation, preferably in a venv
		python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sisl

        # once checked, upload to pypi
		python3 -m twine upload dist/sisl-0.12.0.tar.gz

5. Make release notes by using `tools/changelog.py` to create the output

6. Create conda uploads.

   The conda uploads are based on conda-forge and an associated
   sisl-feedstock is used. To update it, follow these steps:

   1. branch off https://github.com/conda-forge/sisl-feedstock
   2. Edit recipe/meta.yaml by updating version and sha256
   3. Propose merge-request.
   4. Check CI succeeds.
   5. Accept merge and the new version will be uploaded.
