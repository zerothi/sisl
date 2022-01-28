
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

		rm -rf dist
		python setup.py sdist bdist_wheel
		twine upload dist/sisl-VERSION*.tar.gz
		twine upload dist/sisl-VERSION*.whl

5. Create conda uploads.

   The conda uploads are based on conda-forge and an associated
   sisl-feedstock is used. To update it, follow these steps:

   1. branch off https://github.com/conda-forge/sisl-feedstock
   2. Edit recipe/meta.yaml by updating version and sha256
   3. Propose merge-request.
   4. Check CI succeeds.
   5. Accept merge and the new version will be uploaded.
