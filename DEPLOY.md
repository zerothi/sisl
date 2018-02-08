
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

1. Increment the release numbers in the top-directory
   setup.py script
   These are

		MAJOR
		MINOR
		MICRO

	Alltogether the version number _is_:
	`VERSION=MAJOR.MINOR.MICRO`
	In the following `VERSION` should be replaced by the correct release
	numbers
	
2. Set the variable:

	    ISRELEASED = True

3. Set the variable `GIT_REVISION` to the latest commit.
   This means that the revision specification for the release
   actually corresponds to the commit just before the actual release.
   You can get the commit hash by:

        git rev-parse HEAD

        GIT_REVISION = <git rev-parse HEAD>

4. Add `setup.py` to the commit and commit using:

    	git commit -s -m "sisl release: VERSION"

   with the corresponding version number.

5. Tag the commit with:

		git tag -a "vVERSION" -m "Releasing vVERSION"

6. Create tarballs and wheels and upload them

		rm -rf dist
		python setup.py sdist bdist_wheel
		twine upload dist/sisl-VERSION*.tar.gz
		twine upload dist/sisl-VERSION*.whl

7. Create conda uploads.

   The conda uploads are based on conda-forge and an associated
   sisl-feedstock is used. To update it, follow these steps:

   1. branch off https://github.com/conda-forge/sisl-feedstock
   2. Edit recipe/meta.yaml by updating version and sha256
   3. Propose merge-request.
   4. Check CI succeeds.
   5. Accept merge and the new version will be uploaded.
