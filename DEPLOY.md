
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

7. Create conda uploads:

		# Ensure no upload
		conda config --set anaconda_upload no
		rm -rf dist-conda
		conda build --output-folder dist-conda conda --python 2.7
		source activate python3 # or your env for python 3
		conda build --output-folder dist-conda conda --python 3.5
		anaconda login
		anaconda upload dist-conda/*.tar.bz2


Development release
-------------------

Essentially the development release may be performed on every commit.

1. Create conda uploads:

		# Ensure no upload
		conda config --set anaconda_upload no
		rm -rf dist-conda-dev
		conda build --output-folder dist-conda-dev conda-dev --python 2.7
		source activate python3 # or your env for python 3
		conda build --output-folder dist-conda-dev conda-dev --python 3.5
		anaconda login
		anaconda upload dist-conda-dev/*.tar.bz2
