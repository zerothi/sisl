#!/bin/bash

which python
which python3
which sphinx-build

# Ensure single-core
export SISL_NPROCS=1

# First make clean
make clean

# We need to remove the latest folder before
# building to ensure we don't duplicate the
# folder structure
git rm -rf latest

# Now ensure everything is ready...
make html
retval=$?

# If succeeded, we may overwrite the old
# documentation (if it exists)
if [ $retval -eq 0 ]; then

    # Move folder to latest
    mv build/html latest
    rm -rf latest/_sources
    git add latest

    echo "Success = $retval"
else
    # We need to restore the latest folder
    git checkout -- ./latest

    echo "Failure = $retval"
fi
