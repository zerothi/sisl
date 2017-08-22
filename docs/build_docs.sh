#!/bin/bash

# First make clean
make clean

# Now ensure everything is ready...
make html
succeed=$?

# If succeeded, we may overwrite the old
# documentation (if it exists)
if [ $succeed -eq 0 -a -d latest ]; then
    
    # First remove directory
    git rm -rf latest
    mkdir latest
    pushd latest
    cp -rf ../build/html/* .
    rm -rf _sources
    popd
    git add latest
    
fi
