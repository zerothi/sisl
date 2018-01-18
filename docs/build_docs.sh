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
    rm -rf _sources api-generated
    # Clean all rst.dummy files in the
    for f in `find ./ -name "*rst.dummy"`
    do
	rm $f
    done
    popd
    git add latest
    
fi
