#!/bin/bash

# Ensure the version file exists
pushd ..
ls -l
which python3
if [ $? -eq 0 ]; then
   python3 setup.py only-version
else
   python setup.py only-version
fi
popd

# Create the top-level index.rst file
echo ".. include:: docs/index.rst" > ../index.rst

# Ensure directories exist
mkdir -p build

#if [ "x$READTHEDOCS" == "xTrue" ]; then
#    # Make a link to the examples folder
#    ln -s ../../docs/examples examples
#fi

# Clean-up autosummary docs
rm -rf api/generated api/io/generated
