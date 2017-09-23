#!/bin/bash

# Ensure the version file exists
pushd ..
python setup.py only-version
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
rm -rf api/api-generated

exit 0
# Simple documentation script to generate the documentation
rm -rf sisl-api
mkdir sisl-api
if [ -e ../setup.py ]; then
    sphinx-apidoc -fMeET -o sisl-api ../sisl ../sisl/**/setup.py ../sisl/**/tests/*
elif [ -e ../../setup.py ]; then
    sphinx-apidoc -fMeET -o sisl-api ../../sisl ../../sisl/**/setup.py ../../sisl/**/tests/*
fi
