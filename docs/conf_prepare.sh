#!/bin/bash

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
