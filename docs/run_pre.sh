#!/bin/bash

# Ensure the version file exists
pushd ..
python setup.py only-version
popd

# Create the top-level index.rst file
echo ".. include:: docs/index.rst" > ../index.rst

# Ensure directories exist
mkdir -p build

# Simple documentation script to generate the documentation
rm -rf sisl
mkdir sisl
sphinx-apidoc -fe -o sisl ../sisl ../**/setup.py ../**/tests/*

# Ensure the links.rst.dummy is EVERYWHERE
function add_links {
    for d in `ls -d */ 2>/dev/null` ; do
	if [ $d == "." ]; then
	    continue
	fi
	if [ $d == ".." ]; then
	    continue
	fi
	if [ -d $d ]; then
	    pushd $d
	    add_links
	    popd
	fi
	if [ ! -e links.rst.dummy ]; then
	    ln -s $root/links.rst.dummy .
	fi
    done
}

# Create sym-links
root=$(pwd)
add_links
