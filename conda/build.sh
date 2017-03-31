#!/bin/bash

# Create the version number
git describe --tags > __conda_version__.txt
a=`cat __conda_version__.txt`
# Convert "vX.Y.Z" to "X.Y.Z"
a=${a:1}
echo "$a" > __conda_version__.txt
b=${a//-/}
if [[ ${#a} -ne ${#b} ]]; then
   echo "dev" > __conda_version__.txt
fi

$PYTHON setup.py install
