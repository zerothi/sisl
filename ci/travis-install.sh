#!/bin/bash

echo "#### full-env install"
env
echo "#### full-env install done"

echo "Compiler + Python versions:"
gcc --version
gfortran --version
python -c 'import sys ; print(sys.version)'
unset LDFLAGS
pip install --no-deps -vvv -e .


