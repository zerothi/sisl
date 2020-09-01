#!/bin/bash

set -ex

which python
python --version
# Check basic import
python -OO -c "import sisl"

# Before testing, we should move to a different folder
mkdir test-directory
cd test-directory

if [[ ${COVERAGE:-true} == true ]]; then
    py.test -vvv --pyargs sisl -rX --cov=sisl --cov-report term-missing --cov-config=.coveragerc
else
    py.test --doctest-modules sisl
    py.test --pyargs sisl -rX
fi
