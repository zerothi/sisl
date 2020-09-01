#!/bin/bash

# This script is largely based on numpy's travis setup
# We follow this rather closely

# Fail immediately on all errors
# Also print out commands as they are executed.
set -xe

uname -a
free -m
df -h
ulimit -a

# We want to control the environment, fully
mkdir builds
pushd builds
pip install --upgrade virtualenv

virtualenv --python=python venv
ource venv/bin/activate
python -V
gcc --version

popd

# Fetch the files from sisl-files
if [[ ${OPTIONAL_DEPENDENCIES:-false} == true ]]; then
    export SISL_FILES_TESTS=$(pwd)/files/tests
    git submodule update --init files
fi

pip install --upgrade pip setuptools wheel

# fetch all common requirements (for running)
# then append the test specific requirements
pip install --upgrade $(cat requirements.txt) \
    $(cat ci/requirements.txt)

# Optional packages
if [[ ${OPTIONAL_DEPENDENCIES:-false} == true ]]; then
    pip install --upgrade tqdm pathos matplotlib xarray
fi




