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
mkdir -p builds
pushd builds
python -m pip install --upgrade virtualenv

virtualenv --python=python sisl-venv
source sisl-venv/bin/activate
python -V
gcc --version

popd

# setuptools is part of requirements.txt
python -m pip install --upgrade pip wheel

# fetch all common requirements (for running)
# then append the test specific requirements
# Use grep to remove comments
python -m pip install --upgrade $(grep -v -e '^#' requirements.txt) \
    $(grep -v -e '^#' ci/requirements.txt)

# Optional packages
if [[ ${OPTIONAL_DEPENDENCIES:-false} == true ]]; then
    git submodule update --init files
    python -m pip install --upgrade tqdm pathos matplotlib xarray dill plotly scikit-image sisl-gui
fi
