#!/bin/bash
#
# Pre-install commands for read-the-docs
#
set -euxo pipefail

if [[ "${READTHEDOCS:-no}" == "no" ]]; then
  echo "This script is only intended to be run in the read-the-docs environment"
  exit 1
fi

# Try and fetch the shallow submodule
git submodule update --init --depth=1 --single-branch

# Ensure we get a version that lets us use groups
python3 -m pip install --upgrade pip
python3 -m pip install --group docs

# So since kaleido does not install chrome, we have to do it
# This is only necessary when it works, otherwise, skip it!
kaleido_get_chrome || true
