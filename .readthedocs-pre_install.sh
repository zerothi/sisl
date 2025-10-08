#!/bin/bash
#
# Pre-install commands for read-the-docs
#
set -euxo pipefail

if [[ "${READTHEDOCS:-no}" == "no" ]]; then
  echo "This script is only intended to be run in the read-the-docs environment"
  exit 1
fi

python3 -m pip install --group docs
