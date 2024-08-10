#!/bin/bash
#
# Pre-install commands for read-the-docs
#
set -euxo pipefail

if [[ "${READTHEDOCS:-no}" == "no" ]]; then
  echo "This script is only intended to be run in the read-the-docs environment"
  exit 1
fi

# Fix the code in src/sisl/_environ to make it always point to the correct path
sed -i -s -e "s:_THIS_DIRECTORY_DOES_NOT_EXIST_:$READTHEDOCS_REPOSITORY_PATH/files/tests:" src/sisl/_environ.py
