#!/bin/bash

set -e
set -x

WHEEL=$1
DEST_DIR=$2

# By default, the Windows wheels are not repaired.
# In this case, we need to vendor VCRUNTIME140.dll
pip install wheel
wheel unpack "$WHEEL"
WHEEL_DIRNAME=$(ls -d sisl-*)
python tools/build_tools/vendor.py "$WHEEL_DIRNAME"
wheel pack "$WHEEL_DIRNAME" -d "$DEST_DIR"
rm -rf "$WHEEL_DIRNAME"
