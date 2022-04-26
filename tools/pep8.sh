#!/bin/bash

# Enable **/* expansions
shopt -s globstar

# This script will only do removal of whitespace
# It relies on the autopep8 tool and the
# specifics of whitespace error/warnings

# List of errors of whitespace stuff
select="E101,E20,E211,E231,E301,E302,E303,E304,E309,W291,W293,W391"
# Errors of comparison with None and bool
select="$select,E711,E712"
# Imports on single lines
select="$select,E401"
# trailing whitespace
select="$select,C0303"
# imports at top
select="$select,C0413"
# import order
select="$select,C0411"

# pretty print command select running
echo "autopep8 --select \"$select\""
autopep8 -j -1 --select "$select" --in-place -r --exclude build,sisl.egg-info,dist . **/*.pyx **/*.pxd

# Non-Python files
autopep8 --select "W291,W293" --in-place CHANGELOG.md &

# Remove white-space on empty lines
sed -i -s -e 's/^[[:space:]]*$//g' {sisl,toolbox}/**/*.f90
# Remove trailing white-space
sed -i -s -e 's/\([^[:space:]]\)[[:space:]]?$/\1/g' {sisl,toolbox}/**/*.f90
