#!/bin/bash

# This script will only do removal of whitespace
# It relies on the autopep8 tool and the
# specifics of whitespace error/warnings

# List of errors of whitespace stuff
select="E101,E20,E211,E231,E301,E302,E303,E304,E309,W291,W293,W391"
# Errors of comparison with None and bool
select="$select,E711,E712"
autopep8 --select "$select" --in-place -r .

# Non-Python files
autopep8 --select "W291,W293" --in-place NEWS README.md
