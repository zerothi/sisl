#!/bin/sh

# Get project path
PROJECT=$1

if [ -d ${PROJECT}/files/tests ]; then
    export SISL_FILES_TESTS=${PROJECT}/files/tests
    echo "will run with sisl-files tests"
    echo "SISL_FILES_TESTS=$SISL_FILES_TESTS"
else
    echo "will not run with sisl-files tests"
fi
pytest --pyargs sisl
