#!/bin/bash

if [[ ${COVERAGE:-true} == true ]]; then
    bash <(curl -s https://codecov.io/bash)
    python-codacy-coverage -r test-directory/coverage.xml
fi
