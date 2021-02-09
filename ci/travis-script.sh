#!/bin/bash

set -ex

# setup env
if [ -r /usr/lib/libeatmydata/libeatmydata.so ]; then
    # much faster package installation
    export LD_PRELOAD="/usr/lib/libeatmydata/libeatmydata.so"
elif [ -r /usr/lib/*/libeatmydata.so ]; then
    d=$(ls /usr/lib/*/libeatmydata.so)
    # much faster package installation
    export LD_PRELOAD="$d"
fi

source builds/sisl-venv/bin/activate

if [[ ${OPTIONAL_DEPENDENCIES:-false} == true ]]; then
    export SISL_FILES_TESTS=$(pwd)/files/tests
fi

# Before testing, we should move to a different folder
mkdir -p empty-test
cd empty-test

echo "PYTHONPATH=$PYTHONPATH"

which python
python --version
which sgrid

sdata --help
sgrid --version
sgeom --cite
# Check basic import
python -OO -c "import sisl"

if [[ ${COVERAGE:-true} == true ]]; then
    py.test -vv --pyargs sisl -rX --cov=sisl --cov-report term-missing --cov-config=../.coveragerc
    bash <(curl -s https://codecov.io/bash) -d
    bash <(curl -s https://codecov.io/bash) -v
else
    py.test --doctest-modules sisl
    py.test --pyargs sisl -rX
fi

cd ..
