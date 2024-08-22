#!/bin/bash

set -e
set -x

WHEEL=$1

python -c "import joblib; print(f'Number of cores (physical): \
{joblib.cpu_count()} ({joblib.cpu_count(only_physical_cores=True)})')"

FREE_THREADED_BUILD="$(python -c"import sysconfig; print(bool(sysconfig.get_config_var('Py_GIL_DISABLED')))")"
if [[ $FREE_THREADED_BUILD == "True" ]]; then
    # TODO: delete when importing numpy no longer enables the GIL
    # setting to zero ensures the GIL is disabled while running the
    # tests under free-threaded python
    export PYTHON_GIL=0
fi

# First output... simple import and print of version
python -c "import sisl; print(sisl.__version__)"

if [ -d sisl/files/tests ]; then
    export SISL_FILES_TESTS=sisl/files/tests
    echo "will run with sisl-files tests"
    echo "SISL_FILES_TESTS=$SISL_FILES_TESTS"
else
    echo "will not run with sisl-files tests"
fi

if pip show -qq pytest-xdist; then
    XDIST_WORKERS=$(python -c "import joblib; print(joblib.cpu_count(only_physical_cores=True))")
    pytest --pyargs sisl -n $XDIST_WORKERS -m "not slow"
else
    pytest --pyargs sisl -m "not slow"
fi
