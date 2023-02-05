#!/bin/bash

which python
which python3
which sphinx-build

# Ensure single-core
export SISL_NUM_PROCS=1
# Inform to the workflow visualization function that the 
# notebooks are to be exported to html
export SISL_NODES_EXPORT_VIS=1

# First make clean
make clean

# Now ensure everything is ready...
make html
retval=$?

# If succeeded, we may overwrite the old
# documentation (if it exists)
if [ $retval -eq 0 ]; then
    echo "Success = $retval"
else
    echo "Failure = $retval"
fi
