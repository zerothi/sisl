#!/bin/bash

files="
_math_small.pyx
_indices.pyx
_cell.pyx
_sparse.pyx
physics/_bloch.pyx
physics/_phase.pyx
physics/_matrix_phase.pyx
physics/_matrix_phase_nc.pyx
physics/_matrix_phase_so.pyx
physics/_matrix_phase_nc_diag.pyx
physics/_matrix_k.pyx
physics/_matrix_phase3.pyx
physics/_matrix_dk.pyx
physics/_matrix_ddk.pyx
"

# Ensure we are using the correct Cythonize
which cythonize
for file in $files
do
    echo "Parsing: sisl/$file"
    cythonize -a sisl/$file
    [ $? -ne 0 ] && exit 1
done
