#!/bin/bash

files="
_math_small.pyx
_indices.pyx
_supercell.pyx
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

for file in $files
do
    echo "Parsing: $file"
    d=$(dirname $file)
    f=$(basename $file)
    pushd sisl/$d 2>/dev/null
    cythonize -a $f
    [ $? -ne 0 ] && exit 1
    popd 2>/dev/null
done
