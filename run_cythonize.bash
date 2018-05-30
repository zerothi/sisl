#!/bin/bash

files="
_math_small.pyx
_indices.pyx
_supercell.pyx
_sparse.pyx
physics/_matrix_k_dtype.pyx
physics/_matrix_k_nc_dtype.pyx
physics/_matrix_k_so_dtype.pyx
physics/_matrix_diag_k_nc_dtype.pyx
physics/_matrix_k.pyx
physics/_matrix_k_factor_dtype.pyx
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
    popd 2>/dev/null
done
