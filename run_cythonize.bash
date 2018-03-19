#!/bin/bash

files="
_math_small.pyx
_indices.pyx
_supercell.pyx
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
