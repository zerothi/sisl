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

strip_comments=1
if [[ $# -gt 0 ]]; then
    case $1 in
	--keep-comments)
	    strip_comments=0
	    ;;
    esac
fi


# Ensure we are using the correct Cythonize
which cythonize
for file in $files
do
    echo "Parsing: sisl/$file"
    cythonize -a sisl/$file
    if [[ $strip_comments -ne 0 ]]; then
	# Easily strip comments via this command.
	# It should reduce the size of code in the repo
	cfile=${file//pyx/c}
	gcc -fpreprocessed -dD -E sisl/$cfile > .tmp.c
	mv .tmp.c sisl/$cfile
    fi
    [ $? -ne 0 ] && exit 1
done
