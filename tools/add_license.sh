#!/bin/bash
#
# Small snippet script to *recursively* add
# a header describing the license for all *.py files.
shopt -s globstar

declare -a extensions=(py pyx)

function add_header {
  local f=$1 ; shift
  if ! $(grep -Fq "mozilla.org/MPL/2.0" $f) ; then
    local tmpdir=$(mktemp -d)
    local basef=$(basename $f)
    cp $f $tmpdir/$basef
    {
      echo "# This Source Code Form is subject to the terms of the Mozilla Public"
      echo "# License, v. 2.0. If a copy of the MPL was not distributed with this"
      echo "# file, You can obtain one at https://mozilla.org/MPL/2.0/."
      cat $tmpdir/$basef
    } > $f
    rm -fr $tmpdir
  fi
}

for ext in ${extensions[@]}
do
  for f in **/*.$ext
    do
      add_header $f
  done
done
