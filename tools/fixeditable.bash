#!/bin/bash
# When doing editable installs with scikit-build-core
# there will be problems with the paths locating the library
# files.
# This small script will try to fix these inconsistencies

# First we need to locate the editable installation directory
editable_loc=$(python -m pip show sisl | grep "Location:" | awk '{print $2}')

# Get the build location
# If we are in the current directory:
cwd=..
[ -d src ] && cwd=.
_indices=$(find ${cwd}/ -name "_indices*.so")
suffix=${_indices#*_indices}
build_loc=$(readlink -e $(dirname $(dirname $_indices)))
out=_sisl_editable.py.replacement

# edit the sisl_editable.py file
tmpfile=$(mktemp /tmp/sisl-$(basename $0).XXXXXX)

echo "Found this information:"
echo "Build output for libraries: $build_loc"
echo "Suffix for library outputs: $suffix"
echo "Editable location for imports etc: $editable_loc"
echo "   will fix file _sisl_editable.py"
echo "Storing script in temporary file: $tmpfile"
[ ! -e $editable_loc ] && echo "editable_loc" && exit 1
[ ! -e $build_loc ] && echo "build_loc" && exit 1

cat <<EOF > $tmpfile
lines = open('$editable_loc/_sisl_editable.py').readlines()
f = open('$out', 'w')
#f = open('test.file', 'w')
ends = ('.pyx', '.pxd')
def contains(line):
  global ends
  for end in ends:
    if end in line:
      return True
  return False
for line in lines:
  if contains(line):
    splits = line.split(',')
    out_splits = []
    for split in splits:
      for end in ends:
        if end in split:
          pack, path = split.split(':')
          path = '\'${build_loc}/sisl/' + path.split('/sisl/')[-1].replace(end, '$suffix')
          print(f"correcting:\n  {split}")
          split = ':'.join([pack, path])
          print(f"  {split}")
      out_splits.append(split)
    line = ','.join(out_splits)
  f.write(line)
f.close()
EOF

python $tmpfile
if [ -e $out ]; then
  mv $out $editable_loc/_sisl_editable.py
fi
rm -f $out
