#!/bin/bash

# This small bash script tags the git-revision
# by updating the appropriate files with the
# revision numbers

MAJOR=$1 ; shift
MINOR=$1 ; shift
MICRO=$1 ; shift
ISRELEASED=False
v=$MAJOR.$MINOR.$MICRO
MSG="Releasing v$v"
if [ $# -gt 0 ]; then
    case $1 in
	r|rel|release)
	    ISRELEASED=True
	    shift ;;
    esac
fi


# Correct the setup.py
sed -i -e "s:\(MAJOR[[:space:]]*=\).*:\1 $MAJOR:" setup.py
sed -i -e "s:\(MINOR[[:space:]]*=\).*:\1 $MINOR:" setup.py
sed -i -e "s:\(MICRO[[:space:]]*=\).*:\1 $MICRO:" setup.py
sed -i -e "s:\(ISRELEASED[[:space:]]*=\).*:\1 $ISRELEASED:" setup.py

echo "Tagging with message:"
echo " > $MSG"

if [ "$ISRELEASED" == "True" ]; then
    echo "A released version"
fi

# Tagging and releasing
git tag -a "$v" -m "$MSG"
git push --tags


