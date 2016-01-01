#!/bin/bash

# This small bash script tags the git-revision
# by updating the appropriate files with the
# revision numbers

function read_num {
    local n=$1 ; shift
    grep -e "^$n" setup.py | sed -e "s!$n[[:space:]]*=[[:space:]]*\(.*\)!\1!"
}
MAJOR=$(read_num MAJOR)
MINOR=$(read_num MINOR)
MICRO=$(read_num MICRO)

_shifted=0

function _help {
    echo "Use the following commands:"
    echo "  $0 M  :  steps the major revision, minor, micro turns 0"
    echo "  $0 m  :  steps the minor revision, micro turns 0" 
    echo "  $0 u  :  steps the minor revision"
    echo ""
    echo "Example, previous version is 0.1.3"
    echo "  $0 M  => 1.0.0"
    echo "  $0 m  => 0.2.0"
    echo "  $0 u  => 0.1.4"
    echo ""
    echo "Current sids version is $MAJOR.$MINOR.$MICRO"
}


if [ $# -eq 0 ]; then
    echo "You *must* supply at least one step option"
    echo ""
    _help
    exit 1
fi

# Figure out what to step
while [ $# -gt 0 ]; do
    opt=$1 ; shift
    case $opt in
	-h|--h|--help|-help)
	    _help
	    exit 0
	    ;;
	a|A|M|major|ma)
	    let MAJOR++
	    MINOR=0
	    MICRO=0
	    _shifted=1
	    ;;
	b|m|minor|mi)
	    let MINOR++
	    MICRO=0
	    _shifted=1
	    ;;
	c|u|micro|mic)
	    let MICRO++
	    _shifted=1
	    ;;
    esac
done

if [ $_shifted -eq 0 ]; then
    let MICRO++
fi

# Create version string
v=$MAJOR.$MINOR.$MICRO

# Git revision
rev=`git rev-parse HEAD`

# Message for release
MSG="Releasing v$v"

# Set version numbers in setup.py
sed -i -e "s:\(MAJOR[[:space:]]*=\).*:\1 $MAJOR:" setup.py
sed -i -e "s:\(MINOR[[:space:]]*=\).*:\1 $MINOR:" setup.py
sed -i -e "s:\(MICRO[[:space:]]*=\).*:\1 $MICRO:" setup.py
# Update release tag and git revision
sed -i -e "s:\(ISRELEASED[[:space:]]*=\).*:\1 True:" setup.py
sed -i -e "s:\(GIT_REVISION[[:space:]]*=\).*:\1 $rev:" setup.py


echo "Tagging with message:"
echo " -m '$MSG'"

# Tagging and releasing
git add setup.py
git commit -s -m "Prepping for release"
git tag -a "$v" -m "$MSG"
# Publish on pypi
python setup.py sdist upload
# Revert release tag
sed -i -e "s:\(ISRELEASED[[:space:]]*=\).*:\1 False:" setup.py
git add setup.py
git commit -s -m "Reverting internal release"
git push
git push --tags


