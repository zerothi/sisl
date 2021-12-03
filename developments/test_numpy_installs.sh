#!/bin/bash -i

# Try and load a default python and nothing more
# We should not require any dependencies
# and will install everything with pip
source ~/.module_setup
module purge
module load python

# Run different variants of the numpy installations
# using pip
# This should enable us to track down where installations
# goes wrong and why
export CFLAGS='-Og -g'
export FCFLAGS='-Og -g'
export FFLAGS='-Og -g'
export SETUPTOOLS_SCM_DEBUG=true 

# Show to the user which python version is being used
version=($(python3 --version | awk '{print $2}' | tr '.' ' '))
# In bash indices are 0-based
# In zsh indices are 1-based
echo "Python version: ${version[0]}.${version[1]}.${version[2]}"
which python3

# default python version to use as interpreter
pyv=${version[0]}.${version[1]}

# default sisl version to test
version=v0.11.0

while [[ $# -gt 0 ]]; do
    case $1 in
	# This does not really work since it requires other installation
	# flags.
	#--python-version|-pyv)
	#    pyv=$1
	#    shift
	#    ;;
	--version|-v)
	    shift
	    version=$1
	    shift
	    ;;
	--help|-h)
	    shift
	    echo " $0 can install and check various combinations of package conflicts"
	    #echo "   --python-version|-pyv:"
	    #echo "         specify a specific Python version to use."
	    #echo "         This is equivalent to pip3 install --python-version (arg) ..."
	    exit 0
	    ;;
	*)
	    version=$1
	    shift
	    ;;
    esac
done
    
if [[ -d $version ]]; then
    sisl_file=$version
elif [[ -e $version ]]; then
    sisl_file=$version
else
    # from git, this will also fetch the submodule
    sisl_file="git+https://github.com/zerothi/sisl.git@$version"
fi

install_cmd="python3 -u -m pip install --no-color --python-version $pyv"
install_cmd="python3 -u -m pip install --no-color"

echo "Installing sisl from here:"
echo "  $sisl_file"
echo ""

# Filename for the current log output
_log=


# Each installation format has a single required
# input. The pre-installed packages, i.e. what it will
# use once installed.
# They are different when the installation uses
#   --no-build-isolation
declare -A pre


# Define settings
pre[none]=""

case $pyv in
    37|3.7*)
	# numpy lowest available version for 3.7
	pre[n114]="numpy==1.14.6 scipy Cython"
	;;
    38|3.8*)
	# numpy lowest available version for 3.8
	pre[n117]="numpy==1.17.5 scipy Cython"
	;;
    39|3.9*)
	# numpy lowest available version for 3.9
	pre[n119]="numpy==1.19.5 scipy Cython"
	;;
esac
pre[latest]="numpy scipy Cython"


# Now run the different settings
function main() {
    local key

    echo "Starting running installation tests"

    for key in ${!pre[@]}
    do
	install_isolation $key
	install_no_isolation $key
    done
}

declare -A isolation
function install_isolation() {
    local key=$1 ; shift

    local _pre=${pre[$key]}

    local out=$key.isolation
    # force-clean-up
    rm -f $out.*

    echo "[$key]"
    echo " build-isolation"
    echo " pre-installed : $_pre"

    # Run every element in pre and do the following:
    source $(new_venv $out)
    _log=$out.pre
    pip_pre $_pre
    _log=$out.build
    $install_cmd -vvv --log=$_log $sisl_file 2>/dev/null > /dev/null
    _log=$out.test
    test_sisl 2>&1 > $out.test
    if [[ 0 -ne $? ]]; then
	echo "  - failed"
    fi
    clean_venv
}

declare -A no_isolation
function install_no_isolation() {
    local key=$1 ; shift

    local _pre=${pre[$key]}

    local out=$key.no_isolation
    # force-clean-up
    rm -f $out.*
    
    echo "[$key]"
    echo " no-build-isolation"
    echo " pre-installed : $_pre"

    # Run every element in pre and do the following:
    source $(new_venv $out)
    _log=$out.pre
    pip_pre $_pre
    _log=$out.build
    $install_cmd -vvv --log=$_log --no-build-isolation $sisl_file 2>/dev/null > /dev/null
    _log=$out.test
    test_sisl
    if [[ 0 -ne $? ]]; then
	echo "  - failed"
    fi
    clean_venv
}

function pip_pre() {
    echo "running pre:"
    local args="$@"
    if [[ -n "$args" ]]; then
	which python3
	python3 -u -m pip --log=$_log list 2>/dev/null >/dev/null
	$install_cmd --log=$_log $args 2>/dev/null >/dev/null
	echo "post pre:"
	python3 -u -m pip --log=$_log list 2>/dev/null >/dev/null
    fi
}

function test_sisl() {
    local ret
    echo "running test:"
    which python3
    python3 -u -m pip list -l
    python3 -u -c 'import sisl ; print(sisl.__version__)'
}
    

function new_venv() {
    # clean wheels build locally
    local env=$1 ; shift
    rm -rf venv.$env ~/.cache/pip/wheels
    python3 -m venv venv.$env
    printf "%s" "venv.$env/bin/activate"
}

function clean_venv() {
    deactivate
}

main
