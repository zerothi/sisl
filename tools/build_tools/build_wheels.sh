#!/bin/bash

set -e
set -x

# Set environment variables to make our wheel build easier to reproduce byte
# for byte from source. See https://reproducible-builds.org/. The long term
# motivation would be to be able to detect supply chain attacks.
#
# In particular we set SOURCE_DATE_EPOCH to the commit date of the last commit.
#
# XXX: setting those environment variables is not enough. See the following
# issue for more details on what remains to do:
# https://github.com/scikit-learn/scikit-learn/issues/28151
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
export PYTHONHASHSEED=0

# Write out some system information:
uname -a
uname
uname -m

# Maybe we will use this later on...
USE_CONDA=0

function install_arm_flags() {
  SKBUILD_CMAKE_ARGS="$SKBUILD_CMAKE_ARGS -DCMAKE_SYSTEM_NAME=Generic"
  SKBUILD_CMAKE_ARGS="$SKBUILD_CMAKE_ARGS -DCMAKE_SYSTEM_PROCESSOR=arm"
  SKBUILD_CMAKE_ARGS="$SKBUILD_CMAKE_ARGS -DCMAKE_CROSSCOMPILING=1"
}


if [[ $USE_CONDA -eq 1 ]]; then
  echo "CONDA_HOME = $CONDA_HOME"
  ls -l $CONDA_HOME

  # List current conda packages
  conda list
fi

# We have to specify which gfortran we should install
conda_fortran=gfortran
CC=gcc
FC=gfortran

# OpenMP is not present on macOS by default
if [[ "$(uname)" == "Darwin" ]]; then
  # Make sure to use a libomp version binary compatible with the oldest
  # supported version of the macos SDK as libomp will be vendored into the
  # scikit-learn wheels for macos.

  if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
    export MACOSX_DEPLOYMENT_TARGET=14.0
    export MACOS_DEPLOYMENT_TARGET=14.0

    # Check for the actual host
    if [[ "$(uname -m)" == "x86_64" ]]; then
      install_arm_flags
    fi

  else
    export MACOSX_DEPLOYMENT_TARGET=12.0
    export MACOS_DEPLOYMENT_TARGET=12.0
  fi

  CC=clang
  export CPPFLAGS="$CPPFLAGS -Xpreprocessor"
  export CFLAGS="$CFLAGS -I$PREFIX/include"
  export FFLAGS="$FFLAGS -I$PREFIX/include"
  export CXXFLAGS="$CXXFLAGS -I$PREFIX/include"

elif [[ "$(uname)" == "Linux" ]]; then

  # Generic compiler
  conda_fortran=gfortran_linux-64

  if [[ "$CIBW_BUILD" == *-linux_aarch64 ]]; then
    # Check for the actual host
    if [[ $(uname -m) == "x86_64" ]]; then
      install_arm_flags
    else
      conda_fortran=gfortran_linux-aarch64
    fi
  fi

else

  # We need to force cmake to use non-visual studio makefiles
  export CMAKE_GENERATOR="MinGW Makefiles"

fi

export SKBUILD_CMAKE_ARGS


if [[ "$GITHUB_EVENT_NAME" == "schedule" \
        || "$GITHUB_EVENT_NAME" == "workflow_dispatch" \
        || "$CIRRUS_CRON" == "nightly" ]]; then
  # Nightly build:  See also `../github/upload_anaconda.sh` (same branching).
  # To help with NumPy 2.0 transition, ensure that we use the NumPy 2.0
  # nightlies.  This lives on the edge and opts-in to all pre-releases.
  # That could be an issue, in which case no-build-isolation and a targeted
  # NumPy install may be necessary, instead.
  export CIBW_BUILD_FRONTEND='pip; args: --pre --extra-index-url "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple"'
fi

if [[ $USE_CONDA -eq 1 ]]; then
  # Ensure we can use a compiler
  conda install -y $conda_fortran
fi

export CC=$(which $CC)
export FC=$(which $FC)

echo "Found:"
echo " SKBUILD_CMAKE_ARGS: $SKBUILD_CMAKE_ARGS"
echo " CC: $CC ($($CC --version))"
echo " FC: $FC ($($FC --version))"
echo " CXX: $CXX ($($CXX --version))"

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
