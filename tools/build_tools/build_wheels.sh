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

function install_arm_flags() {
  SKBUILD_CMAKE_ARGS="$SKBUILD_CMAKE_ARGS -DCMAKE_SYSTEM_NAME=Generic"
  SKBUILD_CMAKE_ARGS="$SKBUILD_CMAKE_ARGS -DCMAKE_SYSTEM_PROCESSOR=arm"
  SKBUILD_CMAKE_ARGS="$SKBUILD_CMAKE_ARGS -DCMAKE_CROSSCOMPILING=1"
}

echo "CONDA_HOME = $CONDA_HOME"
ls -l $CONDA_HOME

# List current conda packages
conda list

# We have to specify which gfortran we should install
conda_gfortran=gfortran

# OpenMP is not present on macOS by default
if [[ $(uname) == "Darwin" ]]; then
  # Make sure to use a libomp version binary compatible with the oldest
  # supported version of the macos SDK as libomp will be vendored into the
  # scikit-learn wheels for macos.

  if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
    # SciPy requires 12.0 on arm to prevent kernel panics
    # https://github.com/scipy/scipy/issues/14688
    # We use the same deployment target to match SciPy.
    export MACOSX_DEPLOYMENT_TARGET=12.0
    OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"

    # Check for the actual host
    if [[ $(uname -m) == x86_64 ]]; then
      install_arm_flags
    fi

  else
    export MACOSX_DEPLOYMENT_TARGET=10.9
    OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
  fi

  sudo conda create -n build $OPENMP_URL
  PREFIX="$CONDA_HOME/envs/build"

  export CC=/usr/bin/clang
  export CXX=/usr/bin/clang++
  #export FC=/usr/bin/gfortran
  export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
  export CFLAGS="$CFLAGS -I$PREFIX/include"
  export FFLAGS="$FFLAGS -I$PREFIX/include"
  export CXXFLAGS="$CXXFLAGS -I$PREFIX/include"
  export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"

elif [[ $(uname) == "Linux" ]]; then

  # Generic compiler
  conda_gfortran=gfortran_linux-64

  if [[ "$CIBW_BUILD" == *-linux_aarch64 ]]; then
    # Check for the actual host
    if [[ $(uname -m) == x86_64 ]]; then
      install_arm_flags
    else
      conda_gfortran=gfortran_linux-aarch64
    fi
  fi

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

echo "Found:"
echo " SKBUILD_CMAKE_ARGS: $SKBUILD_CMAKE_ARGS"
echo " CC: $CC ($($CC --version))"
echo " CXX: $CXX ($($CXX --version))"

# Ensure we can use a compiler
conda install -y $conda_gfortran
export FC=$(which gfortran)
echo " FC: $FC ($($FC --version))"

conda list

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
