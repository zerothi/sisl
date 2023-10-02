.. _installation:

Installation
============

sisl is easy to install using any of your preferred methods.


Required dependencies
---------------------

Running sisl requires these versions:

- `Python`_ 3.8 or above
- `numpy`_ (1.13 or later)
- `scipy`_ (1.5 or later)
- `pyparsing`_ (1.5.7 or later)
- `xarray`_ (0.10.0 or later)

Optional dependencies:

- `pytest`_ (for running the test suite)
- `pathos`_ (for parallel BrillouinZone calculations)
- `netCDF4-python <netcdf4-py_>`_
- `tqdm`_ (for displaying progress-bars)
- `matplotlib`_
- `plotly`_ (for advanced visualization)


sisl implements certain methods in Cython which speeds up the execution.
Cython is required if one wishes to re-generate the C-sources with a different
Cython version. Note that this is not a necessary step and should typically only
be considered by developers of Cython modules.

.. _installation-pip:


Installation of the stable sisl releases can be done by following the common conventions
using :code:`pip` or :code:`conda` methods:

.. tab:: pip

   pip will install from pre-build wheels when they are found, if not it will try and
   install from a source-distribution.

   .. code-block:: bash

      python3 -m pip install sisl
      # for better analysis
      python3 -m pip install sisl[analysis]
      # for advanced plotting functionality
      python3 -m pip install sisl[viz]

.. tab:: conda

   Conda enviroments are clever, but fragile. It is recommended to contain the
   sisl installation in a separate environment to decouple it from other fragile
   components. Their inter-dependencies may result in problematic installations.

   .. code-block:: bash

      conda create -n sisl
      conda activate sisl
      conda config --add channels conda-forge
      conda install -c conda-forge python sisl

.. tab:: dev|pip

   This is equivalent to a development installation which requires a C and fortran compiler
   as well as some other packages:

   .. code-block:: bash

      python3 -m pip install setuptools_scm "scikit-build-core[pyproject]" Cython
      python3 -m pip install git+https://github.com/zerothi/sisl.git --prefix <prefix>

   The remaining dependencies should automatically be installed.

.. tab:: dev|conda

   Using conda as development environment can be done, but may be a bit more cumbersome
   to work with. To install sisl from sources one needs a conda environment with the following
   content:

   .. code-block:: bash

      conda create -n sisl
      conda activate sisl
      conda config --add channels conda-forge
      conda install -c conda-forge fortran-compiler c-compiler python scikit-build-core pyproject-metadata
      conda install -c conda-forge cython scipy netcdf4 cftime plotly matplotlib

   subsequent installations of sisl should follow :code:`dev|pip` tab

.. tab:: editable|pip

   Editable installs are currently not fully supported by :code:`scikit-build-core` and
   is considered experimental. One *may* get it to work by doing:

   .. code-block:: bash

       git clone git+https://github.com/zerothi/sisl.git
       cd sisl
       python3 -m pip install -e .



Passing options to the build-system through :code:`pip` should de done with
the following convention


.. tab:: pip>=22.1

   .. code-block:: bash

      python3 -m pip install --config-settings=cmake.define.CMAKE_BUILD_PARALLEL_LEVEL=5 ...

.. tab:: pip<22.1

   .. code-block:: bash

      python3 -m pip install --global-option=cmake.define.CMAKE_BUILD_PARALLEL_LEVEL=5 ...


In the above case the compilation of the C/Fortran sources are compiled in parallel using 5
cores. This may greatly reduce compilation times.


There exists a set of compile time definitions that may be handy for developers.
These are all CMake definitions and can be added like this:

.. tab:: pip>=22.1

   .. code-block:: bash

      python3 -m pip install --config-settings=cmake.define.WITH_FORTRAN=YES .

.. tab:: pip<22.1

   .. code-block:: bash

      python3 -m pip install --global-option=cmake.define.WITH_FORTRAN=YES .


The options are:

- ``WITH_FORTRAN`` default to ON
  If OFF, no fortran sources will be compiled, this may be useful in debug
  situations, but are required for full support with externally created fortran
  files, such as output files from DFT codes.
- ``WITH_F2PY_REPORT_EXIT`` default to OFF
  If ON, the compile definition ``-DF2PY_REPORT_ATEXIT`` will be set.
- ``WITH_F2PY_REPORT_COPY`` default to OFF
  If ON, error messages will be printed while running when the array size
  has some certain size (see ``F2PY_REPORT_ON_ARRAY_COPY``)
- ``F2PY_REPORT_ON_ARRAY_COPY`` default 10
  Minimum (total) number of array elements an array should have before
  an error is created when reporting a copy, ``WITH_F2PY_REPORT_COPY`` must
  also be ON for this to take effect.
- ``WITH_LINE_DIRECTIVES`` default to OFF
  Add line-directives when cythonizing sources
- ``WITH_GDB`` default to OFF
  Add information for the GDB debugger
- ``WITH_ANNOTATE`` default to OFF
  create annotation output (html format) that can be viewed

.. warning::

   Only developers should play with these flags at install time.

   And in particular using ``WITH_FORTRAN=OFF`` will reduce the functionality
   of sisl (no fortran binary file support).


Windows
~~~~~~~

To install `sisl` on Windows one will require a specification of
the compilers used. Typically one may do

.. code-block:: bash

   python3 -m pip install . --prefix=<prefix>

but sometimes ``setuptools`` does not intercept the flags in the build process.
Since 3.12 ``distutils`` has been deprecated and one needs to pass explicit linker flags to the CMake environment.
If problems arise, please help out the community by figuring out how this works on Windows.

Adapt to compilers. For an explanation, see `here <https://docs.python.org/3/install/index.html#location-and-names-of-config-files>`_
or the `user issue <https://github.com/zerothi/sisl/issues/244>`_ which spurred this content.

.. _installation-testing:


Testing your installation
-------------------------

After installation (by either of the above listed methods) you are encouraged
to perform the shipped tests to ensure everything got installed correctly.

Note that `pytest`_ needs to be installed to run the tests.
Testing the installation may be done by:

.. code-block:: bash

   pytest --pyargs sisl

The above will run the default test-suite which covers most of the `sisl` tool-box.
Additional tests may be runned by cloning the `sisl-files <sisl-files_>`_
and setting the environment variable `SISL_FILES_TESTS` as the ``tests`` path to the repository.

A basic procedure would be:

.. code-block:: bash

   git clone https://github.com/zerothi/sisl-files.git
   SISL_FILES_TESTS=$(pwd)/sisl-files/tests pytest --pyargs sisl

