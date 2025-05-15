.. _install:

Detailed instructions
=====================

sisl is easy to install using any of your preferred methods.

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
   sisl installation in a separate environment to decouple it from other
   components. Their inter-dependencies may result in problematic installations.

   .. code-block:: bash

      conda create -n sisl
      conda activate sisl
      conda config --add channels conda-forge
      conda install -c conda-forge python sisl

.. tab:: editable|pip

   Editable installs allows one to easily use pure Python code changes
   without having to reinstall all the time. Highly recommended for developers, has the
   same requirements as noted in the `dev|*` tabs.

   .. code-block:: bash

       git clone git+https://github.com/zerothi/sisl.git
       cd sisl
       python3 -m pip install -e .


Windows
~~~~~~~

.. note::

   Currently compiling sisl on Windows is not tested in CI, any contributions
   to get sisl up and running on Windows would be greatly appreciated.
   Please help us out by opening an `issue`_.

The installation process should be equivalent to the other OS's. However,
one will likely be required to adapt the C and Fortran compiler for
the Windows platform.
These needs to be passed through the CMake environment (see
:ref:`install-compile-options`).


.. _installation-deps:

Required dependencies
---------------------

The above installation instructions installs the necessary dependencies
to run sisl, so generally one shouldn't worry about getting correct
packages etc. Here the more detailed requirements are listed.

- `Python`_ 3.9 or above
- `numpy`_
- `scipy`_
- `xarray`_
- `pyparsing`_

Optional dependencies:

- `pytest`_ (for running the test suite)
- `pathos`_ (for parallel `BrillouinZone` calculations)
- `netCDF4-python <netcdf4-py_>`_
- `tqdm`_ (for displaying progress-bars)
- `matplotlib`_
- `plotly`_ (for advanced visualization)

Development dependencies:

- ``C`` and ``fortran`` compilers
- `Cython`_
- `cmake`_ (3.21 or above)
- `pandoc`_ (for locally building the documentation)


.. _installation-testing:

Testing your installation
-------------------------

After installation (by either of the above listed methods) you are encouraged
to perform the shipped tests to ensure everything got installed correctly.

Note that `pytest`_ needs to be installed to run the tests.
Testing the installation may be done by:

.. code-block:: bash

   pytest --pyargs sisl

The above will run the default test-suite which covers most of `sisl`.
Additional tests may be runned by cloning the `stripped` branch of
`sisl-files <sisl-files_>`_
and setting the environment variable `SISL_FILES_TESTS` to the path of the cloned repository.

A basic procedure would be:

.. code-block:: bash

   git clone https://github.com/zerothi/sisl-files.git
   SISL_FILES_TESTS=$(pwd)/sisl-files pytest --pyargs sisl


.. _install-compile-options:

Compile time options
--------------------

By default sisl enables everything that is possible, i.e. the compilation flags
listed here are primarily intended for debugging, performance analysis/regressions
and should typically not be touched.

.. warning::

   It is not recommended to use these flags for production runs.

Passing options to the build-system through :code:`pip` should be done with
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

============================= ======== ======================================================
Option                        Default  Description
============================= ======== ======================================================
``WITH_FORTRAN``              ``ON``   If OFF, no fortran sources will be compiled,
                                       this may be useful in debug situations.
                                       For full support this should be kept ON.
``WITH_F2PY_REPORT_EXIT``     ``OFF``  Other name of ``-DF2PY_REPORT_ATEXIT``.
``WITH_F2PY_REPORT_COPY``     ``OFF``  If ON, warning messages will be printed when arrays
                                       are copied upon fortran routine calls.
``F2PY_REPORT_ON_ARRAY_COPY`` ``10``   Minimum number of elements before
                                       ``WITH_F2PY_REPORT_COPY`` will show a warning.
``WITH_LINES_DIRECTIVES``     ``OFF``  Add line-directives when Cythonizing sources.
``WITH_GDB``                  ``OFF``  Add information to the GDB debugger.
``WITH_ANNOTATE``             ``OFF``  Add annotated output (html) when Cythonizing sources.
============================= ======== ======================================================
