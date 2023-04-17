.. _installation:

Installation
============

sisl is easy to install using any of your preferred methods.


Required dependencies
---------------------

Running sisl requires these versions:

- `Python`_ 3.7 or above
- `numpy`_ (1.13 or later)
- `scipy`_ (0.18 or later)
- `netCDF4-python <netcdf4-py_>`_
- `pyparsing`_ (1.5.7 or later)

Optional dependencies:

- `pytest`_ (for running the test suite)
- `pathos`_ (for parallel BrillouinZone calculations)
- `matplotlib`_
- `tqdm`_ (for displaying progress-bars)
- `xarray`_ (for advanced table data structures in certain methods)
- `plotly`_ (for advanced visualization)


sisl implements certain methods in Cython which speeds up the execution.
Cython is required if one wishes to re-generate the C-sources with a different
Cython version. Note that this is not a necessary step and should typically only
be considered by developers of Cython modules.

.. _installation-pip:

pip
---

Installing sisl using PyPi can be done using

.. code-block:: bash

   python3 -m pip install sisl
   # for better analysis
   python3 -m pip install sisl[analysis]
   # for advanced plotting functionality
   python3 -m pip install sisl[viz]


:code:`pip` will automatically install the required dependencies. The optional dependencies
will be used if later installed.

The latter installations call also installs dependent packages which are part of
extended analysis methods. These are not required and may be installed later if their usage
is desired.

When wanting to pass options to :code:`pip` simply use the following

.. code-block:: bash

   python3 -m pip install --config-settings=cmake.define.CMAKE_BUILD_PARALLEL_LEVEL=5 sisl

note that options are accummulated.


.. _installation-conda:

conda
-----

It is recommended to install sisl in a separate environment to decouple its dependencies
from other packages that may be installed.
To find more information about the conda-forge enabled versions please see
`here <conda-releases_>`_.

conda is a somewhat fragile environment when users want to update/upgrade packages.
Therefore when conda installations fails, or when it will not update to a more recent version it
is advisable to create a new environment (starting from scratch) to ensure that your currently
installed packages are not limiting the upgrading of other packages.

For sisl there are two options, whether one wants to use a stable sisl release, or be
able to install the latest development version from `here <sisl-git_>`_.

Stable
~~~~~~

Installing the stable sisl release in conda all that is needed is:


.. code-block:: bash

   conda create -n sisl
   conda activate sisl
   conda config --add channels conda-forge
   conda install -c conda-forge python=3.9 scipy matplotlib plotly netcdf4 sisl

which will install all dependencies including the graphical visualization
capabilities of sisl.


Development
~~~~~~~~~~~

Installing the development version of sisl requires some other basic packages
while also using :code:`pip` to install sisl, the procedure would be:

.. code-block:: bash

   conda create -n sisl-dev
   conda activate sisl-dev
   conda config --add channels conda-forge
   conda install -c conda-forge fortran-compiler c-compiler python=3.11 scikit-build-core
   conda install -c conda-forge scipy netcdf4 cftime plotly matplotlib


Subsequent installation of sisl in your conda enviroment would follow :ref:`installation-development`.


Manual installation
-------------------

The regular :code:`pip` codes may be used to install git clones or downloaded
tarballs.

Manual installations requires these packages:

- `setuptools_scm`_ (with toml support) 6.2 or later
- `scikit-build-core`_
- `Cython`_ 0.28 or later

Simply download the release tar from `this page <gh-releases_>`_, or clone
the `git repository <sisl-git_>`_ for the latest developments

.. code-block:: bash

   python3 -m pip install . --prefix=<prefix>

There exists a set of compile time definitions that may be handy for developers.
These are all CMake definitions and can be added like this:

.. code-block:: bash

   python3 -m pip install --config-settings=cmake.define.WITH_FORTRAN=YES .

The options are:

- `WITH_FORTRAN` default to ON
  If OFF, no fortran sources will be compiled, this may be useful in debug
  situations, but are required for full support with externally created fortran
  files, such as output files from DFT codes.
- `WITH_F2PY_REPORT_EXIT` default to OFF
  If ON, the compile definition `-DF2PY_REPORT_ATEXIT` will be set.
- `WITH_F2PY_REPORT_COPY` default to OFF
  If ON, error messages will be printed while running when the array size
  has some certain size (see `F2PY_REPORT_ON_ARRAY_COPY`)
- `F2PY_REPORT_ON_ARRAY_COPY` default 10
  Minimum (total) number of array elements an array should have before
  an error is created when reporting a copy, `WITH_F2PY_REPORT_COPY` must
  also be ON for this to take effect.
- `WITH_LINE_DIRECTIVES` default to OFF
  Add line-directives when cythonizing sources
- `WITH_GDB` default to OFF
  Add information for the GDB debugger
- `WITH_ANNOTATE` default to OFF
  create annotation output (html format) that can be viewed


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


.. _installation-development:

Development version
-------------------

For source/development installations some basic packages are required:

- `Cython`_
- C compiler
- fortran compiler (much recommended)
- `scikit-build-core`_

To install the development version using :code:`pip` you may use the URL command:

.. code-block:: bash

   python3 -m pip install -U git+https://github.com/zerothi/sisl.git

Otherwise follow the manual installation by cloning the `git repository <sisl-git_>`_.
