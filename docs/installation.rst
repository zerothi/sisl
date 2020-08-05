.. _installation:

Installation
============

sisl is easy to install using any of your preferred methods.


Required dependencies
---------------------

- `Python`_ 3.6 or above
- `setuptools`_
- `numpy`_ (1.13 or later)
- `scipy`_ (0.18 or later)
- `netCDF4-python <netcdf4-py_>`_
- `pyparsing`_ (1.5.7 or later)
- A C- and fortran-compiler

Optional dependencies:

- `pytest`_ (for running the test suite)
- `pathos`_ (for parallel BrillouinZone calculations)
- `matplotlib`_
- `tqdm`_ (for displaying progress-bars)
- `xarray`_ (for advanced table data structures in certain methods)


sisl implements certain methods in Cython which speeds up the execution.
Cython is required if one wishes to re-generate the C-sources with a different
Cython version. Note that this is not a necessary step and should typically only
be considered by developers of Cython modules.


pip3
----

Installing sisl using PyPi can be done using

.. code-block:: bash

   pip3 install sisl
   # or
   pip3 install sisl[analysis]

:code:`pip3` will automatically install the required dependencies. The optional dependencies
will be used if later installed.

The latter installation call also installs :code:`tqdm` and :code:`xarray` which are part of
extended analysis methods. These are not required and may be installed later if their usage
is desired.

When wanting to pass options to :code:`pip3` simply use the following

.. code-block:: bash

   pip3 install --install-option="--compiler=intelem" --install-option="--fcompiler-intelem" sisl

note that options are accummulated.


conda
-----

Installing sisl using conda can be done by

.. code-block:: bash

   conda config --add channels conda-forge
   conda install sisl

To find more information about the conda-forge installation please see
`here <conda-releases_>`_.


Manual installation
-------------------

sisl may be installed using the regular `setup.py` script.
Ensure the required dependencies are installed before proceeding with the
manual installation (without `numpy`_ installed a spurious error message will
appear). The dependencies may be installed using this :code:`pip3` command:

.. code-block:: bash

   pip3 install -r requirements.txt


Simply download the release tar from `this page <gh-releases_>`_, or clone
the `git repository <sisl-git_>`_ for the latest developments

.. code-block:: bash

   python3 setup.py install --prefix=<prefix>


Windows
~~~~~~~

To install `sisl` on Windows one will require a specification of
the compilers used. Typically one may do

.. code-block:: bash

   python3 setup.py install --prefix=<prefix> --fcompiler=gfortran --compiler=mingw32

but sometimes ``setuptools`` does not intercept the flags in the build process.
To remedy this please ensure ``%HOME%\pydistutils.cfg`` contains the build options:

.. code-block:: bash

   [build]
   compiler = mingw32
   fcompiler = gfortran

Adapt to compilers. For an explanation, see `here <https://docs.python.org/3/install/index.html#location-and-names-of-config-files>`_
or the `user issue <https://github.com/zerothi/sisl/issues/244>`_ which spurred this content.


Testing your installation
-------------------------

After installation (by either of the above listed methods) you are encouraged
to perform the shipped tests to ensure everything got installed correctly.

Note that `pytest`_ needs to be installed to run the tests.
Testing the installation may be done by:

.. code-block:: bash

   pytest --pyargs sisl

The above will run the default test-suite which covers most of the `sisl` tool-box.
Additional tests may be runned by cloning the `sisl-files <sisl-test-files_>`_
and setting the environment variable `SISL_FILES_TESTS` as the path to the repository.

A basic procedure would be:

.. code-block:: bash

   git clone https://github.com/zerothi/sisl-files.git
   SISL_FILES_TESTS=$(pwd)/sisl-files pytest --pyargs sisl


Development version
-------------------

To install the development version using :code:`pip3` you may use the URL command:

.. code-block:: bash

   pip3 install git+https://github.com/zerothi/sisl.git

Otherwise follow the manual installation by cloning the `git repository <sisl-git_>`_.
Remark that the :code:`git+https` protocol is buggy (as of pip v19.0.3) because you cannot pass compiler
options to :code:`setup.py`. If you want to install the development version with e.g.
the Intel compilers you should do:

.. code-block:: bash

   git clone git+https://github.com/zerothi/sisl.git
   cd sisl
   pip3 install --global-option="build" --global-option="--compiler=intelem" --global-option="--fcompiler=intelem" .

which will pass the correct options to the build system.


.. _sisl-test-files: http://github.com/zerothi/sisl-files
