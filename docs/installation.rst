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

   python3 -m pip install --install-option="--compiler=intelem" --install-option="--fcompiler-intelem" sisl

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
   conda install -c conda-forge fortran-compiler c-compiler python=3.9
   conda install -c conda-forge scipy netcdf4 cftime plotly matplotlib


Subsequent installation of sisl in your conda enviroment would follow :ref:`installation-development`.


Manual installation
-------------------

The regular :code:`pip` codes may be used to install git clones or downloaded
tarballs.

Simply download the release tar from `this page <gh-releases_>`_, or clone
the `git repository <sisl-git_>`_ for the latest developments

.. code-block:: bash

   python3 -m pip install . --prefix=<prefix>


Windows
~~~~~~~

To install `sisl` on Windows one will require a specification of
the compilers used. Typically one may do

.. code-block:: bash

   python3 -m pip install . --prefix=<prefix> --install-option='--fcompiler=gfortran' --install-option='--compiler=mingw32'

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
- fortran compiler

To install the development version using :code:`pip` you may use the URL command:

.. code-block:: bash

   python3 -m pip install -U git+https://github.com/zerothi/sisl.git

Otherwise follow the manual installation by cloning the `git repository <sisl-git_>`_.
Remark that the :code:`git+https` protocol is buggy (as of pip v19.0.3) because you cannot pass compiler
options to :code:`setuptools`. If you want to install the development version with e.g.
the Intel compilers you should do:

.. code-block:: bash

   git clone git+https://github.com/zerothi/sisl.git
   cd sisl
   python3 -m pip install . -U --build-option="--compiler=intelem" --build-option="--fcompiler=intelem" .

which will pass the correct options to the build system.

The `-U` flag ensures that prior installations are overwritten.


.. _sisl-test-files: http://github.com/zerothi/sisl-files
