.. _installation:

Installation
============

sisl is easy to install using any of your preferred methods.

Required dependencies
---------------------

- `Python`_ 2.7, 3.4 or above
- `six`_
- `setuptools`_
- `numpy`_ (1.9 or later)
- `scipy`_
- `netCDF4-python <netcdf4-py_>`_

Optional dependencies:

- `pytest`_ (for running the test suite)
- `matplotlib`_
- A Fortran compiler

pip
---

Installing sisl using PyPi can be done using

.. code-block:: bash

   pip install sisl

:code:`pip` will automatically install the required dependencies. The optional dependencies
will be used if later installed.

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
appear).

Simply download the release tar from `this page <gh-releases_>`_, or clone
the `git repository <sisl-git_>`_ for the latest developments

.. code-block:: bash

   python setup.py install --prefix=<prefix>


Testing your installation
-------------------------

After installation (by either of the above listed methods) you are encouraged
to perform the shipped tests to ensure everything got installed correctly.

Note that `pytest`_ needs to be installed to run the tests.
Testing the installation may be done by:

.. code-block:: bash

    pytest --pyargs sisl
