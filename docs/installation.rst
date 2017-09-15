.. _installation:

Installation
============

sisl is easy to install using any of your preferred methods.

.. toctree::
   :local:

pip
---

Installing sisl using PyPi can be done using

.. code-block:: bash

   pip install sisl

conda
-----

Installing sisl using conda can be done by

.. code-block:: bash

    conda install -c zerothi sisl

On conda, sisl is also shipped in a developer installation for more
up-to-date releases, this may be installed using:

.. code-block:: bash

   conda install -c zerothi sisl-dev


Manual installation
-------------------

sisl may be installed using the regular `setup.py` script.
To do this the following packages are required to be in `PYTHONPATH`:

- `six`_
- `setuptools`_
- `numpy`_
- `scipy`_
- `netCDF4-python <netcdf4-py_>`_
- A fortran compiler

If the above listed items are installed, sisl can be installed by first
downloading the latest release on `this page <gh-releases_>`_.
Subsequently install sisl by

.. code-block:: bash

   python setup.py install --prefix=<prefix>


Testing your installation
-------------------------

It may be good practice to test your installation using the shipped test-suite.

To test `sisl`, you are also required having the `pytest` package installed.
Then to test the installation simply run:

.. code-block:: bash

    pytest --pyargs sisl
