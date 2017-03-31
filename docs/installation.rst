
Installation
============

sisl is very easy to install using any of your preferred methods.

.. toctree::
   :local:

pip
---

Installing sisl using PyPi can be done using

    pip install sisl

conda
-----

Installing sisl using conda can be done using

    conda install -c zerothi sisl

On conda sisl is also shipped in a developer installation for more
up-to-date releases, this may be installed using:

    conda install -c zerothi sisl-dev

Manuel installation
-------------------

sisl may also be installed using the regular `setup.py` script.
To do this the following packages are required to be in `PYTHONPATH`:

- `six`_
- `numpy`_
- `scipy`_
- `netCDF4`_
- `setuptools`_
- A fortran compiler

If the above listed items are installed, sisl can be installed
     
    python setup.py install --prefix=<prefix>
