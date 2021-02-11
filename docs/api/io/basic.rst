.. _io.basic:

.. currentmodule:: sisl.io

Basic IO methods/classes
========================

Regular users may only need `get_sile` which retrieves the correct *sile*
based on parsing the filename.


Retrieval methods and warnings
------------------------------

.. autosummary::
   :toctree: generated/

   get_sile
   add_sile - add a file to the list of files that sisl can interact with
   get_siles
   get_sile_class
   SileError - sisl specific error
   SileWarning - sisl specific warning
   SileInfo - sisl specific information


Base classes
------------

All `sisl` files inherit from the `BaseSile` class.
While ASCII files are based on the `Sile` class, NetCDF files are based on the
`SileCDF` and finally binary files inherit from `SileBin`.

.. autosummary::
   :toctree: generated/

   BaseSile - all siles inherit this one
   Sile - sisl specific error
   SileCDF
   SileBin
