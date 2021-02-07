.. _io.basic:

.. currentmodule:: sisl.io

Basic IO methods/classes
========================

Regular users may only need `get_sile` which retrieves the correct *sile*
based on parsing the filename.

.. autosummary::
   :toctree: api-generated/

   get_sile
   add_sile - add a file to the list of files that sisl can interact with
   get_siles
   get_sile_class
   SileError - sisl specific error
   SileWarning - sisl specific warning
   SileInfo - sisl specific information
