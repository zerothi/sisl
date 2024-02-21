.. _io:

============
Input/Output
============

.. module:: sisl.io

Available files for reading/writing

`sisl` handles a large variety of input/output files from a large selection
of DFT software and other post-processing tools.

In `sisl` all files are conventionally named *siles*
to distinguish them from files from other packages.

All files are generally accessed through the `get_sile` method
which exposes all siles through their extension recognition.

The current procedure for determining the file type is based on these
steps:

1. Extract the extension of the filename passed to `get_sile`, for instance
   ``hello.grid.nc`` will both examine extensions of ``grid.nc`` and ``nc``.
   This is necessary for leveled extensions.
2. Determine whether there is some specification in the file name ``hello.grid.nc{<specification>}``
   where ``<specification>`` can be:

   - ``contains=<name>`` (or simply ``<name>``) the class name must contain ``<name>``
   - ``endswith=<name>`` the class name must end with ``<name>``
   - ``startswith=<name>`` the class name must start with ``<name>``

   When a specification is used, only siles that obey that specification will be searched
   for the extension.
   This may be particularly useful when there are multiple codes using the same extension.
   For instance output files exists in several of the code bases.
3. Search all indexed (through `add_sile`) siles which obey the specification (all if no specifier)
   and collect all that matches the longest extension found.
4. If there is only 1 match, then return that class. If there are multiple `sisl` will try all ``read_*``
   methods and if all fail, then the sile will be removed from the eligible list.


.. toctree::
   :maxdepth: 2

   basic
   generic
   bigdft
   dftb
   fhiaims
   gulp
   openmx
   orca
   scaleup
   siesta
   tbtrans
   vasp
   wannier90
