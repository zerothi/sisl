""" Available files for reading/writing

.. currentmodule:: sisl.io

sisl handles a large variety of input/output files from a large selection
of DFT software and other post-processing tools.

Since sisl may be used with many other packages all files are name *siles*
to distinguish them from files from other packages.

.. toctree::


Basic IO routines
-----------------

.. autosummary::
   :toctree: sisl/

   add_sile - add a file to the list of files that sisl can interact with
   get_sile - retrieve a file object via a file name by comparing the extension
   get_siles - retrieve all files with specific attributes or methods
   get_sile_class - retrieve class via a file name by comparing the extension
   BaseSile - the base class for all sisl files
   Sile - a base class for ASCII files
   SileCDF - a base class for NetCDF files
   SileBin - a base class for binary files
   SileError - sisl specific error

Generic files
-------------

These files are generic, in the sense that they are not specific to a
given code.

.. autosummary::
   :toctree: sisl/

   XYZSile - atomic coordinate file
   CUBESile - atomic coordinates *and* 3D grid values
   TableSile - data file in tabular form
   MoldenSile - atomic coordinate file specific for Molden
   XSFSile - atomic coordinate file specific for XCrySDen

"""
from __future__ import print_function, division
import sys

from ._help import extendall
from .sile import *

# Import the different Sile objects
# enabling the actual print-out
from .bigdft import *
from .cube import *
from .gulp import *
from .ham import *
from .molden import *
from .scaleup import *
from .siesta import *
from .table import *
from .vasp import *
from .wannier import *
from .xsf import *
from .xyz import *

# Default functions in this top module
__all__ = []

extendall(__all__, 'sisl.io.sile')

extendall(__all__, 'sisl.io.bigdft')
extendall(__all__, 'sisl.io.cube')
extendall(__all__, 'sisl.io.gulp')
extendall(__all__, 'sisl.io.ham')
extendall(__all__, 'sisl.io.molden')
extendall(__all__, 'sisl.io.scaleup')
extendall(__all__, 'sisl.io.siesta')
extendall(__all__, 'sisl.io.table')
extendall(__all__, 'sisl.io.vasp')
extendall(__all__, 'sisl.io.wannier')
extendall(__all__, 'sisl.io.xsf')
extendall(__all__, 'sisl.io.xyz')
