"""
=============================
Input/Output (:mod:`sisl.io`)
=============================

.. currentmodule:: sisl.io

Available files for reading/writing

sisl handles a large variety of input/output files from a large selection
of DFT software and other post-processing tools.

Since sisl may be used with many other packages all files are name *siles*
to distinguish them from files from other packages.


Basic IO classes
================

.. autosummary::
   :toctree:

   add_sile - add a file to the list of files that sisl can interact with
   get_sile - retrieve a file object via a file name by comparing the extension
   get_siles - retrieve all files with specific attributes or methods
   get_sile_class - retrieve class via a file name by comparing the extension
   BaseSile - the base class for all sisl files
   Sile - a base class for ASCII files
   SileCDF - a base class for NetCDF files
   SileBin - a base class for binary files
   SileError - sisl specific error

.. _toc-io-supported:

External code in/out put supported
==================================

List the relevant codes that `sisl` can interact with. If there are files you think
are missing, please create an issue :ref:`here <issue>`.

- :ref:`toc-io-bigdft`
- :ref:`toc-io-gulp`
- :ref:`toc-io-scaleup`
- :ref:`toc-io-siesta`
- :ref:`toc-io-tbtrans`
- :ref:`toc-io-vasp`
- :ref:`toc-io-wannier`


Generic files
-------------

These files are generic, in the sense that they are not specific to a
given code.

.. autosummary::
   :toctree:

   XYZSile - atomic coordinate file
   CUBESile - atomic coordinates *and* 3D grid values
   TableSile - data file in tabular form
   MoldenSile - atomic coordinate file specific for Molden
   XSFSile - atomic coordinate file specific for XCrySDen


.. _toc-io-bigdft:

BigDFT (:mod:`sisl.io.bigdft`)
------------------------------

.. autosummary::
   :toctree:

   ASCIISileBigDFT - the input for BigDFT


.. _toc-io-gulp:

GULP (:mod:`sisl.io.gulp`)
--------------------------

.. autosummary::
   :toctree:

   gotSileGULP - the output from GULP
   HessianSileGULP - Hessian output from GULP


.. _toc-io-scaleup:

ScaleUp
-------

.. autosummary::
   :toctree:

   orboccSileScaleUp - orbital information
   REFSileScaleUp - reference coordinates
   rhamSileScaleUp - Hamiltonian file


.. _toc-io-siesta:

Siesta (:mod:`sisl.io.siesta`)
------------------------------

.. autosummary::
   :toctree:

   fdfSileSiesta - input file
   outSileSiesta - output file
   XVSileSiesta - xyz and vxyz file
   bandsSileSiesta - band structure information
   eigSileSiesta - EIG file
   GridSileSiesta - Grid charge information (binary)
   gridncSileSiesta - NetCDF grid output files (netcdf)
   EnergyGridSileSiesta - Grid potential information
   TSHSSileSiesta - TranSiesta Hamiltonian
   TSGFSileSiesta - TranSiesta surface Green function files
   ncSileSiesta - NetCDF output file


.. _toc-io-tbtrans:

TBtrans
-------

.. autosummary::
   :toctree:

   tbtncSileTBtrans
   phtncSileTBtrans
   deltancSileTBtrans
   TBTGFSileSiesta - TBtrans surface Green function files
   tbtavncSileTBtrans
   phtavncSileTBtrans


.. _toc-io-vasp:

VASP
----

.. autosummary::
   :toctree:

   CARSileVASP
   POSCARSileVASP
   CONTCARSileVASP


.. _toc-io-wannier:

Wannier90
---------

.. autosummary::
   :toctree:

   winSileWannier90 - input file

"""

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
from .tbtrans import *
from .table import *
from .vasp import *
from .wannier import *
from .xsf import *
from .xyz import *

__all__ = [s for s in dir() if not s.startswith('_')]

#for rm in ['sile', 'bigdft', 'cube', 'gulp',
#           'ham', 'molden', 'scaleup', 'siesta',
#           'tbtrans', 'vasp', 'wannier', 'xsf',
#           'xyz']:
#    __all__.remove(rm)
