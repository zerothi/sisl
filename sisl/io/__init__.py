"""
=============================
Input/Output (:mod:`sisl.io`)
=============================

.. module:: sisl.io

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
- :ref:`toc-io-wannier90`


Generic files
-------------

These files are generic, in the sense that they are not specific to a
given code.

.. autosummary::
   :toctree:

   ~xyz.XYZSile - atomic coordinate file
   ~cube.CUBESile - atomic coordinates *and* 3D grid values
   ~table.TableSile - data file in tabular form
   ~molden.MoldenSile - atomic coordinate file specific for Molden
   ~xsf.XSFSile - atomic coordinate file specific for XCrySDen


.. _toc-io-bigdft:

BigDFT (:mod:`sisl.io.bigdft`)
------------------------------

.. currentmodule:: sisl.io.bigdft

.. autosummary::
   :toctree:

   ASCIISileBigDFT - the input for BigDFT


.. _toc-io-gulp:

GULP (:mod:`sisl.io.gulp`)
--------------------------

.. currentmodule:: sisl.io.gulp

.. autosummary::
   :toctree:

   gotSileGULP - the output from GULP
   HessianSileGULP - Hessian output from GULP


.. _toc-io-scaleup:

ScaleUp (:mod:`sisl.io.scaleup`)
--------------------------------

.. currentmodule:: sisl.io.scaleup

.. autosummary::
   :toctree:

   orboccSileScaleUp - orbital information
   REFSileScaleUp - reference coordinates
   rhamSileScaleUp - Hamiltonian file


.. _toc-io-siesta:

Siesta (:mod:`sisl.io.siesta`)
------------------------------

.. currentmodule:: sisl.io.siesta

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

TBtrans (:mod:`sisl.io.tbtrans`)
--------------------------------

.. currentmodule:: sisl.io.tbtrans

.. autosummary::
   :toctree:

   tbtncSileTBtrans
   phtncSileTBtrans
   deltancSileTBtrans
   TBTGFSileTBtrans - TBtrans surface Green function files
   tbtavncSileTBtrans
   phtavncSileTBtrans

.. autosummary::
   :toctree:
   :hidden:

   dHncSileTBtrans


.. _toc-io-vasp:

VASP (:mod:`sisl.io.vasp`)
--------------------------

.. currentmodule:: sisl.io.vasp

.. autosummary::
   :toctree:

   CARSileVASP
   POSCARSileVASP
   CONTCARSileVASP


.. _toc-io-wannier90:

Wannier90 (:mod:`sisl.io.wannier90`)
------------------------------------

.. currentmodule:: sisl.io.wannier90

.. autosummary::
   :toctree:

   winSileWannier90 - input file



.. ###############################################
.. Add all io modules to the toc (to be reachable)
.. ###############################################

.. currentmodule:: sisl.io

.. autosummary::
   :toctree:
   :hidden:

   bigdft
   gulp
   scaleup
   siesta
   tbtrans
   vasp
   wannier90

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
from .wannier90 import *
from .xsf import *
from .xyz import *

__all__ = [s for s in dir() if not s.startswith('_')]

#for rm in ['sile', 'bigdft', 'cube', 'gulp',
#           'ham', 'molden', 'scaleup', 'siesta',
#           'tbtrans', 'vasp', 'wannier90', 'xsf',
#           'xyz']:
#    __all__.remove(rm)
