"""
=============================
Input/Output (:mod:`sisl.io`)
=============================

.. module:: sisl.io
   :noindex:

Available files for reading/writing

sisl handles a large variety of input/output files from a large selection
of DFT software and other post-processing tools.

Since sisl may be used with many other packages all files are name *siles*
to distinguish them from files from other packages.


Basic IO methods/classes
========================

.. autosummary::
   :toctree:

   add_sile - add a file to the list of files that sisl can interact with
   get_sile - retrieve a file object via a file name by comparing the extension
   SileError - sisl specific error


.. _toc-io-supported:

External code in/out put supported
----------------------------------

List the relevant codes that `sisl` can interact with. If there are files you think
are missing, please create an issue `here <issue>`_.

- :ref:`toc-io-generic`
- :ref:`toc-io-bigdft`
- :ref:`toc-io-gulp`
- :ref:`toc-io-openmx`
- :ref:`toc-io-scaleup`
- :ref:`toc-io-siesta`
- :ref:`toc-io-transiesta`
- :ref:`toc-io-tbtrans`
- :ref:`toc-io-vasp`
- :ref:`toc-io-wannier90`


.. _toc-io-generic:

Generic files
=============

Files not specificly related to any code.

.. autosummary::
   :toctree:

   ~table.tableSile - data file in tabular form
   ~xyz.xyzSile - atomic coordinate file
   ~pdb.pdbSile - atomic coordinates and MD content
   ~cube.cubeSile - atomic coordinates *and* 3D grid values
   ~molden.moldenSile - atomic coordinate file specific for Molden
   ~xsf.xsfSile - atomic coordinate file specific for XCrySDen


.. _toc-io-bigdft:

BigDFT (:mod:`~sisl.io.bigdft`)
===============================

.. currentmodule:: sisl.io.bigdft

.. autosummary::
   :toctree:

   asciiSileBigDFT - the input for BigDFT


.. _toc-io-gulp:

GULP (:mod:`~sisl.io.gulp`)
===========================

.. currentmodule:: sisl.io.gulp

.. autosummary::
   :toctree:

   gotSileGULP - the output from GULP
   fcSileGULP - force constant output from GULP


.. _toc-io-openmx:

OpenMX (:mod:`~sisl.io.openmx`)
===============================

.. currentmodule:: sisl.io.openmx

.. autosummary::
   :toctree:

   omxSileOpenMX - input file


.. _toc-io-scaleup:

ScaleUp (:mod:`~sisl.io.scaleup`)
=================================

.. currentmodule:: sisl.io.scaleup

.. autosummary::
   :toctree:

   orboccSileScaleUp - orbital information
   refSileScaleUp - reference coordinates
   rhamSileScaleUp - Hamiltonian file


.. _toc-io-siesta:

Siesta (:mod:`~sisl.io.siesta`)
===============================

.. currentmodule:: sisl.io.siesta

.. autosummary::
   :toctree:

   fdfSileSiesta - input file
   outSileSiesta - output file
   xvSileSiesta - xyz and vxyz file
   bandsSileSiesta - band structure information
   eigSileSiesta - EIG file
   pdosSileSiesta - PDOS file
   gridSileSiesta - Grid charge information (binary)
   gridncSileSiesta - NetCDF grid output files (netcdf)
   dmSileSiesta - density matrix information
   hsxSileSiesta - Hamiltonian and overlap matrix information
   ncSileSiesta - NetCDF output file
   ionxmlSileSiesta - Basis-information from the ion.xml files
   ionncSileSiesta - Basis-information from the ion.nc files
   orbindxSileSiesta - Basis set information (no geometry information)
   faSileSiesta - Forces on atoms
   fcSileSiesta - Force constant matrix
   kpSileSiesta - k-points from simulation
   rkpSileSiesta - k-points to simulation


.. _toc-io-transiesta:

TranSiesta (:mod:`~sisl.io.siesta`)
===================================

.. autosummary::
   :toctree:

   tshsSileSiesta - TranSiesta Hamiltonian
   tsdeSileSiesta - TranSiesta (energy) density matrix
   tsgfSileSiesta - TranSiesta surface Green function files
   tsvncSileSiesta - TranSiesta specific Hartree potential file


.. _toc-io-tbtrans:

TBtrans (:mod:`~sisl.io.tbtrans`)
=================================

.. currentmodule:: sisl.io.tbtrans

.. autosummary::
   :toctree:

   tbtncSileTBtrans
   deltancSileTBtrans
   tbtgfSileTBtrans - TBtrans surface Green function files
   tbtsencSileTBtrans
   tbtavncSileTBtrans
   tbtprojncSileTBtrans

Additionally the PHtrans code also has these files

.. autosummary::
   :toctree:

   phtncSilePHtrans
   phtsencSilePHtrans
   phtavncSilePHtrans
   phtprojncSilePHtrans


.. _toc-io-vasp:

VASP (:mod:`~sisl.io.vasp`)
===========================

.. currentmodule:: sisl.io.vasp

.. autosummary::
   :toctree:

   carSileVASP
   doscarSileVASP
   eigenvalSileVASP
   chgSileVASP
   locpotSileVASP


.. _toc-io-wannier90:

Wannier90 (:mod:`~sisl.io.wannier90`)
=====================================

.. currentmodule:: sisl.io.wannier90

.. autosummary::
   :toctree:

   winSileWannier90 - input file


.. #################################
.. Switch back to the sisl.io module
.. #################################

.. currentmodule:: sisl.io


Low level methods/classes
=========================


Classes and methods generically only used internally. If you wish to create
your own `Sile` you should inherit either of `Sile` (ASCII), `SileCDF` (NetCDF)
or `SileBin` (binary), then subsequently add it using `add_sile` which enables
its generic use in all routines etc.

.. autosummary::
   :toctree:

   get_siles - retrieve all files with specific attributes or methods
   get_sile_class - retrieve class via a file name by comparing the extension
   BaseSile - the base class for all sisl files
   Sile - a base class for ASCII files
   SileCDF - a base class for NetCDF files
   SileBin - a base class for binary files


.. ###############################################
.. Add all io modules to the toc (to be reachable)
.. ###############################################

.. autosummary::
   :toctree:
   :hidden:

   bigdft
   gulp
   openmx
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
from .openmx import *
from .pdb import *
from .scaleup import *
from .siesta import *
from .tbtrans import *
from .table import *
from .vasp import *
from .wannier90 import *
from .xsf import *
from .xyz import *

__all__ = [s for s in dir() if not s.startswith('_')]
