""" Available files for reading/writing

.. currentmodule:: sisl.io

sisl handles a large variety of input/output files from a large selection
of DFT software and other post-processing tools.

Since sisl may be used with many other packages all files are name *siles*
to distinguish them from files from other packages.


Basic IO classes
================

.. module:: sisl.io.sile

.. autosummary::
   :toctree: api-sisl/

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
=============

These files are generic, in the sense that they are not specific to a
given code.

.. module:: sisl.io

.. autosummary::
   :toctree: api-sisl/

   xyz.XYZSile - atomic coordinate file
   cube.CUBESile - atomic coordinates *and* 3D grid values
   table.TableSile - data file in tabular form
   molden.MoldenSile - atomic coordinate file specific for Molden
   xsf.XSFSile - atomic coordinate file specific for XCrySDen



External code in/out put supported
==================================

List the relevant codes that `sisl` can interact with. If there are files you think
are missing, please create an issue :ref:`here <issue>`.

- `BigDFT`_
- `GULP`_
- `Molden`_
- `ScaleUp`_
- `Siesta`_
- `VASP`_
- `Wannier90`_
- `XCrySDen`_

BigDFT
------

.. module:: sisl.io.bigdft

.. autosummary::
   :toctree: api-sisl/

   ascii.ASCIISileBigDFT - the input for BigDFT

GULP
----

.. module:: sisl.io.gulp

.. autosummary::
   :toctree: api-sisl/

   got.gotSileGULP - the output from GULP
   hessian.HessianSileGULP - Hessian output from GULP


Molden
------

.. module:: sisl.io.molden

.. autosummary::
   :toctree: api-sisl/

   MoldenSile - coordinate file for molden

ScaleUp
-------

.. module:: sisl.io.scaleup

.. autosummary::
   :toctree: api-sisl/

   orbocc.orboccSileScaleUp - orbital information
   ref.REFSileScaleUp - reference coordinates
   rham.rhamSileScaleUp - Hamiltonian file

Siesta
------

.. module:: sisl.io.siesta

.. autosummary::
   :toctree: api-sisl/

   bands.bandsSileSiesta - band structure information
   binaries.TSHSSileSiesta - TranSiesta Hamiltonian
   binaries.GridSileSiesta - Grid charge information
   binaries.EnergyGridSileSiesta - Grid potential information
   binaries.TSGFSileSiesta - TranSiesta surface Green function files
   binaries.TBTGFSileSiesta - TBtrans surface Green function files
   eig.eigSileSiesta - EIG file
   fdf.fdfSileSiesta - input file
   out.outSileSiesta - output file
   siesta.ncSileSiesta - NetCDF output file
   siesta_grid.gridncSileSiesta - NetCDF grid output files
   xv.XVSileSiesta - xyz and vxyz file

TBtrans
-------

.. module:: sisl.io.siesta.tbtrans

.. autosummary::
   :toctree: api-sisl/

   tbtncSileSiesta - output
   tbtavncSileSiesta - k-averaged output
   phtncSileSiesta - output (phtrans)
   phtavncSileSiesta - k-averaged output (phtrans)
   deltancSileSiesta - :math:`\delta` files
   dHncSileSiesta - :math:`\delta H` (deprecated)

.. module:: sisl.io.siesta.tbtrans_proj

.. autosummary::
   :toctree: api-sisl/

   tbtprojncSileSiesta - projection output
   phtprojncSileSiesta - projection output (phtrans)


Wannier90
---------

.. module:: sisl.io.wannier

.. autosummary::
   :toctree: api-sisl/

   seedname.winSileWannier90 - input file


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
from .table import *
from .vasp import *
from .wannier import *
from .xsf import *
from .xyz import *

__all__ = [s for s in dir() if not s.startswith('_')]
