"""
Input/Output
============

Available files for reading/writing

sisl handles a large variety of input/output files from a large selection
of DFT software and other post-processing tools.

Since sisl may be used with many other packages all files are name *siles*
to distinguish them from files from other packages.


Basic IO methods/classes
========================

   add_sile - add a file to the list of files that sisl can interact with
   get_sile - retrieve a file object via a file name by comparing the extension
   SileError - sisl specific error


Generic files
=============

Files not specificly related to any code.

   tableSile - data file in tabular form
   xyzSile - atomic coordinate file
   pdbSile - atomic coordinates and MD content
   cubeSile - atomic coordinates *and* 3D grid values
   moldenSile - atomic coordinate file specific for Molden
   xsfSile - atomic coordinate file specific for XCrySDen

BigDFT
======

   asciiSileBigDFT - the input for BigDFT


GULP
====

   gotSileGULP - the output from GULP
   fcSileGULP - force constant output from GULP


OpenMX
======

   omxSileOpenMX - input file


ScaleUp
=======

   orboccSileScaleUp - orbital information
   refSileScaleUp - reference coordinates
   rhamSileScaleUp - Hamiltonian file


Siesta
======

   fdfSileSiesta - input file
   outSileSiesta - output file
   xvSileSiesta - xyz and vxyz file
   bandsSileSiesta - band structure information
   eigSileSiesta - EIG file
   pdosSileSiesta - PDOS file
   gridSileSiesta - Grid charge information (binary)
   gridncSileSiesta - NetCDF grid output files (netcdf)
   onlysSileSiesta - Overlap matrix information
   dmSileSiesta - density matrix information
   hsxSileSiesta - Hamiltonian and overlap matrix information
   wfsxSileSiesta - wavefunctions
   ncSileSiesta - NetCDF output file
   ionxmlSileSiesta - Basis-information from the ion.xml files
   ionncSileSiesta - Basis-information from the ion.nc files
   orbindxSileSiesta - Basis set information (no geometry information)
   faSileSiesta - Forces on atoms
   fcSileSiesta - Force constant matrix
   kpSileSiesta - k-points from simulation
   rkpSileSiesta - k-points to simulation


TranSiesta
==========

   tshsSileSiesta - TranSiesta Hamiltonian
   tsdeSileSiesta - TranSiesta (energy) density matrix
   tsgfSileSiesta - TranSiesta surface Green function files
   tsvncSileSiesta - TranSiesta specific Hartree potential file


TBtrans
=======

   tbtncSileTBtrans
   deltancSileTBtrans
   tbtgfSileTBtrans - TBtrans surface Green function files
   tbtsencSileTBtrans
   tbtavncSileTBtrans
   tbtprojncSileTBtrans

Additionally the PHtrans code also has these files

   phtncSilePHtrans
   phtsencSilePHtrans
   phtavncSilePHtrans
   phtprojncSilePHtrans


VASP
====

   carSileVASP
   doscarSileVASP
   eigenvalSileVASP
   chgSileVASP
   locpotSileVASP
   outSileVASP


Wannier90
=========

   winSileWannier90 - input file


Low level methods/classes
=========================


Classes and methods generically only used internally. If you wish to create
your own `Sile` you should inherit either of `Sile` (ASCII), `SileCDF` (NetCDF)
or `SileBin` (binary), then subsequently add it using `add_sile` which enables
its generic use in all routines etc.


   get_siles - retrieve all files with specific attributes or methods
   get_sile_class - retrieve class via a file name by comparing the extension
   BaseSile - the base class for all sisl files
   Sile - a base class for ASCII files
   SileCDF - a base class for NetCDF files
   SileBin - a base class for binary files
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
