# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Input/Output
------------

Available files for reading/writing

General file retrieval is done through the file extensions and through
specifications at read time.
For instance

  >>> get_sile("hello.xyz")

will automatically recognize the `xyzSile`. If one wishes to explicitly
specify the filetype one can denote using a specifier:

  >>> get_sile("this_file_is_in_xyz_format.dat{xyz}")

The specifier can be:

- `contains=<name>` which searches for `<name>` in the class name
- `<name>` shorthand for `contains=<name>`
- `endswith=<name>` matches classes which `endswith(<name>)`
- `startswith=<name>` matches classes which `startswith(<name>)`

These classifiers are primarily useful when there are multiple classes
using the same file suffix.


Basic IO methods/classes
------------------------

  add_sile - add a file to the list of files that sisl can interact with
  get_sile - retrieve a file object via a file name by comparing the extension
  SileError - sisl specific error


Generic files
-------------

Files not specificaly related to any code.

  tableSile - data file in tabular form
  xyzSile - atomic coordinate file
  pdbSile - atomic coordinates and MD content
  cubeSile - atomic coordinates *and* 3D grid values
  moldenSile - atomic coordinate file specific for Molden
  xsfSile - atomic coordinate file specific for XCrySDen


For software specific files, see the below list:

BigDFT
------

  asciiSileBigDFT - the input for BigDFT

DFTB+
-----

  overrealSileDFTB - the overlap matrix
  hamrealSileDFTB - the Hamiltonian matrix

FHIaims
-------

  inSileFHIaims - input file for FHIaims

GULP
----

  gotSileGULP - the output from GULP
  fcSileGULP - force constant output from GULP

OpenMX
------

  omxSileOpenMX - input file

ORCA
----

  stdoutSileORCA - standard output file
  outputSileORCA - output file
  txtSileORCA - property.txt file

ScaleUp
-------

  orboccSileScaleUp - orbital information
  refSileScaleUp - reference coordinates
  rhamSileScaleUp - Hamiltonian file

Siesta
------

  aniSileSiesta - xyz file in a trajectory format
  bandsSileSiesta - band structure information
  ionxmlSileSiesta - Basis-information from the ion.xml files
  ionncSileSiesta - Basis-information from the ion.nc files
  onlysSileSiesta - Overlap matrix information
  dmSileSiesta - density matrix information
  hsxSileSiesta - Hamiltonian and overlap matrix information
  gridSileSiesta - Grid charge information (binary)
  gridncSileSiesta - NetCDF grid output files (netcdf)
  ncSileSiesta - NetCDF output file
  tshsSileSiesta - TranSiesta Hamiltonian
  tsdeSileSiesta - TranSiesta (energy) density matrix
  tsgfSileSiesta - TranSiesta surface Green function files
  tsvncSileSiesta - TranSiesta specific Hartree potential file
  wfsxSileSiesta - wavefunctions
  eigSileSiesta - EIG file
  faSileSiesta - Forces on atoms
  fcSileSiesta - Force constant matrix
  fdfSileSiesta - input file
  kpSileSiesta - k-points from simulation
  rkpSileSiesta - k-points to simulation
  orbindxSileSiesta - Basis set information (no geometry information)
  structSileSiesta - structure information
  outSileSiesta - output file
  pdosSileSiesta - PDOS file
  xvSileSiesta - xyz and vxyz file

TBtrans
-------

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
----

  carSileVASP
  doscarSileVASP
  eigenvalSileVASP
  chgSileVASP
  locpotSileVASP
  outcarSileVASP

Wannier90
---------

  winSileWannier90 - input file
  tbSileWannier90
  hrSileWannier90
  centresSileWannier90


Low level methods/classes
-------------------------

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

# isort: split

# Non-code specific files
from .cube import *
from .molden import *
from .pdb import *
from .table import *
from .xsf import *
from .xyz import *

# isort: split

from .bigdft import *
from .dftb import *
from .fhiaims import *
from .gulp import *
from .ham import *
from .openmx import *
from .orca import *
from .scaleup import *
from .siesta import *
from .tbtrans import *
from .vasp import *
from .wannier90 import *
