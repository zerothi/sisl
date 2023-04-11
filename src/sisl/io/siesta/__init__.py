# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Siesta
======

The interaction between sisl and `Siesta`_ is one of the main goals due
to the implicit relationship between the developer of sisl and `Siesta`_.
Additionally the TranSiesta output files are also intrinsically handled by
sisl.

Remark that the `gridSileSiesta` file encompass all ``RHO``, ``RHOINIT``, ``DRHO``,
``RHOXC``, ``BADER``, ``IOCH``, ``TOCH`` ``VH``, ``VNA`` and ``VT`` binary output files.

   aniSileSiesta - ANI file
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
   orbindxSileSiesta - Basis-information (no geometry information, only ranges)
   faSileSiesta - Forces on atoms
   fcSileSiesta - Force constant matrix
   kpSileSiesta - k-points from simulation
   rkpSileSiesta - k-points to simulation
   structSileSiesta - geometry in STRUCT_* files


The TranSiesta specific output files are:

   tshsSileSiesta - TranSiesta Hamiltonian
   tsdeSileSiesta - TranSiesta TSDE
   tsgfSileSiesta - TranSiesta surface Green function files
   tsvncSileSiesta - TranSiesta potential solution input file

"""
from .ani import *
from .sile import *
from .bands import *
from .basis import *
from .binaries import *
from .eig import *
from .fa import *
from .fc import *
from .fdf import *
from .kp import *
from .orb_indx import *
from .out import *
from .pdos import *
from .struct import *
from .siesta_nc import *
from .siesta_grid import *
from .transiesta_grid import *
from .xv import *
