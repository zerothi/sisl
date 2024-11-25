# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Physical objects
================

Implementations of various DFT and tight-binding related quantities
are defined. The implementations range from simple Brillouin zone
perspectives to self-energy calculations from Hamiltonians.

In `sisl` the general usage of physical matrices are considering sparse
matrices. Hence Hamiltonians, density matrices, etc. are considered
sparse. There are exceptions, but it is generally advisable to have this in mind.

Brillouin zone
==============

   BrillouinZone - base class
   MonkhorstPack - MP class
   BandStructure - bandstructure class


Spin configuration
==================

   Spin - spin configuration


Physical quantites
==================

   EnergyDensityMatrix
   DensityMatrix
   Hamiltonian
   DynamicalMatrix
   Overlap
   SelfEnergy
   WideBandSE
   SemiInfinite
   RecursiveSI
   RealSpaceSE
   RealSpaceSI


Bloch's theorem
===============

   Bloch


Distribution functions
======================

   get_distribution
   gaussian
   lorentzian
   fermi_dirac
   bose_einstein
   cold
   step_function
   heaviside


Low level objects
=================

The low level objects are the driving objects for a majority of the physical
objects found here. They are rarely (if ever) required to be used, but they
may be important for developers wishing to extend the functionality of `sisl`
using generic class-structures. For instance the `~Hamiltonian` inherits the
`~SparseOrbitalBZSpin` class and `~EigenvalueElectron` inherits from `~Coefficient`.

States
------

   Coefficient
   State
   StateC


Sparse matrices
---------------

   SparseOrbitalBZ - sparse orbital matrix with k-dependent properties
   SparseOrbitalBZSpin - sparse orbital matrix with k-dependent properties and spin configuration

"""

from ._common import *
from ._feature import *
from .distribution import *
from .sparse import *
from .spin import *
from .state import *

# isort: split

# Patch BrillouinZone objects and import apply classes
from .bloch import *
from .brillouinzone import *
from .densitymatrix import *
from .dynamicalmatrix import *
from .energydensitymatrix import *
from .hamiltonian import *
from .overlap import *
from .self_energy import *

# isort: split

from . import electron, phonon
from .electron import (
    CoefficientElectron,
    EigenstateElectron,
    EigenvalueElectron,
    EigenvectorElectron,
    StateCElectron,
    StateElectron,
)
from .phonon import (
    CoefficientPhonon,
    EigenmodePhonon,
    EigenvaluePhonon,
    EigenvectorPhonon,
    ModeCPhonon,
    ModePhonon,
)

# isort: split

from ._brillouinzone_apply import *
from ._ufuncs_brillouinzone import *
from ._ufuncs_densitymatrix import *
from ._ufuncs_dynamicalmatrix import *
from ._ufuncs_electron import *
from ._ufuncs_energydensitymatrix import *
from ._ufuncs_hamiltonian import *
from ._ufuncs_overlap import *
from ._ufuncs_state import *
