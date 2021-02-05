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
from .distribution import *
from .brillouinzone import *
# Patch BrillouinZone objects and import apply classes
from ._brillouinzone_apply import *

from .bloch import *
from .spin import *
from .sparse import *
from .state import *

from . import electron
from .electron import CoefficientElectron, StateElectron, StateCElectron
from .electron import EigenvalueElectron, EigenvectorElectron, EigenstateElectron

from . import phonon
from .phonon import CoefficientPhonon, ModePhonon, ModeCPhonon
from .phonon import EigenvaluePhonon, EigenvectorPhonon, EigenmodePhonon

from .energydensitymatrix import *
from .densitymatrix import *
from .hamiltonian import *
from .dynamicalmatrix import *
from .overlap import *
from .self_energy import *
