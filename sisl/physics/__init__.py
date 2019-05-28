"""
======================================
Physical objects (:mod:`sisl.physics`)
======================================

.. module:: sisl.physics
   :noindex:

Implementations of various DFT and tight-binding related quantities
are defined. The implementations range from simple Brillouin zone
perspectives to self-energy calculations from Hamiltonians.

In `sisl` the general usage of physical matrices are considering sparse
matrices. Hence Hamiltonians, density matrices, etc. are considered
sparse. There are exceptions, but it is generally advisable to have this in mind.

Brillouin zone (:mod:`~sisl.physics.brillouinzone`)
===================================================

.. autosummary::
   :toctree:

   BrillouinZone - base class
   MonkhorstPack - MP class
   BandStructure - bandstructure class


Spin configuration
==================

.. autosummary::
   :toctree:

   Spin - spin configuration


Physical quantites
==================

.. autosummary::
   :toctree:

   EnergyDensityMatrix
   DensityMatrix
   Hamiltonian
   DynamicalMatrix
   SelfEnergy
   SemiInfinite
   RecursiveSI
   RealSpaceSE
   RealSpaceSI



Electrons (:mod:`~sisl.physics.electron`)
=========================================

.. autosummary::
   :toctree:

   ~electron.DOS
   ~electron.PDOS
   ~electron.velocity
   ~electron.velocity_matrix
   ~electron.berry_phase
   ~electron.wavefunction
   ~electron.spin_moment
   ~electron.spin_squared
   EigenvalueElectron
   EigenvectorElectron
   EigenstateElectron


Phonons (:mod:`~sisl.physics.phonon`)
=====================================

.. autosummary::
   :toctree:

   ~phonon.DOS
   ~phonon.PDOS
   ~phonon.velocity
   ~phonon.displacement
   EigenvaluePhonon
   EigenvectorPhonon
   EigenmodePhonon


Bloch's theorem (:mod:`~sisl.physics.bloch`)
============================================

.. autosummary::
   :toctree:

   Bloch


Distribution functions (:mod:`~sisl.physics.distribution`)
==========================================================

.. autosummary::
   :toctree:

   get_distribution
   gaussian
   lorentzian
   fermi_dirac
   bose_einstein
   cold
   step_function
   heaviside



.. Below lines ensures that the sub-modules gets their own page.

.. autosummary::
   :toctree:
   :hidden:

   sisl.physics.electron
   sisl.physics.phonon
   sisl.physics.distribution
   sisl.physics.brillouinzone


Low level objects
=================

The low level objects are the driving objects for a majority of the physical
objects found here. They are rarely (if ever) required to be used, but they
may be important for developers wishing to extend the functionality of `sisl`
using generic class-structures. For instance the `~Hamiltonian` inherits the
`~SparseOrbitalBZSpin` class and `~EigenvalueElectron` inherits from `~Coefficient`.

States
------

.. autosummary::
   :toctree:

   Coefficient
   State
   StateC


Sparse matrices
---------------

.. autosummary::
   :toctree:

   SparseOrbitalBZ - sparse orbital matrix with k-dependent properties
   SparseOrbitalBZSpin - sparse orbital matrix with k-dependent properties and spin configuration

"""
from .distribution import *
from .brillouinzone import *
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
from .self_energy import *

__all__ = [s for s in dir() if not s.startswith('_')]
