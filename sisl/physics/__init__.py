"""
======================================
Physical objects (:mod:`sisl.physics`)
======================================

.. currentmodule:: sisl.physics

Implementations of various DFT and tight-binding related quantities
are defined. The implementations range from simple Brillouin zone
perspectives to self-energy calculations from Hamiltonians.

In `sisl` the general usage of physical matrices are considering sparse
matrices. Hence Hamiltonians, density matrices, etc. are considered
sparse. There are exceptions, but it is generally advisable to have this in mind.

Brillouin zone
==============

.. autosummary::
   :toctree: api-generated/

   BrillouinZone - base class
   MonkhorstPackBZ - MP class
   PathBZ - bandstructure class

Spin configurations
===================

.. autosummary::
   :toctree: api-generated/

   Spin - spin configurations

Sparse matrices
===============

.. autosummary::
   :toctree: api-generated/

   SparseOrbitalBZ - sparse orbital matrix with k-dependent properties
   SparseOrbitalBZSpin - sparse orbital matrix with k-dependent properties and spin configuration


Physical quantites
==================

.. autosummary::
   :toctree: api-generated/

   EnergyDensityMatrix
   DensityMatrix
   Hamiltonian
   Hessian
   SelfEnergy
   SemiInfinite
   RecursiveSI

"""

from .brillouinzone import *
from .spin import *
from .sparse import *

from .energydensitymatrix import *
from .densitymatrix import *
from .hamiltonian import *
from .hessian import *
from .self_energy import *

__all__ = [s for s in dir() if not s.startswith('_')]

#for rm in ['brillouinzone', 'spin', 'sparse',
#           'energydensitymatrix', 'densitymatrix',
#           'hamiltonian', 'hessian', 'self_energy']:
#    __all__.remove(rm)
