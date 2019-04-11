"""
==================
sisl (:mod:`sisl`)
==================

.. module:: sisl
   :noindex:

sisl is an electronic structure package which may interact with tight-binding
and DFT matrices alike.

The full sisl package consistent of a large variety of classes and methods
which enables large-scale tight-binding calculations as well as post-processing
DFT calculations.

Below a set of classes that are the basis of *everything* in sisl is present.

Generic classes
===============

.. autosummary::
   :toctree:

   PeriodicTable
   Orbital
   SphericalOrbital
   AtomicOrbital
   Atom
   Atoms
   Geometry
   SuperCell
   Grid

Below are a group of advanced classes rarely needed.
A lot of the sub-classes extend these classes, or use them
intrinsically. However, they are not necessarily intended
for users use.

Advanced classes
================

.. autosummary::
   :toctree:

   Quaternion
   SparseCSR
   SparseAtom
   SparseOrbital
   Selector

"""
from __future__ import print_function

__author__ = "Nick Papior"
__copyright__ = "LGPL-3.0"

# Import bibtex, version string and the major, minor, micro as well
from . import info
from .info import bibtex as __bibtex__
from .info import git_revision as __git_revision__
from .info import version as __version__
from .info import major as __major__
from .info import minor as __minor__
from .info import micro as __micro__
from .info import cite

# Import the Selector
from .selector import *

# Import oplist
from .oplist import oplist

# Import plot routine
from ._plot import plot as plot

# Import warning classes
# We currently do not import warn and info
# as they are too generic names in case one does from sisl import *
# Perhaps we should simply remove them from __all__?
from .messages import SislException, SislWarning, SislInfo, SislError
from .messages import SislDeprecation

# load the most commonly, and basic classes
# The unit contain the SI standard conversions using
# all digits (not program specific)
from .unit import unit_group, unit_convert, unit_default, units
from . import unit

# Import numerical constants (they required unit)
from . import constant
# To make it easier to type ;)
C = constant

# Specific linear algebra
from . import linalg

# Utilities
from . import utils

# Below are sisl-specific imports
from .quaternion import *
from .shape import *

from .supercell import *
from .atom import *

from .orbital import *
from .geometry import *
from .grid import *

from .sparse import *
from .sparse_geometry import *

# Physical quantities and required classes
from .physics import *

# The io files requires imports from the above modules
# Hence, we *must* import it last.
# This makes one able to get files through:
#  import sisl
#  sisl.io.tbtgfSileTBtrans
# or
#  sisl.get_sile
# This will reduce the cluttering of the separate entities
# that sisl is made of.
from . import io
from .io.sile import (add_sile, get_sile_class, get_sile,
                      get_siles, SileError,
                      BaseSile, Sile, SileCDF, SileBin)

# Import the default geom structure
# This enables:
# import sisl
# sisl.geom.graphene
from . import geom

# Make these things publicly available
__all__ = [s for s in dir() if not s.startswith('_')]
__all__ += ['__{}__'.format(s) for s in ['bibtex', 'version', 'major', 'minor', 'micro']]
__all__ += ['__{}__'.format(s) for s in ['git_revision']]
__all__ += ['__{}__'.format(s) for s in ['author', 'copyright']]
