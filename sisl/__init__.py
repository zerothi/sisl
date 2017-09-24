"""
==================================
sisl: electronic structure package
==================================

.. currentmodule:: sisl

The sisl package consists of a variety of sub packages enabling
different routines for electronic structure calculations.

Generic classes
===============

.. autosummary::
   :toctree: api-generated

   PeriodicTable
   Atom
   Atoms
   Geometry
   Grid
   SuperCell


..   Quaternion
..   SparseCSR
..   SparseAtom
..   SparseOrbital
..   Selector

"""

# Import version string and the major, minor, micro as well
from .info import version as __version__
from .info import major as __major__
from .info import minor as __minor__
from .info import micro as __micro__

# Import numpy_scipy routines
import sisl._numpy_scipy as math

# Import the Selector
from .selector import *

# Import plot routine
from .plot import *

# load the most commonly, and basic classes
# The units contain the SI standard conversions using
# all digits (not program specific)
from .units import *
from .quaternion import *
from .shape import *

from .supercell import *
from .atom import *

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
#  sisl.io.TBTGFSileSiesta
# or
#  sisl.get_sile
# This will reduce the cluttering of the separate entities
# that sisl is made of.
from .io.sile import (add_sile, get_sile_class, get_sile,
                      get_siles, SileError,
                      BaseSile, Sile, SileCDF, SileBin)
import sisl.io as io

# Import the default geom structure
# This enables:
# import sisl
# sisl.geom.graphene
from . import geom

# Make these things publicly available
__all__ = [s for s in dir() if not s.startswith('_')]
__all__ += ['__{}__'.format(r) for r in ['version', 'major', 'minor', 'micro']]
