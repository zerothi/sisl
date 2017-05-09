""" sisl package

Geometry and tight-binding setups using pure python.
"""

# Import version string and the major, minor, micro as well
from .info import version as __version__
from .info import major as __major__
from .info import minor as __minor__
from .info import micro as __micro__

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

from .io import *

from .sparse import *

# Hamiltonian etc.
from .physics import *


# Import the default geom structure
# This enables:
# import sisl
# sisl.geom.graphene
from . import geom
