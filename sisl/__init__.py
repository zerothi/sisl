""" sisl package

Geometry and tight-binding setups using pure python.
"""

# Import version string and the major, minor, micro as well
from .version import version as __version__
from .version import major as __major__
from .version import minor as __minor__
from .version import micro as __micro__

# load the units
from .units import *

# load the most commonly, and basic classes
from .quaternion import *

from .supercell import *
from .atom import *

from .geometry import *
from .grid import *

from .io import *
from .tb import *

# Import the default geom structure
# This enables:
# import sisl
# sisl.geom.graphene
from . import geom
