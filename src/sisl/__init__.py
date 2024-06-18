# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# isort: skip_file
from __future__ import annotations

"""
sisl
====

sisl is an electronic structure package which may interact with tight-binding
and DFT matrices alike.

The full sisl package consistent of a large variety of classes and methods
which enables large-scale tight-binding calculations as well as post-processing
DFT calculations.

Below a set of classes that are the basis of *everything* in sisl is present.

Generic classes
===============

   PeriodicTable
   Orbital
   SphericalOrbital
   AtomicOrbital
   Atoms
   Geometry
   Lattice
   Grid

Below are a group of advanced classes rarely needed.
A lot of the sub-classes extend these classes, or use them
intrinsically. However, they are not necessarily intended
for users use.

Advanced classes
================

   Quaternion
   SparseCSR
   SparseAtom
   SparseOrbital

"""
import logging
import datetime

year = datetime.datetime.now().year

# instantiate the logger, but we will not use it here...
logging.getLogger(__name__)

__author__ = "Nick Papior"
__license__ = "MPL-2.0"

import sisl._version as _version

__version__ = _version.version
__version_tuple__ = _version.version_tuple
__bibtex__ = f"""# BibTeX information if people wish to cite
@software{{zerothi_sisl,
    author = {{Papior, Nick}},
    title  = {{sisl: v{__version__}}},
    year   = {{ {year} }},
    doi    = {{10.5281/zenodo.597181}},
    url    = {{https://doi.org/10.5281/zenodo.597181}},
}}"""

# do not expose this helper package
del _version, year, datetime

from sisl._environ import get_environ_variable

# Immediately check if the file is logable
log_file = get_environ_variable("SISL_LOG_FILE")
if not log_file.is_dir():
    # Create the logging
    log_lvl = get_environ_variable("SISL_LOG_LEVEL")

    # Start the logging to the file
    logging.basicConfig(filename=str(log_file), level=getattr(logging, log_lvl))
    del log_lvl
del log_file


# import the common options used
from ._common import *

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

# Mixing
from . import mixing

# Below are sisl-specific imports
from .shape import *

from ._core import *

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
from .io.sile import (
    add_sile,
    get_sile_class,
    get_sile,
    get_siles,
    get_sile_rules,
    SileError,
    BaseSile,
    Sile,
    SileCDF,
    SileBin,
)

# Allow geometry to register siles
# Ensure BaseSile works as a str
# We have to do it after loading BaseSile and Geometry
# Since __getitem__ always instantiate the class, we have to use the
# contained lookup table.
Geometry.new.register(BaseSile, Geometry.new._dispatchs[str])
Geometry.new.register("Sile", Geometry.new._dispatchs[str])
Geometry.to.register(BaseSile, Geometry.to._dispatchs[str])
Geometry.to.register("Sile", Geometry.to._dispatchs[str])
Lattice.new.register(BaseSile, Lattice.new._dispatchs[str])
Lattice.new.register("Sile", Lattice.new._dispatchs[str])
Lattice.to.register(BaseSile, Lattice.to._dispatchs[str])
Lattice.to.register("Sile", Lattice.to._dispatchs[str])

# Import the default geom structure
# This enables:
# import sisl
# sisl.geom.graphene
from . import geom

from ._nodify import on_nodify as __nodify__

# Set all the placeholders for the plot attribute
# of sisl classes
from ._lazy_viz import set_viz_placeholders

set_viz_placeholders()

# If someone tries to get the viz attribute, we will load the viz module
_LOADED_VIZ = False


def __getattr__(name):
    global _LOADED_VIZ
    if name == "viz" and not _LOADED_VIZ:
        _LOADED_VIZ = True
        import sisl.viz

        return sisl.viz
    raise AttributeError(f"module {__name__} has no attribute {name}")


from ._ufuncs import expose_registered_methods

expose_registered_methods("sisl")
del expose_registered_methods

# Make these things publicly available
__all__ = [s for s in dir() if not s.startswith("_")]
