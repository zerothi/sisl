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

__author__ = "sisl developers"
__license__ = "MPL-2.0"

import sisl._version as _version

__version__ = _version.version
__version_tuple__ = _version.version_tuple
__bibtex__ = f"""# BibTeX information if people wish to cite
@software{{zerothi_sisl,
    author = {{Papior, Nick and Febrer, Pol}},
    title  = {{sisl: v{__version__}}},
    year   = {{ {year} }},
    doi    = {{10.5281/zenodo.597181}},
    url    = {{https://doi.org/10.5281/zenodo.597181}},
}}"""
__citation__ = __bibtex__

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

from ._core import *

# Import warning classes
# We currently do not import warn and info
# as they are too generic names in case one does from sisl import *
# Perhaps we should simply remove them from __all__?
from .messages import SislException, SislWarning, SislInfo, SislError
from .messages import SislDeprecation

# Simple access
import sisl.constant as C

# load the most commonly, and basic classes
# The unit contain the SI standard conversions using
# all digits (not program specific)
from .unit import unit_group, unit_convert, unit_default, units

# Below are sisl-specific imports
from .shape import *

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

from ._nodify import on_nodify as __nodify__

# Set all the placeholders for the plot attribute
# of sisl classes
from ._lazy_viz import set_viz_placeholders

set_viz_placeholders()

from ._ufuncs import expose_registered_methods

expose_registered_methods("sisl")
expose_registered_methods("sisl.physics")

del expose_registered_methods


# Lazy load modules to easier access sub-modules
def __getattr__(attr):
    """Enables simpler access of sub-modules, without having to import them"""

    # One can test that this is only ever called once
    # per sub-module.
    # Insert a print statement, and you'll see that:
    # import sisl
    # sisl.geom
    # sisl.geom
    # will only print *once*.

    if attr == "geom":
        import sisl.geom as geom

        return geom
    if attr == "io":
        import sisl.io as io

        return io
    if attr == "physics":
        import sisl.physics as physics

        return physics
    if attr == "linalg":
        import sisl.linalg as linalg

        return linalg
    if attr == "shape":
        import sisl.shape as shape

        return shape
    if attr == "mixing":
        import sisl.mixing as mixing

        return mixing
    if attr == "viz":
        import sisl.viz as viz

        return viz
    if attr == "utils":
        import sisl.utils as utils

        return utils
    if attr == "unit":
        import sisl.unit as unit

        return unit
    if attr == "C":
        import sisl.constant as C

        return constant
    if attr == "constant":
        import sisl.constant as constant

        return constant
    if attr == "typing":
        import sisl.typing as typing

        return typing

    if attr in ("print_debug_info", "debug_info"):
        from ._debug_info import print_debug_info

        return print_debug_info

    raise AttributeError(f"module {__name__} has no attribute {attr}")


# Make these things publicly available
__all__ = [s for s in dir() if not s.startswith("_")]
