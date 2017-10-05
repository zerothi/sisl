"""
==========================
Shapes (:mod:`sisl.shape`)
==========================

.. module:: sisl.shape

A variety of default shapes.

All shapes inherit the `Shape` class.

.. autosummary::
   :toctree:

   Shape - base class
   Cuboid - 3d cube
   Cube - 3d box
   Ellipsoid
   Spheroid
   Sphere

"""

from .base import *
from .ellipsoid import *
from .prism4 import *

__all__ = [s for s in dir() if not s.startswith('_')]

#for rm in ['base', 'ellipsoid', 'prism4']:
#    __all__.remove(rm)
