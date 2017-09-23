"""
Shapes (:mod:`sisl.shape`)
==========================

.. currentmodule:: sisl.shape

A variety of default shapes.

All shapes inherit the `Shape` class.

.. autosummary::
   :toctree: api-generated/

   Shape

Cuboids
-------

.. autosummary::
   :toctree: api-generated/

   Cuboid
   Cube

Ellipsoids
----------

.. autosummary::
   :toctree: api-generated/

   Ellipsoid 
   Spheroid
   Sphere

"""

from .shape import *
from .ellipsoid import *
from .prism4 import *

__all__ = [s for s in dir() if not s.startswith('_')]
