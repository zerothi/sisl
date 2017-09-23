"""
Shapes (:mod:`sisl.shape`)
==========================

.. currentmodule:: sisl.shape

A variety of default shapes.

All shapes inherit the `Shape` class.

.. module:: sisl.shape.shape

.. autosummary::

   Shape

Cuboids (:mod:`sisl.shape.prism4`)
----------------------------------

.. module:: sisl.shape.prism4

.. autosummary::

   Cuboid
   Cube

Ellipsoids (:mod:`sisl.shape.ellipsoid`)
----------------------------------------

.. module:: sisl.shape.ellipsoid

.. autosummary::

   Ellipsoid 
   Spheroid
   Sphere

"""

from .shape import *
from .ellipsoid import *
from .prism4 import *

__all__ = [s for s in dir() if not s.startswith('_')]
