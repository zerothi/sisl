"""
==========================
Shapes (:mod:`sisl.shape`)
==========================

.. currentmodule:: sisl.shape

A variety of default shapes.

All shapes inherit the :ref:`Shape` class.

.. autosummary::
   :toctree: api-generated/

   Shape - base class

Cuboids
=======

.. autosummary::
   :toctree: api-generated/

   Cuboid - 3d cube
   Cube - 3d box

Ellipsoids
==========

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

#for rm in ['shape', 'ellipsoid', 'prism4']:
#    __all__.remove(rm)
