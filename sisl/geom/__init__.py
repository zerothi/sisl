"""
Common geometries (:mod:`sisl.geom`)
====================================

.. currentmodule:: sisl.geom

A variety of default geometries.

Basic
-----

.. module:: sisl.geom.basic

.. autosummary::
   :toctree: api-sisl/

   sc - simple cubic
   bcc - body centered cubic
   fcc - face centered cubic
   hcp - hexagonal


Flat
----

.. module:: sisl.geom.flat

.. autosummary::
   :toctree: api-sisl/

   honeycomb - graphene like, but generic
   graphene - graphen

Nanotube
--------

.. module:: sisl.geom.nanotube

.. autosummary::
   :toctree: api-sisl/

   nanotube - a nanotube (default to carbon)

Special
-------

.. module:: sisl.geom.special

.. autosummary::
   :toctree: api-sisl/

   diamond - a diamond lattice

"""

from .basic import *
from .flat import *
from .nanotube import *
from .special import *

__all__ = [s for s in dir() if not s.startswith('_')]
