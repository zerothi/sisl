"""
====================================
Common geometries (:mod:`sisl.geom`)
====================================

.. currentmodule:: sisl.geom

A variety of default geometries.

Basic
=====

.. autosummary::
   :toctree: api-generated/

   sc - simple cubic
   bcc - body centered cubic
   fcc - face centered cubic
   hcp - hexagonal
   diamond - a diamond lattice

2D materials
============

.. autosummary::
   :toctree: api-generated/

   honeycomb - graphene like, but generic
   graphene - graphen

Nanotube
========

.. autosummary::
   :toctree: api-generated/

   nanotube - a nanotube (default to carbon)

"""

from .basic import *
from .flat import *
from .nanotube import *
from .special import *

__all__ = [s for s in dir() if not s.startswith('_')]
for rm in ['basic', 'flat', 'special']:
    __all__.remove(rm)
