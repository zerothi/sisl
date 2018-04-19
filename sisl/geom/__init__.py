"""
====================================
Common geometries (:mod:`sisl.geom`)
====================================

.. module:: sisl.geom
   :noindex:

A variety of default geometries.

Basic (:mod:`sisl.geom.basic`)
==============================

.. autosummary::
   :toctree:

   sc - simple cubic
   bcc - body centered cubic
   fcc - face centered cubic
   hcp - hexagonal
   diamond - a diamond lattice

2D materials
================================

.. autosummary::
   :toctree:

   honeycomb - graphene like, but generic
   graphene - graphen

Nanotube
========

.. autosummary::
   :toctree:

   nanotube - a nanotube (default to carbon)


.. autosummary::
   :toctree:
   :hidden:

   sisl.geom.basic

"""
from .basic import *
from .flat import *
from .nanotube import *
from .special import *

__all__ = [s for s in dir() if not s.startswith('_')]
