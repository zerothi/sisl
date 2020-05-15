"""
====================================
Common geometries (:mod:`sisl.geom`)
====================================

.. module:: sisl.geom
   :noindex:

A variety of default geometries.

.. contents::
   :local:

.. autosummary::
   :toctree:
   :hidden:

   sc
   bcc
   fcc
   hcp
   nanoribbon
   graphene_nanoribbon
   agnr
   zgnr
   nanotube
   honeycomb
   bilayer
   diamond


Basic
=====

.. autofunction:: sc
   :noindex:
.. autofunction:: bcc
   :noindex:
.. autofunction:: fcc
   :noindex:
.. autofunction:: hcp
   :noindex:


1D materials
============

.. autofunction:: nanoribbon
   :noindex:
.. autofunction:: graphene_nanoribbon
   :noindex:
.. autofunction:: agnr
   :noindex:
.. autofunction:: zgnr
   :noindex:
.. autofunction:: nanotube
   :noindex:


2D materials
============

.. autofunction:: honeycomb
   :noindex:
.. autofunction:: graphene
   :noindex:
.. autofunction:: bilayer
   :noindex:


Others
======

.. autofunction:: diamond
   :noindex:

"""
from .basic import *
from .flat import *
from .nanoribbon import *
from .nanotube import *
from .special import *
from .bilayer import *
from .category import *


__all__ = [s for s in dir() if not s.startswith('_')]
