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
   honeycomb
   nanotube
   diamond
   bilayer


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


2D materials
============

.. autofunction:: honeycomb
   :noindex:
.. autofunction:: graphene
   :noindex:
.. autofunction:: bilayer
   :noindex:


Nanotube
========

.. autofunction:: nanotube
   :noindex:


Others
======

.. autofunction:: diamond
   :noindex:

"""
from .basic import *
from .flat import *
from .nanotube import *
from .special import *
from .bilayer import *

__all__ = [s for s in dir() if not s.startswith('_')]
