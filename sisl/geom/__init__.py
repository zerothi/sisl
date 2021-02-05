"""
Common geometries
=================

Bulk
====

   sc
   bcc
   fcc
   hcp
   diamond


1D materials
============

   nanoribbon
   graphene_nanoribbon
   agnr
   zgnr
   nanotube


2D materials
============

   honeycomb
   bilayer
   graphene

"""
from .basic import *
from .flat import *
from .nanoribbon import *
from .nanotube import *
from .special import *
from .bilayer import *
from .category import *


__all__ = [s for s in dir() if not s.startswith('_')]
