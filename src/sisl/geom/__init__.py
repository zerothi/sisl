# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
sisl provides a default set of geometries in various forms.
They return a `Geometry` object.

Bulk
----

  sc
  bcc
  fcc
  hcp
  diamond
  rocksalt


Surfaces
--------

  fcc_slab
  bcc_slab
  rocksalt_slab


0D materials
------------

  graphene_flake
  honeycomb_flake
  triangulene


1D materials
------------

  nanotube
  nanoribbon
  agnr
  zgnr
  cgnr
  heteroribbon
  graphene_nanoribbon
  graphene_heteroribbon


2D materials
------------

  honeycomb
  graphene
  bilayer
  hexagonal
  goldene


They generally take a lattice-parameter argument, and will all allow
to pass an ``atoms`` argument to prefil the atomic species for the
returned geometry:

  >>> graphene = si.geom.graphene(1.42, # bond-length
  ...               atoms=si.Atom('C',
  ...                       orbitals=si.AtomicOrbital("pz"))
  ... )
"""
# TODO add category and neighbors in the above discussion
from ._category import *
from ._composite import *
from ._neighbors import *

# isort: split
from .basic import *
from .bilayer import *
from .flat import *
from .nanoribbon import *
from .nanotube import *
from .special import *
from .surfaces import *
