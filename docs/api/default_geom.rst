.. _geom:

*****************
Common geometries
*****************

.. module:: sisl.geom


A selection of default geometries that `sisl` can construct on the fly.

While this is far from complete we encourage users to contribute additional
geometries via a `pull request <pr>`_.

All methods return a `Geometry` object.


Bulk
====

.. autosummary::
   :toctree: generated/

   sc
   bcc
   bcc_slab
   fcc
   fcc_slab
   rocksalt_slab
   hcp
   diamond


1D materials
============

.. autosummary::
   :toctree: generated/

   nanoribbon
   agnr
   zgnr
   graphene_nanoribbon
   nanotube


2D materials
============

.. autosummary::
   :toctree: generated/

   honeycomb
   bilayer
   graphene
