.. _geom:

*****************
Common geometries
*****************

.. currentmodule:: sisl.geom


A selection of default geometries that `sisl` can construct on the fly.

While this is far from complete we encourage users to contribute additional
geometries via a `pull request <pr>`_.

All methods return a `Geometry` object.


Bulk
====

.. autosummary::
   :toctree: api-generated/

   sc
   bcc
   fcc
   hcp
   diamond


1D materials
============

.. autosummary::
   :toctree: api-generated/

   nanoribbon
   agnr
   zgnr
   graphene_nanoribbon
   nanotube


2D materials
============

.. autosummary::
   :toctree: api-generated/

   honeycomb
   bilayer
   graphene
