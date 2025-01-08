.. _geom-create:

Creating geometries
===================

.. currentmodule:: sisl.geom

A selection of default geometries that `sisl` can construct on the fly.

While this is far from complete we encourage users to contribute additional
geometries via a `pull request <pr>`_.

All methods returns a `~sisl.Geometry` object.

Some of the geometries are created in section based geometries, such as `heteroribbon`.
This functionality is provided through the `composite_geometry`.


Bulk
----

.. autosummary::
   :toctree: generated/

   sc
   bcc
   fcc
   hcp
   diamond
   rocksalt


Surfaces (slabs)
----------------

.. autosummary::
   :toctree: generated/

   bcc_slab
   fcc_slab
   rocksalt_slab

0D materials
------------

.. autosummary::
   :toctree: generated/

   honeycomb_flake
   graphene_flake
   triangulene

1D materials
------------

.. autosummary::
   :toctree: generated/

   nanoribbon
   agnr
   zgnr
   cgnr
   graphene_nanoribbon
   nanotube
   heteroribbon
   graphene_heteroribbon


2D materials
------------

.. autosummary::
   :toctree: generated/

   honeycomb
   bilayer
   graphene
   hexagonal
   goldene


Helpers
-------

.. autosummary::
   :toctree: generated/

   AtomCategory
   composite_geometry
   CompositeGeometrySection
