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


Bulk (3D)
---------

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


Ancillary methods
-----------------

.. autosummary::
   :toctree: generated/

   composite_geometry
   CompositeGeometrySection


Saturation
----------

`saturate` adds atoms to the missing bonds of selected atoms.
The selection uses the categories in :ref:`geom-indexing`.
The first rule that selects an atom is used:

>>> import sisl
>>> from sisl.geom import AtomNeighbors, AtomZ, Saturate, saturate
>>> ribbon = sisl.geom.zgnr(4)
>>> edge = AtomNeighbors(max=2, R=1.5)
>>> saturated = saturate(ribbon, [
...     (edge & AtomZ("N"), Saturate("H", bond=1.01, R=1.5)),
...     (edge, Saturate("H", bond=1.09, R=1.5)),
... ])
>>> saturated.na - ribbon.na
2

.. autosummary::
   :toctree: generated/

   saturate
   Saturate
