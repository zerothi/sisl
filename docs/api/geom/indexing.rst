.. _geom-indexing:

Indexing atoms
===================

.. currentmodule:: sisl.geom

There are *many* methods in `sisl` that will do stuff based no atomic indices.
Generally one can index subsets of atoms via direct indices, but sometimes
it can be beneficial to index atoms based on other things, such as:

- species name
- odd/even indices
- specific coordinates

To accommodate more powerful indices one can use the below class constructs
to select atoms based on particular things:

.. autosummary::
   :toctree: generated/

   AtomCategory
   AtomFracSite
   AtomXYZ
   AtomZ
   AtomIndex
   AtomSeq
   AtomTag
   AtomOdd
   AtomEven
   AtomNeighbors
   NullCategory
