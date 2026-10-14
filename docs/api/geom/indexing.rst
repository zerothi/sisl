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
to select atoms based on particular things.
`AtomCategory` builds them from keywords, and a `dict` works as an index directly:

>>> import sisl
>>> from sisl.geom import AtomCategory, AtomZ
>>> AtomZ("C") == AtomCategory(species="C")
True
>>> ribbon = sisl.geom.zgnr(4)
>>> edge = AtomCategory(Z="C", neighbors=2)  # keywords combine with &
>>> ribbon.sub(edge).na
2
>>> ribbon.sub({"Z": "C", "neighbors": 2}).na
2
>>> ribbon.sub(AtomCategory(odd=True) | AtomZ("N")).na
4


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
