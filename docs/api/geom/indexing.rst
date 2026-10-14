.. _geom-indexing:

Indexing atoms
==============

.. currentmodule:: sisl.geom

Atom categories select atoms by element, position or environment, instead of by index.
Methods that take atomic indices (e.g. `~sisl.Geometry.sub`, `~sisl.Geometry.remove`,
`~sisl.Geometry.translate`) also take a category or a `dict` of category keywords.
This page is the reference. The tutorial :doc:`/tutorials/tutorial_categories`
introduces categories step by step.

>>> import sisl
>>> from sisl.geom import AtomCategory, AtomNeighbors, AtomZ
>>> ribbon = sisl.geom.zgnr(4)
>>> edge = AtomNeighbors(max=2, R=1.5)
>>> ribbon.sub(edge).na
2


Categories
----------

`AtomCategory` builds a category from keywords: ``AtomCategory(Z="C")`` equals ``AtomZ("C")``.
Several keywords combine with ``&``, and a `dict` with the same keywords works as an index.
The keywords are not case-sensitive, unless two keywords differ only by case (``Z`` and ``z``).

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Category
     - Keywords
     - Selects atoms
   * - `AtomZ`
     - ``Z``, ``species``, ``element``
     - by atomic number or symbol: ``AtomZ("C")``, ``AtomZ([6, "N"])``
   * - `AtomTag`
     - ``tag``
     - by the tag of the atom (a regular expression)
   * - `AtomIndex`
     - ``index``
     - by index: ``AtomIndex(1, 4, 5)``, ``AtomIndex(mod=2)``
   * - `AtomSeq`
     - ``seq``
     - by an index sequence: ``AtomSeq("1-3,7:")``
   * - `AtomOdd`, `AtomEven`
     - ``odd``, ``even``
     - with odd or even indices: ``AtomCategory(odd=True)``
   * - `AtomXYZ`
     - ``xyz``, ``x``, ``fx``, ``ax``, ...
     - by Cartesian (``x``), fractional (``fx``) or absolute (``ax``) coordinates,
       or within a `~sisl.shape.Shape`: ``AtomXYZ(x=(None, 3.0))``, ``AtomXYZ.fz > 0.5``
   * - `AtomFracSite`
     - ``fracsite``
     - on the lattice sites of another lattice
   * - `AtomNeighbors`
     - ``neighbors``, ``neighbours``
     - by the number of neighbors within ``R``, optionally only neighbors in another category:
       ``AtomNeighbors(3, R=1.5)``, ``AtomNeighbors(min=1, neighbor=AtomZ("N"))``

A value of ``True`` creates the category without arguments, and a `dict` gives keyword arguments:

>>> AtomCategory(Z="C", neighbors={"max": 2, "R": 1.5})
Z={6} & neighbors ∈ [0;2]
>>> ribbon.sub({"Z": "C", "neighbors": {"max": 2, "R": 1.5}}).na
2


Combining categories
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Operator
     - Selects atoms
   * - ``A & B``
     - in ``A`` and in ``B``
   * - ``A | B``
     - in ``A`` or in ``B``
   * - ``A ^ B``
     - in exactly one of ``A`` and ``B``
   * - ``~A``
     - not in ``A``

>>> ribbon.sub(~edge).na
6
>>> ribbon.sub(edge | AtomCategory(odd=True)).na
5


The result of a category
------------------------

``category.mask(geometry)`` returns a boolean array with one value per atom.
Use `numpy.flatnonzero` to get the indices:

>>> import numpy as np
>>> mask = edge.mask(ribbon)
>>> mask
array([ True, False, False,  True, False, False, False, False])
>>> np.flatnonzero(mask)
array([0, 3])

The parts of a combined category share their calculations, for example one neighbor search
for all `AtomNeighbors` in the category.


Write a category
----------------

A new category inherits from `AtomCategory`. It implements ``categorize``, which returns
the category itself for the atoms in the category, and `NullCategory` for the other atoms.
The class argument ``key`` gives the keyword for `AtomCategory`.
Without ``key``, the keyword is the class name without ``Atom``.

>>> from sisl.geom import NullCategory
>>> class AtomHeavy(AtomCategory, key="heavy"):
...     def categorize(self, geometry, atoms=None):
...         if atoms is None:
...             atoms = range(geometry.na)
...         Z = geometry.atoms.Z
...         return [self if Z[ia] > 2 else NullCategory() for ia in np.atleast_1d(atoms)]
>>> molecule = sisl.Geometry([[0, 0, 0], [1, 0, 0], [2, 0, 0]], atoms=["H", "C", "O"])
>>> molecule.sub(AtomCategory(heavy=True)).atoms.Z.tolist()
[6, 8]
>>> (AtomHeavy() & ~AtomZ("O")).mask(molecule)
array([False,  True, False])


Reference
---------

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
