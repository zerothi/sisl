.. _basic:

*************
Basic classes
*************

.. currentmodule:: sisl

sisl provides basic functionality for interacting with orbitals, atoms, geometries,
unit cells and grid functions.



Simple objects
==============

.. index:: basic, geometry, supercell, atom, atom, orbital

.. autosummary::
   :toctree: generated/

   PeriodicTable
   Orbital
   SphericalOrbital
   AtomicOrbital
   HydrogenicOrbital
   GTOrbital
   STOrbital
   Atom
   Atoms
   Geometry
   SuperCell
   Grid


Advanced classes
================

The physical matrices used internally in `sisl` are constructed
based on these base classes.
However, it may be beneficial to read the specific matrix
in :ref:`physics.matrix`.

.. index:: matrix, sparse-matrix

.. autosummary::
   :toctree: generated/

   Quaternion
   SparseCSR
   SparseAtom
   SparseOrbital



Utility classes
===============

A set of classes are utility classes that are used throughout the `sisl` code
and using them will be encouraged in combination with sisl.

In particular `oplist` is useful when calculating averages in Brillouin zones (see
:ref:`physics.brillouinzone`).

.. autosummary::
   :toctree: generated/

   ~sisl.oplist.oplist
   ~sisl.utils.PropertyDict

