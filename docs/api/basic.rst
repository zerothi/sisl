.. _basic:

*************
Basic classes
*************

.. currentmodule:: sisl

sisl provides basic functionality for interacting with orbitals, atoms, geometries,
unit cells and grid functions.



Generic objects
===============

.. index:: basic, geometry, lattice, supercell, atom

.. autosummary::
   :toctree: generated/

   PeriodicTable
   Atom
   Atoms
   Geometry
   Lattice
   BoundaryCondition
   Grid


.. _basic-orbitals:

Orbitals
========

.. index:: orbital, hydrogenic-orbital, atomic-orbital, spherical-orbital

Each of the following orbitals are specialized for various use cases.


.. autosummary::
   :toctree: generated/

   Orbital
   SphericalOrbital
   AtomicOrbital
   HydrogenicOrbital
   GTOrbital
   STOrbital


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

   oplist
   ~utils.PropertyDict
