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

.. autosummary::
   :toctree: generated/

   oplist
   Quaternion
   SparseCSR
   SparseAtom
   SparseOrbital



