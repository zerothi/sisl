.. _physics.matrix:

Physical quantites
==================

.. currentmodule:: sisl.physics

Physical quantities such as Hamiltonian and density matrices are representated
through specific classes enabling various handlings.


Spin
----

.. autosummary::
   :toctree: generated/

   Spin

Matrices
--------

.. autosummary::
   :toctree: generated/

   EnergyDensityMatrix
   DensityMatrix
   Hamiltonian
   DynamicalMatrix
   Overlap


Self energies
-------------

Self-energies are specific physical quantities that enables integrating
out semi-infinite regions.

.. autosummary::
   :toctree: generated/

   SelfEnergy
   WideBandSE
   SemiInfinite
   RecursiveSI
   RealSpaceSE
   RealSpaceSI


Bloch's theorem
---------------

Bloch's theorem is a very powerful procedure that enables one to utilize
the periodicity of a given direction to describe the complete system.

.. autosummary::
   :toctree: generated/

   Bloch
