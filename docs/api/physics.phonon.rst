.. _physics.phonon:

Phonon related functions
==========================

.. currentmodule:: sisl.physics.phonon

In sisl phonon calculations are relying on routines
specific for phonons. For instance density of states calculations from
phonon eigenvalues and other quantities.

This module implements the necessary tools required for calculating
DOS, PDOS, group-velocities and real-space displacements.

.. autosummary::
   :toctree: generated/

   DOS
   PDOS
   velocity
   displacement


Supporting classes
------------------

Certain classes aid in the usage of the above methods by implementing them
using automatic arguments.

.. autosummary::
   :toctree: generated/

   CoefficientPhonon
   ModePhonon
   ModeCPhonon
   EigenvaluePhonon
   EigenvectorPhonon
   EigenmodePhonon