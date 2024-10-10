.. _physics.electron:

Electron related functions
==========================

.. module:: sisl.physics.electron

In sisl electronic structure calculations are relying on routines
specific for electrons. For instance density of states calculations from
electronic eigenvalues and other quantities.

This module implements the necessary tools required for calculating
DOS, PDOS, band-velocities and spin moments of non-colinear calculations.
One may also plot real-space wavefunctions.

.. autosummary::
   :toctree: generated/

   DOS
   PDOS
   COP
   berry_phase
   ahc
   shc
   conductivity
   wavefunction
   spin_moment
   spin_contamination


Supporting classes
------------------

Certain classes aid in the usage of the above methods by implementing them
using automatic arguments.

For instance, the PDOS method requires the overlap matrix in non-orthogonal
basis sets at the :math:`k`-point corresponding to the eigenstates. Hence, the
argument ``S`` must be :math:`\mathbf S(\mathbf k)`. The `EigenstateElectron` class
automatically passes the correct ``S`` because it knows the states :math:`k`-point.

.. autosummary::
   :toctree: generated/

   CoefficientElectron
   StateElectron
   StateCElectron
   EigenvalueElectron
   EigenvectorElectron
   EigenstateElectron
