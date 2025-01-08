.. _physics:

*********************************
Physical objects (`sisl.physics`)
*********************************

.. module:: sisl.physics

Implementations of various DFT and tight-binding related quantities
are defined. The implementations range from simple Brillouin zone
perspectives to self-energy calculations from Hamiltonians.

In `sisl` the general usage of physical matrices are considering sparse
matrices. Hence Hamiltonians, density matrices, etc. are considered
sparse. There are exceptions, but it is generally advisable to have this in mind.


.. toctree::
   :maxdepth: 2

   physics.brillouinzone
   physics.matrix
   physics.electron
   physics.phonon

.. toctree::
   :maxdepth: 1

   physics.distribution


Low level objects
=================

The low level objects are the driving objects for some of the physical
objects found here. They are rarely (if ever) required to be used, but they
may be important for developers wishing to extend the functionality of `sisl`
using generic class-structures. For instance the `Hamiltonian` inherits the
`SparseOrbitalBZSpin` class and `EigenvalueElectron` inherits from `Coefficient`.

States
------

.. autosummary::
   :toctree: generated/

   degenerate_decouple
   Coefficient
   State
   StateC


Sparse matrices
---------------

.. autosummary::
   :toctree: generated/

   SparseOrbitalBZ
   SparseOrbitalBZSpin
