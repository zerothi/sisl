.. _mpi:

MPI (`sisl.mpi`)
================

.. module:: sisl.mpi

Optional MPI support. All functionality works without `mpi4py`, in which case
the communicator is a serial stand-in.

See :ref:`distribution` for a guide on how to use it.


Communicators
-------------

.. autosummary::
   :toctree: generated/

   get_comm
   BaseCommunicator
   Communicator


Distributions
-------------

Distributions decide which rank owns which indices. Calling one on an
object forwards to the object's ``__sisl_distribute__`` method.

.. autosummary::
   :toctree: generated/

   Distribution
   BlockCyclicDistribution


Implementing distributable objects
----------------------------------

Helpers for objects that implement ``__sisl_distribute__`` and need to track
whether they have changed since they were last distributed.

.. autosummary::
   :toctree: generated/

   DistributionState
   distribute_changes
