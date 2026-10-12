.. _distribution:

Distributing work with MPI
==========================

.. currentmodule:: sisl.mpi

`sisl` can split work across the processes of an MPI job. The functionality
lives in `sisl.mpi` and requires `mpi4py <https://mpi4py.readthedocs.io>`_.
Without `mpi4py` (or with ``SISL_MPI=0``, see :ref:`environment`) every
communicator is a serial stand-in: rank 0 of 1 processes, where all operations
are no-ops. Code written for MPI therefore also runs serially, unchanged.

Scripts are run in parallel through the MPI launcher, e.g.:

.. code-block:: bash

   mpirun -n 4 python script.py


The communicator
----------------

`get_comm` returns the communicator `sisl` uses. It is created on the first
call, and is the same object on every subsequent call:

.. code-block:: python

   from sisl.mpi import get_comm

   comm = get_comm()
   print(comm.rank, comm.size)
   if comm.on_rank0():
       print("only printed once")

All `mpi4py` operations are available directly on it (``comm.Allreduce``,
``comm.Split``, ...), with `mpi4py`'s own signatures.


Distributions
-------------

A `Distribution` decides which process (rank) owns which of ``n`` indices.
It knows nothing about *what* the indices refer to; they may be rows of a
matrix, k-points, energy points or anything else that can be counted.
The same distribution can therefore be re-used on any object of the same
length.

`BlockCyclicDistribution` deals out blocks of ``blocksize`` consecutive
indices to the ranks in turn:

.. code-block:: python

   from sisl.mpi import BlockCyclicDistribution

   # 10 indices on 2 ranks
   BlockCyclicDistribution(10)       # blocksize=1: rank 0 [0 2 4 6 8], rank 1 [1 3 5 7 9]
   BlockCyclicDistribution(10, 0)    # contiguous:  rank 0 [0 1 2 3 4], rank 1 [5 6 7 8 9]
   BlockCyclicDistribution(10, 2)    # blocks of 2: rank 0 [0 1 4 5 8 9], rank 1 [2 3 6 7]

Which layout to choose depends on the work:

``blocksize=0``
   one contiguous block per rank, balanced to within one index.
   Keeps neighbouring indices together, which is useful for k-points or
   energies that are close to each other, and for the sparsity pattern of
   a matrix.

``blocksize=1`` (default)
   plain round-robin. Balances the load when the cost varies systematically
   along the indices, at the cost of locality.

``blocksize>1``
   the block-cyclic layout ScaLAPACK expects (typically 32 or 64).

A distribution answers questions about the layout. Without a ``rank``
argument, the rank of the current process is used:

.. code-block:: python

   dist = BlockCyclicDistribution(10, 2)
   dist.indices()       # indices owned by this process
   dist.range()         # the same, as contiguous (start, stop) runs
   dist.count()         # how many indices this process owns
   dist.counts          # how many indices every rank owns
   dist.owner([0, 3])   # which rank owns indices 0 and 3
   dist.n, dist.size    # number of indices, and number of ranks


Using another communicator
^^^^^^^^^^^^^^^^^^^^^^^^^^

The communicator defaults to `get_comm`, but any communicator can be passed
through the keyword-only ``comm`` argument. This allows distributing over a
subset of the processes, e.g. to distribute energy points within groups of
ranks that each handle a different k-point:

.. code-block:: python

   comm = get_comm()
   group = comm.Split(comm.rank % 2)
   dist = BlockCyclicDistribution(len(E), 0, comm=group)

Two distributions are only equal if they have the same layout *and* the same
communicator.


Distributing an object
----------------------

Calling a distribution with an object hands the object over to the object's
own ``__sisl_distribute__`` method:

.. code-block:: python

   dist(obj, *args, **kwargs)
   # is the same as
   obj.__sisl_distribute__(dist, *args, **kwargs)

The object, not the distribution, decides what distributing it means and
which arguments it accepts. An object that does not implement
``__sisl_distribute__`` raises a :class:`TypeError`.

Implementing the protocol
^^^^^^^^^^^^^^^^^^^^^^^^^

Any class can be made distributable. A typical pattern is to return a
generator over the local indices in batches:

.. code-block:: python

   import numpy as np
   from mpi4py import MPI
   from sisl.mpi import BlockCyclicDistribution, get_comm

   class Energies:
       def __init__(self, E):
           self.E = np.asarray(E)

       def __sisl_distribute__(self, dist, batch=1):
           idx = dist.indices()
           for i in range(0, len(idx), batch):
               yield idx[i : i + batch], self.E[idx[i : i + batch]]

   E = Energies(np.linspace(-1, 1, 101))
   dist = BlockCyclicDistribution(len(E.E), 0)

   dos = np.zeros(len(E.E))
   for idx, e in dist(E, batch=4):
       dos[idx] = ...  # calculate for the local energies only

   # every rank only filled its own entries, sum them up
   get_comm().Allreduce(MPI.IN_PLACE, dos)


Sparse matrices
---------------

`sisl.SparseCSR` implements the protocol by distributing its *rows*.
``dist(csr)`` moves every stored element to the rank owning its row, in
place, and returns the matrix. The matrix keeps its full shape; afterwards
each rank only stores the rows it owns.

.. code-block:: python

   dist = BlockCyclicDistribution(csr.shape[0], 0)
   dist(csr)

The ``op`` argument decides what happens when several ranks hold the same
element:

``op="single"`` (default)
   keep one of them. Use this when the ranks hold the *same* data, e.g. a
   matrix that was read or built on every rank.

``op="sum"``
   sum them. Use this when each rank holds a *part* of the matrix, e.g. when
   the contributions to the elements were computed in parallel.

.. warning::

   Distributing a matrix that is replicated on every rank with ``op="sum"``
   multiplies its values by the number of ranks.

Changing a distributed matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Changes to the sparsity pattern do *not* communicate. Any rank may write to
any row, also rows it does not own; the matrix is only marked as changed.
Calling the distribution again ships the changes to their owners, once, no
matter how many changes were made:

.. code-block:: python

   for i, j, v in contributions:
       csr[i, j] = v       # local, no communication
   dist(csr, op="sum")     # one collective exchange

Calling the distribution on an unchanged matrix does nothing.
Calling a *different* distribution moves every element to its new owner,
so a matrix can be re-distributed into another layout, or onto another
communicator.

.. note::

   Distributing is a collective operation: every rank of the communicator
   must call it, otherwise the program hangs.
