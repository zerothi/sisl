Distributions are now agnostic of the object they distribute

The distribution classes moved to `sisl.mpi`, and were renamed:
``Partition`` is now `sisl.mpi.Distribution`, ``BlockCyclicPartition`` is now
`sisl.mpi.BlockCyclicDistribution`.
A distribution holds only its communicator and the indices it distributes,
so the same distribution can be re-used on any object of the same length.
Calling it, ``dist(obj, ...)``, forwards to ``obj.__sisl_distribute__(dist, ...)``,
which lets the object decide what distributing it means.
The communicator is a keyword-only argument defaulting to
`sisl.mpi.get_comm`, so ``BlockCyclicDistribution(n)`` is enough for basic
distributions.
