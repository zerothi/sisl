# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Distributions of indices across the ranks of a communicator.

A `Distribution` states how ``n`` indices map to the ranks of a communicator
and nothing more.  It knows nothing about what is being distributed, so the
same distribution serves rows of a matrix, k-points or energy points -- and
can be re-used on any object of the same length.

Applying it to an object is the object's business::

    dist = BlockCyclicDistribution(len(obj))
    local = dist(obj, *args, **kwargs)

forwards to ``obj.__sisl_distribute__(dist, *args, **kwargs)``.  The object
decides what distributing it means, and which arguments that takes: a
`SparseCSR` moves its rows to their owners and returns itself, while a set of
k-points could return an iterator over the local batch.

`BlockCyclicDistribution` covers the useful range of layouts: one contiguous
run per rank, which preserves locality (neighbouring k-points, energies close
to each other, the sparsity pattern of a matrix) and keeps extraction a slice;
plain cyclic, which balances load when work varies along the axis but destroys
that locality; and the block-cyclic layout in between that ScaLAPACK expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional

import numpy as np

from sisl._array import array_arange
from sisl._internal import set_module

__all__ = [
    "Distribution",
    "BlockCyclicDistribution",
    "DistributionState",
    "distribute_changes",
]


@set_module("sisl.mpi")
class Distribution(ABC):
    """Maps ``n`` indices onto the ranks of `comm`.

    Deliberately agnostic about what is being distributed.  Calling a
    distribution on an object forwards to the object's ``__sisl_distribute__``
    method, which decides what distributing it means::

        dist(obj, *args, **kwargs)
        # is
        obj.__sisl_distribute__(dist, *args, **kwargs)

    The communicator is held by the distribution.  It defaults to the one
    `sisl.mpi.get_comm` hands out, so basic distributions need no communicator,
    but any communicator may be passed -- e.g. a sub-communicator created by
    splitting the world communicator.  The number of ranks, and the default
    rank queried by the methods, follow from it.

    Subclasses supply `range` and `owner`.  `range` returns every contiguous run
    a rank owns, as ``(start, stop)`` pairs -- one run for a chunked scheme,
    several for a cyclic one -- and `indices`, `count` and `counts` all follow
    from it.

    Parameters
    ----------
    n :
        number of indices to distribute
    comm :
        communicator the indices are distributed across.  Anything with ``rank``
        and ``size`` attributes works; ``None`` means `sisl.mpi.get_comm`.
    """

    __slots__ = ("_comm", "_n")

    def __init__(self, n: int, *, comm=None):
        if comm is None:
            from sisl.mpi import get_comm

            comm = get_comm()
        if comm.size < 1:
            raise ValueError(
                f"{self.__class__.__name__} requires a communicator with size >= 1, "
                f"got {comm.size}"
            )
        if n < 0:
            raise ValueError(f"{self.__class__.__name__} requires n >= 0, got {n}")
        self._comm = comm
        self._n = int(n)

    def __call__(self, obj, *args, **kwargs):
        """Distribute `obj`, by calling ``obj.__sisl_distribute__(self, *args, **kwargs)``."""
        try:
            method = obj.__sisl_distribute__
        except AttributeError:
            raise TypeError(
                f"{self.__class__.__name__} cannot distribute an object of type "
                f"{type(obj).__name__}; it does not implement __sisl_distribute__"
            ) from None
        return method(self, *args, **kwargs)

    @property
    def comm(self):
        """The communicator the indices are distributed across."""
        return self._comm

    @property
    def n(self) -> int:
        """Total number of indices across all ranks."""
        return self._n

    @property
    def size(self) -> int:
        """Number of ranks this is distributed across."""
        return self._comm.size

    @property
    def rank(self) -> int:
        """The rank of this process in `comm`."""
        return self._comm.rank

    @abstractmethod
    def range(self, rank: Optional[int] = None) -> tuple[tuple[int, int], ...]:
        """The contiguous runs owned by `rank`, as ascending ``(start, stop)`` pairs.

        A rank owning nothing gets an empty tuple.  Runs are the primitive
        rather than a flat index array because they slice: a consumer can copy
        each run directly instead of gathering scattered elements.

        Parameters
        ----------
        rank :
            defaults to the rank of this process.
        """

    @abstractmethod
    def owner(self, indices) -> np.ndarray:
        """The rank owning each of `indices`."""

    def indices(self, rank: Optional[int] = None) -> np.ndarray:
        """The indices owned by `rank` (default this process), ascending."""
        ranges = self.range(rank)
        if not ranges:
            return np.empty(0, dtype=np.int32)
        starts = np.fromiter((r[0] for r in ranges), dtype=np.int32, count=len(ranges))
        stops = np.fromiter((r[1] for r in ranges), dtype=np.int32, count=len(ranges))
        return array_arange(starts, stops, dtype=np.int32)

    def count(self, rank: Optional[int] = None) -> int:
        """Number of indices owned by `rank` (default this process)."""
        return sum(stop - start for start, stop in self.range(rank))

    @property
    def counts(self) -> np.ndarray:
        """Number of indices owned by each rank."""
        size = self.size
        return np.fromiter(
            (self.count(r) for r in range(size)), dtype=np.int32, count=size
        )

    def _check_rank(self, rank: Optional[int]) -> int:
        if rank is None:
            return int(self.rank)
        if not 0 <= rank < self.size:
            raise IndexError(f"rank {rank} outside distribution of size {self.size}")
        return int(rank)

    def _check_indices(self, indices) -> np.ndarray:
        indices = np.asarray(indices)
        if indices.size and (indices.min() < 0 or indices.max() >= self._n):
            raise IndexError(f"index outside [0, {self._n}) for this distribution")
        return indices

    def __eq__(self, other) -> bool:
        """Equal when the same indices go to the same ranks of the same communicator."""
        if not isinstance(other, Distribution):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._n == other._n
            and self.size == other.size
            and (self._comm is other._comm or self._comm == other._comm)
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__module__}.{self.__class__.__name__} n={self._n}, "
            f"size={self.size}, counts={self.counts.tolist()}>"
        )


@set_module("sisl.mpi")
class BlockCyclicDistribution(Distribution):
    """Blocks of `blocksize` indices dealt round-robin, as ScaLAPACK lays out a matrix.

    Ranks are visited in order, each taking `blocksize` consecutive indices, and
    the cycle repeats until the indices run out.  A rank therefore receives one
    block, then later another, and so on -- which is why `range` returns several
    runs rather than one.

    Equivalently: index ``i`` belongs to block ``i // blocksize``, and block
    ``k`` to rank ``k % size``.

    ``blocksize=1``, the default, is plain cyclic: index ``i`` to rank
    ``i % size``.  That balances load when work varies systematically along the
    axis, at the cost of destroying locality.  Larger values give the
    block-cyclic layout a distributed dense eigensolver expects, so a matrix
    destined for one should be laid out this way from the start rather than
    redistributed later.

    ``blocksize=0`` is shorthand for one contiguous block per rank -- the whole
    axis split into `size` pieces, which maximises locality, e.g. k-points or
    energies that are close to each other end up on the same rank.

    Contiguous is a special case rather than a large `blocksize` because no
    uniform block size reproduces it.  With ``n=10, size=3`` a contiguous split
    is ``[4, 3, 3]``, while ``blocksize=4`` gives ``[4, 4, 2]``; worse, at
    ``n=5, size=4`` a uniform ``blocksize=2`` yields ``[2, 2, 1, 0]`` and leaves
    a rank idle, where the contiguous split gives ``[2, 1, 1, 1]``.  The
    remainder is therefore spread one index per rank across the leading ranks,
    so the largest block exceeds the smallest by at most one.

    For any other ``blocksize`` the load is only even when the blocks divide the
    indices evenly.  That is inherent -- the block structure is the point, and
    evening out the tail would break it.

    Parameters
    ----------
    n :
        number of indices to distribute
    blocksize :
        indices per block. 1 is plain cyclic, 0 one contiguous block per rank;
        ScaLAPACK codes typically use 32 or 64.
    comm :
        communicator the indices are distributed across, ``None`` means
        `sisl.mpi.get_comm`.

    Examples
    --------
    With ``comm`` having 3 ranks:

    >>> BlockCyclicDistribution(10, comm=comm).indices(1)
    array([1, 4, 7], dtype=int32)
    >>> BlockCyclicDistribution(10, 2, comm=comm).range(0)
    ((0, 2), (6, 8))
    >>> BlockCyclicDistribution(10, 0, comm=comm).counts
    array([4, 3, 3], dtype=int32)
    """

    __slots__ = ("_nb", "_starts")

    def __init__(self, n: int, blocksize: int = 1, *, comm=None):
        super().__init__(n, comm=comm)
        if blocksize < 0:
            raise ValueError(
                f"{self.__class__.__name__} requires blocksize >= 0, got {blocksize}"
            )
        self._nb = int(blocksize)

        if self._nb == 0:
            size = self.size
            base, extra = divmod(self._n, size)
            counts = np.full(size, base, dtype=np.int32)
            counts[:extra] += 1
            # size + 1 entries; the last is n, so consecutive pairs give a run.
            self._starts = np.insert(np.cumsum(counts), 0, 0).astype(np.int32)
            self._starts.flags.writeable = False
        else:
            self._starts = None

    @property
    def blocksize(self) -> int:
        """Indices per block; 0 meaning one contiguous block per rank."""
        return self._nb

    @property
    def contiguous(self) -> bool:
        """Whether each rank holds a single unbroken run."""
        return self._nb == 0 or self.size * self._nb >= self._n

    @property
    def period(self) -> int:
        """Indices consumed by one full cycle over all ranks."""
        if self._nb == 0:
            return self._n
        return self.size * self._nb

    def range(self, rank: Optional[int] = None) -> tuple[tuple[int, int], ...]:
        rank = self._check_rank(rank)

        if self._nb == 0:
            start, stop = int(self._starts[rank]), int(self._starts[rank + 1])
            return ((start, stop),) if stop > start else ()

        period = self.period
        runs = []
        start = rank * self._nb
        while start < self._n:
            runs.append((start, min(start + self._nb, self._n)))
            start += period
        return tuple(runs)

    def owner(self, indices) -> np.ndarray:
        """Which block an index falls in decides the rank."""
        indices = self._check_indices(indices)
        if self._nb == 0:
            # O(k log size) rather than a table of length n -- and n being large
            # is the reason to distribute in the first place.
            return np.searchsorted(self._starts[1:], indices, side="right").astype(
                np.int32
            )
        return ((indices // self._nb) % self.size).astype(np.int32)

    def __eq__(self, other) -> bool:
        equal = super().__eq__(other)
        if equal is NotImplemented or not equal:
            return equal
        return self._nb == other._nb

    def __repr__(self) -> str:
        return (
            f"<{self.__module__}.{self.__class__.__name__} n={self._n}, "
            f"size={self.size}, blocksize={self._nb}, counts={self.counts.tolist()}>"
        )


@set_module("sisl.mpi")
class DistributionState:
    """Distribution state carried by a distributed object.

    Holds the `Distribution` the object is laid out by, and tracks whether the
    object has been mutated since it was last made coherent.  The epoch is a
    plain counter: mutation bumps it, assembly records it.  Nothing here
    communicates.

    This lives on the object, not on the `Distribution`, so that a single
    distribution can be shared by many objects.
    """

    __slots__ = ("_distribution", "_epoch", "_assembled")

    def __init__(self, distribution: Distribution):
        self._distribution = distribution
        self._epoch = 0
        # -1, not 0: a state that has never assembled cannot be coherent.
        # Starting "assembled" would let the first consumer use replicated data
        # as though it had been distributed.
        self._assembled = -1

    @property
    def distribution(self) -> Distribution:
        return self._distribution

    @property
    def epoch(self) -> int:
        """Bumped by every structural change."""
        return self._epoch

    @property
    def is_assembled(self) -> bool:
        """Whether the object is coherent, i.e. usable by a consumer."""
        return self._epoch == self._assembled

    def touch(self) -> None:
        """Record a structural change. Cheap, local, and never communicates."""
        self._epoch += 1

    def mark_assembled(self) -> None:
        """Record that the object has been made coherent at the current epoch."""
        self._assembled = self._epoch

    def __repr__(self) -> str:
        state = "assembled" if self.is_assembled else "unassembled"
        return (
            f"<{self.__module__}.{self.__class__.__name__} {state}, "
            f"epoch={self._epoch}, {self._distribution!r}>"
        )


def distribute_changes(method):
    """Mark `method` as invalidating the distribution of its object.

    Applied to the few methods that change a structure.  The object keeps its
    `DistributionState` in ``_dist_state``; on an undistributed object this
    costs one failed attribute lookup on a path that is already doing array
    surgery.

    The marker attribute is what lets a test assert that every structural
    mutation carries it, so the set cannot silently drift.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        state = getattr(self, "_dist_state", None)
        if state is not None:
            state.touch()
        return result

    wrapper._distribute_changes = True
    return wrapper
