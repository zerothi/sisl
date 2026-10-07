# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Block distributions.

`Partition` states how indices map to ranks and nothing more, so the same
schemes serve rows, columns, k-points or energy points.  `BlockCyclicPartition`
covers the useful range: one contiguous run per rank, which preserves whatever
locality the sparsity pattern has and keeps extraction a slice; plain cyclic,
which balances load when work varies along the axis but destroys that locality;
and the block-cyclic layout in between that ScaLAPACK expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from sisl._array import array_arange
from sisl._internal import set_module

__all__ = [
    "Partition",
    "BlockCyclicPartition",
    "Distribution",
    "distribute_changes",
]


@set_module("sisl")
class Partition(ABC):
    """Maps ``n`` indices onto ``size`` ranks.

    Deliberately agnostic about what is being partitioned.  `local_rows` applies
    one to the row axis of a `SparseCSR`, but nothing here knows that: the same
    partition describes columns, grid planes, k-points or energy points.

    Subclasses supply `range` and `owner`.  `range` returns every contiguous run
    a rank owns, as ``(start, stop)`` pairs -- one run for a chunked scheme,
    several for a cyclic one -- and `indices`, `count` and `counts` all follow
    from it.

    Parameters
    ----------
    n :
        number of indices to distribute
    size :
        number of ranks to distribute across
    """

    __slots__ = ("_n", "_size")

    def __init__(self, n: int, size: int):
        if size < 1:
            raise ValueError(
                f"{self.__class__.__name__} requires size >= 1, got {size}"
            )
        if n < 0:
            raise ValueError(f"{self.__class__.__name__} requires n >= 0, got {n}")
        self._n = int(n)
        self._size = int(size)

    @classmethod
    def from_comm(cls, n: int, comm, **kwargs) -> Partition:
        """Partition `n` indices across every rank of `comm`."""
        return cls(n, comm.size, **kwargs)

    @property
    def n(self) -> int:
        """Total number of indices across all ranks."""
        return self._n

    @property
    def size(self) -> int:
        """Number of ranks this is partitioned across."""
        return self._size

    @abstractmethod
    def range(self, rank: int) -> tuple[tuple[int, int], ...]:
        """The contiguous runs owned by `rank`, as ascending ``(start, stop)`` pairs.

        A rank owning nothing gets an empty tuple.  Runs are the primitive
        rather than a flat index array because they slice: a consumer can copy
        each run directly instead of gathering scattered elements.
        """

    @abstractmethod
    def owner(self, indices) -> np.ndarray:
        """The rank owning each of `indices`."""

    def indices(self, rank: int) -> np.ndarray:
        """The indices owned by `rank`, ascending."""
        ranges = self.range(rank)
        if not ranges:
            return np.empty(0, dtype=np.int32)
        starts = np.fromiter((r[0] for r in ranges), dtype=np.int32, count=len(ranges))
        stops = np.fromiter((r[1] for r in ranges), dtype=np.int32, count=len(ranges))
        return array_arange(starts, stops, dtype=np.int32)

    def count(self, rank: int) -> int:
        """Number of indices owned by `rank`."""
        return sum(stop - start for start, stop in self.range(rank))

    @property
    def counts(self) -> np.ndarray:
        """Number of indices owned by each rank."""
        return np.fromiter(
            (self.count(r) for r in range(self._size)), dtype=np.int32, count=self._size
        )

    def _check_rank(self, rank: int) -> int:
        if not 0 <= rank < self._size:
            raise IndexError(f"rank {rank} outside partition of size {self._size}")
        return int(rank)

    def _check_indices(self, indices) -> np.ndarray:
        indices = np.asarray(indices)
        if indices.size and (indices.min() < 0 or indices.max() >= self._n):
            raise IndexError(f"index outside [0, {self._n}) for this partition")
        return indices

    def __len__(self) -> int:
        return self._size

    def __eq__(self, other) -> bool:
        if not isinstance(other, Partition):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._n == other._n
            and self._size == other._size
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} n={self._n}, size={self._size}, "
            f"counts={self.counts.tolist()}>"
        )


@set_module("sisl")
class BlockCyclicPartition(Partition):
    """Blocks of `blocksize` indices dealt round-robin, as ScaLAPACK lays out a matrix.

    Ranks are visited in order, each taking `blocksize` consecutive indices, and
    the cycle repeats until the indices run out.  A rank therefore receives one
    block, then later another, and so on -- which is why `range` returns several
    runs rather than one.

    Equivalently: index ``i`` belongs to block ``i // blocksize``, and block
    ``k`` to rank ``k % size``.

    ``blocksize=1``, the default, is plain cyclic: index ``i`` to rank
    ``i % size``.  That balances load when work varies systematically along the
    axis, at the cost of destroying locality in the sparsity pattern.  Larger
    values give the block-cyclic layout a distributed dense eigensolver expects,
    so a matrix destined for one should be laid out this way from the start
    rather than redistributed later.

    ``blocksize=0`` is shorthand for one contiguous block per rank -- the whole
    axis split into `size` pieces, which maximises locality.

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
    size :
        number of ranks to distribute across
    blocksize :
        indices per block. 1 is plain cyclic, 0 one contiguous block per rank;
        ScaLAPACK codes typically use 32 or 64.

    Examples
    --------
    >>> BlockCyclicPartition(10, 3).indices(1)
    array([1, 4, 7], dtype=int32)
    >>> BlockCyclicPartition(10, 3, 2).range(0)
    ((0, 2), (6, 8))
    >>> BlockCyclicPartition(10, 3, 0).counts
    array([4, 3, 3], dtype=int32)
    """

    __slots__ = ("_nb", "_starts")

    def __init__(self, n: int, size: int, blocksize: int = 1):
        super().__init__(n, size)
        if blocksize < 0:
            raise ValueError(
                f"{self.__class__.__name__} requires blocksize >= 0, got {blocksize}"
            )
        self._nb = int(blocksize)

        if self._nb == 0:
            base, extra = divmod(self._n, self._size)
            counts = np.full(self._size, base, dtype=np.int32)
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
        return self._nb == 0 or self._size * self._nb >= self._n

    @property
    def period(self) -> int:
        """Indices consumed by one full cycle over all ranks."""
        if self._nb == 0:
            return self._n
        return self._size * self._nb

    def range(self, rank: int) -> tuple[tuple[int, int], ...]:
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
        return ((indices // self._nb) % self._size).astype(np.int32)

    def __eq__(self, other) -> bool:
        equal = super().__eq__(other)
        if equal is NotImplemented or not equal:
            return equal
        return self._nb == other._nb

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} n={self._n}, size={self._size}, "
            f"blocksize={self._nb}, counts={self.counts.tolist()}>"
        )


@set_module("sisl")
class Distribution:
    """The distribution state carried by a distributed `SparseCSR`.

    Holds the communicator and partition, and tracks whether the matrix has
    been mutated since it was last made coherent.  The epoch is a plain
    counter: mutation bumps it, assembly records it.  Nothing here
    communicates.
    """

    __slots__ = ("_comm", "_partition", "_epoch", "_assembled")

    def __init__(self, comm, partition: Partition):
        # None means "whatever sisl.mpi hands out", resolved on first use so
        # that building a Distribution never acquires MPI by itself.
        self._comm = comm
        self._partition = partition
        self._epoch = 0
        # -1, not 0: a distribution that has never assembled cannot be coherent.
        # Starting "assembled" would let the first consumer use replicated data
        # as though it had been distributed.
        self._assembled = -1

    @property
    def comm(self):
        if self._comm is None:
            from sisl.mpi import get_comm

            self._comm = get_comm()
        return self._comm

    @property
    def partition(self) -> Partition:
        return self._partition

    @property
    def epoch(self) -> int:
        """Bumped by every structural change."""
        return self._epoch

    @property
    def is_assembled(self) -> bool:
        """Whether the matrix is coherent, i.e. usable by a consumer."""
        return self._epoch == self._assembled

    def touch(self) -> None:
        """Record a structural change. Cheap, local, and never communicates."""
        self._epoch += 1

    def mark_assembled(self) -> None:
        """Record that the matrix has been made coherent at the current epoch."""
        self._assembled = self._epoch

    def __repr__(self) -> str:
        state = "assembled" if self.is_assembled else "unassembled"
        return (
            f"<{self.__class__.__name__} {state}, epoch={self._epoch}, "
            f"{self._partition!r}>"
        )


def distribute_changes(method):
    """Mark `method` as invalidating the distribution.

    Applied to the few methods that change a structure.  On an
    undistributed matrix -- which is every matrix today -- this costs one failed
    attribute lookup on a path that is already doing array surgery.

    The marker attribute is what lets a test assert that every structural
    mutation carries it, so the set cannot silently drift.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        distribution = getattr(self, "_distribution", None)
        if distribution is not None:
            distribution.touch()
        return result

    wrapper._distribute_changes = True
    return wrapper
