# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.mpi import (
    BlockCyclicDistribution,
    Distribution,
    DistributionState,
    distribute_changes,
)

pytestmark = [pytest.mark.mpi]


class FakeComm:
    """Stand-in communicator: a distribution only needs ``rank`` and ``size``."""

    def __init__(self, size=1, rank=0):
        self.size = size
        self.rank = rank


def BCD(n, size, blocksize=1):
    """A block-cyclic distribution of `n` over a fake communicator of `size`."""
    return BlockCyclicDistribution(n, blocksize, comm=FakeComm(size))


# --- distribution: the contract every scheme must satisfy --------------------


def _contiguous(n, size):
    """blocksize 0: one unbroken run per rank."""
    return BCD(n, size, 0)


def _cyclic(n, size):
    return BCD(n, size)


def _cyclic2(n, size):
    return BCD(n, size, 2)


def _cyclic3(n, size):
    return BCD(n, size, 3)


#: Every scheme, for the invariants that hold universally.
SCHEMES = [_contiguous, _cyclic, _cyclic2, _cyclic3]

#: Schemes that promise even load. A blocksize above 1 deliberately does not:
#: the block structure is the point, and evening out the tail would break it.
BALANCED = [_contiguous, _cyclic]


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("n,size", [(9, 3), (10, 3), (97, 7), (2, 5), (0, 3), (5, 1)])
def test_partition_covers_every_index_exactly_once(make, n, size):
    p = make(n, size)
    covered = np.concatenate([p.indices(r) for r in range(p.size)])
    assert np.array_equal(np.sort(covered), np.arange(n))


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("n,size", [(9, 3), (10, 3), (97, 7), (2, 5)])
def test_partition_owner_agrees_with_indices(make, n, size):
    p = make(n, size)
    owners = p.owner(np.arange(n))
    for rank in range(size):
        assert np.all(owners[p.indices(rank)] == rank)


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("n,size", [(9, 3), (10, 3), (97, 7), (2, 5)])
def test_partition_counts_sum_to_n(make, n, size):
    assert make(n, size).counts.sum() == n


@pytest.mark.parametrize("make", BALANCED)
@pytest.mark.parametrize("n,size", [(9, 3), (10, 3), (97, 7), (2, 5)])
def test_partition_is_balanced(make, n, size):
    """No rank may carry more than one extra index over the lightest."""
    counts = make(n, size).counts
    assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("n,size", [(23, 4), (10, 3), (2, 5), (0, 3)])
def test_partition_ranges_are_ascending_disjoint_and_complete(make, n, size):
    """Runs must be usable as slices: ordered, non-overlapping, and exhaustive."""
    p = make(n, size)
    for rank in range(size):
        ranges = p.range(rank)
        for start, stop in ranges:
            assert 0 <= start < stop <= n, (start, stop)
        for (_, prev_stop), (next_start, _) in zip(ranges, ranges[1:]):
            assert prev_stop < next_start, "runs must be ascending and disjoint"
        flat = np.concatenate([np.arange(a, b) for a, b in ranges]) if ranges else []
        assert np.array_equal(flat, p.indices(rank))


@pytest.mark.parametrize("make", SCHEMES)
def test_partition_count_matches_indices(make):
    p = make(23, 4)
    for rank in range(4):
        assert p.count(rank) == len(p.indices(rank))


def test_partition_rejects_nonsense():
    with pytest.raises(ValueError):
        BCD(10, 0)
    with pytest.raises(ValueError):
        BCD(-1, 2)
    with pytest.raises(IndexError):
        BCD(10, 3).indices(3)
    with pytest.raises(IndexError):
        BCD(10, 3).owner([10])


def test_partition_equality_ignores_identity():
    comm = FakeComm(3)
    assert BlockCyclicDistribution(10, comm=comm) == BlockCyclicDistribution(
        10, comm=comm
    )
    assert BlockCyclicDistribution(10, comm=comm) != BCD(10, 4)
    assert BlockCyclicDistribution(10, comm=comm) != BlockCyclicDistribution(
        11, comm=comm
    )


def test_partition_is_abstract():
    with pytest.raises(TypeError):
        Distribution(10, comm=FakeComm(3))


# --- distribution: scheme-specific behaviour ---------------------------------


def test_blocksize_zero_gives_one_balanced_run_per_rank():
    p = BCD(10, 3, 0)
    # not [3, 3, 4] -- the largest run exceeds the smallest by at most one
    assert p.counts.tolist() == [4, 3, 3]
    assert p.range(1) == ((4, 7),)
    assert p.indices(1).tolist() == [4, 5, 6]
    assert p.contiguous
    assert BCD(11, 3, 0).counts.tolist() == [4, 4, 3]


def test_blocksize_zero_is_not_a_large_uniform_blocksize():
    """Why contiguous is its own mode rather than blocksize=ceil(n/size).

    A uniform block can leave a rank idle while a balanced split does not.
    """
    assert BCD(5, 4, 0).counts.tolist() == [2, 1, 1, 1]
    assert BCD(5, 4, 2).counts.tolist() == [2, 2, 1, 0]

    assert BCD(10, 3, 0).counts.tolist() == [4, 3, 3]
    assert BCD(10, 3, 4).counts.tolist() == [4, 4, 2]


def test_blocksize_zero_handles_more_ranks_than_indices():
    p = BCD(2, 5, 0)
    assert p.counts.tolist() == [1, 1, 0, 0, 0]
    assert p.range(4) == ()
    assert p.indices(4).tolist() == []


def test_contiguous_reports_whether_runs_are_unbroken():
    assert BCD(10, 3, 0).contiguous
    assert not BCD(10, 3, 1).contiguous
    assert not BCD(10, 3, 2).contiguous
    # a block large enough that nobody comes round twice
    assert BCD(10, 3, 4).contiguous


def test_cyclic_defaults_to_plain_round_robin():
    """blocksize 1 is the default, so the plain reading of the name holds."""
    p = BCD(10, 3)
    assert p.blocksize == 1
    assert p.indices(1).tolist() == [1, 4, 7]
    assert p.owner(np.arange(10)).tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


def test_cyclic_handles_more_ranks_than_indices():
    p = BCD(2, 5)
    assert p.counts.tolist() == [1, 1, 0, 0, 0]
    assert p.range(4) == ()
    assert p.indices(4).tolist() == []


def test_cyclic_returns_one_run_per_visit():
    """A rank gets a block, then later another; range must show both."""
    p = BCD(10, 3, 2)
    assert p.range(0) == ((0, 2), (6, 8))
    assert p.range(1) == ((2, 4), (8, 10))
    assert p.range(2) == ((4, 6),)
    assert p.owner(np.arange(10)).tolist() == [0, 0, 1, 1, 2, 2, 0, 0, 1, 1]


def test_cyclic_with_one_block_per_rank_is_contiguous():
    """The scheme spans from cyclic to contiguous as blocksize grows."""
    p = BCD(9, 3, 3)
    for rank in range(3):
        assert len(p.range(rank)) == 1
        # n divides evenly here, so it coincides with the contiguous split
        assert np.array_equal(p.indices(rank), BCD(9, 3, 0).indices(rank))


def test_cyclic_short_final_block():
    """n not a multiple of the cycle leaves the last run short."""
    p = BCD(7, 2, 3)
    assert p.period == 6
    assert p.range(0) == ((0, 3), (6, 7))
    assert p.range(1) == ((3, 6),)
    assert p.counts.sum() == 7


def test_cyclic_blocksize_is_part_of_identity():
    comm = FakeComm(3)
    assert BlockCyclicDistribution(10, 2, comm=comm) != BlockCyclicDistribution(
        10, 4, comm=comm
    )
    assert BlockCyclicDistribution(10, 2, comm=comm) == BlockCyclicDistribution(
        10, 2, comm=comm
    )
    assert BlockCyclicDistribution(10, comm=comm) == BlockCyclicDistribution(
        10, 1, comm=comm
    )


def test_cyclic_rejects_bad_blocksize():
    with pytest.raises(ValueError):
        BCD(10, 3, -1)


# --- distribution state ---------------------------------------------------


def test_distribution_starts_unassembled():
    """It has never communicated, so it cannot claim to be coherent."""
    d = DistributionState(BCD(10, 1, 0))
    assert not d.is_assembled
    assert d.epoch == 0


def test_touch_invalidates_and_assembly_restores():
    d = DistributionState(BCD(10, 1, 0))
    d.touch()
    assert not d.is_assembled
    d.mark_assembled()
    assert d.is_assembled


# --- the protocol ----------------------------------------------------------


class _Recorder:
    def __sisl_distribute__(self, dist, *args, **kwargs):
        return dist, args, kwargs


def test_call_forwards_to_the_object():
    """The object, not the distribution, decides the arguments."""
    dist = BCD(10, 3)
    assert _Recorder().__sisl_distribute__(dist) == (dist, (), {})
    assert dist(_Recorder(), 1, op="sum") == (dist, (1,), {"op": "sum"})


def test_call_iterates_when_the_object_returns_an_iterator():
    class Points:
        def __init__(self, n):
            self.points = np.arange(n) * 0.1

        def __sisl_distribute__(self, dist, batch=1):
            idx = dist.indices()
            for i in range(0, len(idx), batch):
                yield self.points[idx[i : i + batch]]

    dist = BlockCyclicDistribution(10, 0, comm=FakeComm(3, rank=1))
    batches = list(dist(Points(10), batch=2))
    assert np.allclose(np.concatenate(batches), np.arange(4, 7) * 0.1)
    assert [len(b) for b in batches] == [2, 1]


def test_call_refuses_an_object_without_the_protocol():
    with pytest.raises(TypeError, match="__sisl_distribute__"):
        BCD(3, 2)(np.arange(3))


def test_rank_defaults_to_the_communicator_rank():
    dist = BlockCyclicDistribution(10, comm=FakeComm(3, rank=2))
    assert dist.rank == 2
    assert dist.indices().tolist() == dist.indices(2).tolist() == [2, 5, 8]
    assert dist.count() == 3
    assert dist.range() == dist.range(2)


def test_distribution_holds_the_communicator():
    """Any communicator, e.g. a split one, can be used in place of the world."""
    comm = FakeComm(4)
    dist = BlockCyclicDistribution(10, comm=comm)
    assert dist.comm is comm
    assert dist.size == 4


def test_distribution_equality_includes_the_communicator():
    """Same layout on different communicators sends indices to different processes."""
    comm = FakeComm(3)
    assert BlockCyclicDistribution(10, comm=comm) == BlockCyclicDistribution(
        10, comm=comm
    )
    assert BlockCyclicDistribution(10, comm=comm) != BlockCyclicDistribution(
        10, comm=FakeComm(3)
    )


def test_distribution_is_reusable_across_objects():
    """It holds only indices, so one distribution serves any object of that length."""
    dist = BCD(6, 2, 0)
    a, b = _Recorder(), _Recorder()
    assert dist(a)[0] is dist(b)[0] is dist


def test_distribute_changes_touches_the_state():
    class Obj:
        _dist_state = None

        @distribute_changes
        def mutate(self):
            return 1

    o = Obj()
    assert o.mutate() == 1  # no state: nothing happens
    o._dist_state = DistributionState(BCD(4, 1))
    o._dist_state.mark_assembled()
    o.mutate()
    assert not o._dist_state.is_assembled
