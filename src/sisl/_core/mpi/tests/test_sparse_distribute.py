# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from sisl._core.mpi.distribute import (
    BlockCyclicPartition,
    Distribution,
    Partition,
    distribute_changes,
)
from sisl._core.mpi.sparse_distribute import (
    distribute,
    local_rows,
)
from sisl._core.sparse import SparseCSR

pytestmark = [pytest.mark.sparse, pytest.mark.mpi]


def build(nr=10, nc=14, dim=2, seed=3, finalize=False):
    rng = np.random.default_rng(seed)
    csr = SparseCSR((nr, nc, dim), dtype=np.float64)
    for r in range(nr):
        for c in np.sort(rng.choice(nc, size=rng.integers(1, 5), replace=False)):
            csr[r, c] = rng.random(dim)
    if finalize:
        csr.finalize()
    return csr


# --- partition: the contract every scheme must satisfy --------------------


def _contiguous(n, size):
    """blocksize 0: one unbroken run per rank."""
    return BlockCyclicPartition(n, size, 0)


def _cyclic(n, size):
    return BlockCyclicPartition(n, size)


def _cyclic2(n, size):
    return BlockCyclicPartition(n, size, 2)


def _cyclic3(n, size):
    return BlockCyclicPartition(n, size, 3)


#: Every scheme, for the invariants that hold universally.
SCHEMES = [_contiguous, _cyclic, _cyclic2, _cyclic3]

#: Schemes that promise even load. A blocksize above 1 deliberately does not:
#: the block structure is the point, and evening out the tail would break it.
BALANCED = [_contiguous, _cyclic]


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("n,size", [(9, 3), (10, 3), (97, 7), (2, 5), (0, 3), (5, 1)])
def test_partition_covers_every_index_exactly_once(make, n, size):
    p = make(n, size)
    covered = np.concatenate([p.indices(r) for r in range(len(p))])
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
        BlockCyclicPartition(10, 0)
    with pytest.raises(ValueError):
        BlockCyclicPartition(-1, 2)
    with pytest.raises(IndexError):
        BlockCyclicPartition(10, 3).indices(3)
    with pytest.raises(IndexError):
        BlockCyclicPartition(10, 3).owner([10])


def test_partition_equality_ignores_identity():
    assert BlockCyclicPartition(10, 3) == BlockCyclicPartition(10, 3)
    assert BlockCyclicPartition(10, 3) != BlockCyclicPartition(10, 4)
    assert BlockCyclicPartition(10, 3) != BlockCyclicPartition(11, 3)


def test_partition_is_abstract():
    with pytest.raises(TypeError):
        Partition(10, 3)


# --- partition: scheme-specific behaviour ---------------------------------


def test_blocksize_zero_gives_one_balanced_run_per_rank():
    p = BlockCyclicPartition(10, 3, 0)
    # not [3, 3, 4] -- the largest run exceeds the smallest by at most one
    assert p.counts.tolist() == [4, 3, 3]
    assert p.range(1) == ((4, 7),)
    assert p.indices(1).tolist() == [4, 5, 6]
    assert p.contiguous
    assert BlockCyclicPartition(11, 3, 0).counts.tolist() == [4, 4, 3]


def test_blocksize_zero_is_not_a_large_uniform_blocksize():
    """Why contiguous is its own mode rather than blocksize=ceil(n/size).

    A uniform block can leave a rank idle while a balanced split does not.
    """
    assert BlockCyclicPartition(5, 4, 0).counts.tolist() == [2, 1, 1, 1]
    assert BlockCyclicPartition(5, 4, 2).counts.tolist() == [2, 2, 1, 0]

    assert BlockCyclicPartition(10, 3, 0).counts.tolist() == [4, 3, 3]
    assert BlockCyclicPartition(10, 3, 4).counts.tolist() == [4, 4, 2]


def test_blocksize_zero_handles_more_ranks_than_indices():
    p = BlockCyclicPartition(2, 5, 0)
    assert p.counts.tolist() == [1, 1, 0, 0, 0]
    assert p.range(4) == ()
    assert p.indices(4).tolist() == []


def test_contiguous_reports_whether_runs_are_unbroken():
    assert BlockCyclicPartition(10, 3, 0).contiguous
    assert not BlockCyclicPartition(10, 3, 1).contiguous
    assert not BlockCyclicPartition(10, 3, 2).contiguous
    # a block large enough that nobody comes round twice
    assert BlockCyclicPartition(10, 3, 4).contiguous


def test_cyclic_defaults_to_plain_round_robin():
    """blocksize 1 is the default, so the plain reading of the name holds."""
    p = BlockCyclicPartition(10, 3)
    assert p.blocksize == 1
    assert p.indices(1).tolist() == [1, 4, 7]
    assert p.owner(np.arange(10)).tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


def test_cyclic_handles_more_ranks_than_indices():
    p = BlockCyclicPartition(2, 5)
    assert p.counts.tolist() == [1, 1, 0, 0, 0]
    assert p.range(4) == ()
    assert p.indices(4).tolist() == []


def test_cyclic_returns_one_run_per_visit():
    """A rank gets a block, then later another; range must show both."""
    p = BlockCyclicPartition(10, 3, 2)
    assert p.range(0) == ((0, 2), (6, 8))
    assert p.range(1) == ((2, 4), (8, 10))
    assert p.range(2) == ((4, 6),)
    assert p.owner(np.arange(10)).tolist() == [0, 0, 1, 1, 2, 2, 0, 0, 1, 1]


def test_cyclic_with_one_block_per_rank_is_contiguous():
    """The scheme spans from cyclic to contiguous as blocksize grows."""
    p = BlockCyclicPartition(9, 3, 3)
    for rank in range(3):
        assert len(p.range(rank)) == 1
        # n divides evenly here, so it coincides with the contiguous split
        assert np.array_equal(
            p.indices(rank), BlockCyclicPartition(9, 3, 0).indices(rank)
        )


def test_cyclic_short_final_block():
    """n not a multiple of the cycle leaves the last run short."""
    p = BlockCyclicPartition(7, 2, 3)
    assert p.period == 6
    assert p.range(0) == ((0, 3), (6, 7))
    assert p.range(1) == ((3, 6),)
    assert p.counts.sum() == 7


def test_cyclic_blocksize_is_part_of_identity():
    assert BlockCyclicPartition(10, 3, 2) != BlockCyclicPartition(10, 3, 4)
    assert BlockCyclicPartition(10, 3, 2) == BlockCyclicPartition(10, 3, 2)
    assert BlockCyclicPartition(10, 3) == BlockCyclicPartition(10, 3, 1)


def test_cyclic_rejects_bad_blocksize():
    with pytest.raises(ValueError):
        BlockCyclicPartition(10, 3, -1)


# --- row extraction -------------------------------------------------------


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("finalize", [False, True])
@pytest.mark.parametrize("nranks", [3, 4])
def test_local_rows_reconstruct_the_matrix(make, finalize, nranks):
    """Blocks placed back at their own indices must equal the original."""
    csr = build(finalize=finalize)
    p = make(csr.shape[0], nranks)

    rebuilt = np.zeros_like(csr.todense())
    total = 0
    for rank in range(nranks):
        block = local_rows(csr, p, rank)
        rebuilt[p.indices(rank)] = block.todense()
        total += block.nnz

    assert np.allclose(rebuilt, csr.todense())
    assert total == csr.nnz


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("finalize", [False, True])
def test_local_rows_keep_the_full_column_extent(make, finalize):
    """Columns must not be renumbered: they are what a halo exchange works from."""
    csr = build(finalize=finalize)
    p = make(csr.shape[0], 3)
    for rank in range(3):
        block = local_rows(csr, p, rank)
        assert block.shape[1] == csr.shape[1]
        assert block.shape[2] == csr.shape[2]
        assert block.shape[0] == p.count(rank)


def test_local_rows_is_a_copy_not_a_view():
    csr = build()
    block = local_rows(csr, BlockCyclicPartition(csr.shape[0], 2, 0), 0)
    before = csr.todense().copy()
    block._D[:] = -99.0
    assert np.allclose(csr.todense(), before)


@pytest.mark.parametrize("make", SCHEMES)
def test_local_rows_gives_an_empty_block_to_a_rank_owning_nothing(make):
    """A rank with no rows still gets a usable matrix, and joins collectives."""
    csr = build(nr=2)
    p = make(2, 5)
    empty = local_rows(csr, p, 4)
    assert empty.shape == (0, csr.shape[1], csr.shape[2])
    assert empty.nnz == 0


# --- deferred coherence ---------------------------------------------------


class _FakeComm:
    size = 1
    rank = 0


def test_distribution_starts_unassembled():
    """It has never communicated, so it cannot claim to be coherent."""
    d = Distribution(_FakeComm(), BlockCyclicPartition(10, 1, 0))
    assert not d.is_assembled
    assert d.epoch == 0


def test_touch_invalidates_and_assembly_restores():
    d = Distribution(_FakeComm(), BlockCyclicPartition(10, 1, 0))
    d.touch()
    assert not d.is_assembled
    d.mark_assembled()
    assert d.is_assembled


def test_structural_change_bumps_the_epoch():
    """A marked method must invalidate; assembly is deferred to the consumer."""
    csr = build()
    csr._distribution = Distribution(
        _FakeComm(), BlockCyclicPartition(csr.shape[0], 1, 0)
    )
    csr._distribution.mark_assembled()

    csr.finalize()
    assert not csr._distribution.is_assembled


def test_many_mutations_cost_one_assembly():
    """The point of deferring: batching mutations must not batch redistributions."""
    csr = build()
    csr._distribution = Distribution(
        _FakeComm(), BlockCyclicPartition(csr.shape[0], 1, 0)
    )
    csr._distribution.mark_assembled()

    for c in range(5):
        csr[0, c] = 1.0
    csr.translate_columns(np.arange(csr.shape[1]), np.arange(csr.shape[1]))
    csr.finalize()

    assert not csr._distribution.is_assembled
    csr._distribution.mark_assembled()
    assert csr._distribution.is_assembled


def test_undistributed_matrices_are_unaffected():
    """Every matrix today is undistributed; marking must be invisible to them."""
    csr = build()
    assert not hasattr(csr, "_distribution")
    csr.finalize()
    csr[0, 0] = 1.0
    assert csr.nnz > 0


# --- the guard that stops the marking drifting ----------------------------

#: Methods that build a matrix from nothing, so no distribution exists yet.
_CONSTRUCTORS = {"__init__", "_SparseCSR__init_shape", "__init_shape", "__setstate__"}

#: Arrays that define the sparsity structure; writing one invalidates ownership.
_STRUCTURE = {"ptr", "ncol", "col", "_nnz"}


def _methods_writing_structure():
    """Every SparseCSR method that assigns to a sparsity-structure array."""
    # NOT inspect.getmodule(SparseCSR): @set_module("sisl") rewrites __module__,
    # so that returns the sisl package and the scan silently finds nothing.
    from sisl._core import sparse as sparse_module

    source = inspect.getsource(sparse_module)
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "SparseCSR"
    )
    found = set()
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                while isinstance(t, ast.Subscript):
                    t = t.value
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                    and t.attr in _STRUCTURE
                ):
                    found.add(fn.name)
    return found


def test_every_structural_mutation_is_marked():
    """Forgetting the marker would corrupt distributed results silently.

    So this is checked mechanically rather than by review: any new method that
    writes ptr/ncol/col/_nnz must either carry `@distribute_changes` or be
    listed as a constructor.
    """
    unmarked = []
    for name in sorted(_methods_writing_structure() - _CONSTRUCTORS):
        method = getattr(SparseCSR, name, None)
        if method is None or not getattr(method, "_distribute_changes", False):
            unmarked.append(name)

    assert not unmarked, (
        f"SparseCSR methods change the sparsity structure without "
        f"@distribute_changes: {unmarked}. Mark them, or add them to "
        f"_CONSTRUCTORS if they build a matrix from nothing."
    )


def test_the_guard_would_actually_catch_an_omission():
    """A guard nobody has seen fail is not a guard."""
    assert _methods_writing_structure(), "AST scan found nothing -- it is broken"

    def unmarked(self):
        self.col = None

    assert not getattr(unmarked, "_distribute_changes", False)
    assert getattr(distribute_changes(unmarked), "_distribute_changes", False)


# --- assembly: serial -----------------------------------------------------


def _distributed(csr, size=1, blocksize=0, comm=None):
    csr._distribution = Distribution(
        comm or _FakeComm(), BlockCyclicPartition(csr.shape[0], size, blocksize)
    )
    return csr


def test_assembly_preserves_data_on_one_rank():
    """With one rank everything is owned, so assembly must be a faithful rebuild."""
    csr = build()
    ref = csr.todense().copy()
    _distributed(csr)

    csr.__sisl_distribute__()

    assert np.allclose(csr.todense(), ref)
    assert csr.nnz == int((np.abs(ref).sum(-1) > 0).sum())
    assert csr._distribution.is_assembled


def test_assembly_leaves_the_matrix_finalized():
    """The rebuild sorts and compacts, so claiming otherwise would waste a pass."""
    csr = _distributed(build())
    csr.__sisl_distribute__()
    assert csr.finalized
    assert csr.ptr[-1] == csr.nnz


def test_assembly_is_idempotent_while_nothing_changes():
    csr = _distributed(build())
    csr.__sisl_distribute__()
    ref = csr.todense().copy()
    epoch = csr._distribution.epoch

    csr.__sisl_distribute__()

    assert csr._distribution.epoch == epoch
    assert np.allclose(csr.todense(), ref)


def test_mutating_after_assembly_requires_another():
    csr = _distributed(build())
    csr.__sisl_distribute__()
    csr[0, 1] = 3.0
    assert not csr._distribution.is_assembled
    csr.__sisl_distribute__()
    assert csr._distribution.is_assembled
    assert np.allclose(csr.todense()[0, 1], 3.0)


def test_assembly_handles_an_empty_matrix():
    csr = _distributed(SparseCSR((6, 8, 1), dtype=np.float64))
    csr.__sisl_distribute__()
    assert csr.nnz == 0
    assert csr._distribution.is_assembled


def test_assembly_refuses_an_undistributed_matrix():
    with pytest.raises(ValueError, match="no distribution"):
        build().__sisl_distribute__()


def test_assembly_refuses_an_unknown_op():
    with pytest.raises(ValueError, match="sum.*single"):
        _distributed(build()).__sisl_distribute__(op="multiply")


def test_assembly_refuses_a_partition_of_the_wrong_length():
    csr = build()
    csr._distribution = Distribution(_FakeComm(), BlockCyclicPartition(999, 1, 0))
    with pytest.raises(ValueError, match="partition covers"):
        csr.__sisl_distribute__()


def test_assembly_refuses_a_partition_for_other_ranks():
    """A partition for 4 ranks on a communicator of 1 would silently drop rows."""
    csr = build()
    csr._distribution = Distribution(
        _FakeComm(), BlockCyclicPartition(csr.shape[0], 4, 0)
    )
    with pytest.raises(ValueError, match="partition is for"):
        csr.__sisl_distribute__()


# --- assembly: genuinely parallel -----------------------------------------

MPIRUN = shutil.which("mpirun")
HAS_MPI4PY = importlib.util.find_spec("mpi4py") is not None

needs_mpirun = pytest.mark.skipif(
    MPIRUN is None or not HAS_MPI4PY, reason="mpirun or mpi4py unavailable"
)

_PARALLEL = """
    import numpy as np
    from mpi4py import MPI

    from sisl._core.sparse import SparseCSR
    from sisl._core.mpi import BlockCyclicPartition, Distribution
    import sisl.mpi as smpi

    comm = smpi.get_comm()
    N, NC, DIM = 12, 20, 2
    BLOCKSIZE = {blocksize}
    part = BlockCyclicPartition(N, comm.size, BLOCKSIZE)
    own = part.indices(comm.rank)
    other = np.ones(N, bool)
    other[own] = False

    def attach(c):
        c._distribution = Distribution(None, part)
        return c

    def build_replicated():
        rng = np.random.default_rng(42)          # identical on every rank
        c = SparseCSR((N, NC, DIM), dtype=np.float64)
        for r in range(N):
            for col in np.sort(rng.choice(NC, size=3, replace=False)):
                c[r, col] = rng.random(DIM)
        return c

    ref = build_replicated().todense()
    csr = attach(build_replicated())
    csr.__sisl_distribute__()
    dense = csr.todense()
    assert np.allclose(dense[own], ref[own]), "owned rows were not preserved"
    assert np.allclose(dense[other], 0.0), "rows owned elsewhere were not dropped"

    nnz = np.array([csr.nnz], dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, nnz)
    assert nnz[0] == int((np.abs(ref).sum(-1) > 0).sum()), "nnz not conserved"

    # every rank writes every row, including rows it does not own
    csr = attach(SparseCSR((N, NC, DIM), dtype=np.float64))
    for r in range(N):
        csr[r, comm.rank] = float(comm.rank + 1)
    csr.__sisl_distribute__()
    expect = np.zeros((N, NC, DIM))
    for r in own:
        for k in range(comm.size):
            expect[r, k] = float(k + 1)
    assert np.allclose(csr.todense(), expect), "off-rank writes did not reach the owner"

    # every rank writes the SAME position; 'sum' must sum them
    csr = attach(SparseCSR((N, NC, DIM), dtype=np.float64))
    for r in range(N):
        csr[r, 0] = 1.0
    csr.__sisl_distribute__(op="sum")
    dense = csr.todense()
    assert np.allclose(dense[own, 0], float(comm.size)), "sum did not sum duplicates"
    assert np.allclose(dense[other], 0.0)

    comm.Barrier()
    if comm.rank == 0:
        print("OK")
"""


@needs_mpirun
@pytest.mark.parametrize("blocksize", [0, 1, 2])
@pytest.mark.parametrize("nprocs", [2, 3])
def test_assembly_across_ranks(blocksize, nprocs):
    """Assembly must move every element to its owner, for any layout.

    blocksize 0 gives each rank one contiguous run, 1 and 2 give several, which
    exercises the scattered path through Alltoallv.
    """
    proc = subprocess.run(
        [
            MPIRUN,
            "--oversubscribe",
            "-n",
            str(nprocs),
            sys.executable,
            "-c",
            textwrap.dedent(_PARALLEL.format(blocksize=blocksize)),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=dict(os.environ),
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "OK" in proc.stdout, combined
