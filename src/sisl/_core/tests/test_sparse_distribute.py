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

from sisl._core._sparse_distribute import distribute, local_rows
from sisl._core.sparse import SparseCSR
from sisl.mpi import BlockCyclicDistribution, DistributionState, distribute_changes

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


class FakeComm:
    """Stand-in communicator: a distribution only needs ``rank`` and ``size``."""

    def __init__(self, size=1, rank=0):
        self.size = size
        self.rank = rank


def _contiguous(n, size):
    """blocksize 0: one unbroken run per rank."""
    return BlockCyclicDistribution(n, 0, comm=FakeComm(size))


def _cyclic(n, size):
    return BlockCyclicDistribution(n, comm=FakeComm(size))


def _cyclic2(n, size):
    return BlockCyclicDistribution(n, 2, comm=FakeComm(size))


def _cyclic3(n, size):
    return BlockCyclicDistribution(n, 3, comm=FakeComm(size))


#: Every scheme.
SCHEMES = [_contiguous, _cyclic, _cyclic2, _cyclic3]


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
    block = local_rows(csr, _contiguous(csr.shape[0], 2), 0)
    before = csr.todense().copy()
    block._D[:] = -99.0
    assert np.allclose(csr.todense(), before)


def test_local_rows_gives_an_empty_block_to_a_rank_owning_nothing():
    """A rank with no rows still gets a usable matrix, and joins collectives."""
    csr = build(nr=2)
    p = _cyclic(2, 5)
    empty = local_rows(csr, p, 4)
    assert empty.shape == (0, csr.shape[1], csr.shape[2])
    assert empty.nnz == 0


# --- deferred coherence ---------------------------------------------------


def test_structural_change_bumps_the_epoch():
    """A marked method must invalidate; assembly is deferred to the consumer."""
    csr = build()
    csr._dist_state = DistributionState(_contiguous(csr.shape[0], 1))
    csr._dist_state.mark_assembled()

    csr.finalize()
    assert not csr._dist_state.is_assembled


def test_many_mutations_cost_one_assembly():
    """The point of deferring: batching mutations must not batch redistributions."""
    csr = build()
    csr._dist_state = DistributionState(_contiguous(csr.shape[0], 1))
    csr._dist_state.mark_assembled()

    for c in range(5):
        csr[0, c] = 1.0
    csr.translate_columns(np.arange(csr.shape[1]), np.arange(csr.shape[1]))
    csr.finalize()

    assert not csr._dist_state.is_assembled
    csr._dist_state.mark_assembled()
    assert csr._dist_state.is_assembled


def test_undistributed_matrices_are_unaffected():
    """Every matrix today is undistributed; marking must be invisible to them."""
    csr = build()
    assert not hasattr(csr, "_dist_state")
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


def _distributed(csr, size=1, blocksize=0):
    dist = BlockCyclicDistribution(csr.shape[0], blocksize, comm=FakeComm(size))
    return dist(csr)


def test_assembly_preserves_data_on_one_rank():
    """With one rank everything is owned, so assembly must be a faithful rebuild."""
    csr = build()
    ref = csr.todense().copy()
    _distributed(csr)

    csr.__sisl_distribute__()

    assert np.allclose(csr.todense(), ref)
    assert csr.nnz == int((np.abs(ref).sum(-1) > 0).sum())
    assert csr._dist_state.is_assembled


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
    epoch = csr._dist_state.epoch

    csr.__sisl_distribute__()

    assert csr._dist_state.epoch == epoch
    assert np.allclose(csr.todense(), ref)


def test_mutating_after_assembly_requires_another():
    csr = _distributed(build())
    csr.__sisl_distribute__()
    csr[0, 1] = 3.0
    assert not csr._dist_state.is_assembled
    csr.__sisl_distribute__()
    assert csr._dist_state.is_assembled
    assert np.allclose(csr.todense()[0, 1], 3.0)


def test_assembly_handles_an_empty_matrix():
    csr = _distributed(SparseCSR((6, 8, 1), dtype=np.float64))
    csr.__sisl_distribute__()
    assert csr.nnz == 0
    assert csr._dist_state.is_assembled


def test_assembly_refuses_an_undistributed_matrix():
    with pytest.raises(ValueError, match="no distribution"):
        build().__sisl_distribute__()


def test_assembly_refuses_an_unknown_op():
    with pytest.raises(ValueError, match="sum.*single"):
        _distributed(build()).__sisl_distribute__(op="multiply")


def test_assembly_refuses_a_distribution_of_the_wrong_length():
    csr = build()
    with pytest.raises(ValueError, match="distribution covers"):
        _contiguous(999, 1)(csr)


def test_calling_the_distribution_distributes_in_place():
    csr = build()
    dist = _contiguous(csr.shape[0], 1)
    assert dist(csr) is csr
    assert csr._dist_state.distribution is dist
    assert csr._dist_state.is_assembled


def test_one_distribution_serves_several_matrices():
    """The distribution holds no per-object state, so it can be shared."""
    dist = _cyclic(10, 1)
    a, b = dist(build(seed=1)), dist(build(seed=2))
    assert a._dist_state is not b._dist_state
    a[0, 0] = 1.0
    assert not a._dist_state.is_assembled
    assert b._dist_state.is_assembled


def test_a_new_distribution_forces_reassembly():
    csr = build()
    _contiguous(csr.shape[0], 1)(csr)
    state = csr._dist_state

    new = _cyclic(csr.shape[0], 1)
    new(csr)
    assert csr._dist_state is not state
    assert csr._dist_state.distribution is new
    assert csr._dist_state.is_assembled

    # an equal distribution keeps the state, and nothing is redone
    epoch = csr._dist_state.epoch
    BlockCyclicDistribution(csr.shape[0], comm=new.comm)(csr)
    assert csr._dist_state.distribution is new
    assert csr._dist_state.epoch == epoch


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
    from sisl.mpi import BlockCyclicDistribution
    import sisl.mpi as smpi

    comm = smpi.get_comm()
    N, NC, DIM = 12, 20, 2
    BLOCKSIZE = {blocksize}
    dist = BlockCyclicDistribution(N, BLOCKSIZE)
    assert dist.comm is comm
    own = dist.indices()
    other = np.ones(N, bool)
    other[own] = False

    def build_replicated():
        rng = np.random.default_rng(42)          # identical on every rank
        c = SparseCSR((N, NC, DIM), dtype=np.float64)
        for r in range(N):
            for col in np.sort(rng.choice(NC, size=3, replace=False)):
                c[r, col] = rng.random(DIM)
        return c

    ref = build_replicated().todense()
    csr = dist(build_replicated())
    dense = csr.todense()
    assert np.allclose(dense[own], ref[own]), "owned rows were not preserved"
    assert np.allclose(dense[other], 0.0), "rows owned elsewhere were not dropped"

    nnz = np.array([csr.nnz], dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, nnz)
    assert nnz[0] == int((np.abs(ref).sum(-1) > 0).sum()), "nnz not conserved"

    # another layout ships every element from its old owner to its new one
    new = BlockCyclicDistribution(N, 0 if BLOCKSIZE else 1, comm=comm)
    new(csr)
    own_new = new.indices()
    other_new = np.ones(N, bool)
    other_new[own_new] = False
    dense = csr.todense()
    assert np.allclose(dense[own_new], ref[own_new]), "redistribution lost rows"
    assert np.allclose(dense[other_new], 0.0), "redistribution kept foreign rows"

    # every rank writes every row, including rows it does not own
    csr = SparseCSR((N, NC, DIM), dtype=np.float64)
    for r in range(N):
        csr[r, comm.rank] = float(comm.rank + 1)
    dist(csr)
    expect = np.zeros((N, NC, DIM))
    for r in own:
        for k in range(comm.size):
            expect[r, k] = float(k + 1)
    assert np.allclose(csr.todense(), expect), "off-rank writes did not reach the owner"

    # every rank writes the SAME position; 'sum' must sum them
    csr = SparseCSR((N, NC, DIM), dtype=np.float64)
    for r in range(N):
        csr[r, 0] = 1.0
    dist(csr, op="sum")
    dense = csr.todense()
    assert np.allclose(dense[own, 0], float(comm.size)), "sum did not sum duplicates"
    assert np.allclose(dense[other], 0.0)

    comm.Barrier()
    if comm.rank == 0:
        print("OK")
"""


@pytest.mark.slow
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
