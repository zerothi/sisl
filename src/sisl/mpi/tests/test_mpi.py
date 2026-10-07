# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Tests for `sisl.mpi`.

Almost every test here shells out to a fresh interpreter.  That is not
defensiveness, it is a hard constraint: MPI can be initialized exactly once per
process, so the acquisition states (sisl owns MPI / sisl attaches to a foreign
owner / no MPI at all) are mutually exclusive *within* a process and cannot be
exercised in a single pytest session.

These tests are the executable form of the probes in
``docs/ideas/mpi-core-probes/``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.mpi]

MPIRUN = shutil.which("mpirun")

needs_mpirun = pytest.mark.skipif(MPIRUN is None, reason="mpirun unavailable")


def run(code, *, nprocs=None, env=None, timeout=120):
    """Execute `code` in a fresh interpreter, optionally under ``mpirun``."""
    launcher = ()
    if nprocs is not None:
        launcher = (MPIRUN, "--oversubscribe", "-n", str(nprocs))
    environ = dict(os.environ)
    if env:
        environ.update(env)
    return subprocess.run(
        [*launcher, sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environ,
    )


def assert_ok(proc):
    """Require a clean exit, and no launcher complaint about improper teardown."""
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "OK" in proc.stdout, combined
    assert "improperly" not in combined, combined


# --- laziness -------------------------------------------------------------


def test_import_sisl_does_not_import_mpi4py():
    """`import sisl` must not drag in mpi4py, nor initialize MPI."""
    proc = run("""
        import sys
        import sisl
        leaked = sorted(m for m in sys.modules if m.startswith("mpi4py"))
        assert not leaked, leaked
        print("OK")
        """)
    assert_ok(proc)


# --- serial fallbacks -----------------------------------------------------


def test_serial_shim_when_mpi4py_missing():
    """With mpi4py unimportable, sisl.mpi must still work and report serial."""
    proc = run("""
        import sys

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name == "mpi4py" or name.startswith("mpi4py."):
                    raise ImportError("blocked for testing")
                return None

        sys.meta_path.insert(0, Block())

        import sisl.mpi as m
        comm = m.get_comm()
        assert comm.rank == 0, comm.rank
        assert comm.size == 1, comm.size
        assert comm.is_parallel is False
        assert comm.owns_mpi is False
        # operations outside the contract are no-ops, not errors
        assert comm.Gather(None) is None
        assert comm.Split(0) is None
        print("OK")
        """)
    assert_ok(proc)


def test_sisl_mpi_env_var_disables():
    """SISL_MPI=0 forces the serial shim even when mpi4py is installed."""
    proc = run(
        """
        import sys
        import sisl.mpi as m
        comm = m.get_comm()
        assert comm.size == 1, comm.size
        assert comm.rank == 0, comm.rank
        assert comm.is_parallel is False
        assert comm.owns_mpi is False
        assert "mpi4py.MPI" not in sys.modules
        print("OK")
        """,
        env={"SISL_MPI": "0"},
    )
    assert_ok(proc)


def test_serial_collectives_are_identity():
    """Serial collectives must leave buffers holding exactly the reduced result."""
    proc = run(
        """
        import numpy as np
        import sisl.mpi as m

        comm = m.get_comm()

        buf = np.array([1.0, 2.0, 3.0])
        comm.Bcast(buf)
        assert np.allclose(buf, [1.0, 2.0, 3.0]), buf

        send = np.array([1.0, 2.0])
        recv = np.zeros(2)
        comm.Allreduce(send, recv)
        assert np.allclose(recv, [1.0, 2.0]), recv

        # both in-place spellings must leave the buffer untouched: MPI.IN_PLACE,
        # which is what parallel code writes, and None for code that must also
        # run with mpi4py absent entirely.
        from mpi4py import MPI

        inplace = np.array([3.0, 4.0])
        comm.Allreduce(MPI.IN_PLACE, inplace)
        assert np.allclose(inplace, [3.0, 4.0]), inplace

        inplace = np.array([3.0, 4.0])
        comm.Allreduce(None, inplace)
        assert np.allclose(inplace, [3.0, 4.0]), inplace

        comm.Barrier()
        assert comm.on_rank0() is True
        print("OK")
        """,
        env={"SISL_MPI": "0"},
    )
    assert_ok(proc)


@needs_mpirun
def test_parallel_buffer_collectives():
    """The upper-case collectives must actually reduce and broadcast across ranks."""
    proc = run(
        """
        import numpy as np
        from mpi4py import MPI

        import sisl.mpi as m

        comm = m.get_comm()
        assert comm.size == 3, comm.size

        inplace = np.ones(4)
        comm.Allreduce(MPI.IN_PLACE, inplace)
        assert np.allclose(inplace, 3.0), inplace

        send = np.full(2, comm.rank + 1.0)
        recv = np.zeros(2)
        comm.Allreduce(send, recv)
        assert np.allclose(recv, 6.0), recv

        buf = np.zeros(2)
        if comm.on_rank0():
            buf[:] = [7.0, 8.0]
        comm.Bcast(buf)
        assert np.allclose(buf, [7.0, 8.0]), (comm.rank, buf)

        comm.Barrier()
        if comm.on_rank0():
            print("OK")
        """,
        nprocs=3,
    )
    assert_ok(proc)


def test_no_module_level_wrappers_and_one_shared_contract():
    """Lock in the API shape: use the object, and both classes share a contract."""
    import sisl.mpi as m

    for name in ("bcast", "allreduce", "barrier", "on_rank0", "rank", "size"):
        assert not hasattr(m, name), f"module-level {name!r} should not exist"

    # the contract is only what cannot be delegated
    contract = ("comm", "rank", "size", "owns_mpi")
    for cls in (m.Communicator, m._SerialComm):
        assert issubclass(cls, m.BaseCommunicator), cls
        for name in contract:
            assert hasattr(cls, name), f"{cls.__name__} is missing {name!r}"

    # derived once, on the base, so the two cannot drift apart
    for name in ("is_parallel", "on_rank0"):
        assert hasattr(m.BaseCommunicator, name), name

    # operations are NOT wrapped -- they reach mpi4py through __getattr__
    for name in ("Barrier", "Bcast", "Allreduce", "Abort", "Get_rank", "Get_size"):
        assert name not in vars(m.Communicator), f"{name!r} should not be wrapped"


def test_owns_mpi_when_nobody_else_did():
    """No launcher, no foreign owner: sisl claims MPI and cleans up after itself."""
    proc = run("""
        import sisl.mpi as m
        comm = m.get_comm()
        assert comm.owns_mpi is True, comm.owns_mpi
        assert comm.size == 1, comm.size
        assert comm.rank == 0, comm.rank
        assert comm.is_parallel is False
        print("OK")
        """)
    assert_ok(proc)


@needs_mpirun
def test_owns_mpi_under_launcher():
    """Under mpirun the world size must be reported without inspecting the environment."""
    proc = run(
        """
        import sisl.mpi as m
        comm = m.get_comm()
        if comm.on_rank0():
            assert comm.owns_mpi is True, comm.owns_mpi
            assert comm.size == 3, comm.size
            assert comm.is_parallel is True
            print("OK")
        """,
        nprocs=3,
    )
    assert_ok(proc)


def test_attaches_to_foreign_owner():
    """If another component initialized MPI first, sisl attaches and never finalizes."""
    proc = run("""
        from mpi4py import MPI          # foreign auto-initialization
        assert MPI.Is_initialized()
        import sisl.mpi as m
        comm = m.get_comm()
        assert comm.owns_mpi is False, comm.owns_mpi
        assert comm.size == 1, comm.size
        print("OK")
        """)
    assert_ok(proc)


def test_repeated_acquisition_is_idempotent():
    """A second MPI_Init aborts uncatchably, so acquisition must happen once."""
    proc = run("""
        import sisl.mpi as m
        first = m.get_comm()
        for _ in range(5):
            assert m.get_comm() is first
        assert first.size == 1
        print("OK")
        """)
    assert_ok(proc)


def test_initialized_with_init_thread_not_init():
    """sisl must request FUNNELED; a plain MPI.Init would leave us at SINGLE.

    This asserts the behaviour rather than a stored attribute, so it keeps
    working regardless of whether the granted level is recorded anywhere.
    """
    proc = run("""
        import sisl.mpi as m

        # sisl must acquire first; importing mpi4py here would initialize MPI
        # and leave sisl merely attaching to it.
        comm = m.get_comm()
        assert comm.owns_mpi is True

        from mpi4py import MPI

        provided = MPI.Query_thread()
        assert provided >= MPI.THREAD_FUNNELED, provided
        print("provided thread level:", provided)
        print("OK")
        """)
    assert_ok(proc)


# --- rank-divergent failure -----------------------------------------------


@needs_mpirun
def test_excepthook_aborts_instead_of_hanging():
    """One rank raising must kill the job, not leave the others in a collective.

    Without the excepthook rank 0 blocks in `barrier` forever and this test
    fails by timeout rather than by assertion.
    """
    proc = run(
        """
        import sisl.mpi as m
        comm = m.get_comm()
        if comm.rank == 1:
            raise RuntimeError("boom from rank 1")
        comm.Barrier()
        print("rank 0 should never get here")
        """,
        nprocs=2,
        timeout=60,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert "boom from rank 1" in combined, combined
    assert "should never get here" not in proc.stdout, combined


def test_no_excepthook_when_serial():
    """Serial users must keep ordinary Python tracebacks and exit codes."""
    proc = run("""
        import sisl.mpi as m
        assert m.get_comm().size == 1
        raise RuntimeError("ordinary failure")
        """)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 1, combined
    assert "RuntimeError: ordinary failure" in combined, combined
    assert "Traceback" in combined, combined


def test_unknown_operations_delegate_to_the_raw_communicator():
    """The full MPI surface stays reachable without a wrapper per operation."""
    proc = run("""
        import sisl.mpi as m

        comm = m.get_comm()

        # not wrapped by Communicator, but real MPI -> forwarded untouched
        assert comm.Get_size() == 1
        assert callable(comm.Gather)
        assert callable(comm.Split)

        # the lower-case pickle-based calls pass through as well; nothing is
        # rewritten or refused on the caller's behalf
        assert comm.bcast(42) == 42
        assert comm.allreduce(3) == 3
        assert comm.comm.bcast(42) == 42

        assert comm.Get_rank() == 0
        print("OK")
        """)
    assert_ok(proc)


def test_serial_operations_outside_the_contract_are_noops():
    """Code written for MPI must run unchanged in serial."""
    proc = run(
        """
        import numpy as np

        import sisl.mpi as m

        comm = m.get_comm()

        # outside the contract -> silently does nothing
        assert comm.Gather(None) is None
        assert comm.Alltoallv(None) is None
        assert comm.Isend(None, dest=0) is None

        # inside the contract -> real (serial) semantics still apply
        send = np.array([2.0, 5.0])
        recv = np.zeros(2)
        comm.Reduce(send, recv)
        assert np.allclose(recv, send), recv

        comm.Barrier()
        assert comm.on_rank0() is True
        assert comm.owns_mpi is False
        assert comm.comm is comm
        print("OK")
        """,
        env={"SISL_MPI": "0"},
    )
    assert_ok(proc)
