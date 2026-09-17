# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Optional MPI support for sisl.

sisl works identically whether or not MPI is present.  `get_comm` returns a
`BaseCommunicator`: a `Communicator` wrapping mpi4py when MPI is available, or
a serial stand-in when `mpi4py` is missing or ``SISL_MPI=0``.  The stand-in is
rank 0 of 1 and every operation is a no-op, so calling code needs no branches.

Everything goes through that object::

    from sisl.mpi import get_comm

    comm = get_comm()
    if comm.on_rank0():
        ...

There are deliberately no module-level convenience wrappers around it: the
object carries the provenance (who initialized MPI, which thread level was
actually granted) that makes its behaviour correct, so hiding it behind free
functions would hide exactly what callers need to reason about.

Operations
----------
`Communicator` wraps no operations at all: everything is forwarded to the
underlying mpi4py communicator unchanged, with mpi4py's own signatures and
defaults intact.  Callers write ordinary mpi4py.

Bear in mind mpi4py's convention while doing so -- the lower-case calls
(`bcast`, `allreduce`) pickle, the upper-case ones (`Bcast`, `Allreduce`) send
buffers.  sisl moves numpy arrays, so the upper-case forms are almost always
what is wanted.

Acquisition
-----------
mpi4py initializes MPI when ``mpi4py.MPI`` is imported, which makes
``MPI_Initialized()`` unconditionally true and useless as a signal.  Disabling
that auto-initialization restores its meaning: *has something else already
claimed MPI?*

- Nothing has claimed it: sisl claims it, and is responsible for finalizing it.
- Something already has: sisl attaches, and must never finalize it, because
  doing so would break the host application at exit.

No launcher or scheduler environment variables are inspected.  ``COMM_WORLD``
already reports the correct size under ``mpirun``, because the runtime is
populated before Python starts.

See ``docs/ideas/mpi-core.md`` for the design, and
``docs/ideas/mpi-core-probes/`` for the evidence behind each rule.
"""

from __future__ import annotations

import atexit
import sys
from abc import ABC, abstractmethod

from sisl._environ import get_environ_variable

__all__ = ["BaseCommunicator", "Communicator", "get_comm"]


class BaseCommunicator(ABC):
    """The communicator contract sisl relies on.

    Only what cannot be delegated appears here: which rank this is, how many
    there are, and who owns MPI.  Every actual operation -- collectives,
    point-to-point, communicator management -- is reached through
    ``__getattr__`` on the implementations, so this contract does not have to
    track mpi4py's surface.

    Two classes implement it: `Communicator`, wrapping a real mpi4py
    communicator, and the serial stand-in used when there is no MPI.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def comm(self):
        """The underlying communicator object."""

    @property
    @abstractmethod
    def rank(self) -> int:
        """Index of this rank within the communicator."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of ranks in the communicator."""

    @property
    @abstractmethod
    def owns_mpi(self) -> bool:
        """Whether sisl initialized MPI, and must therefore finalize it."""

    # -- derived from the above, shared by every implementation --------------

    @property
    def is_parallel(self) -> bool:
        return self.size > 1

    def on_rank0(self) -> bool:
        """True on the rank that should perform single-writer work (IO, printing)."""
        return self.rank == 0


def _reduce_serially(sendbuf, recvbuf) -> None:
    """Perform a reduction over a single rank.

    Reducing one contribution yields that contribution under every operator, so
    the only work is making sure the caller can read it back.  With one buffer
    it is already in place; with two it must be copied across, or the caller
    reads an untouched `recvbuf`.
    """
    if recvbuf is None or sendbuf is None:
        return

    # `MPI.IN_PLACE` also means the data is already where the caller wants it.
    # Looked up lazily because this class exists precisely for when mpi4py is
    # absent -- in which case the caller cannot have passed it either.
    try:
        from mpi4py import MPI
    except ImportError:
        pass
    else:
        if sendbuf is MPI.IN_PLACE:
            return

    recvbuf[...] = sendbuf


class _SerialComm(BaseCommunicator):
    """A communicator shaped object for when there is no MPI.

    Every operation is a no-op.  The single exception is a reduction given
    separate send and receive buffers, which must still deliver the result; see
    `_reduce_serially`.
    """

    __slots__ = ()

    @property
    def comm(self):
        return self

    @property
    def rank(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    @property
    def owns_mpi(self) -> bool:
        # There is no MPI here, so there is nothing to own or to finalize.
        return False

    def Reduce(self, sendbuf, recvbuf=None, op=None, root: int = 0) -> None:
        _reduce_serially(sendbuf, recvbuf)

    def Allreduce(self, sendbuf, recvbuf=None, op=None) -> None:
        # With one rank this coincides with `Reduce`.
        _reduce_serially(sendbuf, recvbuf)

    def __getattr__(self, name):
        """Every operation outside the contract does nothing, and returns None.

        Dunders are excluded so that `copy`, `pickle` and friends still see a
        normal object rather than a callable for every protocol they probe.
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _noop(*args, **kwargs):
            return None

        return _noop


class Communicator(BaseCommunicator):
    """An `mpi4py` communicator plus the provenance needed to behave correctly.

    A `Communicator` always wraps real MPI; the serial case is a different
    implementation of `BaseCommunicator`, not this one holding a stand-in.

    ``owns_mpi`` is not bookkeeping: it is the guard that stops sisl finalizing
    an MPI it did not initialize.
    """

    __slots__ = ("_comm", "_owns")

    def __init__(self, comm, owns: bool):
        self._comm = comm
        self._owns = owns

    @property
    def comm(self):
        """The underlying `mpi4py` communicator."""
        return self._comm

    @property
    def rank(self) -> int:
        return self._comm.rank

    @property
    def size(self) -> int:
        return self._comm.size

    @property
    def owns_mpi(self) -> bool:
        return self._owns

    def __getattr__(self, name):
        """Pass anything not defined here through to the underlying communicator.

        This keeps the whole mpi4py surface -- collectives, point-to-point,
        communicator management, both the buffer-based and the pickle-based
        calls -- reachable without a wrapper per operation.  Nothing is
        rewritten or refused: what the caller asks for is what mpi4py receives,
        with mpi4py's own defaults intact.
        """
        return getattr(object.__getattribute__(self, "_comm"), name)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} rank={self.rank}/{self.size}, "
            f"owns={self._owns}>"
        )


_COMM = None
_MPI = None


def _serial() -> BaseCommunicator:
    """The communicator used when there is no MPI to talk to."""
    return _SerialComm()


def _finalize() -> None:
    """Finalize MPI, but only if it is still ours to finalize."""
    global _MPI

    if not _MPI.Is_finalized():
        _MPI.Finalize()


def _install_excepthook(comm) -> None:
    """Turn a rank-local exception into a job-wide abort.

    Without this, one rank raising leaves every other rank blocked in a
    collective until walltime -- confirmed behaviour, not a precaution; see
    ``docs/ideas/mpi-core-probes/probe_divergence.py``.  The previous hook runs
    first so the traceback is still reported, and by the host application's
    reporting if it installed any.
    """
    previous = sys.excepthook

    def _abort_on_exception(exc_type, exc, tb):
        try:
            previous(exc_type, exc, tb)
            sys.stderr.flush()
            sys.stdout.flush()
        finally:
            comm.Abort(1)

    sys.excepthook = _abort_on_exception


def _acquire() -> BaseCommunicator:
    """Determine, exactly once, which communicator sisl should use."""
    global _MPI
    if not get_environ_variable("SISL_MPI"):
        return _serial()

    try:
        import mpi4py

        # Best effort, and deliberately so: if mpi4py.MPI was already imported
        # these are ignored, which is precisely the case the Is_initialized()
        # branch below handles.
        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False

        from mpi4py import MPI as _MPI
    except ImportError:
        return _serial()

    owns = False
    if not _MPI.Is_initialized():
        # The most dangerous line in this module.  A second MPI_Init aborts the
        # process uncatchably -- no exception, no traceback -- so this must run
        # exactly once.  `_COMM` being a module-level singleton is what
        # guarantees that; do not call `_acquire` from anywhere but `get_comm`.
        #
        # Init_thread rather than Init: the level cannot be raised afterwards.
        # FUNNELED means other threads may exist (threaded BLAS) but only the
        # main thread calls MPI, which is what sisl does.
        _MPI.Init_thread(_MPI.THREAD_FUNNELED)
        owns = True
        atexit.register(_finalize)

    communicator = Communicator(
        _MPI.COMM_WORLD,
        owns=owns,
    )

    # Only when parallel: a serial run must keep ordinary Python tracebacks and
    # exit codes.
    if communicator.is_parallel:
        _install_excepthook(communicator)

    return communicator


def get_comm() -> BaseCommunicator:
    """The communicator sisl is using, acquiring it on first call."""
    global _COMM
    if _COMM is None:
        _COMM = _acquire()
    return _COMM
