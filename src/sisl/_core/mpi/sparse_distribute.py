# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Row-block distribution for `SparseCSR`.

`SparseCSR` is geometry-blind: it is ``(nr, nc, dim)`` over ``ptr``/``ncol``/
``col``/``_D`` and knows nothing about where its rows sit in space.  The
distribution here is blind in the same way -- rows are divided by index alone.
That is general, working for any CSR, at the cost of ignoring spatial locality;
a geometry-aware partition belongs one layer up, in `SparseOrbital`, where the
`Geometry` can bound the halo by interaction range.

`Partition` states how indices map to ranks and nothing more, so the same
schemes serve rows, columns, k-points or energy points.  `BlockCyclicPartition`
covers the useful range: one contiguous run per rank, which preserves whatever
locality the sparsity pattern has and keeps extraction a slice; plain cyclic,
which balances load when work varies along the axis but destroys that locality;
and the block-cyclic layout in between that ScaLAPACK expects.

Other schemes -- an explicit permutation, or a split balanced by nnz rather
than by row count -- fit the same contract by implementing `range` and `owner`
alone.

Coherence is deferred, not eager
--------------------------------
Redistribution is not triggered by mutation.  A method that changes the
sparsity structure bumps an integer and returns; the redistribution happens at
the point of *use*.  This follows PETSc's assembly model, and it matters
because `transpose` on a row-distributed CSR is a full all-to-all: assembling
eagerly would turn ``H.transpose().transform(...).eliminate_zeros()`` into three
redistributions instead of one, and sisl's construction loops perform thousands
of ``__setitem__`` calls.

The marking is by decorator rather than a bare attribute so that forgetting to
mark a method -- which would corrupt results silently, the worst failure mode in
a physics code -- can be caught mechanically; see
``tests/test_sparse_distribute.py``.

Assembly is reached through ``__sisl_distribute__``, the protocol any object
carrying a distribution implements, so a consumer can make what it was handed
coherent without knowing what it is.  `distribute` is `SparseCSR`'s
implementation of that protocol; a distributed `Grid`, or a set of k-points,
would supply its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from sisl._array import array_arange
from sisl._core.sparse import SparseCSR
from sisl._internal import set_module
from sisl._ufuncs import register_sisl_dispatch

from .distribute import Partition

__all__ = [
    "local_rows",
    "distribute",
]


def local_rows(csr, partition: Partition, rank: int):
    """Extract the rows of `csr` owned by `rank` as a new `SparseCSR`.

    Purely local: no communication.  Correct for an unfinalized matrix, where
    ``ptr`` is over-allocated and only ``ncol[r]`` entries per row are valid,
    and for any `Partition` -- a contiguous split yields one run per rank, a
    cyclic one several.

    A rank owning no rows gets a matrix with zero rows, which is a perfectly
    good `SparseCSR` and still participates in collectives.

    The column dimension is left at its full extent.  Columns index the whole
    matrix -- for `SparseOrbital` the supercell -- so renumbering them here
    would destroy the only information a later halo exchange has to work with.
    """
    # Imported here: this module is imported from sparse.py, so the dependency
    # can only run the other way at call time.
    from ..sparse import SparseCSR

    # A single run keeps the common contiguous case a view rather than a gather.
    ranges = partition.range(rank)
    if len(ranges) == 1:
        selection = slice(*ranges[0])
    else:
        selection = partition.indices(rank)

    ncol = csr.ncol[selection]
    idx = array_arange(csr.ptr[selection], n=ncol)

    ptr = np.insert(np.cumsum(ncol), 0, 0).astype(np.int32)
    return SparseCSR(
        (csr._D[idx].copy(), csr.col[idx].copy(), ptr),
        shape=(partition.count(rank), csr.shape[1]),
        dim=csr.shape[2],
        dtype=csr.dtype,
    )


def _stored(csr):
    """Every stored element of `csr`, as ``(row, col, data)`` arrays.

    Correct for an unfinalized matrix: only ``ncol[r]`` entries per row are
    valid, whatever ``ptr`` has over-allocated.
    """
    idx = array_arange(csr.ptr[:-1], n=csr.ncol, dtype=np.int32)
    # array_arange concatenates the rows in order, so repeating each row index
    # ncol times lines up element for element
    rows = np.repeat(np.arange(csr.shape[0], dtype=np.int32), csr.ncol)
    return rows, csr.col[idx], csr._D[idx]


def _exchange(comm, dest, rows, cols, data):
    """Send every element to the rank in `dest`, and return what arrives.

    Elements destined for the sending rank travel through the same call.  That
    costs a local copy and removes the special case, which is worth it: a
    forgotten self-contribution is a silent wrong answer.
    """
    size = comm.size

    # Ialltoallv needs each destination's elements contiguous.
    order = np.argsort(dest, kind="stable")
    rows, cols, data = rows[order], cols[order], data[order]

    send_counts = np.bincount(dest, minlength=size).astype(np.int32)
    recv_counts = np.empty(size, dtype=np.int32)
    comm.Alltoall(send_counts, recv_counts)

    send_displ = np.insert(np.cumsum(send_counts), 0, 0)[:-1].astype(np.int32)
    recv_displ = np.insert(np.cumsum(recv_counts), 0, 0)[:-1].astype(np.int32)
    total = int(recv_counts.sum())

    recv_rows = np.empty(total, dtype=rows.dtype)
    req_rows = comm.Ialltoallv(
        [rows, (send_counts, send_displ)], [recv_rows, (recv_counts, recv_displ)]
    )
    recv_cols = np.empty(total, dtype=cols.dtype)
    req_cols = comm.Ialltoallv(
        [cols, (send_counts, send_displ)], [recv_cols, (recv_counts, recv_displ)]
    )

    # the data carries `dim` values per element, so every count and offset scales
    dim = data.shape[1]
    recv_data = np.empty((total, dim), dtype=data.dtype)
    req_data = comm.Ialltoallv(
        [np.ascontiguousarray(data), (send_counts * dim, send_displ * dim)],
        [recv_data, (recv_counts * dim, recv_displ * dim)],
    )

    # We can send all at once
    req_rows.Waitall([req_rows, req_cols, req_data])

    return recv_rows, recv_cols, recv_data


def _rebuild(csr, rows, cols, data, op: str) -> None:
    """Replace `csr`'s contents with the given elements, merging duplicates.

    Writes the CSR arrays directly rather than going through ``__setitem__``:
    the elements are sorted into place in one pass, where element-by-element
    insertion would be quadratic in the number of received entries.
    """
    nr, nc, dim = csr.shape

    if rows.size:
        # one key per element orders by row then column in a single sort
        key = rows.astype(np.int64) * nc + cols
        order = np.argsort(key, kind="stable")
        key, rows, cols, data = key[order], rows[order], cols[order], data[order]

        first = np.empty(key.size, dtype=bool)
        first[0] = True
        np.not_equal(key[1:], key[:-1], out=first[1:])
        starts = np.flatnonzero(first)

        if op == "sum":
            data = np.add.reduceat(data, starts, axis=0)
        elif op == "single":
            # "insert": the last writer of a duplicate wins
            ends = np.append(starts[1:], key.size) - 1
            data = data[ends]
        else:
            raise ValueError(
                f"rebuild received {op = } which is not one of " "sum|single"
            )
        rows, cols = rows[starts], cols[starts]
    else:
        data = np.zeros((0, dim), dtype=csr.dtype)
        cols = np.zeros(0, dtype=np.int32)

    ncol = np.bincount(rows, minlength=nr).astype(np.int32)
    csr.ncol = ncol
    csr.ptr = np.insert(np.cumsum(ncol), 0, 0).astype(np.int32)
    csr.col = np.ascontiguousarray(cols, dtype=np.int32)
    csr._D = np.ascontiguousarray(data, dtype=csr.dtype)
    csr._nnz = int(ncol.sum())
    # rows are contiguous and columns ascending within each row
    csr._finalized = True


@register_sisl_dispatch(SparseCSR, module="sisl")
def distribute(csr: SparseCSR, op: str = "insert"):
    """Make a distributed `SparseCSR` coherent, moving each element to its owner.

    `SparseCSR`'s implementation of the ``__sisl_distribute__`` protocol; call it
    through ``csr.__sisl_distribute__()`` in code that should work for any
    distributed object.

    This is the assembly step of the deferred scheme: structural changes only
    bump an epoch, and the communication happens here, once, at the point a
    consumer needs a coherent matrix.  Afterwards every rank holds exactly the
    rows its partition assigns it, and nothing else.

    Ranks may write to rows they do not own; those entries are shipped to the
    owner.  All ranks must call this together -- it is collective.

    Parameters
    ----------
    csr :
        a `SparseCSR` carrying a `Distribution` in ``_distribution``
    op :
        how to merge entries several ranks wrote to the same position.
        ``"insert"`` keeps one of them, ``"add"`` sums them -- which is what
        accumulating contributions into a matrix needs.

    Returns
    -------
    SparseCSR
        `csr` itself, assembled in place.
    """
    if op not in ("sum", "single"):
        raise ValueError(f"distribute: op must be 'sum' or 'single', got {op!r}")

    distribution = getattr(csr, "_distribution", None)
    if distribution is None:
        raise ValueError(
            "distribute: this matrix carries no distribution; attach one to "
            "`_distribution` before assembling"
        )

    partition = distribution.partition
    if partition.n != csr.shape[0]:
        raise ValueError(
            f"distribute: partition covers {partition.n} rows but the matrix "
            f"has {csr.shape[0]}"
        )

    comm = distribution.comm
    if partition.size != comm.size:
        raise ValueError(
            f"distribute: partition is for {partition.size} ranks but the "
            f"communicator has {comm.size}"
        )

    if distribution.is_assembled:
        return csr

    rows, cols, data = _stored(csr)
    if comm.size > 1:
        rows, cols, data = _exchange(comm, partition.owner(rows), rows, cols, data)

    _rebuild(csr, rows, cols, data, op)
    distribution.mark_assembled()
    return csr
