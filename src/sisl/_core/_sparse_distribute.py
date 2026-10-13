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

The rows are divided by a `sisl.mpi.Distribution`, which is agnostic of what
it distributes; ``dist(csr)`` forwards to ``csr.__sisl_distribute__(dist)``,
which lands in `distribute` here.

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

Assembly is reached through ``__sisl_distribute__``, the protocol any
distributable object implements, so a consumer can make what it was handed
coherent without knowing what it is.  `distribute` is `SparseCSR`'s
implementation of that protocol; a distributed `Grid`, or a set of k-points,
would supply its own.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sisl._array import array_arange
from sisl._ufuncs import register_sisl_dispatch
from sisl.mpi._distribute import Distribution, DistributionState

from .sparse import SparseCSR

__all__ = [
    "local_rows",
    "distribute",
]


def local_rows(csr, dist: Distribution, rank: Optional[int] = None):
    """Extract the rows of `csr` owned by `rank` as a new `SparseCSR`.

    Purely local: no communication.  Correct for an unfinalized matrix, where
    ``ptr`` is over-allocated and only ``ncol[r]`` entries per row are valid,
    and for any `Distribution` -- a contiguous split yields one run per rank, a
    cyclic one several.

    `rank` defaults to the rank of this process in ``dist.comm``.

    A rank owning no rows gets a matrix with zero rows, which is a perfectly
    good `SparseCSR` and still participates in collectives.

    The column dimension is left at its full extent.  Columns index the whole
    matrix -- for `SparseOrbital` the supercell -- so renumbering them here
    would destroy the only information a later halo exchange has to work with.
    """
    # A single run keeps the common contiguous case a view rather than a gather.
    ranges = dist.range(rank)
    if len(ranges) == 1:
        selection = slice(*ranges[0])
    else:
        selection = dist.indices(rank)

    ncol = csr.ncol[selection]
    idx = array_arange(csr.ptr[selection], n=ncol)

    ptr = np.insert(np.cumsum(ncol), 0, 0)
    return SparseCSR(
        (csr._D[idx].copy(), csr.col[idx].copy(), ptr),
        shape=(dist.count(rank), csr.shape[1]),
        dim=csr.shape[2],
        dtype=csr.dtype,
    )


def _stored(csr):
    """Every stored element of `csr`, as ``(row, col, data)`` arrays.

    Correct for an unfinalized matrix: only ``ncol[r]`` entries per row are
    valid, whatever ``ptr`` has over-allocated.
    """
    idx = array_arange(csr.ptr[:-1], n=csr.ncol)
    # array_arange concatenates the rows in order, so repeating each row index
    # ncol times lines up element for element
    rows = np.repeat(np.arange(csr.shape[0], dtype=idx.dtype), csr.ncol)
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
        cols = np.zeros(0, dtype=cols.dtype)

    ncol = np.bincount(rows, minlength=nr).astype(rows.dtype, copy=False)
    csr.ncol = ncol
    csr.ptr = np.insert(np.cumsum(ncol), 0, 0)
    csr.col = np.ascontiguousarray(cols)
    csr._D = np.ascontiguousarray(data, dtype=csr.dtype)
    csr._nnz = int(ncol.sum())
    # rows are contiguous and columns ascending within each row
    csr._finalized = True


@register_sisl_dispatch(SparseCSR, module="sisl")
def distribute(csr: SparseCSR, dist: Optional[Distribution] = None, op: str = "single"):
    """Distribute the rows of `csr` according to `dist`, moving each element to its owner.

    `SparseCSR`'s implementation of the ``__sisl_distribute__`` protocol; call it
    through ``dist(csr)`` in code that should work for any distributable object.

    This is the assembly step of the deferred scheme: structural changes only
    bump an epoch, and the communication happens here, once, at the point a
    consumer needs a coherent matrix.  Afterwards every rank holds exactly the
    rows `dist` assigns it, and nothing else.  The matrix keeps its full shape.

    Ranks may write to rows they do not own; those entries are shipped to the
    owner.  The same holds when `dist` differs from the distribution the matrix
    was previously assembled with: every stored element is shipped to its new
    owner.  All ranks must call this together -- it is collective.

    Parameters
    ----------
    csr :
        the matrix to distribute, it is changed in place
    dist :
        the distribution of the rows.  Defaults to the distribution `csr` was
        last distributed with.
    op :
        how to merge entries several ranks wrote to the same position.
        ``"single"`` keeps one of them, ``"sum"`` sums them -- which is what
        accumulating contributions into a matrix needs.

    Returns
    -------
    SparseCSR
        `csr` itself, assembled in place.
    """
    if op not in ("sum", "single"):
        raise ValueError(f"distribute: op must be 'sum' or 'single', got {op!r}")

    state = getattr(csr, "_dist_state", None)
    if dist is None:
        if state is None:
            raise ValueError(
                "distribute: this matrix carries no distribution; pass one "
                "through `dist`"
            )
        dist = state.distribution

    if dist.n != csr.shape[0]:
        raise ValueError(
            f"distribute: distribution covers {dist.n} rows but the matrix "
            f"has {csr.shape[0]}"
        )

    if state is None or state.distribution != dist:
        # A new layout invalidates whatever was assembled before.
        state = DistributionState(dist)
        csr._dist_state = state
    elif state.is_assembled:
        return csr

    comm = dist.comm
    rows, cols, data = _stored(csr)
    if comm.size > 1:
        rows, cols, data = _exchange(comm, dist.owner(rows), rows, cols, data)

    _rebuild(csr, rows, cols, data, op)
    state.mark_assembled()
    return csr
