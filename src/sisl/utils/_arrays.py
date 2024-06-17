# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""Special handlers for arrays """
import numpy as np

__all__ = ["batched_indices"]


def batched_indices(ref, y, axis=-1, atol=1e-8, batch_size=200, diff_func=None):
    """Locate `x` in `ref` by examining ``np.abs(diff_func(ref - y)) <= atol``

    This method is necessary for very large groups of data since the ``ref-y``
    calls will use broad-casting to create very large memory chunks.

    This method will allow broad-casting arrays up to a size of `batch_size`.

    The memory is calculating using ``np.prod(ref.shape) / ref.shape[axis] * n``
    such that ``n`` chunks of `y` is using the `batch_size` MB of memory.
    ``n`` will minimally be 1, regardless of `batch_size`.

    Parameters
    ----------
    ref : array_like
       reference array where we wish to locate the indices of `y` in
    y : array_like of 1D or 2D
       array to locate in `ref`. For 2D arrays and `axis` not None,
    axis : int or None, optional
       which axis to do a logical reduction along, if `None` it means that they
       are 1D arrays and no axis will be reduced, i.e. same as ``ref.ravel() - y.reshape(-1, 1)``
       but retaining indices of the same dimension as `ref`.
    atol : float or array_like, optional
       absolute tolerance for comparing values, for array_like values it must have the same
       length as ``ref.shape[axis]``
    batch_size : float, optional
       maximum size of broad-casted array. Internal algorithms will determine
       the chunk size of `y`
    diff_func : callable, optional
       function to post-process the difference values, defaults to do nothing.
       Should have an interface ``def diff_func(diff)``

    Returns
    -------
    tuple of ndarray:
       the indices for each `ref` dimension that matches the values in `y`.
       the returned indices behave like `numpy.nonzero`.
    """
    # first ensure the arrays are numpy arrays
    ref = np.asarray(ref)
    y = np.asarray(y)
    # atol may be an array of the same dimension as the axis
    atol = np.asarray(atol)

    # determine best chunk-size of `y`
    n = batch_size / (np.prod(ref.shape) * ref.itemsize / 1024**2)
    n = max(1, n)

    if diff_func is None:

        def diff_func(d):
            """Do nothing"""
            return d

    def yield_cs(n, size):
        n = max(1, int(n))
        i = 0
        for j in range(n, size, n):
            yield range(i, j)
            i += n
        if i < size:
            yield range(i, size)

    indices = []

    # determine the batch size
    if axis is None:
        if atol.size != 1:
            raise ValueError(
                f"batched_indices: for 1D comparisons atol can only be a single number."
            )
        if y.ndim != 1:
            raise ValueError(
                f"batched_indices: axis is None and y.ndim != 1 ({y.ndim}). For ravel comparisons the "
                "dimensionality of y must be 1D."
            )

        # a 1D array comparison
        # we might as well ravel y (here to ensure we do not
        # overwrite the input y's shape
        n = min(n, y.size)

        # create shapes
        y = np.expand_dims(y, tuple(range(ref.ndim)))
        ref = np.expand_dims(ref, ref.ndim)

        # b-cast size is
        for idx in yield_cs(n, y.size):
            idx = (np.abs(diff_func(ref - y[..., idx])) <= atol).nonzero()[:-1]
            indices.append(idx)

        # concatenate each indices array
        return tuple(map(np.concatenate, zip(*indices)))

    # Axis is specified
    # y must have 2 dimensions (or 1 with the same size as ref.shape[axis])
    if y.ndim == 1:
        if y.size != ref.shape[axis]:
            raise ValueError(
                f"batched_indices: when y is a single value it must have same length as ref.shape[axis]"
            )
        y = y.reshape(1, -1)
    elif y.ndim == 2:
        if y.shape[1] != ref.shape[axis]:
            raise ValueError(
                f"batched_indices: the comparison axis of y (y[0, :]) should have the same length as ref.shape[axis]"
            )
    else:
        raise ValueError(f"batched_indices: y should be either 1D or 2D")

    # create shapes, since we change the shapes we must fix the axis'
    if axis < 0:
        axis = axis + ref.ndim

    y = np.expand_dims(y, tuple(range(ref.ndim - 1)))
    y = np.moveaxis(y, -1, axis)
    ref = np.expand_dims(ref, ref.ndim)

    if atol.size > 1:
        if atol.size != ref.shape[axis]:
            raise ValueError(
                f"batched_indices: atol size does not match the axis {axis} for ref argument."
            )
        atol = np.expand_dims(atol.ravel(), tuple(range(ref.ndim - 1)))
        atol = np.moveaxis(atol, -1, axis)

    # b-cast size is
    for idx in yield_cs(n, y.shape[-1]):
        idx = np.logical_and.reduce(
            np.abs(diff_func(ref - y[..., idx])) <= atol, axis=axis
        ).nonzero()[:-1]
        indices.append(idx)

    # concatenate each indices array
    return tuple(map(np.concatenate, zip(*indices)))
