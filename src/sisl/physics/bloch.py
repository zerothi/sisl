# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

r"""Bloch's theorem
===================

Bloch's theorem is a very powerful proceduce that enables one to utilize
the periodicity of a given direction to describe the complete system.
"""
from collections.abc import Sequence

import numpy as np
from numpy import empty

import sisl._array as _a
from sisl._help import dtype_float_to_complex
from sisl._internal import set_module
from sisl.typing import KPoint

from ._bloch import bloch_unfold

__all__ = ["Bloch"]


@set_module("sisl.physics")
class Bloch:
    r""" Bloch's theorem object containing unfolding factors and unfolding algorithms

    This class is a wrapper for expanding *any* matrix from a smaller matrix cell into
    a larger, using Bloch's theorem.
    The general idea may be summarized in the following equation:

    .. math::
        \mathbf M_{K}^N =\frac1N
        \;
        \sum_{
         \substack{j=0\\
           k_j=\frac{K+j}N
         }
        }^{N-1}
        \quad
        \begin{bmatrix}
         1 & \cdots & e^{i (1-N)k_j}
         \\
         e^{i k_j} & \cdots & e^{i (2-N)k_j}
         \\
         \vdots & \ddots & \vdots
         \\
         e^{i (N-1)k_j} & \cdots & 1
        \end{bmatrix}
        \otimes
        \mathbf M_{k_j}^1.


    Parameters
    ----------
    bloch : (3,) int
       Bloch repetitions along each direction

    Examples
    --------
    >>> bloch = Bloch([2, 1, 2])
    >>> k_unfold = bloch.unfold_points([0] * 3)
    >>> M = [func(*args, k=k) for k in k_unfold]
    >>> bloch.unfold(M, k_unfold)
    """

    def __init__(self, *bloch):
        """Create `Bloch` object"""
        self._bloch = _a.arrayi(bloch).ravel()
        self._bloch = np.where(self._bloch < 1, 1, self._bloch).astype(
            np.int32, copy=False
        )
        if len(self._bloch) != 3:
            raise ValueError(self.__class__.__name__ + " requires 3 input values")
        if np.any(self._bloch < 1):
            raise ValueError(
                self.__class__.__name__ + " requires all unfoldings to be larger than 0"
            )

    def __len__(self):
        """Return unfolded size"""
        return np.prod(self.bloch)

    def __str__(self):
        """Representation of the Bloch model"""
        B = self._bloch
        return f"{self.__class__.__name__}{{{B[0]}, {B[1]}, {B[2]}}}"

    def __repr__(self):
        """Representation of the Bloch model"""
        B = self._bloch
        cls = self.__class__
        return f"<{cls.__module__}.{cls.__name__}{{{B[0]}, {B[1]}, {B[2]}}}>"

    @property
    def bloch(self):
        """Number of Bloch expansions along each lattice vector"""
        return self._bloch

    def unfold_points(self, k: KPoint):
        r"""Return a list of k-points to be evaluated for this objects unfolding

        The k-point `k` is with respect to the unfolded geometry.
        The return list of `k` points are the k-points required to be sampled in the
        folded geometry.

        Parameters
        ----------
        k : (3,) of float
           k-point evaluation corresponding to the unfolded unit-cell

        Returns
        -------
        numpy.ndarray
            a list of ``np.prod(self.bloch)`` k-points used for the unfolding
        """
        k = _a.arrayd(k)

        # Create expansion points
        B = self._bloch
        unfold = _a.emptyd([B[2], B[1], B[0], 3])
        # Use B-casting rules (much simpler)
        unfold[:, :, :, 0] = (_a.aranged(B[0]).reshape(1, 1, -1) + k[0]) / B[0]
        unfold[:, :, :, 1] = (_a.aranged(B[1]).reshape(1, -1, 1) + k[1]) / B[1]
        unfold[:, :, :, 2] = (_a.aranged(B[2]).reshape(-1, 1, 1) + k[2]) / B[2]
        # Back-transform shape
        return unfold.reshape(-1, 3)

    def __call__(self, func, k: KPoint, *args, **kwargs):
        """Return a functions return values as the Bloch unfolded equivalent according to this object

        Calling the `Bloch` object is a shorthand for the manual use of the `Bloch.unfold_points` and `Bloch.unfold`
        methods.

        This call structure is a shorthand for:

        >>> bloch = Bloch([2, 1, 2])
        >>> k_unfold = bloch.unfold_points([0] * 3)
        >>> M = [func(*args, k=k) for k in k_unfold]
        >>> bloch.unfold(M, k_unfold)

        Notes
        -----
        The function passed *must* have a keyword argument ``k``.

        Parameters
        ----------
        func : callable
           method called which returns a matrix.
        k : (3, ) of float
           k-point to be unfolded
        *args : list
           arguments passed directly to `func`
        **kwargs : dict
           keyword arguments passed directly to `func`

        Returns
        -------
        numpy.ndarray
            unfolded Bloch matrix
        """
        K_unfold = self.unfold_points(k)
        M0 = func(*args, k=K_unfold[0, :], **kwargs)
        shape = (K_unfold.shape[0], M0.shape[0], M0.shape[1])
        M = empty(shape, dtype=dtype_float_to_complex(M0.dtype))
        M[0] = M0
        del M0
        for i in range(1, K_unfold.shape[0]):
            M[i] = func(*args, k=K_unfold[i, :], **kwargs)
        return bloch_unfold(self._bloch, K_unfold, M)

    def unfold(self, M, k_unfold: Sequence[KPoint]):
        r"""Unfold the matrix list of matrices `M` into a corresponding k-point (unfolding k-points are `k_unfold`)

        Parameters
        ----------
        M : (:, :, :)
            an ``*``-N-M matrix used for unfolding
        k_unfold : (:, 3) of float
            unfolding k-points as returned by `Bloch.unfold_points`

        Returns
        -------
        numpy.ndarray
            unfolded matrix of size ``M[0].shape * k_unfold.shape[0] ** 2``
        """
        if isinstance(M, (list, tuple)):
            M = np.stack(M)
        return bloch_unfold(self._bloch, k_unfold, M)
