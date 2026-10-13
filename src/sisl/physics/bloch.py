# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

r"""Bloch's theorem
===================

Bloch's theorem is a very powerful procedure that enables one to utilize
the periodicity of a given direction to describe the complete system.
"""
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from numpy import empty

import sisl._array as _a
from sisl._help import dtype_float_to_complex
from sisl._internal import set_module
from sisl.typing import KPoint

from ._bloch import bloch_unfold, bloch_unfold_finalize

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


    Notes
    -----
    When doing a Brillouin-zone average of Bloch-expansions (e.g.
    self-energies) one can get better performance by using
    ``finalize=False`` and then call a subsequent ``bloch.unfold_finalize(M)``.

    Only a sub-matrix of the unfolded matrix may be calculated, see the ``indices``
    argument of `unfold` and `__call__`. This greatly reduces the memory
    requirements when only few elements are needed, e.g. the elements
    on the edges of a large Bloch expansion.


    Parameters
    ----------
    bloch : (3,) int
        Bloch repetitions along each direction
    finalize :
        whether the unfold routine also copies out the
        Toeplitz matrix. If false, the user should call
        ``bloch.unfold_finalize(M)`` on the returned matrix
        to ensure it is complete, see examples.

    Examples
    --------
    Auto-use finalize of the Toeplitz matrix.

    >>> bloch = Bloch([2, 1, 2])
    >>> k_unfold = bloch.unfold_points([0] * 3)
    >>> M = [func(*args, k=k) for k in k_unfold]
    >>> M = bloch.unfold(M, k_unfold)

    Manual finalization of the Toeplitz matrix.

    >>> bloch = Bloch([2, 1, 2], finalize=False)
    >>> k_unfold = bloch.unfold_points([0] * 3)
    >>> M = [func(*args, k=k) for k in k_unfold]
    >>> M = bloch.unfold(M, k_unfold)
    >>> if not bloch.finalize:
    ...    bloch.unfold_finalize(M)

    Only calculate the sub-matrix ``M[np.ix_(indices, indices)]`` of the unfolded matrix.

    >>> bloch = Bloch([2, 1, 2])
    >>> M = bloch(func, [0] * 3, indices=[0, 1, -2, -1])
    """

    def __init__(self, *bloch, finalize: bool = True):
        """Create `Bloch` object"""
        self._bloch = _a.arrayi(bloch).ravel()  # ensure 1D
        self._bloch = np.where(self._bloch < 1, 1, self._bloch).astype(
            np.int32, copy=False
        )
        if len(self._bloch) != 3:
            raise ValueError(self.__class__.__name__ + " requires 3 input values")
        if np.any(self._bloch < 1):
            raise ValueError(
                self.__class__.__name__ + " requires all unfoldings to be larger than 0"
            )
        self.finalize = finalize

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
        k :
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

    def __call__(
        self,
        func,
        k: KPoint,
        *args,
        indices: Optional[
            Union[npt.ArrayLike, tuple[npt.ArrayLike, npt.ArrayLike]]
        ] = None,
        **kwargs,
    ):
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
        k :
           k-point to be unfolded
        *args :
           arguments passed directly to `func`
        indices :
           only calculate this sub-matrix of the unfolded matrix, see `unfold`.
           Only the needed elements of the matrices returned by `func` are stored.
        **kwargs :
           keyword arguments passed directly to `func`

        Returns
        -------
        numpy.ndarray
            unfolded Bloch matrix (or the sub-matrix if `indices` is specified)
        """
        K_unfold = self.unfold_points(k)
        M0 = func(*args, k=K_unfold[0, :], **kwargs)
        n0, n1 = M0.shape
        if indices is not None:
            rows, cols, R, C = self._indices(indices, n0, n1)
            # only store the local elements required
            M0 = M0[np.ix_(R, C)]
        shape = (K_unfold.shape[0], M0.shape[0], M0.shape[1])
        M = empty(shape, dtype=dtype_float_to_complex(M0.dtype))
        M[0] = M0
        del M0
        for i in range(1, K_unfold.shape[0]):
            if indices is None:
                M[i] = func(*args, k=K_unfold[i, :], **kwargs)
            else:
                M[i] = func(*args, k=K_unfold[i, :], **kwargs)[np.ix_(R, C)]
        if indices is not None:
            return self._unfold_indices(M, K_unfold, rows, cols, n0, n1, R, C)
        return bloch_unfold(self._bloch, K_unfold, M, self.finalize)

    def unfold(
        self,
        M,
        k_unfold: Sequence[KPoint],
        indices: Optional[
            Union[npt.ArrayLike, tuple[npt.ArrayLike, npt.ArrayLike]]
        ] = None,
    ) -> np.ndarray:
        r"""Unfold the matrix list of matrices `M` into a corresponding k-point (unfolding k-points are `k_unfold`)

        Parameters
        ----------
        M : (:, :, :)
            an ``*``-N-M matrix used for unfolding
        k_unfold :
            unfolding k-points as returned by `Bloch.unfold_points`
        indices :
            only calculate this sub-matrix of the unfolded matrix (indices
            in the unfolded matrix):

            - an array: the sub-matrix ``unfold(M, k_unfold)[np.ix_(indices, indices)]``
            - a `tuple` ``(rows, cols)``: the sub-matrix ``unfold(M, k_unfold)[np.ix_(rows, cols)]``

            Negative indices count from the end. Only the elements of the
            unique displacements between the Bloch unit-cells are calculated,
            so the memory scales with the size of the sub-matrix.

        Returns
        -------
        numpy.ndarray
            unfolded matrix of size ``M[0].shape * k_unfold.shape[0] ** 2``,
            or the sub-matrix if `indices` is specified (the `finalize` flag has no effect).
        """
        if isinstance(M, (list, tuple)):
            M = np.stack(M)
        if indices is not None:
            _, n0, n1 = M.shape
            rows, cols, R, C = self._indices(indices, n0, n1)
            return self._unfold_indices(
                M[:, R][:, :, C], k_unfold, rows, cols, n0, n1, R, C
            )
        return bloch_unfold(self._bloch, k_unfold, M, self.finalize)

    def _indices(self, indices, n0: int, n1: int):
        """Convert `indices` to (non-negative) rows and columns of the unfolded matrix, and the local rows and columns (of a block) required"""
        N = len(self)

        def conv(idx, n):
            idx = np.asarray(idx, dtype=np.intp).ravel()
            if np.any(idx < -N * n) or np.any(idx >= N * n):
                raise ValueError(
                    f"{self.__class__.__name__} indices are outside the unfolded matrix"
                )
            return idx % (N * n)

        if isinstance(indices, tuple):
            if len(indices) != 2:
                raise ValueError(
                    f"{self.__class__.__name__} indices as a tuple must be (rows, cols)"
                )
            rows = conv(indices[0], n0)
            cols = conv(indices[1], n1)
        else:
            if n0 != n1:
                raise ValueError(
                    f"{self.__class__.__name__} indices as a single array requires "
                    "square matrices, use (rows, cols)"
                )
            rows = cols = conv(indices, n0)
        return rows, cols, np.unique(rows % n0), np.unique(cols % n1)

    def _unfold_indices(
        self,
        m: np.ndarray,
        k_unfold,
        rows: np.ndarray,
        cols: np.ndarray,
        n0: int,
        n1: int,
        R: np.ndarray,
        C: np.ndarray,
    ) -> np.ndarray:
        r"""Unfold only the sub-matrix ``[rows, cols]`` of the unfolded matrix

        Block :math:`(I, J)` only depends on the unit-cell displacement
        :math:`\mathbf d = \mathbf c_I - \mathbf c_J`:

        .. math::

            \mathbf M_{IJ} = \frac1N \sum_j e^{i 2\pi \mathbf k_j\cdot\mathbf d} \mathbf M_{\mathbf k_j}

        Each unique displacement is calculated once (in chunks, a single matrix product)
        for the local elements ``m[:, R, C]``, and copied into the requested elements of all
        the unit-cell pairs with that displacement.

        Parameters
        ----------
        m : (N, len(R), len(C))
            matrices at the unfolding k-points, only the local rows `R` and
            columns `C` (of a block) are present
        rows, cols :
            requested (non-negative) indices in the unfolded matrix
        n0, n1 :
            block size of the (non-sliced) matrices
        R, C :
            sorted local rows and columns of a block present in `m`
        """
        B = self._bloch
        N = len(self)
        if m.ndim != 3 or m.shape[0] != N:
            raise ValueError(
                f"{self.__class__.__name__}.unfold requires M to have shape "
                f"({N}, :, :), got {m.shape}"
            )
        k_unfold = _a.asarrayd(k_unfold).reshape(N, 3)
        dtype = dtype_float_to_complex(m.dtype)
        nR, nC = len(R), len(C)
        m = m.reshape(N, -1)

        def split(blk, loc):
            """Unique blocks, and for each the output indices and local indices"""
            ublk, inv = np.unique(blk, return_inverse=True)
            order = np.argsort(inv, kind="stable")
            out = np.split(order, np.cumsum(np.bincount(inv))[:-1])
            return ublk, out, [loc[o] for o in out]

        def blk2cell(blk):
            return np.stack(
                [blk % B[0], blk // B[0] % B[1], blk // (B[0] * B[1])], axis=-1
            )

        r_blk, r_out, r_loc = split(rows // n0, np.searchsorted(R, rows % n0))
        c_blk, c_out, c_loc = split(cols // n1, np.searchsorted(C, cols % n1))

        # unique displacements of all block pairs
        disp = blk2cell(r_blk)[:, None, :] - blk2cell(c_blk)[None, :, :]
        disp, dinv = np.unique(disp.reshape(-1, 3), axis=0, return_inverse=True)
        dinv = dinv.ravel()
        # loop pairs ordered by displacements
        pairs = np.argsort(dinv, kind="stable")
        dsorted = dinv[pairs]
        nc_blk = len(c_blk)

        out = np.zeros([len(rows), len(cols)], dtype=dtype)

        # limit the temporary memory of the displacement matrices
        chunk = max(1, 2**22 // max(1, nR * nC))
        for d0 in range(0, len(disp), chunk):
            d1 = min(d0 + chunk, len(disp))
            phases = np.exp(2j * np.pi * (disp[d0:d1] @ k_unfold.T)) / N
            Md = (phases.astype(dtype, copy=False) @ m).reshape(-1, nR, nC)
            p0, p1 = np.searchsorted(dsorted, [d0, d1])
            for p in pairs[p0:p1]:
                i, j = divmod(p, nc_blk)
                out[np.ix_(r_out[i], c_out[j])] = Md[dinv[p] - d0][
                    np.ix_(r_loc[i], c_loc[j])
                ]
        return out

    def unfold_finalize(self, M):
        """Copy out the Toeplitz matrix from the sub-calculated elements.

        This is necessary to be called if the `Bloch` object has
        been initialized with ``Bloch(..., finalize=False)``.
        Otherwise it should not be used.

        Parameters
        ----------
        M :
            the full matrix, where only the Toeplitz elements are filled.

        Returns
        -------
        the full matrix with copied out elements.
        """
        return bloch_unfold_finalize(self._bloch, M)
