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
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy import empty

import sisl._array as _a
from sisl._help import dtype_float_to_complex
from sisl._internal import set_module
from sisl.typing import KPoint

if TYPE_CHECKING:
    from sisl_toolbox.btd import BlockArray

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


    Only parts of the unfolded matrix may be calculated, see the ``parts``
    argument of `unfold` and `__call__`.
    The unfolded matrix is then returned as a `~sisl_toolbox.btd.BlockArray`
    containing only the blocks coupling the requested unit-cells.

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

    Only calculate the blocks coupling the unit-cells ``[0, 0, 0]`` and ``[1, 0, 1]``
    (and their self-couplings).

    >>> bloch = Bloch([2, 1, 2])
    >>> M = bloch(func, [0] * 3, parts=[[0, 0, 0], [1, 0, 1]])
    >>> M.block_indexer[0, 3] # coupling between the two cells
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
        self, func, k: KPoint, *args, parts: Optional[npt.ArrayLike] = None, **kwargs
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
        parts :
           only calculate the blocks coupling these unit-cells, see `unfold`.
        **kwargs :
           keyword arguments passed directly to `func`

        Returns
        -------
        numpy.ndarray or sisl_toolbox.btd.BlockArray
            unfolded Bloch matrix, a `BlockArray` if `parts` is specified
        """
        K_unfold = self.unfold_points(k)
        M0 = func(*args, k=K_unfold[0, :], **kwargs)
        shape = (K_unfold.shape[0], M0.shape[0], M0.shape[1])
        M = empty(shape, dtype=dtype_float_to_complex(M0.dtype))
        M[0] = M0
        del M0
        for i in range(1, K_unfold.shape[0]):
            M[i] = func(*args, k=K_unfold[i, :], **kwargs)
        if parts is not None:
            return self._unfold_parts(M, K_unfold, parts)
        return bloch_unfold(self._bloch, K_unfold, M, self.finalize)

    def unfold(
        self,
        M,
        k_unfold: Sequence[KPoint],
        parts: Optional[npt.ArrayLike] = None,
    ) -> Union[np.ndarray, BlockArray]:
        r"""Unfold the matrix list of matrices `M` into a corresponding k-point (unfolding k-points are `k_unfold`)

        The unfolded matrix consists of ``len(self) x len(self)`` blocks, each
        of shape ``M[0].shape``. Block ``I`` corresponds to the unit-cell with
        indices ``(i0, i1, i2)`` (``0 <= ix < self.bloch[x]``) with
        ``I = (i2 * self.bloch[1] + i1) * self.bloch[0] + i0``.

        Parameters
        ----------
        M : (:, :, :)
            an ``*``-N-M matrix used for unfolding
        k_unfold :
            unfolding k-points as returned by `Bloch.unfold_points`
        parts :
            only calculate the blocks coupling these unit-cells (all pairs of
            the unit-cells, including the self-couplings).
            Either an ``(N, 3)`` array of unit-cell indices, or a `tuple` of up to 3
            arrays (one per lattice vector) that are broadcasted against each other,
            e.g. ``(np.arange(2).reshape(-1, 1), np.arange(2), [1])``.
            Lattice vectors missing in the `tuple` take all unit-cells along that
            direction (like `numpy` indexing), e.g. ``([-2],)`` is all unit-cells
            with the second last index along the first lattice vector.
            Negative indices count from the end (like `numpy` indexing),
            i.e. ``-1`` is the last unit-cell along that lattice vector.
            Duplicate unit-cells are ignored.

        Returns
        -------
        numpy.ndarray
            unfolded matrix of size ``M[0].shape * k_unfold.shape[0] ** 2``
            (if `parts` is None)
        sisl_toolbox.btd.BlockArray
            unfolded matrix of the same shape, but only with the blocks
            of `parts` defined, all other blocks are zero (if `parts` is specified).
            The `finalize` flag has no effect.
        """
        if isinstance(M, (list, tuple)):
            M = np.stack(M)
        if parts is not None:
            return self._unfold_parts(M, k_unfold, parts)
        return bloch_unfold(self._bloch, k_unfold, M, self.finalize)

    def _parts_cells(self, parts) -> np.ndarray:
        """Convert `parts` into a unique ``(N, 3)`` array of unit-cell indices"""
        if isinstance(parts, tuple):
            if len(parts) > 3:
                raise ValueError(
                    f"{self.__class__.__name__} parts as a tuple accepts at most "
                    "one array per lattice vector (3)"
                )
            if len(parts) == 0:
                cells = _a.emptyi([1, 0])
            else:
                cells = np.stack(
                    [a.ravel() for a in np.broadcast_arrays(*map(_a.asarrayi, parts))],
                    axis=-1,
                )
            # missing lattice vectors take all unit-cells (like numpy indexing)
            for b in self._bloch[len(parts) :]:
                cells = np.concatenate(
                    [
                        np.repeat(cells, b, axis=0),
                        np.tile(_a.arangei(b), len(cells)).reshape(-1, 1),
                    ],
                    axis=1,
                )
        else:
            cells = _a.asarrayi(parts)
            if cells.ndim not in (1, 2) or cells.shape[-1] != 3:
                raise ValueError(
                    f"{self.__class__.__name__} parts must be an (N, 3) array of "
                    "unit-cell indices"
                )
            cells = cells.reshape(-1, 3)
        if np.any(cells < -self._bloch) or np.any(cells >= self._bloch):
            raise ValueError(
                f"{self.__class__.__name__} parts has unit-cell indices outside "
                f"the Bloch expansion {self._bloch.tolist()}"
            )
        # wrap negative indices (counting from the end)
        return np.unique(cells % self._bloch, axis=0)

    def _unfold_parts(self, M: np.ndarray, k_unfold, parts) -> BlockArray:
        r"""Unfold only the blocks coupling the unit-cells in `parts`

        Block :math:`(I, J)` only depends on the unit-cell displacement
        :math:`\mathbf d = \mathbf c_I - \mathbf c_J`:

        .. math::

            \mathbf M_{IJ} = \frac1N \sum_j e^{i 2\pi \mathbf k_j\cdot\mathbf d} \mathbf M_{\mathbf k_j}

        so only the unique displacements are calculated (a single matrix product).
        """
        from sisl_toolbox.btd import BlockArray

        B = self._bloch
        N = len(self)
        if M.ndim != 3 or M.shape[0] != N:
            raise ValueError(
                f"{self.__class__.__name__}.unfold requires M to have shape "
                f"({N}, :, :), got {M.shape}"
            )
        k_unfold = _a.asarrayd(k_unfold).reshape(N, 3)
        cells = self._parts_cells(parts)
        # block index of the cells
        idx = (cells[:, 2] * B[1] + cells[:, 1]) * B[0] + cells[:, 0]

        # all pairs of cells, and their unique displacements
        disp = (cells[:, None, :] - cells[None, :, :]).reshape(-1, 3)
        disp, inv = np.unique(disp, axis=0, return_inverse=True)

        _, n0, n1 = M.shape
        dtype = dtype_float_to_complex(M.dtype)
        phases = np.exp(2j * np.pi * (disp @ k_unfold.T)) / N
        Md = (phases.astype(dtype, copy=False) @ M.reshape(N, -1)).reshape(-1, n0, n1)

        ba = BlockArray([np.full(N, n0), np.full(N, n1)], dtype=dtype)
        grid = ba._grid
        nc = len(idx)
        for i, I in enumerate(idx):
            for j, J in enumerate(idx):
                # copy, the blocks must be independent (in-place operations)
                grid[I, J] = Md[inv[i * nc + j]].copy()
        return ba

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
        from sisl_toolbox.btd import BlockArray

        if isinstance(M, BlockArray):
            # parts are always fully calculated
            return M
        return bloch_unfold_finalize(self._bloch, M)
