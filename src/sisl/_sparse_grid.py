# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import Optional, Union

import numpy as np
from scipy.sparse import issparse, spmatrix

from sisl import Grid, Lattice, SparseCSR
from sisl.physics import Overlap

from ._sparse_grid_ops import (
    reduce_grid_matvec_multiply,
    reduce_grid_matvecs_multiply,
    reduce_grid_sum,
    reduce_sc_products,
    reduce_sc_products_multicoeffs,
    reduce_sc_products_multicoeffs_sparse_denseindex,
)
from .physics._phase import phase_rsc


class Sparse4DGrid:
    """Stores information on a 3D grid with an extra sparse dimension.

    The information is stored as a matrix where the rows are the raveled indices of the
    grid and the columns represent the extra dimension. The sparsity of the matrix is
    handled by the usual methods.
    """

    def __init__(self, grid_shape, *args, geometry=None, lattice=None, **kwargs):
        self.grid_shape = tuple(grid_shape)

        self.geometry = geometry
        if geometry is not None:
            self.lattice = geometry.lattice
        else:
            self.lattice = lattice

        self._csr = SparseCSR(*args, **kwargs)

    def _transposed_grid_indices(self, *axes):
        """Function that returns the transposed raveled indices.

        Might be useful to reorder rows in the CSR matrix that stores the grid.

        Not used for now.
        """
        csr = self._csr

        # Transform the indices
        old_indices = np.arange(csr.shape[0], dtype=np.int32)
        old_indices_g = old_indices.reshape(self.grid_shape).ravel()

        new_indices_g = old_indices_g.transpose(*axes)
        new_indices = new_indices_g.ravel()
        new_shape = new_indices_g.shape

        return new_indices, new_shape

    def reduce_dimension(
        self, weights: Optional[np.ndarray] = None, reduce_grid: tuple[int, ...] = ()
    ) -> Union[Grid, np.ndarray]:
        """Reduces the extra dimension on the grid, possibly applying weights.

        It can also reduce grid axes on the same go. See the `reduce_grid` parameter.

        Parameters
        ----------
        weights:
            Array of weights for the columns. The first dimension of this array must have length
            equal to the extra dimension of this grid object.

            The array might optionally have a second dimension,
            which will also be reflected as an extra dimension in the output grid.
        reduce_grid:
            Axes that you wish to reduce on the grid. Along these axes, the grid values will be
            accumulated. This is very useful to reduce memory footprint (and probably also
            computation time) if your final goal is to reduce over the grid axes, because with
            this approach the grid axes are reduced on the fly (instead of first constructing the
            full grid and then reducing).

            This might be the difference between being able to perform the calculation or not.

        Returns
        -------
        Union[Grid, np.ndarray]
            The reduced grid. If the weights array has more than one dimension, it will be a
            numpy array. Otherwise it will be a sisl.Grid object.
        """
        # Assess if we are dealing with multiple weights
        multi_weights = weights is not None and (
            weights.ndim == 2 and weights.shape[1] != 1
        )
        if not multi_weights and weights is not None:
            weights = weights.ravel()

        # If we don't need to reduce the grid axes, we use scipy functionality
        if len(reduce_grid) == 0:
            csr = self._csr.tocsr()
            if weights is None:
                reduced = csr.sum(axis=1)
            else:
                # Perform matrix-vector multiplication to reduce the extra dimension.
                reduced = csr @ weights

            if multi_weights:
                grid = reduced.reshape(*self.grid_shape, -1)
            else:
                # Create a new grid and set the result of the operation as the values
                grid = Grid(
                    self.grid_shape, geometry=self.geometry, dtype=reduced.dtype
                )
                del grid.grid
                grid.grid = reduced.reshape(self.grid_shape)
        # If we wish to reduce the axes, we use our own functionality that reduces the grid
        # on the fly, because it is faster and uses much less memory than reducing a posteriori.
        else:
            csr = self._csr

            # Find out the reduced shape, and the reduce factor. The reduced factor is the number
            # by which we have to divide the index to get the reduced index.
            reduced_shape = list(self.grid_shape)
            reduce_factor = 1

            if len(reduce_grid) > 3:
                raise ValueError(
                    f"There are only 3 grid axes, it is not possible to reduce axes {reduce_grid}"
                )

            # Check if we are reducing the last dimensions. Otherwise we will need to transpose the grid to
            # put the dimension to reduce on the last axis. Note that we will not do that explicilty, transposing
            # is done inside the cython routines (per index) to reduce memory footprint.
            need_transpose = False
            sorted_reduce_axes = np.sort(reduce_grid)
            if len(reduce_grid) > 0:
                need_transpose = sorted_reduce_axes[-1] != 2
            if len(reduce_grid) > 1:
                need_transpose = need_transpose or sorted_reduce_axes[-2] != 1
            if len(reduce_grid) > 2:
                need_transpose = need_transpose or sorted_reduce_axes[0] != 0

            # Calculate the reducing factor and the new reduced shape.
            for reduce_axis in sorted_reduce_axes:
                reduce_factor *= self.grid_shape[reduce_axis]
                reduced_shape[reduce_axis] = 1

            # Prepare the quantities needed to transpose.
            grid_shape = np.array(self.grid_shape, dtype=np.int32)
            if need_transpose:
                not_reduced_axes = np.sort(tuple(set([0, 1, 2]) - set(reduce_grid)))
                new_axes_order = np.array(
                    [*not_reduced_axes, *reduce_grid], dtype=np.int32
                )
            else:
                # A dummy array to pass to cython functions so that they don't complain
                new_axes_order = np.zeros(0, dtype=np.int32)

            # Determine the datatype of the output
            if weights is not None:
                dtype = np.result_type(weights, csr.data)
            else:
                dtype = csr.dtype

            # If we do not have multiple weights, we store it in a sisl Grid, otherwise
            # we store it in a raw numpy array, since sisl.Grid doesn't support extra
            # dimensions.
            if not multi_weights:
                grid = Grid(reduced_shape, geometry=self.geometry, dtype=dtype)
                out = grid.grid.ravel()
            else:
                grid = np.zeros((*reduced_shape, weights.shape[1]), dtype=dtype)
                out = grid.reshape(-1, weights.shape[1])

            # Apply the corresponding function to do the operation that we need to do.
            if weights is None:
                reduce_grid_sum(
                    csr.data[:, 0].astype(dtype),
                    csr.ptr,
                    csr.col,
                    reduce_factor=int(reduce_factor),
                    grid_shape=grid_shape,
                    new_axes=new_axes_order,
                    out=out,
                )
            elif multi_weights:
                reduce_grid_matvecs_multiply(
                    csr.data[:, 0].astype(dtype),
                    csr.ptr,
                    csr.col,
                    weights.astype(dtype),
                    reduce_factor=int(reduce_factor),
                    grid_shape=grid_shape,
                    new_axes=new_axes_order,
                    out=out,
                )
            else:
                reduce_grid_matvec_multiply(
                    csr.data[:, 0].astype(dtype),
                    csr.ptr,
                    csr.col,
                    weights.astype(dtype),
                    reduce_factor=int(reduce_factor),
                    grid_shape=grid_shape,
                    new_axes=new_axes_order,
                    out=out,
                )

        return grid


class SparseGridOrbitalBZ(Sparse4DGrid):
    """Stores information on a 3D grid with an extra orbital dimension, which is sparse."""

    def get_overlap_matrix(self) -> Overlap:
        """Computes the overlap matrix.

        It does so by integrating the product between each pair of orbitals on the whole grid.
        """

        # Get the volume of each grid cell, because we will need to multiply
        # the sum of the products by it.
        dvolume = self.lattice.volume / np.prod(self.grid_shape)

        # Get the grid values as a scipy matrix
        csr = self._csr.tocsr()

        # Compute the overlap matrix. It is a simple matrix multiplication.
        all_overlap = csr.T @ csr

        # We would be done here, if it wasn't because the matrix multiplication
        # returns a square matrix. We need to acumulate all the extra rows into
        # the first set of rows (unit cell). For example, the overlap between
        # supercell [0, 0, 1] and supercell [0, 0, 1] is an overlap within the same
        # cell. We should add it to the unit cell overlap.
        # In practice, what we do is to take the rows of the matrix that correspond
        # to a supercell, shift the columns, and add them to the unit cell rows.

        # Number of orbitals in the unit cell
        no = self.geometry.no

        # Part of the overlap matrix where we will fold everything. Shape (no, no * n_supercells)
        overlap = all_overlap[:no]
        # Compute number of supercells
        n_s = overlap.shape[-1] // no

        # Then loop over supercell rows, shifting them and folding into the main matrix,
        # as explained above.
        for isc_row in range(1, n_s):
            next_rows = all_overlap[isc_row * no : (isc_row + 1) * no]
            for isc_col in range(n_s):
                start_col = (isc_col - isc_row) * no
                if start_col < 0:
                    start_col = overlap.shape[-1] + start_col

                overlap[:, start_col : start_col + no] += next_rows[
                    :, isc_col * no : (isc_col + 1) * no
                ]

        # Multiply by the volume of each grid cell and return an Overlap object.
        return Overlap.fromsp(self.geometry, overlap * dvolume)

    def reduce_orbitals(
        self, weights: np.ndarray, k: tuple[float, float, float] = (0, 0, 0), **kwargs
    ) -> Union[Grid, np.ndarray]:
        """Reduces the orbitals dimension, applying phases if needed.

        Parameters
        ----------
        weights:
            Array of weights for the orbitals. The first dimension of this array must have length
            equal to the number of orbitals in the unit cell.

            The array might optionally have a second dimension,
            which will also be reflected as an extra dimension in the output grid.
        k:
            The k point for which the values are to be projected. This will be used to compute a
            phase for the contributions of orbitals outside the unit cell.
        **kwargs:
            passed directly to reduce_dimension

        Returns
        -------
        Union[Grid, np.ndarray]
            The reduced grid. If the weights array has more than one dimension, it will be a
            numpy array. Otherwise it will be a sisl.Grid object.
        """
        # Assess if we are dealing with multiple weights
        multi_weights = weights.ndim == 2 and weights.shape[1] != 1
        if not multi_weights:
            weights = weights.reshape(-1, 1)

        if weights.shape[0] == self.geometry.no:
            # We should expand the coefficients to the whole supercell, applying phases.
            k = np.array(k, dtype=np.float64)

            if np.any(k > 1e-5):
                # Compute phases
                phases = phase_rsc(self.geometry.lattice, k, dtype=np.complex128)
                # Expand the coefficients array.
                weights = (
                    phases.reshape(-1, 1)
                    .dot(weights.reshape(1, -1))
                    .reshape(-1, weights.shape[1])
                )
            else:
                # At gamma we just need to tile the coefficients array
                n_s = self.geometry.n_s
                weights = (
                    np.tile(weights.T, n_s)
                    .reshape(weights.shape[1], weights.shape[0] * n_s)
                    .T
                )

        elif weights.shape[0] != self.geometry.no_s:
            raise ValueError(
                f"The coefficients array must be of length no ({self.geometry.no}) or no_s ({self.geometry.no_s}). It is of length {weights.shape[0]}"
            )

        return self.reduce_dimension(weights, **kwargs)

    def reduce_orbital_products(
        self,
        weights: Union[np.ndarray, SparseCSR, spmatrix],
        weights_lattice: Lattice,
        reduce_grid: tuple[int, ...] = (),
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reduces the orbital dimension by taking products of orbitals with weights.

        Parameters
        ----------
        weights:
            The coefficient to apply to each orbital product. The first two dimensions of this
            array must be ``(no, no_s)``. Therefore each entry ``weights[i, j]`` is the coefficient
            to apply to the product of orbitals ``i`` and ``j``.

            The array might optionally have a third dimension, which will also be reflected as an
            extra dimension in the output grid. This is useful to compute multiple quantities at
            the same time, because orbital products are only computed once.

            The ``weights`` array be either a dense or sparse array.
        weights_lattice:
            The lattice associated to the weights array. It might be different from the one
            associated to the ``SparseGridOrbitalBZ`` object.
        reduce_grid:
            Axes that you wish to reduce on the grid. Along these axes, the grid values will be
            accumulated. This is very useful to reduce memory footprint (and probably also
            computation time) if your final goal is to reduce over the grid axes, because with
            this approach the grid axes are reduced on the fly (instead of first constructing the
            full grid and then reducing).

            This might be the difference between being able to perform the calculation or not.
        out:
            If you wish to store the result in a preallocated array, you can pass it here.
            It is your responsibility to ensure that the array has the correct shape and dtype.

        Returns
        -------
        np.ndarray
            The output grid. If ``out`` is not ``None``, it will be the same as ``out``. Otherwise
            it will be a newly created array.
        """
        if weights.shape[0] != self.geometry.no:
            raise ValueError(
                f"Mismatch in weights array shape: Number of unit cell orbitals is {weights.shape[0]}, while orbital values are stored for {self.geometry.no} orbitals."
            )
        elif weights.shape[1] != self.geometry.no_s:
            raise ValueError(
                f"Mismatch in weights array shape: The number of unit cell orbitals ({self.geometry.no})"
                f" is correct, but supercell orbitals in the weights array is {weights.shape[1]},"
                f" while it should be {self.geometry.no_s}. It is likely that the weights array"
                f" has been obtained from a matrix with the wrong number of auxiliary cells."
                f"  The correct number of auxiliary cells is {self.geometry.nsc}."
            )

        csr = self._csr

        # Find out the reduced shape, and the reduce factor. The reduced factor is the number
        # by which we have to divide the index to get the reduced index.
        reduced_shape = list(self.grid_shape)
        reduce_factor = 1

        if len(reduce_grid) > 3:
            raise ValueError(
                f"There are only 3 grid axes, it is not possible to reduce axes {reduce_grid}"
            )

        # Check if we are reducing the last dimensions. Otherwise we will need to transpose the grid to
        # put the dimension to reduce on the last axis. Note that we will not do that explicilty, transposing
        # is done inside the cython routines (per index) to reduce memory footprint.
        need_transpose = False
        sorted_reduce_axes = np.sort(reduce_grid)
        if len(reduce_grid) > 0:
            need_transpose = sorted_reduce_axes[-1] != 2
        if len(reduce_grid) > 1:
            need_transpose = need_transpose or sorted_reduce_axes[-2] != 1
        if len(reduce_grid) > 2:
            need_transpose = need_transpose or sorted_reduce_axes[0] != 0

        # Calculate the reducing factor and the new reduced shape.
        for reduce_axis in sorted_reduce_axes:
            reduce_factor *= self.grid_shape[reduce_axis]
            reduced_shape[reduce_axis] = 1

        # Prepare the quantities needed to transpose.
        grid_shape = np.array(self.grid_shape, dtype=np.int32)
        if need_transpose:
            not_reduced_axes = np.sort(tuple(set([0, 1, 2]) - set(reduce_grid)))
            new_axes_order = np.array([*not_reduced_axes, *reduce_grid], dtype=np.int32)
        else:
            # A dummy array to pass to cython functions so that they don't complain
            new_axes_order = np.zeros(0, dtype=np.int32)

        # If it is a scipy sparse matrix, convert it to a sisl sparse matrix
        if issparse(weights):
            weights = SparseCSR(weights)

        sparse_coeffs = isinstance(weights, SparseCSR)
        multi_coeffs = len(weights.shape) == 3 and weights.shape[2] > 1

        if out is None:
            dtype = np.result_type(weights, csr)
        else:
            dtype = out.dtype

        # Decide which function to use.
        if sparse_coeffs:
            if multi_coeffs:
                reduce_func = reduce_sc_products_multicoeffs_sparse_denseindex
            else:
                reduce_func = reduce_sc_products
                weights = weights.tocsr().toarray(order="C")
        else:
            if multi_coeffs:
                reduce_func = reduce_sc_products_multicoeffs
            else:
                reduce_func = reduce_sc_products

        if out is not None:
            grid = out

        if multi_coeffs:
            if out is None:
                grid = np.zeros((*reduced_shape, weights.shape[2]), dtype=dtype)
                out = grid

            out = grid.reshape(-1, weights.shape[2])
        else:
            if out is None:
                grid = Grid(reduced_shape, geometry=self.geometry, dtype=dtype)
                out = grid.grid

            out = out.ravel()

        if isinstance(weights, SparseCSR):
            weights = weights.copy(dtype=dtype)
        else:
            weights = weights.astype(dtype)

        reduce_func(
            csr.data[:, 0].astype(dtype),
            csr.ptr,
            csr.col,
            weights,
            self.geometry.no,
            self.lattice.sc_off,
            weights_lattice.isc_off,
            reduce_factor=int(reduce_factor),
            grid_shape=grid_shape,
            new_axes=new_axes_order,
            out=out,
        )

        return grid
