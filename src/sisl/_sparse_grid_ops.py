# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import cython
import cython.cimports.numpy as cnp
import numpy as np

from sisl import SparseCSR


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.ccall
@cython.exceptval(check=False)
def transpose_raveled_index(
    index: cython.int, grid_shape: cnp.int32_t[:], new_order: cnp.int32_t[:]
) -> cython.int:
    """Transposes a raveled index, given the shape that raveled it and the new axis order.

    Currently it only works for 3D indices.

    Parameters
    -----------
    index:
        The index to transpose.
    grid_shape:
        The old grid shape.
    new_order:
        The new axes order. E.g. [1, 2, 0] will move the first axis to the last dimension.
    """

    unraveled: cython.int[3]
    new_grid_shape: cython.int[3]
    new_unraveled: cython.int[3]

    remaining: cython.int
    iaxis: cython.int

    new_raveled: cython.int

    remaining = index
    for iaxis in range(2, -1, -1):
        unraveled[iaxis] = remaining % grid_shape[iaxis]
        remaining = remaining // grid_shape[iaxis]

    for iaxis in range(3):
        new_grid_shape[iaxis] = grid_shape[new_order[iaxis]]
        new_unraveled[iaxis] = unraveled[new_order[iaxis]]

    new_raveled = (
        new_unraveled[2]
        + new_unraveled[1] * new_grid_shape[2]
        + new_unraveled[0] * new_grid_shape[1] * new_grid_shape[2]
    )

    return new_raveled


# This function should be in sisl._sparse, but I don't know how to import it from there
@cython.boundscheck(False)
@cython.wraparound(False)
def dense_index(shape, ptr: cnp.int32_t[:], col: cnp.int32_t[:]) -> cnp.int32_t[:, :]:
    """Returns a dense array containing the value index for each nonzero (row, col) element.

    Zero elements are asigned an index of -1, so routines using this dense index should
    take this into account.

    Parameters
    -----------
    shape: tuple of shape (2, )
        The shape of the sparse matrix
    ptr, col:
        ptr and col descriptors of the matrix in CSR format

    Returns
    --------
    np.ndarray:
        Numpy array of shape = matrix shape and data type np.int32.
    """
    nrows = ptr.shape[0] - 1
    row: cython.int
    ival: cython.int

    # Initialize the dense index
    dense_index: cnp.int32_t[:, :] = np.full(shape, -1, dtype=np.int32)

    for row in range(nrows):
        row_start = ptr[row]
        row_end = ptr[row + 1]

        for ival in range(row_start, row_end):
            val_col = col[ival]
            dense_index[row, val_col] = ival

    return np.asarray(dense_index)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def reduce_grid_sum(
    data: cython.numeric[:],
    ptr: cnp.int32_t[:],
    col: cnp.int32_t[:],
    reduce_factor: cython.int,
    grid_shape: cnp.int32_t[:],
    new_axes: cnp.int32_t[:],
    out: cython.numeric[:],
):
    """Performs sum over the extra dimension while reducing other dimensions of the grid.

    Parameters
    ----------
    data:
        The data values of the sparse matrix
    ptr:
        The array with pointers to the start of each row for the sparse matrix.
    col:
        The array containing the column index for each value in the sparse matrix.
    reduce_factor:
        Each row index is integer divided (//) by this factor to get the row index where
        the result should be stored.
    grid_shape:
        If you want to transpose the grid indices before reducing it, you need to specify
        the (old) grid shape. If not, pass here an array of shape 0.
    new_axes:
        If you want to transpose the grid indices before reducing it, you need to specify
        the nex axes. As an example, [1, 2, 0] will move the first axis to the last dimension.

        If you don't want to transpose, pass an array of shape 0.
    out:
        The array where the output should be stored.
    """
    nrows = ptr.shape[0] - 1

    i: cython.int
    j: cython.int

    need_transpose: cython.bint = new_axes.shape[0] > 1

    for i in range(nrows):

        if need_transpose:
            reduced_i = (
                transpose_raveled_index(i, grid_shape, new_axes) // reduce_factor
            )
        else:
            reduced_i = i // reduce_factor

        row_value: cython.numeric = 0
        for j in range(ptr[i], ptr[i + 1]):
            row_value += data[j]

        out[reduced_i] = out[reduced_i] + row_value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def reduce_grid_matvec_multiply(
    data: cython.numeric[:],
    ptr: cnp.int32_t[:],
    col: cnp.int32_t[:],
    V: cython.numeric[:],
    reduce_factor: cython.int,
    grid_shape: cnp.int32_t[:],
    new_axes: cnp.int32_t[:],
    out: cython.numeric[:],
):
    """Performs a matrix-vector multiplication while reducing other dimensions of the grid.

    Parameters
    ----------
    data:
        The data values of the sparse matrix
    ptr:
        The array with pointers to the start of each row for the sparse matrix.
    col:
        The array containing the column index for each value in the sparse matrix.
    V:
        The dense vector by which the sparse matrix is to be multiplied.
    reduce_factor:
        Each row index is integer divided (//) by this factor to get the row index where
        the result should be stored.
    grid_shape:
        If you want to transpose the grid indices before reducing it, you need to specify
        the (old) grid shape. If not, pass here an array of shape 0.
    new_axes:
        If you want to transpose the grid indices before reducing it, you need to specify
        the nex axes. As an example, [1, 2, 0] will move the first axis to the last dimension.

        If you don't want to transpose, pass an array of shape 0.
    out:
        The array where the output should be stored.
    """

    nrows: cython.int = ptr.shape[0] - 1

    i: cython.int
    reduced_i: cython.int
    j: cython.int
    jcol: cython.int

    need_transpose: cython.bint = new_axes.shape[0] > 1

    for i in range(nrows):
        reduced_i = i // reduce_factor

        if need_transpose:
            reduced_i = (
                transpose_raveled_index(i, grid_shape, new_axes) // reduce_factor
            )
        else:
            reduced_i = i // reduce_factor

        row_value: cython.numeric = 0
        for j in range(ptr[i], ptr[i + 1]):
            jcol = col[j]
            row_value += data[j] * V[jcol]

        out[reduced_i] = out[reduced_i] + row_value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def reduce_grid_matvecs_multiply(
    data: cython.numeric[:],
    ptr: cnp.int32_t[:],
    col: cnp.int32_t[:],
    V: cython.numeric[:, :],
    reduce_factor: cython.int,
    grid_shape: cnp.int32_t[:],
    new_axes: cnp.int32_t[:],
    out: cython.numeric[:, :],
):
    """Performs a matrix-matrix multiplication while reducing other dimensions of the grid.

    Parameters
    ----------
    data:
        The data values of the sparse matrix
    ptr:
        The array with pointers to the start of each row for the sparse matrix.
    col:
        The array containing the column index for each value in the sparse matrix.
    V:
        The dense matrix by which the sparse matrix is to be multiplied.
    reduce_factor:
        Each row index is integer divided (//) by this factor to get the row index where
        the result should be stored.
    grid_shape:
        If you want to transpose the grid indices before reducing it, you need to specify
        the (old) grid shape. If not, pass here an array of shape 0.
    new_axes:
        If you want to transpose the grid indices before reducing it, you need to specify
        the nex axes. As an example, [1, 2, 0] will move the first axis to the last dimension.

        If you don't want to transpose, pass an array of shape 0.
    out:
        The array where the output should be stored.
    """

    nrows = ptr.shape[0] - 1
    nvecs = V.shape[1]

    i: cython.int
    j: cython.int
    ivec: cython.int

    need_transpose: cython.bint = new_axes.shape[0] > 1

    row_value: cython.numeric[:] = np.zeros(nvecs, np.asarray(data).dtype)

    for i in range(nrows):
        if need_transpose:
            reduced_i = (
                transpose_raveled_index(i, grid_shape, new_axes) // reduce_factor
            )
        else:
            reduced_i = i // reduce_factor

        row_value[:] = 0
        for j in range(ptr[i], ptr[i + 1]):
            jcol = col[j]

            for ivec in range(nvecs):
                row_value[ivec] = row_value[ivec] + data[j] * V[jcol, ivec]

        for ivec in range(nvecs):
            out[reduced_i, ivec] = out[reduced_i, ivec] + row_value[ivec]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def reduce_sc_products(
    data: cython.floating[:],
    ptr: cnp.int32_t[:],
    col: cnp.int32_t[:],
    coeffs: cython.floating[:, :],
    uc_ncol: cython.int,
    data_sc_off: cnp.int32_t[:, :],
    coeffs_isc_off: cnp.int32_t[:, :, :],
    reduce_factor: cython.int,
    grid_shape: cnp.int32_t[:],
    new_axes: cnp.int32_t[:],
    out: cython.floating[:],
):
    """For each row, sums all possible products between column pairs.

    A coefficients array is accepted to weight each product.

    This function is to be used when columns are some local property of the cell (orbital, atom...) and
    there are interactions between neighbouring cells.

    For each pair of columns ij, the offset between columns is calculated. Then, the function looks
    in the coeffs array (which potentially contains information for interactions between cells) for the ij pair
    with that offset.

    This

    Parameters
    ----------
    data:
        The data values of the sparse matrix
    ptr:
        The array with pointers to the start of each row for the sparse matrix.
    col:
        The array containing the column index for each value in the sparse matrix.
    coeffs:
        The dense matrix containing weights for all possible products.
    uc_ncol:
        Number of columns for each unit cell.
    reduce_factor:
        Each row index is integer divided (//) by this factor to get the row index where
        the result should be stored.
    grid_shape:
        If you want to transpose the grid indices before reducing it, you need to specify
        the (old) grid shape. If not, pass here an array of shape 0.
    new_axes:
        If you want to transpose the grid indices before reducing it, you need to specify
        the nex axes. As an example, [1, 2, 0] will move the first axis to the last dimension.

        If you don't want to transpose, pass an array of shape 0.
    out:
        The array where the output should be stored.
    """
    nrows = ptr.shape[0] - 1

    # Indices to handle rows
    row: cython.int
    reduced_row: cython.int

    # Indices to handle pairs of columns (ij)
    i: cython.int
    icol: cython.int
    uc_icol: cython.int
    sc_icol: cython.int
    icol_sc: cython.int
    ipair_sc: cython.int

    j: cython.int
    jcol: cython.int
    uc_jcol: cython.int
    sc_jcol: cython.int
    jcol_sc: cython.int
    jpair_sc: cython.int

    # Index to loop over axes of the grid.
    iaxis: cython.int

    # Variables that will help managing orbital pairs that are not within the same cell.
    sc_diff: cython.int[3]
    inv_sc_diff: cython.int[3]
    force_same_cell: cython.bint
    same_cell: cython.bint

    # Boolean to store whether we should reduce row indices
    grid_reduce: cython.bint = reduce_factor > 1
    # Do we need to transpose while reducing?
    need_transpose: cython.bint = new_axes.shape[0] > 1

    # Calculate the number of cells in each direction that the supercell is built of.
    # This will be useful just to convert negative supercell indices to positive ones.
    # If the number of supercells is 1, we assume that even intercell overlaps
    # (if any) have been stored in the unit cell. This is what SIESTA does for gamma point calculations
    # with nsc <= 3.
    force_same_cell = True
    coeffs_nsc = coeffs_isc_off.shape
    for iaxis in range(3):
        if coeffs_nsc[iaxis] != 1:
            force_same_cell = False

    # Loop over rows.
    for row in range(nrows):
        # Get the potentially reduced index of the output row where we should store the
        # results for this row.
        reduced_row = row
        if grid_reduce:

            if need_transpose:
                reduced_row = transpose_raveled_index(row, grid_shape, new_axes)

            reduced_row = reduced_row // reduce_factor

        # Get the limits of this row
        row_start = ptr[row]
        row_end = ptr[row + 1]

        # Initialize the row value.
        row_value: cython.floating = 0

        # For each row, loop over pairs of columns (ij).
        # We add both ij and ji contributions, therefore we only need to loop over j greater than i.
        # We do this because it is very easy if orbitals i and j are in the same cell. We also save
        # some computation if they are not.
        for i in range(row_start, row_end):
            icol = col[i]

            # Initialize the value for all pairs that we found for i
            i_row_value: cython.floating = 0

            # Precompute the supercell index of icol (will compare it to that of jcol)
            icol_sc = icol // uc_ncol
            # And also its unit cell index
            uc_icol = icol % uc_ncol

            # Get also the
            for j in range(i, row_end):
                jcol = col[j]

                jcol_sc = jcol // uc_ncol
                # Get the unit cell index of jcol
                uc_jcol = jcol % uc_ncol

                same_cell = force_same_cell
                # If same cell interactions are not forced, we need to discover if this pair
                # of columns is within the same cell.
                if not force_same_cell:
                    same_cell = icol_sc == jcol_sc

                # If the columns are not in the same cell, we need to
                # (1) Calculate the supercell offset between icol and jcol
                # (2) And then calculate the new index for jcol, moving icol to the unit cell
                # (3) Do the same in the reverse direction (jcol -> icol)
                if not same_cell:
                    # Calculate the sc offset between both orbitals.
                    for iaxis in range(3):
                        sc_diff[iaxis] = (
                            data_sc_off[jcol_sc, iaxis] - data_sc_off[icol_sc, iaxis]
                        )
                        # Calculate also the offset in the reverse direction
                        inv_sc_diff[iaxis] = -sc_diff[iaxis]

                        # If the sc_difference is negative, convert it to positive so that we can
                        # use it to index the isc_off array (we switched off the handling of negative
                        # indices in cython with wraparound(False))
                        if sc_diff[iaxis] < 0:
                            sc_diff[iaxis] = coeffs_nsc[iaxis] + sc_diff[iaxis]
                        elif inv_sc_diff[iaxis] < 0:
                            inv_sc_diff[iaxis] = coeffs_nsc[iaxis] + inv_sc_diff[iaxis]

                    # Get the supercell offset index of jcol with respect to icol
                    jpair_sc = coeffs_isc_off[sc_diff[0], sc_diff[1], sc_diff[2]]
                    # And use it to calculate the supercell index of the j orbital in this ij pair
                    sc_jcol = jpair_sc * uc_ncol + uc_jcol

                    # Do the same for the ji pair
                    ipair_sc = coeffs_isc_off[
                        inv_sc_diff[0], inv_sc_diff[1], inv_sc_diff[2]
                    ]
                    sc_icol = ipair_sc * uc_ncol + uc_icol

                # Add the contribution of this column pair to the row total value. Note that we only
                # multiply the coefficients by data[j] here. This is because this loop is over all j
                # that pair with a given i. data[i] is a common factor and therefore we can multiply
                # after the loop to save operations.
                if same_cell:
                    if icol == jcol:
                        i_row_value += coeffs[uc_icol, uc_jcol] * data[j]
                    else:
                        i_row_value += (
                            coeffs[uc_icol, uc_jcol] + coeffs[uc_jcol, uc_icol]
                        ) * data[j]
                else:
                    i_row_value += (
                        coeffs[uc_icol, sc_jcol] + coeffs[uc_jcol, sc_icol]
                    ) * data[j]

            # Multiply all the contributions of ij pairs with this i by data[i], as explained inside the j loop.
            i_row_value *= data[i]

            # Add the contribution of all ij pairs for this i to the row value.
            row_value += i_row_value

        # Store the row value in the output
        out[reduced_row] = out[reduced_row] + row_value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def reduce_sc_products_multicoeffs(
    data: cython.floating[:],
    ptr: cnp.int32_t[:],
    col: cnp.int32_t[:],
    coeffs: cython.floating[:, :, :],
    uc_ncol: cython.int,
    data_sc_off: cnp.int32_t[:, :],
    coeffs_isc_off: cnp.int32_t[:, :, :],
    reduce_factor: cython.int,
    grid_shape: cnp.int32_t[:],
    new_axes: cnp.int32_t[:],
    out: cython.floating[:, :],
):
    """For each row, sums all possible products between column pairs.

    A coefficients array is accepted to weight each product. The with reduce_sc_products is that
    this function accepts multiple

    This function is to be used when columns are some local property of the cell (orbital, atom...) and
    there are interactions between neighbouring cells.

    For each pair of columns ij, the offset between columns is calculated. Then, the function looks
    in the coeffs array (which potentially contains information for interactions between cells) for the ij pair
    with that offset.

    Parameters
    ----------
    data:
        The data values of the sparse matrix
    ptr:
        The array with pointers to the start of each row for the sparse matrix.
    col:
        The array containing the column index for each value in the sparse matrix.
    coeffs:
        The dense matrix containing weights for all possible products.
    uc_ncol:
        Number of columns for each unit cell.
    reduce_factor:
        Each row index is integer divided (//) by this factor to get the row index where
        the result should be stored.
    grid_shape:
        If you want to transpose the grid indices before reducing it, you need to specify
        the (old) grid shape. If not, pass here an array of shape 0.
    new_axes:
        If you want to transpose the grid indices before reducing it, you need to specify
        the nex axes. As an example, [1, 2, 0] will move the first axis to the last dimension.

        If you don't want to transpose, pass an array of shape 0.
    out:
        The array where the output should be stored.
    """
    nrows = ptr.shape[0] - 1

    # Indices to handle rows
    row: cython.int
    reduced_row: cython.int

    # Indices to handle pairs of columns (ij)
    i: cython.int
    icol: cython.int
    uc_icol: cython.int
    sc_icol: cython.int
    icol_sc: cython.int
    ipair_sc: cython.int

    j: cython.int
    jcol: cython.int
    uc_jcol: cython.int
    sc_jcol: cython.int
    jcol_sc: cython.int
    jpair_sc: cython.int

    # Variables to handle multiple productcoefficients
    ncoeffs: cython.int = coeffs.shape[2]
    icoeff: cython.int

    # Temporal storage to build values
    row_value: cython.floating[:] = np.zeros(ncoeffs, dtype=np.asarray(data).dtype)
    i_row_value: cython.floating[:] = np.zeros(ncoeffs, dtype=np.asarray(data).dtype)

    # Index to loop over axes of the grid.
    iaxis: cython.int

    # Variables that will help managing orbital pairs that are not within the same cell.
    sc_diff: cython.int[3]
    inv_sc_diff: cython.int[3]
    force_same_cell: cython.bint
    same_cell: cython.bint

    # Boolean to store whether we should reduce row indices
    grid_reduce: cython.bint = reduce_factor > 1
    # Do we need to transpose while reducing?
    need_transpose: cython.bint = new_axes.shape[0] > 1

    # Calculate the number of cells in each direction that the supercell is built of.
    # This will be useful just to convert negative supercell indices to positive ones.
    # If the number of supercells is 1, we assume that even intercell overlaps
    # (if any) have been stored in the unit cell. This is what SIESTA does for gamma point calculations
    # with nsc <= 3.
    force_same_cell = True
    coeffs_nsc = coeffs_isc_off.shape
    for iaxis in range(3):
        if coeffs_nsc[iaxis] != 1:
            force_same_cell = False

    # Loop over rows.
    for row in range(nrows):
        # Get the potentially reduced index of the output row where we should store the
        # results for this row.
        reduced_row = row
        if grid_reduce:

            if need_transpose:
                reduced_row = transpose_raveled_index(row, grid_shape, new_axes)

            reduced_row = reduced_row // reduce_factor

        # Get the limits of this row
        row_start = ptr[row]
        row_end = ptr[row + 1]

        # Initialize the row value.
        row_value[:] = 0

        # For each row, loop over pairs of columns (ij).
        # We add both ij and ji contributions, therefore we only need to loop over j greater than i.
        # We do this because it is very easy if orbitals i and j are in the same cell. We also save
        # some computation if they are not.
        for i in range(row_start, row_end):
            icol = col[i]
            # Initialize the value for all pairs that we found for i
            i_row_value[:] = 0

            # Precompute the supercell index of icol (will compare it to that of jcol)
            icol_sc = icol // uc_ncol
            # And also its unit cell index
            uc_icol = icol % uc_ncol

            # Get also the
            for j in range(i, row_end):
                jcol = col[j]

                jcol_sc = jcol // uc_ncol
                # Get the unit cell index of jcol
                uc_jcol = jcol % uc_ncol

                same_cell = force_same_cell
                # If same cell interactions are not forced, we need to discover if this pair
                # of columns is within the same cell.
                if not force_same_cell:
                    same_cell = icol_sc == jcol_sc

                # If the columns are not in the same cell, we need to
                # (1) Calculate the supercell offset between icol and jcol
                # (2) And then calculate the new index for jcol, moving icol to the unit cell
                # (3) Do the same in the reverse direction (jcol -> icol)
                if not same_cell:
                    # Calculate the sc offset between both orbitals.
                    for iaxis in range(3):
                        sc_diff[iaxis] = (
                            data_sc_off[jcol_sc, iaxis] - data_sc_off[icol_sc, iaxis]
                        )
                        # Calculate also the offset in the reverse direction
                        inv_sc_diff[iaxis] = -sc_diff[iaxis]

                        # If the sc_difference is negative, convert it to positive so that we can
                        # use it to index the isc_off array (we switched off the handling of negative
                        # indices in cython with wraparound(False))
                        if sc_diff[iaxis] < 0:
                            sc_diff[iaxis] = coeffs_nsc[iaxis] + sc_diff[iaxis]
                        elif inv_sc_diff[iaxis] < 0:
                            inv_sc_diff[iaxis] = coeffs_nsc[iaxis] + inv_sc_diff[iaxis]

                    # Get the supercell offset index of jcol with respect to icol
                    jpair_sc = coeffs_isc_off[sc_diff[0], sc_diff[1], sc_diff[2]]
                    # And use it to calculate the supercell index of the j orbital in this ij pair
                    sc_jcol = jpair_sc * uc_ncol + uc_jcol

                    # Do the same for the ji pair
                    ipair_sc = coeffs_isc_off[
                        inv_sc_diff[0], inv_sc_diff[1], inv_sc_diff[2]
                    ]
                    sc_icol = ipair_sc * uc_ncol + uc_icol

                # Add the contribution of this column pair to the row total value. Note that we only
                # multiply the coefficients by data[j] here. This is because this loop is over all j
                # that pair with a given i. data[i] is a common factor and therefore we can multiply
                # after the loop to save operations.

                # Do it for all coefficients.
                for icoeff in range(ncoeffs):
                    if same_cell:
                        if icol == jcol:
                            i_row_value[icoeff] += (
                                coeffs[uc_icol, uc_jcol, icoeff] * data[j]
                            )
                        else:
                            i_row_value[icoeff] += (
                                coeffs[uc_icol, uc_jcol, icoeff]
                                + coeffs[uc_jcol, uc_icol, icoeff]
                            ) * data[j]
                    else:
                        i_row_value[icoeff] += (
                            coeffs[uc_icol, sc_jcol, icoeff]
                            + coeffs[uc_jcol, sc_icol, icoeff]
                        ) * data[j]

            for icoeff in range(ncoeffs):
                # Multiply all the contributions of ij pairs with this i by data[i], as explained inside the j loop.
                i_row_value[icoeff] *= data[i]

                # Add the contribution of all ij pairs for this i to the row value.
                row_value[icoeff] += i_row_value[icoeff]

        for icoeff in range(ncoeffs):
            # Store the row value in the output
            out[reduced_row, icoeff] = out[reduced_row, icoeff] + row_value[icoeff]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def reduce_sc_products_multicoeffs_sparse_denseindex(
    data: cython.floating[:],
    ptr: cnp.int32_t[:],
    col: cnp.int32_t[:],
    coeffs_csr: SparseCSR,
    uc_ncol: cython.int,
    data_sc_off: cnp.int32_t[:, :],
    coeffs_isc_off: cnp.int32_t[:, :, :],
    reduce_factor: cython.int,
    grid_shape: cnp.int32_t[:],
    new_axes: cnp.int32_t[:],
    out: cython.floating[:, :],
):
    """For each row, sums all possible products between column pairs.

    A coefficients array is accepted to weight each product. The with reduce_sc_products is that
    this function accepts multiple

    This function is to be used when columns are some local property of the cell (orbital, atom...) and
    there are interactions between neighbouring cells.

    For each pair of columns ij, the offset between columns is calculated. Then, the function looks
    in the coeffs array (which potentially contains information for interactions between cells) for the ij pair
    with that offset.

    Parameters
    ----------
    data:
        The data values of the sparse matrix
    ptr:
        The array with pointers to the start of each row for the sparse matrix.
    col:
        The array containing the column index for each value in the sparse matrix.
    coeffs:
        The dense matrix containing weights for all possible products.
    uc_ncol:
        Number of columns for each unit cell.
    reduce_factor:
        Each row index is integer divided (//) by this factor to get the row index where
        the result should be stored.
    grid_shape:
        If you want to transpose the grid indices before reducing it, you need to specify
        the (old) grid shape. If not, pass here an array of shape 0.
    new_axes:
        If you want to transpose the grid indices before reducing it, you need to specify
        the nex axes. As an example, [1, 2, 0] will move the first axis to the last dimension.

        If you don't want to transpose, pass an array of shape 0.
    out:
        The array where the output should be stored.
    """
    nrows = ptr.shape[0] - 1

    # Indices to handle rows
    row: cython.int
    reduced_row: cython.int

    # Indices to handle pairs of columns (ij)
    i: cython.int
    icol: cython.int
    uc_icol: cython.int
    sc_icol: cython.int
    icol_sc: cython.int
    ipair_sc: cython.int

    j: cython.int
    jcol: cython.int
    uc_jcol: cython.int
    sc_jcol: cython.int
    jcol_sc: cython.int
    jpair_sc: cython.int

    # Extra variables to handle coefficients
    coeffs: cython.floating[:, :] = coeffs_csr.data
    # Get the array containing all the dense indices.
    dense_idx: cnp.int32_t[:, :] = dense_index(
        coeffs_csr.shape[:2], coeffs_csr.ptr, coeffs_csr.col
    )
    coeff_index: cython.int
    coeff_index2: cython.int

    # Variables to handle multiple productcoefficients
    ncoeffs: cython.int = coeffs.shape[1]
    icoeff: cython.int

    # Temporal storage to build values
    row_value: cython.floating[:] = np.zeros(ncoeffs, dtype=np.asarray(data).dtype)
    i_row_value: cython.floating[:] = np.zeros(ncoeffs, dtype=np.asarray(data).dtype)

    # Index to loop over axes of the grid.
    iaxis: cython.int

    # Variables that will help managing orbital pairs that are not within the same cell.
    sc_diff: cython.int[3]
    inv_sc_diff: cython.int[3]
    force_same_cell: cython.bint
    same_cell: cython.bint

    # Boolean to store whether we should reduce row indices
    grid_reduce: cython.bint = reduce_factor > 1
    # Do we need to transpose while reducing?
    need_transpose: cython.bint = new_axes.shape[0] > 1

    # Calculate the number of cells in each direction that the supercell is built of.
    # This will be useful just to convert negative supercell indices to positive ones.
    # If the number of supercells is 1, we assume that even intercell overlaps
    # (if any) have been stored in the unit cell. This is what SIESTA does for gamma point calculations
    # with nsc <= 3.
    force_same_cell = True
    coeffs_nsc = coeffs_isc_off.shape
    for iaxis in range(3):
        if coeffs_nsc[iaxis] != 1:
            force_same_cell = False

    # Loop over rows.
    for row in range(nrows):
        # Get the potentially reduced index of the output row where we should store the
        # results for this row.
        reduced_row = row
        if grid_reduce:

            if need_transpose:
                reduced_row = transpose_raveled_index(row, grid_shape, new_axes)

            reduced_row = reduced_row // reduce_factor

        # Get the limits of this row
        row_start = ptr[row]
        row_end = ptr[row + 1]

        # Initialize the row value.
        row_value[:] = 0

        # For each row, loop over pairs of columns (ij).
        # We add both ij and ji contributions, therefore we only need to loop over j greater than i.
        # We do this because it is very easy if orbitals i and j are in the same cell. We also save
        # some computation if they are not.
        for i in range(row_start, row_end):
            icol = col[i]
            # Initialize the value for all pairs that we found for i
            i_row_value[:] = 0

            # Precompute the supercell index of icol (will compare it to that of jcol)
            icol_sc = icol // uc_ncol
            # And also its unit cell index
            uc_icol = icol % uc_ncol

            # Get also the
            for j in range(i, row_end):
                jcol = col[j]

                jcol_sc = jcol // uc_ncol
                # Get the unit cell index of jcol
                uc_jcol = jcol % uc_ncol

                same_cell = force_same_cell
                # If same cell interactions are not forced, we need to discover if this pair
                # of columns is within the same cell.
                if not force_same_cell:
                    same_cell = icol_sc == jcol_sc

                # If the columns are not in the same cell, we need to
                # (1) Calculate the supercell offset between icol and jcol
                # (2) And then calculate the new index for jcol, moving icol to the unit cell
                # (3) Do the same in the reverse direction (jcol -> icol)
                if not same_cell:
                    # Calculate the sc offset between both orbitals.
                    for iaxis in range(3):
                        sc_diff[iaxis] = (
                            data_sc_off[jcol_sc, iaxis] - data_sc_off[icol_sc, iaxis]
                        )
                        # Calculate also the offset in the reverse direction
                        inv_sc_diff[iaxis] = -sc_diff[iaxis]

                        # If the sc_difference is negative, convert it to positive so that we can
                        # use it to index the isc_off array (we switched off the handling of negative
                        # indices in cython with wraparound(False))
                        if sc_diff[iaxis] < 0:
                            sc_diff[iaxis] = coeffs_nsc[iaxis] + sc_diff[iaxis]
                        elif inv_sc_diff[iaxis] < 0:
                            inv_sc_diff[iaxis] = coeffs_nsc[iaxis] + inv_sc_diff[iaxis]

                    # Get the supercell offset index of jcol with respect to icol
                    jpair_sc = coeffs_isc_off[sc_diff[0], sc_diff[1], sc_diff[2]]
                    # And use it to calculate the supercell index of the j orbital in this ij pair
                    sc_jcol = jpair_sc * uc_ncol + uc_jcol

                    # Do the same for the ji pair
                    ipair_sc = coeffs_isc_off[
                        inv_sc_diff[0], inv_sc_diff[1], inv_sc_diff[2]
                    ]
                    sc_icol = ipair_sc * uc_ncol + uc_icol

                # Get the index needed to find the coefficients that we want from the coeffs array.
                if same_cell:
                    if icol == jcol:
                        coeff_index = dense_idx[uc_icol, uc_jcol]
                        coeff_index2 = 0
                    else:
                        coeff_index = dense_idx[uc_icol, uc_jcol]
                        coeff_index2 = dense_idx[uc_jcol, uc_icol]
                else:
                    coeff_index = dense_idx[uc_icol, sc_jcol]
                    coeff_index2 = dense_idx[uc_jcol, sc_icol]

                # If the index for the needed (row, col) element is -1, it means that the element is 0.
                # Just go to next iteration if all elements that we need are 0. Note that we assume here
                # that if (row, col) is zero (col, row) is also zero.
                if coeff_index < 0:
                    continue

                # Add the contribution of this column pair to the row total value. Note that we only
                # multiply the coefficients by data[j] here. This is because this loop is over all j
                # that pair with a given i. data[i] is a common factor and therefore we can multiply
                # after the loop to save operations.

                # Do it for all coefficients.
                for icoeff in range(ncoeffs):
                    if same_cell:
                        if icol == jcol:
                            i_row_value[icoeff] += coeffs[coeff_index, icoeff] * data[j]
                        else:
                            i_row_value[icoeff] += (
                                coeffs[coeff_index, icoeff]
                                + coeffs[coeff_index2, icoeff]
                            ) * data[j]
                    else:
                        i_row_value[icoeff] += (
                            coeffs[coeff_index, icoeff] + coeffs[coeff_index2, icoeff]
                        ) * data[j]

            for icoeff in range(ncoeffs):
                # Multiply all the contributions of ij pairs with this i by data[i], as explained inside the j loop.
                i_row_value[icoeff] *= data[i]

                # Add the contribution of all ij pairs for this i to the row value.
                row_value[icoeff] += i_row_value[icoeff]

        for icoeff in range(ncoeffs):
            # Store the row value in the output
            out[reduced_row, icoeff] = out[reduced_row, icoeff] + row_value[icoeff]
