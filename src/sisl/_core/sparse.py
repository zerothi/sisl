# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import reduce, singledispatchmethod
from numbers import Integral

import numpy as np

# To speed up the _extend algorithm we limit lookups
from numpy import all as np_all
from numpy import (
    allclose,
)
from numpy import any as np_any
from numpy import (
    arange,
    argsort,
    asarray,
    atleast_1d,
    bool_,
    broadcast,
    concatenate,
    copyto,
    count_nonzero,
    delete,
    diff,
    empty,
    full,
    insert,
    int32,
    intersect1d,
    isin,
    isnan,
    isscalar,
    lexsort,
    ndarray,
    repeat,
    setdiff1d,
    split,
    take,
    unique,
    zeros,
)
from numpy.lib.mixins import NDArrayOperatorsMixin
from scipy.sparse import csr_matrix, issparse

from sisl import _array as _a
from sisl._array import array_arange
from sisl._help import array_fill_repeat, isiterable
from sisl._indices import indices, indices_only
from sisl._internal import set_module
from sisl.messages import SislError, warn
from sisl.typing import OrSequence, SeqOrScalarFloat, SeqOrScalarInt, SparseMatrix
from sisl.utils.mathematics import intersect_and_diff_sets

from ._sparse import sparse_dense

# Although this re-implements the CSR in scipy.sparse.csr_matrix
# we use it slightly differently and thus require this new sparse pattern.

__all__ = ["SparseCSR", "ispmatrix", "ispmatrixd"]


def _to_cd(csr, data: bool = True, rows=None):
    """Retrieve the CSR information in a sanitized manner

    This is equivalent to `_to_coo` except it does not return the
    `rows` data.

    Parameters
    ----------
    csr: SparseCSR
        matrix to sanitize
    data:
        whether the data should also be returned sanitized
    rows:
        only return for a subset of rows

    Returns
    -------
    cols : always
    matrix_data : when `data` is True
    """
    ptr = csr.ptr
    ncol = csr.ncol
    col = csr.col
    D = csr._D

    if csr.nnz == csr.ptr[-1] and rows is None:
        cols = col.copy()
        if data:
            D = D.copy()
    else:
        if rows is None:
            idx = array_arange(ptr[:-1], n=ncol, dtype=int32)
        else:
            rows = csr._sanitize(rows).ravel()
            ncol = ncol[rows]
            idx = array_arange(ptr[rows], n=ncol, dtype=int32)
        if data:
            D = D[idx]
        cols = col[idx]

    if data:
        return cols, D
    return cols


def _to_coo(csr, data: bool = True, rows=None):
    """Retrieve the CSR information in a sanitized manner

    I.e. as COO format with data stripped, or only rows and columns

    Parameters
    ----------
    csr: SparseCSR
        matrix to sanitize
    data:
        whether the data should also be returned sanitized
    rows:
        only return for a subset of rows

    Returns
    -------
    rows, cols : always
    matrix_data : when `data` is True
    """
    ptr = csr.ptr
    ncol = csr.ncol
    col = csr.col
    D = csr._D

    if csr.nnz == csr.ptr[-1] and rows is None:
        cols = col.copy()
        if data:
            D = D.copy()
    else:
        if rows is None:
            idx = array_arange(ptr[:-1], n=ncol, dtype=int32)
        else:
            rows = csr._sanitize(rows).ravel()
            ncol = ncol[rows]
            idx = array_arange(ptr[rows], n=ncol, dtype=int32)
        if data:
            D = D[idx]
        cols = col[idx]
    idx = (ncol > 0).nonzero()[0]
    rows = repeat(idx.astype(int32, copy=False), ncol[idx])

    if data:
        return rows, cols, D
    return rows, cols


def _ncol_to_indptr(ncol):
    """Convert the ncol array into a pointer array"""
    ptr = _a.emptyi(ncol.size + 1)
    ptr[0] = 0
    _a.cumsumi(ncol, out=ptr[1:])
    return ptr


def valid_index(idx, shape: int):
    """Check that all indices in `idx` is between [0; shape["""
    return np.logical_and(0 <= idx, idx < shape)


def invalid_index(idx, shape: int):
    """Check that all indices in `idx` is not between [0; shape["""
    return np.logical_or(idx < 0, shape <= idx)


@set_module("sisl")
class SparseCSR(NDArrayOperatorsMixin):
    """
    A compressed sparse row matrix, slightly different than :class:`~scipy.sparse.csr_matrix`.

    This class holds all required information regarding the CSR matrix format.

    Note that this sparse matrix of data does not retain the number of columns
    in the matrix, i.e. it has no way of determining whether the input is
    correct.


    This sparse matrix class tries to resemble the
    :class:`~scipy.sparse.csr_matrix` as much as possible with
    the difference of this class being multi-dimensional.

    Creating a new sparse matrix is much similar to the
    `scipy` equivalent.

    `nnz` is only used if ``nnz > nr * nnzpr``.

    This class may be instantiated by verious means.

    - ``SparseCSR(S)``
      where ``S`` is a `scipy.sparse` matrix
    - ``SparseCSR((M,N)[, dtype])``
      the shape of the sparse matrix (equivalent
      to ``SparseCSR((M,N,1)[, dtype])``.
    - ``SparseCSR((M,N), dim=K, [, dtype])``
      the shape of the sparse matrix (equivalent
      to ``SparseCSR((M,N,K)[, dtype])``.
    - ``SparseCSR((M,N,K)[, dtype])``
      creating a sparse matrix with ``M`` rows, ``N`` columns
      and ``K`` elements per sparse element.
    - ``SparseCSR((data, ptr, indices), [shape, dtype])``
      creating a sparse matrix with specific data as would
      be used when creating `scipy.sparse.csr_matrix`.

    Additionally these parameters control the
    creation of the sparse matrix.

    Parameters
    ----------
    arg1 : tuple
       various initialization methods as described above
    dim : int, optional
       number of elements stored per sparse element, only used if (M,N) is passed
    dtype : numpy.dtype, optional
       data type of the matrix, defaults to `numpy.float64`
    nnzpr : int, optional
       initial number of non-zero elements per row.
       Only used if `nnz` is not supplied
    nnz : int, optional
       initial total number of non-zero elements
       This quantity has precedence over `nnzpr`
    """

    # We don't really need slots, but it is useful
    # to keep a good overview of which variables are present
    __slots__ = ("_shape", "_ns", "_finalized", "_nnz", "ptr", "ncol", "col", "_D")

    def __init__(self, arg1, dim=1, dtype=None, nnzpr: int = 20, nnz=None, **kwargs):
        """Initialize a new sparse CSR matrix"""

        # step size in sparse elements
        # If there isn't enough room for adding
        # a non-zero element, the # of elements
        # for the insert row is increased at least by this number
        self._ns = 10
        self._finalized = False

        if issparse(arg1):
            # This is a sparse matrix
            # The data-type is infered from the
            # input sparse matrix.
            arg1 = arg1.tocsr()
            # Default shape to the CSR matrix
            kwargs["shape"] = kwargs.get("shape", arg1.shape)
            self.__init__(
                (arg1.data, arg1.indices, arg1.indptr), dim=dim, dtype=dtype, **kwargs
            )

        elif isinstance(arg1, (tuple, list)):
            if isinstance(arg1[0], Integral):
                self.__init_shape(
                    arg1, dim=dim, dtype=dtype, nnzpr=nnzpr, nnz=nnz, **kwargs
                )

            elif len(arg1) != 3:
                raise ValueError(
                    self.__class__.__name__ + " sparse array *must* be created "
                    "with data, indices, indptr"
                )
            else:
                # Correct dimension according to passed array
                if len(arg1[0].shape) == 2:
                    dim = max(dim, arg1[0].shape[1])

                if dtype is None:
                    # The first element is the data
                    dtype = arg1[0].dtype

                # The first *must* be some sort of array
                if "shape" in kwargs:
                    shape = kwargs["shape"]

                else:
                    M = len(arg1[2]) - 1
                    N = ((np.amax(arg1[1]) // M) + 1) * M
                    shape = (M, N)

                self.__init_shape(shape, dim=dim, dtype=dtype, nnz=1, **kwargs)

                # Copy data to the arrays
                self.ptr = arg1[2].astype(int32, copy=False)
                """ int-array, ``self.shape[0]+1``
pointer index in the 1D column indices of the corresponding row
                """
                self.ncol = diff(self.ptr)
                """ int-array, ``self.shape[0]``
number of entries per row
                """
                self.col = arg1[1].astype(int32, copy=False)
                """ int-array
column indices of the sparse elements
                """
                self._nnz = len(self.col)
                self._D = empty([len(arg1[1]), self.shape[-1]], dtype=self.dtype)
                if len(arg1[0].shape) == 2:
                    self._D[:, :] = arg1[0]
                else:
                    self._D[:, 0] = arg1[0]
                if np.all(self.ncol <= 1):
                    self._finalized = True

    def __init_shape(self, arg1, dim=1, dtype=None, nnzpr=20, nnz=None, **kwargs):
        # The shape of the data...
        if len(arg1) == 2:
            # extend to extra dimension
            arg1 = arg1 + (dim,)
        elif len(arg1) != 3:
            raise ValueError(
                self.__class__.__name__
                + " unrecognized shape input, either a 2-tuple or 3-tuple is required"
            )

        # Set default dtype
        if dtype is None:
            dtype = np.float64

        # unpack size and check the sizes are "physical"
        M, N, K = arg1
        if M <= 0 or N <= 0 or K <= 0:
            raise ValueError(
                self.__class__.__name__
                + f" invalid size of sparse matrix, one of the dimensions is zero: M={M}, N={N}, K={K}"
            )

        # Store shape
        self._shape = (M, N, K)

        # Check default construction of sparse matrix
        nnzpr = max(nnzpr, 1)

        # Re-create options
        if nnz is None:
            # number of non-zero elements is NOT given
            nnz = M * nnzpr

        else:
            # number of non-zero elements is give AND larger
            # than the provided non-zero elements per row
            nnzpr = nnz // M

        # Correct input in case very few elements are requested
        nnzpr = max(nnzpr, 1)
        nnz = max(nnz, nnzpr * M)

        # Store number of columns currently hold
        # in the sparsity pattern
        self.ncol = _a.zerosi([M])
        # Create pointer array
        self.ptr = _a.cumsumi(_a.fulli(M + 1, nnzpr)) - nnzpr
        # Create column array
        self.col = _a.fulli(nnz, -1)
        # Store current number of non-zero elements
        self._nnz = 0

        # Important that this is zero
        # For instance one may set one dimension at a time
        # thus automatically zeroing the other dimensions.
        self._D = zeros([nnz, K], dtype)

    @classmethod
    def sparsity_union(
        cls,
        sparse_matrices: OrSequence[SparseMatrix],
        dtype=None,
        dim: Optional[int] = None,
        value: float = 0,
    ) -> Self:
        """Create a `SparseCSR` with constant fill value in all places that `sparse_matrices` have nonzeros

        By default the returned matrix will be sorted.

        Parameters
        ----------
        sparse_matrices :
            SparseCSRs to find the sparsity pattern union of.
        dtype : dtype, optional
            Output dtype. If not given, use the result dtype of the spmats.
        dim :
            If given, the returned SparseCSR will have this as dim.
            By default the first sparse matrix in `sparse_matrices` determines
            the resulting 3rd dimension.
        value :
            The used fill value.
        """
        if issparse(sparse_matrices) or isinstance(sparse_matrices, SparseCSR):
            sparse_matrices = [sparse_matrices]

        # short-hand
        spmats = sparse_matrices

        shape2 = spmats[0].shape[:2]
        if not all(shape2 == m.shape[:2] for m in spmats):
            raise ValueError(
                f"Cannot find sparsity union of differently shaped sparse matrices: "
                " & ".join(str(m.shape) for m in spmats)
            )

        if dim is not None:
            shape = shape2 + (dim,)
        elif len(spmats[0].shape) == 3:
            shape = shape2 + (spmats[0].shape[2],)
        else:  # csr_matrix
            shape = shape2 + (1,)

        if dtype is None:
            dtype = np.result_type(*(m.dtype for m in spmats))

        out = cls(shape, dtype=dtype, nnzpr=1, nnz=2)

        # Create sparsity union (columns/indices array)
        out_col = []
        for row in range(shape[0]):
            row_cols = []
            for mat in spmats:
                if isinstance(mat, SparseCSR):
                    row_cols.append(
                        mat.col[mat.ptr[row] : mat.ptr[row] + mat.ncol[row]]
                    )
                else:
                    # we have to ensure it is a csr matrix
                    mat = mat.tocsr()
                    row_cols.append(mat.indices[mat.indptr[row] : mat.indptr[row + 1]])
            out_col.append(np.unique(concatenate(row_cols)))
        # Put into the output
        out.ncol = _a.arrayi([len(cols) for cols in out_col])
        out.ptr = _ncol_to_indptr(out.ncol)
        out.col = concatenate(out_col).astype(np.int32, copy=False)
        out._nnz = len(out.col)
        out._D = full([out._nnz, out.dim], value, dtype=dtype)
        return out

    def diagonal(self) -> np.ndarray:
        r"""Return the diagonal elements from the matrix"""
        # get the diagonal components
        diag = np.zeros([self.shape[0], self.shape[2]], dtype=self.dtype)

        rows, cols = _to_coo(self, data=False)

        # Now retrieve rows and cols
        idx = array_arange(self.ptr[:-1], n=self.ncol, dtype=int32)
        # figure out the indices where we have a diagonal index
        diag_idx = np.equal(rows, cols)
        idx = idx[diag_idx]
        diag[rows[diag_idx]] = self._D[idx]
        if self.shape[2] == 1:
            return diag.ravel()
        return diag

    def diags(self, diagonals, offsets=0, dim: Optional[int] = None, dtype=None):
        """Create a `SparseCSR` with diagonal elements with the same shape as the routine

        Parameters
        ----------
        diagonals : scalar or array_like
           the diagonal values, if scalar the `shape` *must* be present.
        offsets : scalar or array_like
           the offsets from the diagonal for each of the components (defaults
           to the diagonal)
        dim :
           the extra dimension of the new diagonal matrix (default to the current
           extra dimension)
        dtype : numpy.dtype, optional
           the data-type to create (default to `numpy.float64`)
        """
        if dim is None:
            dim = self.shape[2]
        diagonals = np.asarray(diagonals)
        if dtype is None:
            dtype = np.result_type(self.dtype, diagonals.dtype)

        # Now create the sparse matrix
        shape = list(self.shape)
        shape[2] = dim
        shape = tuple(shape)

        offsets = array_fill_repeat(offsets, shape[0], cls=np.int32)

        # Create the index-pointer, data and values
        data = array_fill_repeat(diagonals, shape[0], axis=0, cls=dtype)
        indices = _a.arangei(shape[0]) + offsets

        # create the pointer.
        idx_ok = valid_index(indices, shape[1])
        data = data[idx_ok]
        ptr1 = _a.onesi(shape[0])
        ptr1[~idx_ok] = 0
        indices = indices[idx_ok]
        ptr = _a.emptyi(shape[0] + 1)
        ptr[0] = 0
        ptr[1:] = np.cumsum(ptr1)

        # Delete the last entry, regardless of the size, the diagonal
        D = self.__class__((data, indices, ptr), shape=shape, dtype=dtype)

        return D

    def empty(self, keep_nnz: bool = False) -> None:
        """Delete all sparse information from the sparsity pattern

        Essentially this deletes all entries.

        Parameters
        ----------
        keep_nnz :
           if ``True`` keeps the sparse elements *as is*.
           I.e. it will merely set the stored sparse elements to zero.
           This may be advantagegous when re-constructing a new sparse
           matrix from an old sparse matrix
        """
        self._D[:, :] = 0.0

        if not keep_nnz:
            self._finalized = False
            # The user does not wish to retain the
            # sparse pattern
            self.ncol[:] = 0
            self._nnz = 0
            # We do not mess with the other arrays
            # they may be obscure data any-way.

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the sparse matrix"""
        return self._shape

    @property
    def dim(self) -> int:
        """The extra dimensionality of the sparse matrix (elements per matrix element)"""
        return self.shape[2]

    @property
    def data(self) -> np.ndarray:
        """Data contained in the sparse matrix (numpy array of elements)"""
        return self._D

    @property
    def dtype(self):
        """The data-type in the sparse matrix"""
        return self._D.dtype

    @property
    def dkind(self):
        """The data-type in the sparse matrix (in str)"""
        return np.dtype(self._D.dtype).kind

    @property
    def nnz(self) -> int:
        """Number of non-zero elements in the sparse matrix"""
        return self._nnz

    def __len__(self) -> int:
        """Number of rows in the sparse matrix"""
        return self.shape[0]

    @property
    def finalized(self) -> bool:
        """Whether the contained data is finalized and non-used elements have been removed"""
        return self._finalized

    def finalize(self, sort: bool = True) -> None:
        """Finalizes the sparse matrix by removing all non-set elements

        One may still interact with the sparse matrix as one would previously.

        NOTE: This is mainly an internal used routine to ensure data structure
        when converting to :class:`~scipy.sparse.csr_matrix`

        Parameters
        ----------
        sort :
           sort the column indices for each row
        """
        if self.finalized:
            return

        # Create and index array to retain the indices we want
        row, col, D = _to_coo(self)
        self.col = col
        self._D = D
        self.ptr = _ncol_to_indptr(self.ncol)

        if sort:
            # first sort according to row (which is already sorted)
            # then sort by col
            idx = lexsort((col, row))
            # direct re-order
            col[:] = col[idx]
            D[:, :] = D[idx, :]

            # check that we don't have any duplicate values
            duplicates = np.logical_and(diff(col) == 0, diff(row) == 0).nonzero()[0]

            if len(duplicates) > 0:
                raise SislError(
                    "You cannot have two elements between the same "
                    + f"i,j index (i={row[duplicates]}, j={col[duplicates]})"
                )

        else:
            ptr = self.ptr
            for r in range(self.shape[0]):
                ptr1 = ptr[r]
                ptr2 = ptr[r + 1]
                if unique(col[ptr1:ptr2]).shape[0] != ptr2 - ptr1:
                    raise SislError(
                        "You cannot have two elements between the same "
                        + f"i,j index (i={r}), something has went terribly wrong."
                    )

        if len(col) != self.nnz:
            raise SislError(
                "Final size in the sparse matrix finalization went wrong."
            )  # pragma: no cover

        # Check that all column indices are within the expected shape
        if invalid_index(self.col, self.shape[1]).any():
            warn(
                "Sparse matrix contains column indices outside the shape "
                "of the matrix. Data may not represent what is expected!"
            )

        # Signal that we indeed have finalized the data
        self._finalized = sort

    @singledispatchmethod
    def _sanitize(self, idx, axis: int = 0) -> ndarray:
        """Sanitize the input indices to a conforming numpy array"""
        if idx is None:
            if axis < 0:
                return _a.arangei(np.max(self.shape))
            return _a.arangei(self.shape[axis])
        idx = _a.asarrayi(idx)
        if idx.size == 0:
            return _a.asarrayi([])
        if idx.dtype == bool_:
            return idx.nonzero()[0].astype(np.int32, copy=False)
        return idx

    @_sanitize.register
    def _(self, idx: ndarray, axis: int = 0) -> ndarray:
        if idx.dtype == bool_:
            return np.flatnonzero(idx).astype(np.int32)
        return idx.astype(np.int32, copy=False)

    @_sanitize.register
    def _(self, idx: slice, axis: int = 0) -> ndarray:
        idx = idx.indices(self.shape[axis])
        return _a.arangei(*idx)

    def edges(self, rows: SeqOrScalarInt, exclude: Optional[SeqOrScalarInt] = None):
        """Retrieve edges (connections) of given `rows`

        The returned edges are unique and sorted (see `numpy.unique`).

        Parameters
        ----------
        rows :
            the edges are returned only for the given row
        exclude :
           remove edges which are in the `exclude` list.
        """
        rows = unique(self._sanitize(rows))
        if exclude is None:
            exclude = []
        else:
            exclude = unique(self._sanitize(exclude))

        # Now get the edges
        ptr = self.ptr
        ncol = self.ncol

        # Create column indices
        edges = unique(self.col[array_arange(ptr[rows], n=ncol[rows])])

        if len(exclude) > 0:
            # Return the difference to the exclude region, we know both are unique
            return setdiff1d(edges, exclude, assume_unique=True)
        return edges

    def delete_columns(self, cols: SeqOrScalarInt, keep_shape: bool = False):
        """Delete all columns in `columns` (in-place action)

        Parameters
        ----------
        columns :
           columns to delete from the sparse pattern
        keep_shape :
           whether the ``shape`` of the object should be retained, if ``True`` all higher
           columns will be shifted according to the number of columns deleted below,
           if ``False``, only the elements will be deleted.
        """
        # Shorthand function for retrieval
        cnz = count_nonzero

        # Sort the columns
        cols = unique(self._sanitize(cols, axis=1))
        n_cols = cnz(cols < self.shape[1])

        # Grab pointers
        ptr = self.ptr
        ncol = self.ncol
        col = self.col

        # Get indices of deleted columns
        idx = array_arange(ptr[:-1], n=ncol)
        # Convert to boolean array where we have columns to be deleted
        lidx = isin(col[idx], cols)
        # Count number of deleted entries per row
        ndel = _a.fromiteri(map(count_nonzero, split(lidx, _a.cumsumi(ncol[:-1]))))
        # Backconvert lidx to deleted indices
        lidx = idx[lidx]
        del idx

        if len(lidx) == 0:
            # Simply update the shape and return
            # We have nothing to delete!
            if not keep_shape:
                shape = list(self.shape)
                shape[1] -= n_cols
                self._shape = tuple(shape)
            return

        # Reduce the column indices and the data
        self.col = delete(self.col, lidx)
        self._D = delete(self._D, lidx, axis=0)
        del lidx

        # Update pointer
        #  col
        col = self.col

        # Figure out if it is necessary to update columns
        # This is only necessary when the deleted columns
        # are not the *last* columns
        update_col = not keep_shape
        if update_col:
            # Check that we really do have to update
            update_col = np_any(cols < self.shape[1] - n_cols)

        # Correct number of elements per column, and the pointers
        ncol[:] -= ndel
        ptr[1:] -= _a.cumsumi(ndel)

        if update_col:
            # Create a count array to subtract
            count = _a.zerosi(self.shape[1])
            count[cols] = 1
            count = _a.cumsumi(count)

            # Recreate pointers due to deleted indices
            idx = array_arange(ptr[:-1], n=ncol)
            col[idx] -= count[col[idx]]
            del idx

        # Update number of non-zeroes
        self._nnz = int(ncol.sum())

        if not keep_shape:
            shape = list(self.shape)
            shape[1] -= n_cols
            self._shape = tuple(shape)

    def _clean_columns(self):
        """Remove all intrinsic columns that are not defined in the sparse matrix
        (below 0 or above nc)"""
        # Grab pointers
        ptr = self.ptr
        ncol = self.ncol
        col = self.col

        # Number of columns
        nc = self.shape[1]

        # Get indices of columns
        idx = array_arange(ptr[:-1], n=ncol)
        # Convert to boolean array where we have columns to be deleted
        lidx = invalid_index(col[idx], nc)
        # Count number of deleted entries per row
        ndel = _a.fromiteri(map(count_nonzero, split(lidx, _a.cumsumi(ncol[:-1]))))
        # Backconvert lidx to deleted indices
        lidx = idx[lidx]
        del idx

        # Reduce column indices and data
        self.col = delete(self.col, lidx)
        self._D = delete(self._D, lidx, axis=0)
        del lidx

        # Update number of entries per row, and pointers
        ncol[:] -= ndel
        ptr[1:] -= _a.cumsumi(ndel)

        # Update number of non-zeroes
        self._nnz = int(ncol.sum())

        # We are *only* deleting columns, so if it is finalized,
        # it will still be

    def translate_columns(
        self,
        old: SeqOrScalarInt,
        new: SeqOrScalarInt,
        rows: Optional[SeqOrScalarInt] = None,
        clean: bool = True,
    ):
        """Takes all `old` columns and translates them to `new`.

        Parameters
        ----------
        old :
           old column indices
        new :
           new column indices
        rows :
           only translate columns for the given rows
        clean :
           whether the new translated columns, outside the shape, should be deleted or not (default delete)
        """
        old = self._sanitize(old, axis=1)
        new = self._sanitize(new, axis=1)

        if len(old) != len(new):
            raise ValueError(
                f"{self.__class__.__name__}.translate_columns requires input and output columns with "
                "equal length"
            )

        if allclose(old, new):
            # No need to translate anything...
            return

        if invalid_index(old, self.shape[1]).any():
            raise ValueError(
                f"{self.__class__.__name__}.translate_columns has non-existing old column values"
            )

        # Now do the translation
        pvt = _a.arangei(self.shape[1])
        pvt[old] = new

        # Get indices of valid column entries
        if rows is None:
            idx = array_arange(self.ptr[:-1], n=self.ncol)
        else:
            idx = array_arange(self.ptr[rows], n=self.ncol[rows])
            # Convert the old column indices to new ones
        self.col[idx] = pvt[self.col[idx]]

        # After translation, set to not finalized
        self._finalized = False
        if clean:
            if invalid_index(new, self.shape[1]).any():
                self._clean_columns()

    def scale_columns(
        self,
        cols: SeqOrScalarInt,
        scale: SeqOrScalarFloat,
        rows: Optional[SeqOrScalarInt] = None,
    ):
        r"""Scale all values with certain column values with a number

        This will multiply all values with certain column values with `scale`

        .. math::
            \mathbf M\[\mathrm{rows}, \mathrm{cols}\] *= \mathrm{scale}

        This is an in-place operation.

        Parameters
        ----------
        cols :
           column indices to scale
        scale :
           scale value for each value (if array-like it has to have the same
           dimension as the sparsity dimension)
        rows :
           only scale the column values that exists in these rows, default to all
        """
        cols = self._sanitize(cols, axis=1)

        if invalid_index(cols, self.shape[1]).any():
            raise ValueError(
                f"{self.__class__.__name__}.scale_columns has non-existing old column values"
            )

        # Find indices
        if rows is None:
            idx = array_arange(self.ptr[:-1], n=self.ncol)
        else:
            idx = array_arange(self.ptr[rows], n=self.ncol[rows])
        scale_idx = np.isin(self.col[idx], cols).nonzero()[0]

        # Scale values where columns coincide with scaling factor
        self._D[idx[scale_idx]] *= scale

    def toarray(self):
        """Return a dense `numpy.ndarray` which has 3 dimensions (self.shape)"""
        return sparse_dense(self)

    def todense(self):
        """Return a dense `numpy.ndarray` which has 3 dimensions (self.shape)"""
        return sparse_dense(self)

    def spsame(self, other) -> bool:
        """Check whether two sparse matrices have the same non-zero elements

        Parameters
        ----------
        other : SparseCSR

        Returns
        -------
        bool
           true if the same non-zero elements are in the matrices (but not necessarily the same values)
        """
        if self.shape[:2] != other.shape[:2]:
            return False

        sptr = self.ptr
        sncol = self.ncol
        scol = self.col
        optr = other.ptr
        oncol = other.ncol
        ocol = other.col

        # Easy check for non-equal number of elements
        if not np.array_equal(self.ncol, other.ncol):
            return False

        for r in range(self.shape[0]):
            inter = intersect1d(
                scol[sptr[r] : sptr[r] + sncol[r]],
                ocol[optr[r] : optr[r] + oncol[r]],
            )
            if len(inter) != sncol[r]:
                return False
        return True

    def align(self, other):
        """Aligns this sparse matrix with the sparse elements of the other sparse matrix

        Routine for ensuring that all non-zero elements in `other` are also in this
        object.

        I.e. this will, possibly, change the sparse elements in-place.

        A ``ValueError`` will be raised if the shapes are not mergeable.

        Parameters
        ----------
        other : SparseCSR
           the other sparse matrix to align.
        """

        if self.shape[:2] != other.shape[:2]:
            raise ValueError("Aligning two sparse matrices requires same shapes")

        sptr = self.ptr
        sncol = self.ncol
        scol = self.col
        optr = other.ptr
        oncol = other.ncol
        ocol = other.col
        for r in range(self.shape[0]):
            # pointers
            sn = sncol[r]
            op = optr[r]
            on = oncol[r]

            if sn == 0:
                self._extend(r, ocol[op : op + on], False)
                continue

            sp = sptr[r]
            adds = setdiff1d(ocol[op : op + on], scol[sp : sp + sn])
            if len(adds) > 0:
                # simply extend the elements
                self._extend(r, adds, False)

    def iter_nnz(self, rows: Optional[SeqOrScalarInt] = None):
        """Iterations of the non-zero elements, returns a tuple of row and column with non-zero elements

        An iterator returning the current row index and the corresponding column index.

        >>> for r, c in self:

        In the above case ``r`` and ``c`` are rows and columns such that

        >>> self[r, c]

        returns the non-zero element of the sparse matrix.

        Parameters
        ----------
        row : int or array_like of int
           only loop on the given row(s) default to all rows
        """
        if rows is None:
            # loop on rows
            for r in range(self.shape[0]):
                n = self.ncol[r]
                ptr = self.ptr[r]
                for c in self.col[ptr : ptr + n]:
                    yield r, c
        else:
            for r in self._sanitize(rows).ravel():
                n = self.ncol[r]
                ptr = self.ptr[r]
                for c in self.col[ptr : ptr + n]:
                    yield r, c

    # Define default iterator
    __iter__ = iter_nnz

    def _extend(self, i, j, ret_indices=True):
        """Extends the sparsity pattern to retain elements `j` in row `i`

        Parameters
        ----------
        i : int
           the row of the matrix
        j : int or array_like
           columns belonging to row `i` where a non-zero element is stored.
        ret_indices : bool, optional
           also return indices (otherwise, return nothing)

        Returns
        -------
        numpy.ndarray
           indices of existing/added elements (only for `ret_indices` true)

        Raises
        ------
        IndexError
            for indices out of bounds
        """
        i = self._sanitize(i)
        if i.size == 0:
            return _a.arrayi([])
        if i.size > 1:
            raise ValueError(
                "extending the sparse matrix is only allowed for single rows at a time"
            )
        if invalid_index(i, self.shape[0]):
            raise IndexError(f"row index is out-of-bounds {i} : {self.shape[0]}")
        i1 = i + 1

        # We skip this check and let sisl die if wrong input is given...
        # if not isinstance(i, Integral):
        #    raise ValueError("Retrieving/Setting elements in a sparse matrix"
        #                     " must only be performed at one row-element at a time.\n"
        #                     "However, multiple columns at a time are allowed.")
        # Ensure flattened array...
        j = self._sanitize(j, axis=1).ravel()
        if len(j) == 0:
            return _a.arrayi([])
        if invalid_index(j, self.shape[1]).any():
            raise IndexError(f"column index is out-of-bounds {j} : {self.shape[1]}")

        # fast reference
        ptr = self.ptr
        col = self.col

        # Get index pointer
        ptr_i = int(ptr[i])
        ncol_i = int(self.ncol[i])

        # To create the indices for the sparse elements
        # we first find which values are _not_ in the sparse
        # matrix
        if ncol_i > 0:
            # Checks whether any non-zero elements are
            # already in the sparse pattern
            # If so we remove those from the j
            new_j = j[
                isin(j, col[ptr_i : ptr_i + ncol_i], invert=True, assume_unique=True)
            ]
        else:
            new_j = j

        # Get list of new elements to be added
        # astype(...) is necessary since len(...) returns a long
        # and adding long and 32 is horribly slow in Python!
        new_n = len(new_j)

        ncol_ptr_i = ptr_i + ncol_i

        # Check how many elements cannot fit in the currently
        # allocated sparse matrix...
        new_nnz = new_n - int(ptr[i1]) + ncol_ptr_i

        if new_nnz > 0:
            # Ensure that it is not-set as finalized
            # There is no need to set it all the time.
            # Simply because the first call to finalize
            # will reduce the sparsity pattern, which
            # on first expansion calls this part.
            self._finalized = False

            # Get how much larger we wish to create the sparse matrix...
            ns = max(self._ns, new_nnz)

            # ...expand size of the sparsity pattern...

            # Insert new empty elements in the column index
            # after the column
            self.col = insert(self.col, ncol_ptr_i, full(ns, -1, col.dtype))

            # update reference
            col = self.col

            # Insert zero data in the data array
            # We use `zeros` as then one may set each dimension
            # individually...
            self._D = insert(
                self._D, ptr[i1], zeros([ns, self.shape[2]], self._D.dtype), axis=0
            )

            # Lastly, shift all pointers above this row to account for the
            # new non-zero elements
            ptr[i1:] += int32(ns)

        if new_n > 0:
            # Ensure that we write the new elements to the matrix...

            # assign the column indices for the new entries
            # NOTE that this may not assign them in the order
            # of entry as new_j is sorted and thus new_j != j
            col[ncol_ptr_i : ncol_ptr_i + new_n] = new_j

            # Step the size of the stored non-zero elements
            self.ncol[i] += int32(new_n)

            ncol_ptr_i += new_n

            # Step the number of non-zero elements
            self._nnz += new_n

        # Now we have extended the sparse matrix to hold all
        # information that is required...

        # ... retrieve the indices and return
        if ret_indices:
            return indices(col[ptr_i:ncol_ptr_i], j, ptr_i)

    def _extend_empty(self, i, n):
        """Extends the sparsity pattern with `n` elements in row `i`

        Parameters
        ----------
        i : int
           the row of the matrix
        n : int
           number of elements to add in the space for row `i`

        Raises
        ------
        IndexError
            for indices out of bounds
        """
        if invalid_index(i, self.shape[0]):
            raise IndexError("row index is out-of-bounds")

        # fast reference
        i1 = i + 1

        # Ensure that it is not-set as finalized
        # There is no need to set it all the time.
        # Simply because the first call to finalize
        # will reduce the sparsity pattern, which
        # on first expansion calls this part.
        self._finalized = False

        # Insert new empty elements in the column index
        # after the column
        self.col = insert(
            self.col, self.ptr[i] + self.ncol[i], full(n, -1, self.col.dtype)
        )

        # Insert zero data in the data array
        # We use `zeros` as then one may set each dimension
        # individually...
        self._D = insert(
            self._D, self.ptr[i1], zeros([n, self.shape[2]], self._D.dtype), axis=0
        )

        # Lastly, shift all pointers above this row to account for the
        # new non-zero elements
        self.ptr[i1:] += int32(n)

    def _get(self, i, j):
        """Retrieves the data pointer arrays of the elements, if it is non-existing, it will return ``-1``

        Parameters
        ----------
        i : int
           the row of the matrix
        j : int or array_like of int
           columns belonging to row `i` where a non-zero element is stored.

        Returns
        -------
        numpy.ndarray
            indices of the existing elements
        """
        j = self._sanitize(j, axis=1)

        # Make it a little easier
        ptr = self.ptr[i]

        if j.ndim == 0:
            return indices(self.col[ptr : ptr + self.ncol[i]], j.ravel(), ptr)[0]
        return indices(self.col[ptr : ptr + self.ncol[i]], j, ptr)

    def _get_only(self, i, j):
        """Retrieves the data pointer arrays of the elements, only return elements in the sparse array

        Parameters
        ----------
        i : int
           the row of the matrix
        j : int or array_like of int
           columns belonging to row `i` where a non-zero element is stored.

        Returns
        -------
        numpy.ndarray
            indices of existing elements
        """
        j = self._sanitize(j, axis=1).ravel()

        # Make it a little easier
        ptr = self.ptr[i]

        return indices_only(self.col[ptr : ptr + self.ncol[i]], j) + ptr

    def __delitem__(self, key):
        """Remove items from the sparse patterns"""
        # Get indices of sparse data (-1 if non-existing)
        if len(key) > 2:
            raise ValueError(
                f"{self.__class__.__name__}.__delitem__ requires "
                "key to only be of length 2 (cannot delete sub-sets of "
                "the last dimension."
            )

        i = self._sanitize(key[0], axis=0)
        key1 = self._sanitize(key[1], axis=1)
        if i.size > 1:
            for I in i:
                del self[I, key1]
            return

        # We can only delete unique values, trying to delete
        # the same value twice is just not a good idea!
        index = unique(self._get_only(i, key1))

        if len(index) == 0:
            # There are no elements to delete...
            return

        # Get short-hand
        ptr = self.ptr
        ncol = self.ncol

        # Get original values
        sl = slice(ptr[i], ptr[i] + ncol[i], None)
        oC = self.col[sl].copy()
        oD = self._D[sl, :].copy()
        self.col[sl] = -1
        self._D[sl, :] = 0

        # Now create the compressed data...
        index -= ptr[i]
        keep = isin(_a.arangei(ncol[i]), index, invert=True)

        # Update new count of the number of
        # non-zero elements
        n_index = int32(len(index))
        ncol[i] -= n_index

        # Now update the column indices and the data
        sl = slice(ptr[i], ptr[i] + ncol[i], None)
        self.col[sl] = oC[keep]
        self._D[sl, :] = oD[keep, :]

        self._finalized = False
        self._nnz -= n_index

    def __getitem__(self, key):
        """Intrinsic sparse matrix retrieval of a non-zero element"""

        # Get indices of sparse data (-1 if non-existing)
        get_idx = self._get(key[0], key[1])
        dim0 = get_idx.ndim == 0
        if dim0:
            n = 1
        else:
            n = len(get_idx)

        # Indices of existing values in return array
        ret_idx = atleast_1d(get_idx >= 0).nonzero()[0]
        # Indices of existing values in get array
        get_idx = get_idx.ravel()[ret_idx]

        # Check which data to retrieve
        if len(key) > 2:
            # user requests a specific element
            # get dimension retrieved
            r = zeros(n, dtype=self._D.dtype)
            r[ret_idx] = self._D[get_idx, key[2]]

        else:
            # user request all stored data

            s = self.shape[2]
            if s == 1:
                r = zeros(n, dtype=self._D.dtype)
                r[ret_idx] = self._D[get_idx, 0]
            else:
                r = zeros([n, s], dtype=self._D.dtype)
                r[ret_idx, :] = self._D[get_idx, :]

        if dim0:
            if r.size == 1:
                return r.ravel()[0]
            return np.squeeze(r, axis=-2)
        return r

    def __setitem__(self, key, data):
        """Intrinsic sparse matrix assignment of the item.

        It will only allow to set the data in the sparse
        matrix if the dimensions match.

        If the `data` parameter is ``None`` or an array
        only with ``None`` then the data will not be stored.
        """
        # Ensure data type... possible casting...
        if data is None:
            return

        # Sadly, converting integers with None
        # will NOT produce nan's.
        # Hence, this will only work with floats, etc.
        # TODO we need some way to reduce these things
        # for integer stuff.
        data = asarray(data, self._D.dtype)
        # Places where there are nan will be set to zero
        data[isnan(data)] = 0

        # Determine how the indices should work
        i = self._sanitize(key[0])
        j = self._sanitize(key[1], axis=1)
        if i.size > 1 and isinstance(j, (list, ndarray)):
            # Create a b-cast object to iterate
            # Note that this does not do the actual b-casting and thus
            # we can iterate and operate as though it was an actual array
            # When doing *array* assignments the data array *have* to be
            # a matrix-like object (one cannot assume linear data indices
            # works)
            ij = broadcast(i, j)
            if data.ndim == 0:
                # enables checking for shape values
                data.shape = (1,)
            elif data.ndim == 1:
                # this corresponds to:
                #  [:, :, :] = data.reshape(1, 1, -1)
                if ij.ndim == 2:
                    data.shape = (1, -1)
                # if ij.ndim == 1 we shouldn't need
                # to reshape data since it should correspond to each value
            elif data.ndim == 2:
                if ij.ndim == 2:
                    # this means we have two matrices
                    # We will *not* allow any b-casting:
                    # [:, :, :] = data[:, :]
                    # *only* if
                    # do a sanity check
                    if self.dim > 1 and len(key) == 2:
                        raise ValueError(
                            "could not broadcast input array from shape {} into shape {}".format(
                                data.shape, ij.shape + (self.dim,)
                            )
                        )
                    if len(key) == 3:
                        if atleast_1d(key[2]).size > 1:
                            raise ValueError(
                                "could not broadcast input array from shape {} into shape {}".format(
                                    data.shape, ij.shape + (atleast_1d(key[2]).size,)
                                )
                            )
                    # flatten data
                    data.shape = (-1,)
                # ij.ndim == 1
                # this should correspond to the diagonal specification case
                # and we don't need to do anything
                # if ij.size != data.shape[0] an error should occur down below
            elif data.ndim == 3:
                if ij.ndim != 2:
                    raise ValueError(
                        "could not broadcast input array from 3 dimensions into 2"
                    )
                data.shape = (-1, data.shape[2])

            # Now we need to figure out the final dimension and how to
            # assign elements.
            if len(key) == 3:
                # we are definitely assigning the final dimension
                k = atleast_1d(key[2])

                if len(k) == 1 and data.ndim == 2:
                    data.shape = (-1,)

                if data.shape[0] == ij.size:
                    for (i, j), d in zip(ij, data):
                        self.__setitem__((i, j, key[2]), d)
                else:
                    for i, j in ij:
                        self.__setitem__((i, j, key[2]), data)
            else:
                if data.shape[0] == ij.size:
                    for (i, j), d in zip(ij, data):
                        self.__setitem__((i, j), d)
                else:
                    for i, j in ij:
                        self.__setitem__((i, j), data)

            return

        # Retrieve indices in the 1D data-structure
        index = self._extend(i, j)

        if len(key) > 2:
            # Explicit data of certain dimension
            self._D[index, key[2]] = data

        else:
            # Ensure correct shape
            if data.size == 1:
                data.shape = (1, 1)
            else:
                data.shape = (-1, self.shape[2])

            # Now there are two cases
            if data.shape[0] == 1:
                # we copy all elements
                self._D[index, :] = data[None, :]

            else:
                # each element have different data
                self._D[index, :] = data[:, :]

    def __contains__(self, key):
        """Check whether a sparse index is non-zero"""
        # Get indices of sparse data (-1 if non-existing)
        return np_all(self._get(key[0], key[1]) >= 0)

    def nonzero(self, rows: Optional[SeqOrScalarInt] = None, only_cols: bool = False):
        """Row and column indices where non-zero elements exists

        Parameters
        ----------
        rows :
           only return the tuples for the requested rows, default is all rows
        only_cols :
           only return the non-zero columns
        """
        if only_cols:
            return _to_cd(self, data=False, rows=rows)
        return _to_coo(self, data=False, rows=rows)

    def eliminate_zeros(self, atol: float = 0.0) -> None:
        """Remove all zero elememts from the sparse matrix

        This is an *in-place* operation

        Parameters
        ----------
        atol :
            absolute tolerance below or equal this value will be considered 0.
        """
        shape2 = self.shape[2]

        ptr = self.ptr
        ncol = self.ncol
        col = self.col
        D = self._D

        # Get short-hand
        nsum = np.sum
        nabs = np.abs
        arangei = _a.arangei

        # Fast check to see for return (skips loop)
        idx = array_arange(ptr[:-1], n=ncol)
        if (nsum(nabs(D[idx, :]) <= atol, axis=1) == shape2).nonzero()[0].sum() == 0:
            return

        for r in range(self.shape[0]):
            # Create short-hand slice
            idx = arangei(ptr[r], ptr[r] + ncol[r])

            # Retrieve columns with zero values (summed over all elements)
            C0 = (nsum(nabs(D[idx, :]) <= atol, axis=1) == shape2).nonzero()[0]
            if len(C0) == 0:
                continue

            # Remove all entries with 0 values
            del self[r, col[idx[C0]]]

    def copy(self, dims: Optional[SeqOrScalarInt] = None, dtype=None):
        """A deepcopy of the sparse matrix

        Parameters
        ----------
        dims :
           which dimensions to store in the copy, defaults to all.
        dtype : `numpy.dtype`
           this defaults to the dtype of the object,
           but one may change it if supplied.
        """
        # Create sparse matrix (with only one entry per
        # row, we overwrite it immediately afterward)
        if dims is None:
            dims = range(self.dim)
        elif isinstance(dims, Integral):
            dims = [dims]
        dim = len(dims)

        if dtype is None:
            dtype = self.dtype

        # Create correct input
        shape = list(self.shape[:])
        shape[2] = dim

        new = self.__class__(shape, dtype=dtype, nnz=1)

        # The default sizes are not passed
        # Hence we *must* copy the arrays directly
        new.ptr[:] = self.ptr[:]
        new.ncol[:] = self.ncol[:]
        new.col = self.col.copy()
        new._nnz = self.nnz

        new._D = empty([len(self.col), dim], dtype)
        for i, dim in enumerate(dims):
            new._D[:, i] = self._D[:, dim]

        # Mark it as the same state as the other one
        new._finalized = self._finalized

        return new

    def tocsr(self, dim: int = 0, **kwargs) -> csr_matrix:
        """Convert dimension `dim` into a :class:`~scipy.sparse.csr_matrix` format

        Parameters
        ----------
        dim :
           dimension of the data returned in a scipy sparse matrix format
        **kwargs:
           arguments passed to the :class:`~scipy.sparse.csr_matrix` routine
        """
        shape = self.shape[:2]
        if self.finalized:
            # Easy case...
            return csr_matrix(
                (
                    self._D[:, dim].copy(),
                    self.col.astype(int32, copy=True),
                    self.ptr.astype(int32, copy=True),
                ),
                shape=shape,
                **kwargs,
            )

        # Use array_arange
        idx = array_arange(self.ptr[:-1], n=self.ncol)
        # create new pointer
        ptr = _ncol_to_indptr(self.ncol)

        return csr_matrix(
            (self._D[idx, dim].copy(), self.col[idx], ptr.astype(int32, copy=False)),
            shape=shape,
            **kwargs,
        )

    def transform(self, matrix, dtype=None):
        r"""Apply a linear transformation :math:`R^n \rightarrow R^m` to the :math:`n`-dimensional elements of the sparse matrix

        Notes
        -----
        The transformation matrix does *not* act on the rows and columns, only on the
        final dimension of the matrix.

        Parameters
        ----------
        matrix : array_like
            transformation matrix of shape :math:`m \times n`, :math:`n` should correspond to
            the number of elements in ``self.shape[2]``
        dtype : numpy.dtype, optional
            defaults to the common dtype of the object and the transformation matrix
        """
        matrix = np.asarray(matrix)

        if dtype is None:
            # no need for result_type
            # result_type differs from promote_types, only in the case
            # where the input arguments mixes scalars and arrays (not the case here)
            dtype = np.promote_types(self.dtype, matrix.dtype)

        if matrix.shape[1] != self.shape[2]:
            raise ValueError(
                f"{self.__class__.__name__}.transform incompatible "
                f"transformation matrix and spin dimensions: "
                f"matrix.shape={matrix.shape} and self.spin={self.shape[2]} ; out.spin={matrix.shape[0]}"
            )

        # set dimension of new sparse matrix
        new_dim = matrix.shape[0]
        shape = list(self.shape[:])
        shape[2] = new_dim

        new = self.__class__(shape, dtype=dtype, nnz=1)

        copyto(new.ptr, self.ptr, casting="no")
        copyto(new.ncol, self.ncol, casting="no")
        new.col = self.col.copy()
        new._nnz = self.nnz

        new._D = self._D.dot(matrix.T).astype(dtype, copy=False)

        new._finalized = self._finalized

        return new

    def astype(self, dtype, copy: bool = True) -> Self:
        """Convert the stored data-type to something else

        Parameters
        ----------
        dtype :
            the new dtype for the sparse matrix
        copy :
            copy when needed, or do not copy when not needed.
        """
        old_dtype = np.dtype(self.dtype)
        new_dtype = np.dtype(dtype)

        if old_dtype == new_dtype:
            if copy:
                return self.copy()
            return self

        new = self.copy()
        new._D = new._D.astype(dtype, copy=copy)
        return new

    @classmethod
    def fromsp(cls, sparse_matrices: OrSequence[SparseMatrix], dtype=None):
        """Combine multiple single-dimension sparse matrices into one SparseCSR matrix

        The different sparse matrices need not have the same sparsity pattern.

        Parameters
        ----------
        sparse_matrices :
            any sparse matrix which can convert to a `scipy.sparse.csr_matrix` matrix
        dtype : numpy.dtype, optional
            data-type to store in the matrix, default to largest ``dtype`` for the
            passed sparse matrices
        """
        if issparse(sparse_matrices) or isinstance(sparse_matrices, SparseCSR):
            sparse_matrices = list(sparse_matrices)

        # short-hand
        spmats = sparse_matrices

        if dtype is None:
            dtype = np.result_type(*[sp.dtype for sp in spmats])

        # Number of dimensions
        def get_3rddim(spmat):
            if isinstance(spmat, SparseCSR):
                return spmat.shape[2]
            return 1

        dim = sum(map(get_3rddim, spmats))

        if len(spmats) == 1:
            spmat = spmats[0]
            if isinstance(spmat, SparseCSR):
                return spmat.copy(dtype=dtype)

            # We are dealing with something different from a SparseCSR
            # Likely some scipy.sparse matrix.
            # Use that one.
            m = spmat.tocsr()
            out = cls(m.shape + (1,), nnzpr=1, nnz=1, dtype=dtype)
            out.col = m.indices.astype(np.int32, copy=True)
            out.ptr = m.indptr.astype(np.int32, copy=True)
            out.ncol = np.diff(out.ptr)
            out._nnz = len(out.col)
            out._D = m.data.reshape(-1, 1).astype(dtype, copy=True)
            return out

        # Pre-allocate by finding sparsity pattern union
        out = cls.sparsity_union(spmats, dim=dim, dtype=dtype)

        # For all non-sisl objects, finalize it by sorting
        # indices, that should be really fast!
        def finalize(spmat):
            if isinstance(spmat, SparseCSR):
                return spmat

            spmat = spmat.tocsr()
            spmat.sort_indices()
            return spmat

        spmats = list(map(finalize, spmats))

        # Now transfer the data
        for r in range(out.shape[0]):
            # loop across all rows of the sparse matrix
            osl = slice(out.ptr[r], out.ptr[r] + out.ncol[r])
            ocol = out.col[osl]
            if len(ocol) == 0:
                continue

            im = 0
            for m in spmats:
                sorted = True
                dims = 1
                if isinstance(m, SparseCSR):
                    sorted = m.finalized
                    dims = m.shape[2]

                    msl = slice(m.ptr[r], m.ptr[r] + m.ncol[r])
                    mcol = m.col[msl]
                    D = m._D
                else:
                    msl = slice(m.indptr[r], m.indptr[r + 1])
                    mcol = m.indices[msl]
                    D = m.data.reshape(-1, 1)

                out_idx = indices(ocol, mcol, osl.start, both_sorted=sorted)

                if len(out_idx) > 0:
                    out._D[out_idx, im : im + dims] = D[msl, :]

                # step the dimensions we write to
                im += dims

        return out

    def remove(self, indices: SeqOrScalarInt) -> Self:
        """Return a new sparse CSR matrix with all the indices removed

        Parameters
        ----------
        indices :
           the indices of the rows *and* columns that are removed in the sparse pattern
        """
        indices = self._sanitize(indices, axis=-1)

        # Check if we have a square matrix or a rectangular one
        if self.shape[0] >= self.shape[1]:
            rindices = delete(_a.arangei(self.shape[0]), indices)

        else:
            rindices = delete(_a.arangei(self.shape[1]), indices)

        return self.sub(rindices)

    def sub(self, indices: SeqOrScalarInt) -> Self:
        """Create a new sparse CSR matrix with the data only for the given rows and columns

        All rows and columns in `indices` are retained, everything else is removed.

        Parameters
        ----------
        indices :
           the indices of the rows *and* columns that are retained in the sparse pattern
        """
        indices = self._sanitize(indices, axis=-1).ravel()

        # Check if we have a square matrix or a rectangular one
        if self.shape[0] == self.shape[1]:
            ridx = indices

        elif self.shape[0] < self.shape[1]:
            ridx = indices[indices < self.shape[0]]

        elif self.shape[0] > self.shape[1]:
            ridx = indices

        # Number of rows, columns
        nr = len(ridx)
        nc = count_nonzero(indices < self.shape[1])

        # Fix the pivoting indices with the new indices
        pvt = _a.fulli([max(self.shape[0], self.shape[1])], -1)
        pvt[indices] = _a.arangei(len(indices))

        # Create the new SparseCSR
        # We use nnzpr = 1 because we will overwrite all quantities afterwards.
        csr = self.__class__((nr, nc, self.shape[2]), dtype=self.dtype, nnz=1)
        # Limit memory
        csr._D = empty([1])

        # Get views
        ptr1 = csr.ptr
        ncol1 = csr.ncol

        # Create the sub data
        take(self.ptr, ridx, out=ptr1[1:])
        # Place directly where it should be (i.e. re-use space)
        take(self.ncol, ridx, out=ncol1)

        # Create a 2D array to contain
        #   [0, :] the column indices to keep
        #   [1, :] the new column data
        # We do this because we can then use take on this array
        # and not two arrays.
        col_data = _a.emptyi([2, ncol1.sum()])

        # Create a list of ndarrays with indices of elements per row
        # and transfer to a linear index
        col_data[0, :] = array_arange(ptr1[1:], n=ncol1)

        # Reduce the column indices (note this also ensures that
        # it will work on non-finalized sparse matrices)
        col_data[1, :] = pvt[take(self.col, col_data[0, :])]

        # Count the number of items that are left in the sparse pattern
        # First recreate the new (temporary) pointer
        ptr1[0] = 0
        # Place it directly where it should be
        _a.cumsumi(ncol1, out=ptr1[1:])

        # Count number of entries
        idx_take = col_data[1, :] >= 0
        ncol1[:] = _a.fromiteri(map(count_nonzero, split(idx_take, ptr1[1:-1]))).ravel()

        # Convert to indices
        idx_take = idx_take.nonzero()[0]

        # Decrease column data and also extract the data
        col_data = take(col_data, idx_take, axis=1)
        del idx_take
        csr._D = take(self._D, col_data[0, :], axis=0)
        csr.col = col_data[1, :].copy()
        del col_data

        # Set the data for the new sparse csr
        csr.ptr[0] = 0
        _a.cumsumi(ncol1, out=csr.ptr[1:])
        csr._nnz = len(csr.col)

        return csr

    def transpose(self, sort: bool = True):
        """Create the transposed sparse matrix

        Parameters
        ----------
        sort :
           the returned columns for the transposed structure will be sorted
           if this is true, default

        Notes
        -----
        The components for each sparse element are not changed in this method.

        Returns
        -------
        object
            an equivalent sparse matrix with transposed matrix elements
        """
        # Create a temporary copy to put data into
        T = self.copy()
        # properly set the shape!
        T._shape = (self.shape[1], self.shape[0], self.shape[2])

        # clean memory to not crowd memory too much
        T.ptr = None
        T.col = None
        T.ncol = None
        T._D = None

        # First extract the actual data in COO format
        row, col, D = _to_coo(self)

        # Now we can re-create the sparse matrix
        # All we need is to count the number of non-zeros per column.
        rows, nrow = unique(col, return_counts=True)
        T.ncol = _a.zerosi(T.shape[0])
        T.ncol[rows] = nrow
        del rows

        if sort:
            # also sort individual rows for each column
            idx = lexsort((row, col))
        else:
            # sort columns to get transposed values.
            # This will randomize the rows
            idx = argsort(col)

        # Our new data will then be
        T.col = row[idx]
        del row
        T._D = D[idx]
        del D
        T.ptr = _ncol_to_indptr(T.ncol)

        # If `sort` we have everything sorted, otherwise it
        # is not ensured
        T._finalized = sort

        return T

    def __str__(self) -> str:
        """Representation of the sparse matrix model"""
        ints = self.shape[:] + (self.nnz,)
        return (
            self.__class__.__name__
            + "{{dim={2}, kind={kind},\n  rows: {0}, columns: {1},\n  non-zero: {3}\n}}".format(
                *ints, kind=self.dkind
            )
        )

    def __repr__(self) -> str:
        return f"<{self.__module__}.{self.__class__.__name__} shape={self.shape}, kind={self.dkind}, nnz={self.nnz}>"

    # numpy dispatch methods
    __array_priority__ = 14

    def __array__(self, dtype=None, *, copy: bool = False):
        out = self.toarray()
        if dtype is None:
            return out
        # Always no copy, since it will always be a copy!
        return out.astype(dtype, copy=False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop("out", None)

        if getattr(ufunc, "signature", None) is not None:
            # The signature is not a scalar operation
            return NotImplemented

        if out is not None:
            (out,) = out

        if method == "__call__":
            result = _ufunc_call(ufunc, *inputs, **kwargs)
        elif method == "reduce":
            result = _ufunc_reduce(ufunc, *inputs, **kwargs)
        elif method == "outer":
            # Currently I don't know what to do here
            # We don't have multidimensional sparse matrices,
            # but perhaps that could be needed later?
            return NotImplemented
        else:
            return NotImplemented

        if out is None:
            out = result
        elif isinstance(out, ndarray):
            out[...] = result[...]
        elif isinstance(out, SparseCSR):
            if out.shape != result.shape:
                raise ValueError(
                    f"non-broadcastable output operand with shape {out.shape} "
                    "doesn't match the broadcast shape {result.shape}"
                )
            out._finalized = result._finalized
            out.ncol[:] = result.ncol[:]
            out.ptr[:] = result.ptr[:]
            # this will copy
            out.col = result.col.copy()
            out._D = result._D.astype(out.dtype)
            out._nnz = result.nnz
            del result
        else:
            out = NotImplemented
        return out

    def __getstate__(self):
        """Return dictionary with the current state (finalizing the object may reduce memory footprint)"""
        d = {
            "shape": self._shape[:],
            "ncol": self.ncol.copy(),
            "col": self.col.copy(),
            "D": self._D.copy(),
            "finalized": self._finalized,
        }
        if not self.finalized:
            d["ptr"] = self.ptr.copy()
        return d

    def __setstate__(self, state):
        """Reset state of the object"""
        self._shape = tuple(state["shape"][:])
        self.ncol = state["ncol"]
        self.col = state["col"]
        self._D = state["D"]
        self._nnz = self.ncol.sum()
        self._finalized = state["finalized"]
        if self.finalized:
            self.ptr = _ncol_to_indptr(self.ncol)
        else:
            self.ptr = state["ptr"]


def _get_reduced_shape(arr):
    # return the reduced shape by removing any dimensions with length 1
    if isscalar(arr):
        return tuple()
    if isinstance(arr, (tuple, list)):
        n = len(arr)
        if n > 1:
            return (n,) + _get_reduced_shape(arr[0])
        return tuple()
    return np.squeeze(arr).shape


def _ufunc(ufunc, a, b, **kwargs):
    if issparse(a) or isinstance(a, (SparseCSR, tuple)):
        if issparse(b) or isinstance(b, (SparseCSR, tuple)):
            return _ufunc_sp_sp(ufunc, a, b, **kwargs)
        return _ufunc_sp_ndarray(ufunc, a, b, **kwargs)
    if isinstance(b, SparseCSR):
        return _ufunc_ndarray_sp(ufunc, a, b, **kwargs)
    return ufunc(a, b, **kwargs)


def _sp_data_cast(sp, result, dtype=None):
    """Converts the data in `sp` to the result data-type"""
    if dtype is None:
        dtype = result.dtype
    if sp.dtype != dtype:
        sp._D = sp._D.astype(dtype)


def _ufunc_sp_ndarray(ufunc, a, b, **kwargs):
    if len(_get_reduced_shape(b)) > 1:
        # there are shapes for individiual
        # we will now calculate a full matrix
        return ufunc(a.toarray(), b, **kwargs)

    # create a copy
    out = a.copy(dtype=kwargs.get("dtype", None))
    if out.ptr[-1] == out.nnz:
        out._D = ufunc(a._D, b, **kwargs)
    else:
        # limit the values
        # since slicing non-uniform ranges does not return
        # a view, we can't use
        #   ufunc(..., out=out._D[idx, :])
        idx = array_arange(a.ptr[:-1], n=a.ncol)
        D = ufunc(a._D[idx, :], b, **kwargs)
        _sp_data_cast(out, D)
        out._D[idx] = D
        del idx
    return out


def _ufunc_ndarray_sp(ufunc, a, b, **kwargs):
    if len(_get_reduced_shape(a)) > 1:
        # there are shapes for individiual
        # we will now calculate a full matrix
        return ufunc(a, b.toarray(), **kwargs)

    # create a copy
    out = b.copy(dtype=kwargs.get("dtype", None))
    if out.ptr[-1] == out.nnz:
        out._D = ufunc(a, b._D, **kwargs)
    else:
        # limit the values
        # since slicing non-uniform ranges does not return
        # a view, we can't use
        #   ufunc(..., out=out._D[idx, :])
        idx = array_arange(b.ptr[:-1], n=b.ncol)
        D = ufunc(a, b._D[idx, :], **kwargs)
        _sp_data_cast(out, D)
        out._D[idx] = D
        del idx
    return out


def _ufunc_sp_sp(ufunc, a, b, **kwargs):
    """Calculate ufunc on sparse matrices"""
    if isinstance(a, tuple):
        a = SparseCSR.fromsp(a)
    if isinstance(b, tuple):
        b = SparseCSR.fromsp(b)

    def accessors(mat):
        if isinstance(mat, SparseCSR):

            def rowslice(r):
                return slice(mat.ptr[r], mat.ptr[r] + mat.ncol[r])

            accessors = mat.dim, mat.col, mat._D, rowslice
            # check whether they are actually sorted
            if mat.finalized:
                rows, cols = _to_coo(mat, data=False)
                rows, cols = np.diff(rows), np.diff(cols)
                issorted = np.all(cols[rows == 0] > 0)
            else:
                issorted = False
        else:
            # makes this work for all matrices
            # and csr_matrix.tocsr is a no-op
            mat = mat.tocsr()
            mat.sort_indices()

            def rowslice(r):
                return slice(mat.indptr[r], mat.indptr[r + 1])

            accessors = 1, mat.indices, mat.data.reshape(-1, 1), rowslice
            issorted = mat.has_sorted_indices
        if issorted:
            indexfunc = lambda ocol, matcol, offset: indices(
                ocol, matcol, offset, both_sorted=True
            )
        else:
            indexfunc = (
                lambda ocol, matcol, offset: np.searchsorted(ocol, matcol) + offset
            )
        return accessors + (indexfunc,)

    adim, acol, adata, arow, afindidx = accessors(a)
    bdim, bcol, bdata, brow, bfindidx = accessors(b)

    if a.shape[:2] != b.shape[:2] or (adim != bdim and not (adim == 1 or bdim == 1)):
        raise ValueError(f"could not broadcast sparse matrices {a.shape} and {b.shape}")

    # create union of the sparsity pattern
    # create a fake *out* to grap the dtype
    dtype = kwargs.get("dtype")
    if dtype is None:
        out = ufunc(adata[:1, :], bdata[:1, :], **kwargs)
        dtype = out.dtype
    out = SparseCSR.sparsity_union([a, b], dim=max(adim, bdim), dtype=dtype)

    for r in range(out.shape[0]):
        offset = out.ptr[r]
        ocol = out.col[offset : offset + out.ncol[r]]

        asl = arow(r)
        aidx = afindidx(ocol, acol[asl], offset)
        asl = _a.arangei(asl.start, asl.stop)

        bsl = brow(r)
        bidx = bfindidx(ocol, bcol[bsl], offset)
        bsl = _a.arangei(bsl.start, bsl.stop)

        # Common indices
        iover, aover, bover, iaonly, ibonly = intersect_and_diff_sets(aidx, bidx)

        # the following size checks should solve ufunc calls with no elements (numpy >= 1.24)
        #   TypeError: No loop matching the specified signature and casting was found for ufunc equal

        # overlapping indices
        if iover.size > 0:
            out._D[iover, :] = ufunc(
                adata[asl[aover], :], bdata[bsl[bover], :], **kwargs
            )

        if iaonly.size > 0:
            # only a
            out._D[aidx[iaonly]] = ufunc(adata[asl[iaonly], :], 0, **kwargs)
        if ibonly.size > 0:
            # only b
            out._D[bidx[ibonly]] = ufunc(0, bdata[bsl[ibonly], :], **kwargs)

    return out


def _ufunc_call(ufunc, *in_args, **kwargs):
    # first process in_args to args
    # by numpy-fying and checking for sparsecsr
    args = []
    for arg in in_args:
        if isinstance(arg, SparseCSR):
            args.append(arg)
        elif isscalar(arg) or isinstance(arg, (tuple, list, ndarray)):
            args.append(arg)
        elif issparse(arg):
            args.append(arg)
        else:
            return

    if len(args) == 1:
        a = args[0]
        # create a copy
        out = a.copy(dtype=kwargs.get("dtype", None))
        if out.ptr[-1] == out.nnz:
            out._D = ufunc(a._D, **kwargs)
        else:
            # limit the values
            idx = array_arange(a.ptr[:-1], n=a.ncol)
            D = ufunc(a._D[idx, :], **kwargs)
            _sp_data_cast(out, D)
            out._D[idx] = D
            del idx
        return out

    def _(a, b):
        return _ufunc(ufunc, a, b, **kwargs)

    return reduce(_, args)


def _ufunc_reduce(ufunc, array, axis=0, *args, **kwargs):
    # currently the initial argument does not work properly if the
    # size isn't correct
    if np.asarray(kwargs.get("initial", 0.0)).ndim > 1:
        raise ValueError(
            f"{array.__class__.__name__}.{ufunc.__name__}.reduce currently does not implement initial values in different dimensions"
        )

    if isinstance(axis, (tuple, list, np.ndarray)):
        if len(axis) == 1:
            axis = axis[0]
        else:

            def wrap_axis(axis):
                if axis < 0:
                    return axis + len(array.shape)
                return axis

            axis = tuple(wrap_axis(ax) for ax in axis)
            if axis == (0, 1) or axis == (1, 0):
                return ufunc.reduce(array._D, axis=0, *args, **kwargs)
            raise NotImplementedError

    # correct axis
    if axis is None:
        # reduction on all axes
        return ufunc.reduce(array._D, axis=None, *args, **kwargs)
    elif axis < 0:
        # correct for negative axis specification
        axis = axis + len(array.shape)

    if axis == 0:
        # no need to sorting
        array = array.transpose(sort=False)
    elif axis == 1:
        pass
    elif axis == 2:
        out = array.copy(dims=0, dtype=kwargs.get("dtype", None))
        D = ufunc.reduce(array._D, axis=1, *args, **kwargs)
        _sp_data_cast(out, D, kwargs.get("dtype"))
        out._D[:, 0] = D
        return out
    else:
        raise ValueError(
            f"Unknown axis argument in ufunc.reduce call on {array.__class__.__name__}"
        )

    ret = ufunc.reduce(array._D[0:1, :], axis=0, *args, **kwargs)
    ret = empty([array.shape[0], array.shape[2]], dtype=kwargs.get("dtype", ret.dtype))

    # Now do ufunc calculations, note that initial gets passed directly
    ptr = array.ptr
    ncol = array.ncol
    for r in range(array.shape[0]):
        ret[r, :] = ufunc.reduce(
            array._D[ptr[r] : ptr[r] + ncol[r], :], axis=0, *args, **kwargs
        )
    return ret


@set_module("sisl")
def ispmatrix(matrix, map_row=None, map_col=None):
    """Iterator for iterating rows and columns for non-zero elements in a `scipy.sparse.*_matrix` (or `SparseCSR`)

    If either `map_row` or `map_col` are not None the generator will only yield
    the unique values.

    Parameters
    ----------
    matrix : scipy.sparse.sp_matrix
      the sparse matrix to iterate non-zero elements
    map_row : func, optional
      map each row entry through the function `map_row`, defaults to ``None`` which is
      equivalent to no mapping.
    map_col : func, optional
      map each column entry through the function `map_col`, defaults to ``None`` which is
      equivalent to no mapping.

    Yields
    ------
    int, int
       the row, column indices of the non-zero elements
    """

    if map_row is None and map_col is None:
        # Skip unique checks
        yield from _ispmatrix_all(matrix)
        return

    if map_row is None:
        map_row = lambda x: x
    if map_col is None:
        map_col = lambda x: x
    map_row = np.vectorize(map_row)
    map_col = np.vectorize(map_col)

    nrow = len(unique(map_row(arange(matrix.shape[0], dtype=int32))))
    ncol = len(unique(map_col(arange(matrix.shape[1], dtype=int32))))
    rows = zeros(nrow, dtype=np.bool_)
    cols = zeros(ncol, dtype=np.bool_)

    # Initialize the unique arrays
    rows[:] = False

    # Consider using the numpy nditer function for buffered iterations
    # it = np.nditer([geom.o2a(tmp.row), geom.o2a(tmp.col % geom.no), tmp.data],
    #               flags=['buffered'], op_flags=['readonly'])

    if issparse(matrix) and matrix.format == "csr":
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            if rows[rr]:
                continue
            rows[rr] = True
            cols[:] = False
            for ind in range(matrix.indptr[r], matrix.indptr[r + 1]):
                c = map_col(matrix.indices[ind])
                if cols[c]:
                    continue
                cols[c] = True
                yield rr, c

    elif issparse(matrix) and matrix.format == "lil":
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            if rows[rr]:
                continue
            rows[rr] = True
            cols[:] = False
            if len(matrix.rows[r]) == 0:
                continue
            for c in map_col(matrix.rows[r]):
                if cols[c]:
                    continue
                cols[c] = True
                yield rr, c

    elif issparse(matrix) and matrix.format == "coo":
        raise ValueError(
            "mapping and unique returns are not implemented for COO matrix"
        )

    elif issparse(matrix) and matrix.format == "csc":
        raise ValueError(
            "mapping and unique returns are not implemented for CSC matrix"
        )

    elif isinstance(matrix, SparseCSR):
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            if rows[rr]:
                continue
            rows[rr] = True
            cols[:] = False
            n = matrix.ncol[r]
            if n == 0:
                continue
            ptr = matrix.ptr[r]
            for c in map_col(matrix.col[ptr : ptr + n]):
                if cols[c]:
                    continue
                cols[c] = True
                yield rr, c

    else:
        raise NotImplementedError(
            "The iterator for this sparse matrix has not been implemented"
        )


def _ispmatrix_all(matrix):
    """Iterator for iterating rows and columns for non-zero elements in a ``scipy.sparse.*_matrix`` (or `SparseCSR`)

    Parameters
    ----------
    matrix : ``scipy.sparse.*_matrix``
      the sparse matrix to iterate non-zero elements

    Yields
    ------
    int, int
       the row, column indices of the non-zero elements
    """
    if issparse(matrix) and matrix.format == "csr":
        for r in range(matrix.shape[0]):
            for ind in range(matrix.indptr[r], matrix.indptr[r + 1]):
                yield r, matrix.indices[ind]

    elif issparse(matrix) and matrix.format == "lil":
        for r in range(matrix.shape[0]):
            for c in matrix.rows[r]:
                yield r, c

    elif issparse(matrix) and matrix.format == "coo":
        yield from zip(matrix.row, matrix.col)

    elif issparse(matrix) and matrix.format == "csc":
        for c in range(matrix.shape[1]):
            for ind in range(matrix.indptr[c], matrix.indptr[c + 1]):
                yield matrix.indices[ind], c

    elif isinstance(matrix, SparseCSR):
        for r in range(matrix.shape[0]):
            n = matrix.ncol[r]
            ptr = matrix.ptr[r]
            for c in matrix.col[ptr : ptr + n]:
                yield r, c

    else:
        raise NotImplementedError(
            "The iterator for this sparse matrix has not been implemented"
        )


@set_module("sisl")
def ispmatrixd(matrix, map_row=None, map_col=None):
    """Iterator for iterating rows, columns and data for non-zero elements in a ``scipy.sparse.*_matrix`` (or `SparseCSR`)

    Parameters
    ----------
    matrix : scipy.sparse.sp_matrix
      the sparse matrix to iterate non-zero elements
    map_row : func, optional
      map each row entry through the function `map_row`, defaults to ``None`` which is
      equivalent to no mapping.
    map_col : func, optional
      map each column entry through the function `map_col`, defaults to ``None`` which is
      equivalent to no mapping.

    Yields
    ------
    int, int, <>
       the row, column and data of the non-zero elements
    """
    if map_row is None:
        map_row = lambda x: x
    if map_col is None:
        map_col = lambda x: x

    # Consider using the numpy nditer function for buffered iterations
    # it = np.nditer([geom.o2a(tmp.row), geom.o2a(tmp.col % geom.no), tmp.data],
    #               flags=['buffered'], op_flags=['readonly'])

    if issparse(matrix) and matrix.format == "csr":
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            for ind in range(matrix.indptr[r], matrix.indptr[r + 1]):
                yield rr, map_col(matrix.indices[ind]), matrix.data[ind]

    elif issparse(matrix) and matrix.format == "lil":
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            for c, m in zip(map_col(matrix.rows[r]), matrix.data[r]):
                yield rr, c, m

    elif issparse(matrix) and matrix.format == "coo":
        yield from zip(map_row(matrix.row), map_col(matrix.col), matrix.data)

    elif issparse(matrix) and matrix.format == "csc":
        for c in range(matrix.shape[1]):
            cc = map_col(c)
            for ind in range(matrix.indptr[c], matrix.indptr[c + 1]):
                yield map_row(matrix.indices[ind]), cc, matrix.data[ind]

    elif isinstance(matrix, SparseCSR):
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            n = matrix.ncol[r]
            if n == 0:
                continue
            ptr = matrix.ptr[r]
            sl = slice(ptr, ptr + n, None)
            for c, d in zip(map_col(matrix.col[sl]), matrix._D[sl, :]):
                yield rr, c, d

    else:
        raise NotImplementedError(
            "The iterator for this sparse matrix has not been implemented"
        )
