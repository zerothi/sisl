from __future__ import print_function, division

import warnings
from numbers import Integral
from collections import Iterable

# To speed up the extension algorithm we limit
# the lookup table
import numpy as np
from numpy import empty, zeros, asarray, arange
from numpy import insert, take, delete, copyto, split
from numpy import intersect1d, setdiff1d, unique
from numpy import diff, count_nonzero
from numpy import argsort
try:
    isin = np.isin
except Exception:
    isin = np.in1d

from scipy.sparse import isspmatrix
from scipy.sparse import isspmatrix_coo
from scipy.sparse import csr_matrix, isspmatrix_csr
from scipy.sparse import isspmatrix_csc
from scipy.sparse import isspmatrix_lil

import sisl._array as _a
from ._help import array_fill_repeat, ensure_array, get_dtype
from ._help import _range as range, _zip as zip, _map as map
from .utils.ranges import array_arange

# Although this re-implements the CSR in scipy.sparse.csr_matrix
# we use it slightly differently and thus require this new sparse pattern.

__all__ = ['SparseCSR', 'ispmatrix', 'ispmatrixd']


def indices_single(col, value, offset=0):
    """ Return indices of values in col with a possible offset """
    w = (col == value).nonzero()[0]
    if len(w) == 0:
        return -1
    else:
        return offset + w[0]

# Vectorize the function,
# The return-type is always numpy.int32
# The column indices are passed "as-is" via the
# excluded keyword
indices = np.vectorize(indices_single, otypes=[np.int32],
                       excluded=[0, 'col'])


class SparseCSR(object):
    """
    A compressed sparse row matrix, slightly different than ``scipy.sparse.csr_matrix``.

    This class holds all required information regarding the CSR matrix format.

    Note that this sparse matrix of data does not retain the number of columns
    in the matrix, i.e. it has no way of determining whether the input is
    correct.
    """

    def __init__(self, arg1, dim=1, dtype=None, nnzpr=20, nnz=None,
                 **kwargs):
        """ Initialize a new sparse CSR matrix

        This sparse matrix class tries to resemble the
        ``scipy.sparse.csr_matrix`` as much as possible with
        the difference of this class being multi-dimensional.

        Creating a new sparse matrix is much similar to the
        ``scipy`` equivalent.

        `nnz` is only used if ``nnz > nr * nnzpr``.

        This class may be instantiated by verious means.

        - ``SparseCSR(S)``
          where ``S`` is a ``scipy.sparse`` matrix
        - ``SparseCSR((M,N)[, dtype])``
          the shape of the sparse matrix (equivalent
          to ``SparseCSR((M,N,1)[, dtype])``.
        - ``SparseCSR((M,N), dim=K, [, dtype])``
          the shape of the sparse matrix (equivalent
          to ``SparseCSR((M,N,K)[, dtype])``.
        - ``SparseCSR((M,N,K)[, dtype])``
          creating a sparse matrix with ``M`` rows, ``N`` columns
          and ``K`` elements per sparse element.

        Additionally these parameters control the
        creation of the sparse matrix

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

        Attributes
        ----------
        ncol: int-array, ``self.shape[0]``
           number of entries per row
        ptr: int-array, ``self.shape[0]+1``
           pointer index in the 1D column indices of the corresponding row
        col: int-array
           column indices of the sparse elements
        data:
           the data in the sparse matrix
        dim: int
           the extra dimension of the sparse matrix
        nnz: int
           number of contained sparse elements
        shape: tuple, 3*(,)
           size of contained matrix, M, N, K
        finalized: boolean
           whether the sparse matrix is finalized and non-set elements
           are removed
        """

        # step size in sparse elements
        # If there isn't enough room for adding
        # a non-zero element, the # of elements
        # for the insert row is increased at least by this number
        self._ns = 10

        if isspmatrix(arg1):
            # This is a sparse matrix
            # The data-type is infered from the
            # input sparse matrix.
            arg1 = arg1.tocsr()
            self.__init__((arg1.data, arg1.indices, arg1.indptr),
                          dim=dim, dtype=dtype, **kwargs)

        elif isinstance(arg1, (tuple, list)):

            if isinstance(arg1[0], Integral):
                self.__init_shape(arg1, dim=dim, dtype=dtype,
                                  nnzpr=nnzpr, nnz=nnz,
                                  **kwargs)

            elif len(arg1) != 3:
                raise ValueError(('The sparse array *must* be created '
                                  'with data, indices, indptr'))
            else:

                # Correct dimension according to passed array
                if len(arg1[0].shape) == 2:
                    dim = max(dim, arg1[0].shape[1])

                if dtype is None:
                    # The first element is the data
                    dtype = arg1[0].dtype

                # The first *must* be some sort of array
                if 'shape' in kwargs:
                    shape = kwargs['shape']

                else:
                    M = len(arg1[2])-1
                    N = ((np.amax(arg1[1]) // M) + 1) * M
                    shape = (M, N)

                self.__init_shape(shape, dim=dim, dtype=dtype,
                                  nnz=1, **kwargs)

                # Copy data to the arrays
                self.ptr = arg1[2].astype(np.int32, copy=False)
                self.ncol = diff(self.ptr)
                self.col = arg1[1].astype(np.int32, copy=False)
                self._nnz = len(self.col)
                self._D = empty([len(arg1[1]), self.shape[-1]], dtype=self.dtype)
                if len(arg1[0].shape) == 2:
                    self._D[:, :] = arg1[0]
                else:
                    self._D[:, 0] = arg1[0]

    def __init_shape(self, arg1, dim=1, dtype=None, nnzpr=20, nnz=None,
                     **kwargs):

        # The shape of the data...
        if len(arg1) == 2:
            # extend to extra dimension
            arg1 = arg1 + (dim,)
        elif len(arg1) != 3:
            raise ValueError("unrecognized shape input, either a 2-tuple or 3-tuple is required")

        # Set default dtype
        if dtype is None:
            dtype = np.float64

        # unpack size
        M, N, K = arg1

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
        self.ptr = _a.cumsumi(_a.arrayi([nnzpr] * (M+1))) - nnzpr
        # Create column array
        self.col = _a.emptyi(nnz)
        # Store current number of non-zero elements
        self._nnz = 0

        # Important that this is zero
        # For instance one may set one dimension at a time
        # thus automatically zeroing the other dimensions.
        self._D = zeros([nnz, K], dtype)

        # Denote that this sparsity pattern hasn't been finalized
        self._finalized = False

    def diags(self, diagonals, offsets=0, dim=None, dtype=None):
        """ Create a `SparseCSR` with diagonal elements with the same shape as the routine

        Parameters
        ----------
        diagonals : scalar or array_like
           the diagonal values, if scalar the `shape` *must* be present.
        offsets : scalar or array_like
           the offsets from the diagonal for each of the components (defaults
           to the diagonal)
        dim : int, optional
           the extra dimension of the new diagonal matrix (default to the current
           extra dimension)
        dtype : numpy.dtype, optional
           the data-type to create (default to `numpy.float64`)
        """
        if dim is None:
            dim = self.shape[2]
        if dtype is None:
            dtype = self.dtype

        # Now create the sparse matrix
        shape = list(self.shape)
        shape[2] = dim
        shape = tuple(shape)

        # Delete the last entry, regardless of the size, the diagonal
        D = self.__class__(shape, dtype=dtype)

        diagonals = array_fill_repeat(diagonals, D.shape[0], cls=dtype)
        offsets = array_fill_repeat(offsets, D.shape[0], cls=dtype)

        # Create diagonal elements
        for i in range(D.shape[0]):
            D[i, i + offsets[i]] = diagonals[i]

        return D

    def empty(self, keep_nnz=False):
        """ Delete all sparse information from the sparsity pattern

        Essentially this deletes all entries.

        Parameters
        ----------
        keep_nnz: boolean, optional
           if ``True`` keeps the sparse elements *as is*.
           I.e. it will merely set the stored sparse elements to zero.
           This may be advantagegous when re-constructing a new sparse
           matrix from an old sparse matrix
        """
        self._D[:, :] = 0.

        if not keep_nnz:
            self._finalized = False
            # The user does not wish to retain the
            # sparse pattern
            self.ncol[:] = 0
            self._nnz = 0
            # We do not mess with the other arrays
            # they may be obscure data any-way.

    @property
    def shape(self):
        """ The shape of the sparse matrix """
        return self._shape

    @property
    def dim(self):
        """ The extra dimensionality of the sparse matrix (elements per matrix element) """
        return self.shape[2]

    @property
    def data(self):
        """ Data contained in the sparse matrix (numpy array of elements) """
        return self._D

    @property
    def dtype(self):
        """ The data-type in the sparse matrix """
        return self._D.dtype

    @property
    def dkind(self):
        """ The data-type in the sparse matrix (in str) """
        return np.dtype(self._D.dtype).kind

    @property
    def nnz(self):
        """ Number of non-zero elements in the sparse matrix """
        return self._nnz

    def __len__(self):
        """ Number of rows in the sparse matrix """
        return self.shape[0]

    @property
    def finalized(self):
        """ Whether the contained data is finalized and non-used elements have been removed """
        return self._finalized

    def finalize(self, sort=True):
        """ Finalizes the sparse matrix by removing all non-set elements

        One may still interact with the sparse matrix as one would previously.

        NOTE: This is mainly an internal used routine to ensure data structure
        when converting to ``scipy.sparse.csr_matrix``

        Parameters
        ----------
        sort: bool, optional
           sort the column indices for each row
        """
        if self.finalized:
            return

        # Create and index array to retain the indices we want
        ptr = self.ptr
        ncol = self.ncol
        idx = array_arange(ptr[:-1], n=ncol)

        self.col = take(self.col, idx)
        self._D = take(self._D, idx, 0)
        del idx
        self.ptr[0] = 0
        _a.cumsumi(ncol, out=self.ptr[1:])

        ptr = self.ptr.view()
        col = self.col.view()
        D = self._D.view()

        # We truncate all the connections
        if sort:
            def func(r):
                """ Sort and check whether there are double entries """
                ptr1 = ptr[r]
                ptr2 = ptr[r+1]
                sl = slice(ptr1, ptr2)
                ccol = col[sl].view()
                DD = D[sl, :].view()
                if unique(ccol).shape[0] != ptr2 - ptr1:
                    raise ValueError(('You cannot have two elements between the same ' +
                                      'i,j index ({}), something has went terribly wrong.'.format(ptr1)))
                idx = argsort(ccol)
                # Do in-place sorting
                ccol[:] = ccol[idx]
                DD[:, :] = DD[idx, :]

        else:
            def func(r):
                ptr1 = ptr[r]
                ptr2 = ptr[r+1]
                if unique(col[ptr1:ptr2]).shape[0] != ptr2 - ptr1:
                    raise ValueError(('You cannot have two elements between the same ' +
                                      'i,j index ({}), something has went terribly wrong.'.format(ptr1)))
        # Since map puts it on the stack, we have to force the evaluation.
        list(map(func, range(self.shape[0])))

        assert len(col) == self.nnz, ('Final size in the sparse matrix finalization '
                                      'went wrong.')

        # Check that all column indices are within the expected shape
        if np.any(self.shape[1] <= self.col):
            warnings.warn("Sparse matrix contains column indices outside the shape "
                          "of the matrix. Data may not represent what you had expected")

        # Signal that we indeed have finalized the data
        self._finalized = True

    def edges(self, row, exclude=None):
        """ Retrieve edges (connections) of a given `row` or list of `row`'s

        The returned edges are unique and sorted (see `numpy.unique`).

        Parameters
        ----------
        row : int or list of int
            the edges are returned only for the given row
        exclude : int or list of int, optional
           remove edges which are in the `exclude` list.
           Default to `row`.
        """
        row = unique(ensure_array(row))
        if exclude is None:
            exclude = row.view()
        else:
            exclude = unique(ensure_array(exclude))

        # Now get the edges
        ptr = self.ptr.view()
        ncol = self.ncol.view()

        # Create column indices
        edges = unique(self.col[array_arange(ptr[row], n=ncol[row])])

        # Return the difference to the exclude region, we know both are unique
        return setdiff1d(edges, exclude, assume_unique=True)

    def delete_columns(self, columns, keep_shape=False):
        """ Delete all columns in `columns`

        Parameters
        ----------
        columns : int or array_like
           columns to delete from the sparse pattern
        keep_shape : bool, optional
           whether the ``shape`` of the object should be retained, if ``True`` all higher
           columns will be shifted according to the number of columns deleted below,
           if ``False``, only the elements will be deleted.
        """
        # Shorthand function for retrieval
        cnz = count_nonzero

        # Sort the columns
        columns = unique(ensure_array(columns))
        n_cols = cnz(columns < self.shape[1])

        # Grab pointers
        ptr = self.ptr.view()
        ncol = self.ncol.view()
        col = self.col.view()

        # Get indices of deleted columns
        idx = array_arange(ptr[:-1], n=ncol)
        # Convert to boolean array where we have columns to be deleted
        lidx = isin(col[idx], columns)
        # Count number of deleted entries per row
        ndel = ensure_array(map(count_nonzero, split(lidx, _a.cumsumi(ncol[:-1]))))
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
        col = self.col.view()

        # Figure out if it is necessary to update columns
        # This is only necessary when the deleted columns
        # are not the *last* columns
        update_col = not keep_shape
        if update_col:
            # Check that we really do have to update
            update_col = np.any(columns < self.shape[1] - n_cols)

        # Correct number of elements per column, and the pointers
        ncol[:] -= ndel
        ptr[1:] -= _a.cumsumi(ndel)

        if update_col:
            # Create a count array to subtract
            count = _a.zerosi(self.shape[1])
            count[columns] = 1
            count = _a.cumsumi(count)

            # Recreate pointers due to deleted indices
            idx = array_arange(ptr[:-1], n=ncol)
            col[idx] -= count[col[idx]]
            del idx

        # Update number of non-zeroes
        self._nnz = np.sum(ncol)

        if not keep_shape:
            shape = list(self.shape)
            shape[1] -= n_cols
            self._shape = tuple(shape)

    def _clean_columns(self):
        """ Remove all intrinsic columns that are not defined in the sparse matrix """
        # Grab pointers
        ptr = self.ptr.view()
        ncol = self.ncol.view()
        col = self.col.view()

        # Number of columns
        nc = self.shape[1]

        # Get indices of columns
        idx = array_arange(ptr[:-1], n=ncol)
        # Convert to boolean array where we have columns to be deleted
        lidx = col[idx] >= nc
        # Count number of deleted entries per row
        ndel = ensure_array(map(count_nonzero, split(lidx, _a.cumsumi(ncol[:-1]))))
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
        self._nnz = np.sum(ncol)

        # We are *only* deleting columns, so if it is finalized,
        # it will still be

    def translate_columns(self, old, new):
        """ Takes all `old` columns and translates them to `new`.

        Parameters
        ----------
        old : int or array_like
           old column indices
        new : int or array_like
           new column indices
        """
        # Sort the columns
        old = ensure_array(old)
        new = ensure_array(new)

        if np.any(old >= self.shape[1]):
            raise ValueError(self.__class__.__name__+".translate_columns has non-existing old column values")

        end_clean = False
        if np.any(new >= self.shape[1]):
            end_clean = True

        # Now do the translation
        pvt = _a.arangei(self.shape[1])
        pvt[old] = new

        # Get indices of valid column entries
        idx = array_arange(self.ptr[:-1], n=self.ncol)
        # Convert the old column indices to new ones
        col = self.col.view()
        col[idx] = pvt[col[idx]]

        # After translation, set to not finalized
        self._finalized = False
        if end_clean:
            self._clean_columns()

    def spsame(self, other):
        """ Check whether two sparse matrices have the same non-zero elements

        Parameters
        ----------
        other : SparseCSR

        Returns
        -------
        True if the same non-zero elements are in the matrices (but not necessarily the same values)
        """
        if self.shape[:2] != other.shape[:2]:
            return False

        sptr = self.ptr.view()
        sncol = self.ncol.view()
        scol = self.col.view()
        optr = other.ptr.view()
        oncol = other.ncol.view()
        ocol = other.col.view()

        # Easy check for non-equal number of elements
        if (sncol == oncol).sum() != self.shape[0]:
            return False

        llen = len
        lintersect1d = intersect1d
        for r in range(self.shape[0]):
            if llen(lintersect1d(scol[sptr[r]:sptr[r]+sncol[r]],
                                 ocol[optr[r]:optr[r]+oncol[r]])) != sncol[r]:
                return False
        return True

    def align(self, other):
        """ Aligns this sparse matrix with the sparse elements of the other sparse matrix

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
            raise ValueError('Aligning two sparse matrices requires same shapes')

        lsetdiff1d = setdiff1d
        sptr = self.ptr.view()
        sncol = self.ncol.view()
        scol = self.col.view()
        optr = other.ptr.view()
        oncol = other.ncol.view()
        ocol = other.col.view()
        for r in range(self.shape[0]):
            # pointers
            sp = sptr[r]
            sn = sncol[r]
            op = optr[r]
            on = oncol[r]

            adds = lsetdiff1d(ocol[op:op+on], scol[sp:sp+sn])
            if len(adds) > 0:
                # simply extend the elements
                self._extend(r, adds)

    def iter_nnz(self, row=None):
        """ Iterations of the non-zero elements, returns a tuple of row and column with non-zero elements

        An iterator returning the current row index and the corresponding column index.

        >>> for r, c in self: # doctest: +SKIP

        In the above case ``r`` and ``c`` are rows and columns such that

        >>> self[r, c] # doctest: +SKIP

        returns the non-zero element of the sparse matrix.

        Parameters
        ----------
        row : int or array_like of int
           only loop on the given row(s) default to all rows
        """
        if row is None:
            # loop on rows
            for r in range(self.shape[0]):
                n = self.ncol[r]
                ptr = self.ptr[r]
                for c in self.col[ptr:ptr+n]:
                    yield r, c
        else:
            for r in ensure_array(row):
                n = self.ncol[r]
                ptr = self.ptr[r]
                for c in self.col[ptr:ptr+n]:
                    yield r, c

    # Define default iterator
    __iter__ = iter_nnz

    def _slice2list(self, slc, axis):
        """ Convert a slice to a list depending on the provided details """
        if not isinstance(slc, slice):
            return slc

        # Do conversion
        N = self.shape[axis]
        # Get the indices
        idx = slc.indices(N)
        return range(idx[0], idx[1], idx[2])

    def _extend(self, i, j):
        """ Extends the sparsity pattern to retain elements `j` in row `i`

        Parameters
        ----------
        i : int
           the row of the matrix
        j : int or array_like
           columns belonging to row `i` where a non-zero element is stored.

        Returns
        -------
        index : array_like
           the indicies of the existing/added elements
        """

        # We skip this check and let sisl die if wrong input is given...
        #if not isinstance(i, Integral):
        #    raise ValueError("Retrieving/Setting elements in a sparse matrix"
        #                     " must only be performed at one row-element at a time.\n"
        #                     "However, multiple columns at a time are allowed.")

        # Ensure flattened array...
        j = ensure_array(j)
        if len(j) == 0:
            return _a.arrayi([])

        # fast reference
        ptr = self.ptr
        ncol = self.ncol
        col = self.col

        # To create the indices for the sparse elements
        # we first find which values are _not_ in the sparse
        # matrix
        if ncol[i] > 0:

            # Checks whether any non-zero elements are
            # already in the sparse pattern
            # If so we remove those from the j
            exists = intersect1d(j, col[ptr[i]:ptr[i]+ncol[i]],
                                 assume_unique=True)
        else:
            exists = _a.arrayi([])

        # Get list of new elements to be added
        new_j = setdiff1d(j, exists, assume_unique=True)
        new_n = len(new_j)

        # Check how many elements cannot fit in the currently
        # allocated sparse matrix...
        new_nnz = ncol[i] + new_n - (ptr[i + 1] - ptr[i])

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

            # Insert pointer of new data
            iptr = ptr[i] + ncol[i]

            # Insert new empty elements in the column index
            # after the column
            self.col = insert(self.col, iptr,
                              empty(ns, self.col.dtype))

            # update reference
            col = self.col

            # Insert zero data in the data array
            # We use `zeros` as then one may set each dimension
            # individually...
            self._D = insert(self._D, ptr[i+1],
                             zeros([ns, self.shape[2]], self._D.dtype), axis=0)

            # Lastly, shift all pointers above this row to account for the
            # new non-zero elements
            ptr[i + 1:] += ns

        if new_n > 0:
            # Ensure that we write the new elements to the matrix...

            # new data begins from this index location
            old_ptr = ptr[i] + ncol[i]

            # assign the column indices for the new entries
            # NOTE that this may not assign them in the order
            # of entry as new_j is sorted and thus new_j != j
            col[old_ptr:old_ptr + new_n] = new_j[:]

            # Step the size of the stored non-zero elements
            ncol[i] += new_n

            # Step the number of non-zero elements
            self._nnz += new_n

        # Now we have extended the sparse matrix to hold all
        # information that is required...

        # ... retrieve the indices and return
        return indices(col[ptr[i]:ptr[i] + ncol[i]], j, ptr[i])

    def _get(self, i, j):
        """ Retrieves the data pointer arrays of the elements, if it is non-existing, it will return ``-1``

        Parameters
        ----------
        i : int
           the row of the matrix
        j : int or array_like of int
           columns belonging to row `i` where a non-zero element is stored.

        Returns
        -------
        numpy.ndarray : indicies of the existing elements
        """

        # Ensure flattened array...
        j = _a.asarrayi(j).flatten()

        # Make it a little easier
        ptr = self.ptr[i]

        return indices(self.col[ptr:ptr+self.ncol[i]], j, ptr)

    def __delitem__(self, key):
        """ Remove items from the sparse patterns """
        # Get indices of sparse data (-1 if non-existing)
        key = list(key)
        key[0] = self._slice2list(key[0], 0)
        if isinstance(key[0], Iterable):
            if len(key) == 2:
                for i in key[0]:
                    del self[i, key[1]]
            elif len(key) == 3:
                for i in key[0]:
                    del self[i, key[1], key[2]]
            return

        i = key[0]
        key[1] = self._slice2list(key[1], 1)
        index = self._get(i, key[1])

        # First remove all negative indices.
        # The element isn't there anyway...
        index = index[index >= 0]
        index.sort()

        if len(index) == 0:
            # There are no elements to delete...
            return

        # Get short-hand
        ptr = self.ptr.view()
        ncol = self.ncol.view()

        # Get original values
        sl = slice(ptr[i], ptr[i] + ncol[i], None)
        oC = self.col[sl]
        oD = self._D[sl, :]

        # Now create the compressed data...
        index -= ptr[i]
        keep = isin(_a.arangei(ncol[i]), index, invert=True)

        # Update new count of the number of
        # non-zero elements
        ncol[i] -= len(index)

        # Now update the column indices and the data
        sl = slice(ptr[i], ptr[i] + ncol[i], None)
        self.col[sl] = oC[keep]
        self._D[sl, :] = oD[keep, :]

        # Once we remove some things, it is NOT
        # finalized...
        self._finalized = False
        self._nnz -= len(index)

    def __getitem__(self, key):
        """ Intrinsic sparse matrix retrieval of a non-zero element """

        # Get indices of sparse data (-1 if non-existing)
        get_idx = self._get(key[0], key[1])
        n = len(get_idx)

        # Indices of existing values in return array
        ret_idx = (get_idx >= 0).nonzero()[0]
        # Indices of existing values in get array
        get_idx = get_idx[ret_idx]

        # Check which data to retrieve
        if len(key) > 2:

            # user requests a specific element
            # get dimension retrieved
            r = np.zeros(n, dtype=self.dtype)
            r[ret_idx] = self._D[get_idx, key[2]]

        else:

            # user request all stored data

            s = self.shape[2]
            if s == 1:
                r = np.zeros(n, dtype=self.dtype)
                r[ret_idx] = self._D[get_idx, 0]
            else:
                r = np.zeros([n, s], dtype=self.dtype)
                r[ret_idx, :] = self._D[get_idx, :]

        return r

    def __setitem__(self, key, data):
        """ Intrinsic sparse matrix assignment of the item.

        It will only allow to set the data in the sparse
        matrix if the dimensions match.

        If the `data` parameter is ``None`` or an array
        only with `None` then the data will not be stored.
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
        isnan = np.isnan(data)
        if np.all(isnan):
            # If the entries are nan's
            # then we return without adding the
            # entry.
            return
        else:
            # Places where there are nan will be set
            # to zero
            data[isnan] = 0
            del isnan

        # Retrieve indices in the 1D data-structure
        index = self._extend(key[0], key[1])

        if len(key) > 2:
            # Explicit data of certain dimension
            self._D[index, key[2]] = data

        else:
            # Ensure correct shape
            if data.ndim == 0:
                data = np.array([data])
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
        """ Check whether a sparse index is non-zero """
        # Get indices of sparse data (-1 if non-existing)
        return np.all(self._get(key[0], key[1]) >= 0)

    def nonzero(self, row=None, only_col=False):
        """ Row and column indices where non-zero elements exists

        Parameters
        ----------
        row : int or array_like of int, optional
           only return the tuples for the requested rows, default is all rows
        only_col : bool, optional
           only return then non-zero columns
        """
        if row is None:
            idx = array_arange(self.ptr[:-1], n=self.ncol)
            if not only_col:
                rows = _a.emptyi([self.nnz])
                j = 0
                for r, N in enumerate(self.ncol):
                    rows[j:j+N] = r
                    j += N
        else:
            row = ensure_array(row)
            idx = array_arange(self.ptr[row], n=self.ncol[row])
            if not only_col:
                N = _a.sumi(self.ncol[row])
                rows = _a.emptyi([N])
                j = 0
                for r, N in zip(row, self.ncol[row]):
                    rows[j:j+N] = r
                    j += N

        if only_col:
            return self.col[idx]
        return rows, self.col[idx]

    def eliminate_zeros(self):
        """ Remove all zero elememts from the sparse matrix

        This is an *in-place* operation
        """
        ptr = self.ptr.view()
        ncol = self.ncol.view()
        col = self.col.view()
        D = self._D.view()

        # Get short-hand
        nsum = np.sum
        nabs = np.abs
        for i in range(self.shape[0]):

            # Create short-hand slice
            sl = slice(ptr[i], ptr[i] + ncol[i], None)

            # Get current column entries for the row
            C = col[sl]
            # Retrieve columns with zero values (summed over all elements)
            C0 = (nsum(nabs(D[sl, :]), axis=1) == 0).nonzero()[0]
            if len(C0) == 0:
                continue
            # Remove all entries with 0 values
            del self[i, C[C0]]

    def copy(self, dims=None, dtype=None):
        """ Returns an exact copy of the sparse matrix

        Parameters
        ----------
        dims: array-like, optional
           which dimensions to store in the copy, defaults to all.
        dtype : `numpy.dtype`
           this defaults to the dtype of the object,
           but one may change it if supplied.
        """
        # Create sparse matrix (with only one entry per
        # row, we overwrite it immediately afterward)
        if dims is None:
            dim = self.dim
        else:
            dim = len(dims)

        if dtype is None:
            dtype = self.dtype

        # Create correct input
        shape = list(self.shape[:])
        shape[2] = dim

        new = self.__class__(shape, dtype=dtype, nnz=1)

        # The default sizes are not passed
        # Hence we *must* copy the arrays
        # directly
        copyto(new.ptr, self.ptr, casting='no')
        copyto(new.ncol, self.ncol, casting='no')
        new.col = self.col.copy()
        new._nnz = self.nnz

        if dims is None:
            new._D = self._D.astype(dtype)
        else:
            new._D = empty([len(self.col), dim], dtype)
            for i, dim in enumerate(dims):
                new._D[:, i] = self._D[:, dim]

        # Mark it as the same state as the other one
        new._finalized = self._finalized

        return new

    def tocsr(self, dim=0, **kwargs):
        """ Return the data in ``scipy.sparse.csr_matrix`` format

        Parameters
        ----------
        dim: int, optional
           the dimension of the data to create the sparse matrix
        **kwargs:
           arguments passed to the ``scipy.sparse.csr_matrix`` routine
        """
        # We finalize because we do not expect the sparse pattern to change once
        # we request a csr matrix in another format.
        # Otherwise we *could* do array_arange
        self.finalize()

        shape = self.shape[:2]
        return csr_matrix((self._D[:, dim], self.col.astype(np.int32, copy=False),
                           self.ptr.astype(np.int32, copy=False)),
                          shape=shape, **kwargs)

    def remove(self, indices):
        """ Return a new sparse CSR matrix with all the indices removed

        Parameters
        ----------
        indices : array_like
           the indices of the rows *and* columns that are removed in the sparse pattern
        """
        indices = ensure_array(indices)

        # Check if we have a square matrix or a rectangular one
        if self.shape[0] >= self.shape[1]:
            rindices = delete(_a.arangei(self.shape[0]), indices)

        else:
            rindices = delete(_a.arangei(self.shape[1]), indices)

        return self.sub(rindices)

    def sub(self, indices):
        """ Return a new sparse CSR matrix with the data only for the given indices

        Parameters
        ----------
        indices : array_like
           the indices of the rows *and* columns that are retained in the sparse pattern
        """
        indices = ensure_array(indices)

        # Check if we have a square matrix or a rectangular one
        if self.shape[0] == self.shape[1]:
            ridx = indices.view()

        elif self.shape[0] < self.shape[1]:
            ridx = indices[indices < self.shape[0]]

        elif self.shape[0] > self.shape[1]:
            ridx = indices.view()

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
        ptr1 = csr.ptr.view()
        ncol1 = csr.ncol.view()

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
        ncol1[:] = ensure_array(map(count_nonzero,
                                    split(idx_take, ptr1[1:-1])))

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

    def sum(self, axis=None):
        """ Calculate the sum, if `axis` is ``None`` the sum of all elements are returned, else a new sparse matrix is returned

        Parameters
        ----------
        axis : int, optional
           which axis to perform the sum of. If ``None`` the element sum is returned, if either ``0`` or ``1`` is passed a
           vector is returned, and for ``2`` it returns a new sparse matrix with the last dimension reduced to 1 (summed).

        Raises
        ------
        NotImplementedError : when ``axis = 1``
        """
        if axis is None:
            return np.sum(self._D)
        if axis == -1 or axis == 2:
            # We simply create a new sparse matrix with only one entry
            ret = self.copy([0])
            ret._D[:, 0] = self._D.sum(1)
        elif axis == -2 or axis == 1:
            ret = zeros(self.shape[1], dtype=self.dtype)
            raise NotImplementedError('Currently performing a sum on the columns is not implemented')
        elif axis == 0:
            ret = empty([self.shape[0], self.shape[2]], dtype=self.dtype)
            ptr = self.ptr.view()
            ncol = self.ncol.view()
            col = self.col.view()
            for r in range(self.shape[0]):
                ret[r, :] = self._D[ptr[r]:ptr[r]+ncol[r], :].sum(0)

        return ret

    def __repr__(self):
        """ Representation of the sparse matrix model """
        ints = self.shape[:] + (self.nnz,)
        return self.__class__.__name__ + '{{dim={2}, kind={kind},\n  rows: {0}, columns: {1},\n  non-zero: {3}\n}}'.format(*ints, kind=self.dkind)

    ###############################
    # Overload of math operations #
    ###############################
    def __add__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c += b
        return c
    __radd__ = __add__

    def __iadd__(a, b):
        if isinstance(b, SparseCSR):
            if a.shape != b.shape:
                raise ValueError('Adding two sparse matrices requires the same shape')
            # Ensure that a is aligned with b
            a.align(b)

            # loop and add elements
            for r in range(a.shape[0]):
                # pointers
                bptr = b.ptr[r]
                bn = b.ncol[r]
                sl = slice(bptr, bptr+bn, None)

                # Get positions of b-elements in a:
                in_a = a._get(r, b.col[sl])
                a._D[in_a, :] += b._D[sl, :]

        else:
            a._D += b
        return a

    def __sub__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c -= b
        return c

    def __rsub__(a, b):
        if isinstance(b, SparseCSR):
            c = b.copy(dtype=get_dtype(a, other=b.dtype))
            c += -1 * a
        else:
            c = b + (-1) * a
        return c

    def __isub__(a, b):
        if isinstance(b, SparseCSR):
            if a.shape != b.shape:
                raise ValueError('Subtracting two sparse matrices requires the same shape')
            # Ensure that a is aligned with b
            a.align(b)

            # loop and add elements
            for r in range(a.shape[0]):
                # pointers
                bptr = b.ptr[r]
                bn = b.ncol[r]
                sl = slice(bptr, bptr+bn, None)

                # Get positions of b-elements in a:
                in_a = a._get(r, b.col[sl])
                a._D[in_a, :] -= b._D[sl, :]

        else:
            a._D -= b
        return a

    def __mul__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c *= b
        return c
    __rmul__ = __mul__

    def __imul__(a, b):
        if isinstance(b, SparseCSR):
            if a.shape != b.shape:
                raise ValueError('Multiplication of two sparse matrices requires the same shape')

            # Note that for multiplication of these two matrices
            # it is not required that they are aligned...
            # 0 * float == 0
            # Hence aligning is superfluous

            # loop and add elements
            for r in range(a.shape[0]):
                # pointers
                aptr = a.ptr[r]
                an = a.ncol[r]
                bptr = b.ptr[r]
                bn = b.ncol[r]

                acol = a.col[aptr:aptr+an]
                bcol = b.col[bptr:bptr+bn]

                # Get positions of b-elements in a:
                in_a = a._get(r, bcol)
                # remove all -1's
                in_a = in_a[in_a > -1]
                # Everything else *must* be zeroes! :)
                a._D[in_a, :] *= b._D[bptr:bptr+bn, :]

                # Now set everything *not* in b but in a, to zero
                not_in_b = isin(acol, bcol, invert=True).nonzero()[0]
                a._D[aptr+not_in_b, :] = 0

        else:
            a._D *= b
        return a

    def __div__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c /= b
        return c

    def __rdiv__(a, b):
        c = b.copy(dtype=get_dtype(a, other=b.dtype))
        c /= a
        return c

    def __idiv__(a, b):
        if isinstance(b, SparseCSR):
            if a.shape != b.shape:
                raise ValueError('Division of two sparse matrices requires the same shape')

            # Ensure that a is aligned with b
            a.align(b)

            # loop and add elements
            for r in range(a.shape[0]):
                # pointers
                bptr = b.ptr[r]
                bn = b.ncol[r]

                # Get positions of b-elements in a:
                in_a = a._get(r, b.col[bptr:bptr+bn])
                a._D[in_a, :] /= b._D[bptr:bptr+bn, :]

        else:
            a._D /= b
        return a

    def __floordiv__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c //= b
        return c

    def __ifloordiv__(a, b):
        if isinstance(b, SparseCSR):
            if a.shape != b.shape:
                raise ValueError('Floor-division of two sparse matrices requires the same shape')
            # Ensure that a is aligned with b
            a.align(b)

            # loop and add elements
            for r in range(a.shape[0]):
                # pointers
                bptr = b.ptr[r]
                bn = b.ncol[r]

                # Get positions of b-elements in a:
                in_a = a._get(r, b.col[bptr:bptr+bn])
                a._D[in_a, :] //= b._D[bptr:bptr+bn, :]

        else:
            a._D //= b
        return a

    def __truediv__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c /= b
        return c

    def __itruediv__(a, b):
        if isinstance(b, SparseCSR):
            if a.shape != b.shape:
                raise ValueError('True-division of two sparse matrices requires the same shape')
            # Ensure that a is aligned with b
            a.align(b)

            # loop and add elements
            for r in range(a.shape[0]):
                # pointers
                bptr = b.ptr[r]
                bn = b.ncol[r]

                # Get positions of b-elements in a:
                in_a = a._get(r, b.col[bptr:bptr+bn])
                a._D[in_a, :].__itruediv__(b._D[bptr:bptr+bn, :])

        else:
            a._D /= b
        return a

    def __pow__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c **= b
        return c

    def __rpow__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._D[...] = b ** c._D[...]
        return c

    def __ipow__(a, b):
        if isinstance(b, SparseCSR):
            if a.shape != b.shape:
                raise ValueError('True-division of two sparse matrices requires the same shape')
            # Ensure that a is aligned with b
            # 0 ** float == 1.
            a.align(b)

            # loop and add elements
            for r in range(a.shape[0]):
                # pointers
                aptr = a.ptr[r]
                an = a.ncol[r]
                bptr = b.ptr[r]
                bn = b.ncol[r]

                acol = a.col[aptr:aptr+an]
                bcol = b.col[bptr:bptr+bn]

                # Get positions of b-elements in a:
                in_a = a._get(r, bcol)
                a._D[in_a, :] **= b._D[bptr:bptr+bn, :]

                # Now set everything *not* in b but in a, to 1
                #  float ** 0 == 1
                not_in_b = isin(acol, bcol, invert=True).nonzero()[0]
                a._D[aptr+not_in_b, :] = 1

        else:
            a._D **= b
        return a


def ispmatrix(matrix, map_row=None, map_col=None):
    """ Iterator for iterating rows and columns for non-zero elements in a `scipy.sparse.*_matrix` (or `SparseCSR`)

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
        for r, c in _ispmatrix_all(matrix):
            yield r, c
        return

    if map_row is None:
        map_row = lambda x: x
    if map_col is None:
        map_col = lambda x: x
    map_row = np.vectorize(map_row)
    map_col = np.vectorize(map_col)

    nrow = len(unique(map_row(arange(matrix.shape[0], dtype=np.int32))))
    ncol = len(unique(map_col(arange(matrix.shape[1], dtype=np.int32))))
    rows = zeros(nrow, dtype=np.bool_)
    cols = zeros(ncol, dtype=np.bool_)

    # Initialize the unique arrays
    rows[:] = False

    # Consider using the numpy nditer function for buffered iterations
    #it = np.nditer([geom.o2a(tmp.row), geom.o2a(tmp.col % geom.no), tmp.data],
    #               flags=['buffered'], op_flags=['readonly'])

    if isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            if rows[rr]: continue
            rows[rr] = True
            cols[:] = False
            for ind in range(matrix.indptr[r], matrix.indptr[r+1]):
                c = map_col(matrix.indices[ind])
                if cols[c]: continue
                cols[c] = True
                yield rr, c

    elif isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            if rows[rr]: continue
            rows[rr] = True
            cols[:] = False
            if len(matrix.rows[r]) == 0:
                continue
            for c in map_col(matrix.rows[r]):
                if cols[c]: continue
                cols[c] = True
                yield rr, c

    elif isspmatrix_coo(matrix):
        raise ValueError("mapping and unique returns are not implemented for COO matrix")

    elif isspmatrix_csc(matrix):
        raise ValueError("mapping and unique returns are not implemented for CSC matrix")

    elif isinstance(matrix, SparseCSR):
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            if rows[rr]: continue
            rows[rr] = True
            cols[:] = False
            n = matrix.ncol[r]
            if n == 0:
                continue
            ptr = matrix.ptr[r]
            for c in map_col(matrix.col[ptr:ptr+n]):
                if cols[c]: continue
                cols[c] = True
                yield rr, c

    else:
        raise NotImplementedError("The iterator for this sparse matrix has not been implemented")


def _ispmatrix_all(matrix):
    """ Iterator for iterating rows and columns for non-zero elements in a ``scipy.sparse.*_matrix`` (or `SparseCSR`)

    Parameters
    ----------
    matrix : ``scipy.sparse.*_matrix``
      the sparse matrix to iterate non-zero elements

    Yields
    ------
    int, int
       the row, column indices of the non-zero elements
    """
    if isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            for ind in range(matrix.indptr[r], matrix.indptr[r+1]):
                yield r, matrix.indices[ind]

    elif isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            for c in matrix.rows[r]:
                yield r, c

    elif isspmatrix_coo(matrix):
        for r, c in zip(matrix.row, matrix.col):
            yield r, c

    elif isspmatrix_csc(matrix):
        for c in range(matrix.shape[1]):
            for ind in range(matrix.indptr[c], matrix.indptr[c+1]):
                yield matrix.indices[ind], c

    elif isinstance(matrix, SparseCSR):
        for r in range(matrix.shape[0]):
            n = matrix.ncol[r]
            ptr = matrix.ptr[r]
            for c in matrix.col[ptr:ptr+n]:
                yield r, c

    else:
        raise NotImplementedError("The iterator for this sparse matrix has not been implemented")


def ispmatrixd(matrix, map_row=None, map_col=None):
    """ Iterator for iterating rows, columns and data for non-zero elements in a ``scipy.sparse.*_matrix`` (or `SparseCSR`)

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
    #it = np.nditer([geom.o2a(tmp.row), geom.o2a(tmp.col % geom.no), tmp.data],
    #               flags=['buffered'], op_flags=['readonly'])

    if isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            for ind in range(matrix.indptr[r], matrix.indptr[r+1]):
                yield rr, map_col(matrix.indices[ind]), matrix.data[ind]

    elif isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            for c, m in zip(map_col(matrix.rows[r]), matrix.data[r]):
                yield rr, c, m

    elif isspmatrix_coo(matrix):
        for r, c, m in zip(map_row(matrix.row), map_col(matrix.col), matrix.data):
            yield r, c, m

    elif isspmatrix_csc(matrix):
        for c in range(matrix.shape[1]):
            cc = map_col(c)
            for ind in range(matrix.indptr[c], matrix.indptr[c+1]):
                yield map_row(matrix.indices[ind]), cc, matrix.data[ind]

    elif isinstance(matrix, SparseCSR):
        for r in range(matrix.shape[0]):
            rr = map_row(r)
            n = matrix.ncol[r]
            if n == 0:
                continue
            ptr = matrix.ptr[r]
            sl = slice(ptr, ptr+n, None)
            for c, d in zip(map_col(matrix.col[sl]), matrix._D[sl, :]):
                yield rr, c, d

    else:
        raise NotImplementedError("The iterator for this sparse matrix has not been implemented")
