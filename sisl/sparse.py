from __future__ import print_function, division

from numbers import Integral

# To speed up the extension algorithm we limit
# the lookup table
import numpy as np
from numpy import ndarray, int32
from numpy import empty, zeros, full, asarray, arange
from numpy import insert, take, delete, copyto, split
from numpy import intersect1d, setdiff1d, unique, in1d
from numpy import diff, count_nonzero, allclose
from numpy import any as np_any
from numpy import all as np_all
from numpy import atleast_1d
from numpy import isnan, broadcast
from numpy import argsort

from scipy.sparse import isspmatrix
from scipy.sparse import isspmatrix_coo
from scipy.sparse import csr_matrix, isspmatrix_csr
from scipy.sparse import isspmatrix_csc
from scipy.sparse import isspmatrix_lil

from . import _array as _a
from ._array import asarrayi, arrayi, fulli
from ._indices import indices, indices_only, sorted_unique
from .messages import warn, SislError
from ._help import array_fill_repeat, get_dtype, isiterable
from ._help import _range as range, _zip as zip, _map as map
from .utils.ranges import array_arange

# Although this re-implements the CSR in scipy.sparse.csr_matrix
# we use it slightly differently and thus require this new sparse pattern.

__all__ = ['SparseCSR', 'ispmatrix', 'ispmatrixd']


class SparseCSR(object):
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
      where ``S`` is a :module:`scipy.sparse` matrix
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

    Attributes
    ----------
    ncol : int-array, ``self.shape[0]``
       number of entries per row
    ptr : int-array, ``self.shape[0]+1``
       pointer index in the 1D column indices of the corresponding row
    col : int-array
       column indices of the sparse elements
    data:
       the data in the sparse matrix
    dim : int
       the extra dimension of the sparse matrix
    nnz : int
       number of contained sparse elements
    shape : tuple, 3*(,)
       size of contained matrix, M, N, K
    finalized : boolean
       whether the sparse matrix is finalized and non-set elements
       are removed
    """

    # These overrides are necessary to be able to perform
    # ufunc operations with numpy.
    # The reason is that the ufunc in numpy arrays are first
    # tried when encountering operations:
    #   np.int + object will invoke __add__ from ndarray, regardless
    # of objects __radd__ routine.
    # We thus need to define the ufunc method in this object
    # to tell numpy that using numpy.ndarray.__array_ufunc__ won't work.
    # Prior to 1.13 the ufunc is named numpy_ufunc, subsequent versions
    # are using array_ufunc.
    __numpy_ufunc__ = None
    __array_ufunc__ = None

    def __init__(self, arg1, dim=1, dtype=None, nnzpr=20, nnz=None,
                 **kwargs):
        """ Initialize a new sparse CSR matrix """

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
            # Default shape to the CSR matrix
            kwargs['shape'] = kwargs.get('shape', arg1.shape)
            self.__init__((arg1.data, arg1.indices, arg1.indptr),
                          dim=dim, dtype=dtype, **kwargs)

        elif isinstance(arg1, (tuple, list)):

            if isinstance(arg1[0], Integral):
                self.__init_shape(arg1, dim=dim, dtype=dtype,
                                  nnzpr=nnzpr, nnz=nnz,
                                  **kwargs)

            elif len(arg1) != 3:
                raise ValueError(self.__class__.__name__ + ' sparse array *must* be created '
                                 'with data, indices, indptr')
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
                self.ptr = arg1[2].astype(int32, copy=False)
                self.ncol = diff(self.ptr)
                self.col = arg1[1].astype(int32, copy=False)
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
            raise ValueError(self.__class__.__name__ + " unrecognized shape input, either a 2-tuple or 3-tuple is required")

        # Set default dtype
        if dtype is None:
            dtype = np.float64

        # unpack size and check the sizes are "physical"
        M, N, K = arg1
        if M <= 0 or N <= 0 or K <= 0:
            raise ValueError(self.__class__.__name__ + " invalid size of sparse matrix, one of the dimensions is zero: M={}, N={}, K={}".format(M, N, K))

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
        self.ptr = _a.cumsumi(fulli(M + 1, nnzpr)) - nnzpr
        # Create column array
        self.col = _a.fulli(nnz, -1)
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
        keep_nnz : boolean, optional
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
        when converting to :class:`~scipy.sparse.csr_matrix`

        Parameters
        ----------
        sort : bool, optional
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

        ptr = self.ptr
        col = self.col
        D = self._D

        # We truncate all the connections
        if sort:
            for r in range(self.shape[0]):
                # Sort and check whether there are double entries
                sl = slice(ptr[r], ptr[r+1])
                ccol = col[sl]
                DD = D[sl, :]
                idx = argsort(ccol)
                # Do in-place sorting
                ccol[:] = ccol[idx]
                if not sorted_unique(ccol):
                    raise SislError('You cannot have two elements between the same ' +
                                    'i,j index (i={}), something has went terribly wrong.'.format(r))
                DD[:, :] = DD[idx, :]

        else:
            for r in range(self.shape[0]):
                ptr1 = ptr[r]
                ptr2 = ptr[r+1]
                if unique(col[ptr1:ptr2]).shape[0] != ptr2 - ptr1:
                    raise SislError('You cannot have two elements between the same ' +
                                    'i,j index (i={}), something has went terribly wrong.'.format(r))

        if len(col) != self.nnz:
            raise SislError('Final size in the sparse matrix finalization went wrong.') # pragma: no cover

        # Check that all column indices are within the expected shape
        if np_any(self.shape[1] <= self.col):
            warn("Sparse matrix contains column indices outside the shape "
                 "of the matrix. Data may not represent what is expected!")

        # Signal that we indeed have finalized the data
        self._finalized = sort

    def edges(self, row, exclude=None):
        """ Retrieve edges (connections) of a given `row` or list of `row`'s

        The returned edges are unique and sorted (see `numpy.unique`).

        Parameters
        ----------
        row : int or list of int
            the edges are returned only for the given row
        exclude : int or list of int, optional
           remove edges which are in the `exclude` list.
        """
        row = unique(_a.asarrayi(row))
        if exclude is None:
            exclude = []
        else:
            exclude = unique(_a.asarrayi(exclude))

        # Now get the edges
        ptr = self.ptr
        ncol = self.ncol

        # Create column indices
        edges = unique(self.col[array_arange(ptr[row], n=ncol[row])])

        if len(exclude) > 0:
            # Return the difference to the exclude region, we know both are unique
            return setdiff1d(edges, exclude, assume_unique=True)
        return edges

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
        columns = unique(_a.asarrayi(columns))
        n_cols = cnz(columns < self.shape[1])

        # Grab pointers
        ptr = self.ptr
        ncol = self.ncol
        col = self.col

        # Get indices of deleted columns
        idx = array_arange(ptr[:-1], n=ncol)
        # Convert to boolean array where we have columns to be deleted
        lidx = in1d(col[idx], columns)
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
            update_col = np_any(columns < self.shape[1] - n_cols)

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
        self._nnz = int(ncol.sum())

        if not keep_shape:
            shape = list(self.shape)
            shape[1] -= n_cols
            self._shape = tuple(shape)

    def _clean_columns(self):
        """ Remove all intrinsic columns that are not defined in the sparse matrix """
        # Grab pointers
        ptr = self.ptr
        ncol = self.ncol
        col = self.col

        # Number of columns
        nc = self.shape[1]

        # Get indices of columns
        idx = array_arange(ptr[:-1], n=ncol)
        # Convert to boolean array where we have columns to be deleted
        lidx = col[idx] >= nc
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

    def translate_columns(self, old, new, clean=True):
        """ Takes all `old` columns and translates them to `new`.

        Parameters
        ----------
        old : int or array_like
           old column indices
        new : int or array_like
           new column indices
        clean : bool, optional
           whether the new translated columns, outside the shape, should be deleted or not (default delete)
        """
        old = _a.asarrayi(old)
        new = _a.asarrayi(new)

        if len(old) != len(new):
            raise ValueError(self.__class__.__name__+".translate_columns requires input and output columns with "
                             "equal length")

        if allclose(old, new):
            # No need to translate anything...
            return

        if np_any(old >= self.shape[1]):
            raise ValueError(self.__class__.__name__+".translate_columns has non-existing old column values")

        # Now do the translation
        pvt = _a.arangei(self.shape[1])
        pvt[old] = new

        # Get indices of valid column entries
        idx = array_arange(self.ptr[:-1], n=self.ncol)
        # Convert the old column indices to new ones
        col = self.col
        col[idx] = pvt[col[idx]]

        # After translation, set to not finalized
        self._finalized = False
        if clean:
            if np_any(new >= self.shape[1]):
                self._clean_columns()

    def scale_columns(self, col, scale):
        r""" Scale all values with certain column values with a number

        This will multiply all values with certain column values with `scale`

        .. math::
            M[:, cols] *= scale

        This is an in-place operation.

        Parameters
        ----------
        col : int or array_like
           column indices
        scale : float or array_like
           scale value for each value (if array-like it has to have the same
           dimension as the sparsity dimension)
        """
        col = _a.asarrayi(col)

        if np_any(col >= self.shape[1]):
            raise ValueError(self.__class__.__name__+".scale_columns has non-existing old column values")

        # Find indices
        idx = array_arange(self.ptr[:-1], n=self.ncol)
        scale_idx = np.isin(self.col[idx], col).nonzero()[0]

        # Scale values where columns coincide with scaling factor
        self._D[idx[scale_idx]] *= scale

    def spsame(self, other):
        """ Check whether two sparse matrices have the same non-zero elements

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
        if (sncol == oncol).sum() != self.shape[0]:
            return False

        for r in range(self.shape[0]):
            if len(intersect1d(scol[sptr[r]:sptr[r]+sncol[r]],
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
                self._extend(r, ocol[op:op+on], False)
                continue

            sp = sptr[r]
            adds = lsetdiff1d(ocol[op:op+on], scol[sp:sp+sn])
            if len(adds) > 0:
                # simply extend the elements
                self._extend(r, adds, False)

    def iter_nnz(self, row=None):
        """ Iterations of the non-zero elements, returns a tuple of row and column with non-zero elements

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
        if row is None:
            # loop on rows
            for r in range(self.shape[0]):
                n = self.ncol[r]
                ptr = self.ptr[r]
                for c in self.col[ptr:ptr+n]:
                    yield r, c
        else:
            for r in _a.asarrayi(row).ravel():
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

    def _extend(self, i, j, ret_indices=True):
        """ Extends the sparsity pattern to retain elements `j` in row `i`

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
        IndexError for indices out of bounds
        """
        if asarray(i).size == 0:
            return arrayi([])
        if i < 0 or i >= self.shape[0]:
            raise IndexError('row index is out-of-bounds')
        i1 = int(i) + 1
        # We skip this check and let sisl die if wrong input is given...
        #if not isinstance(i, Integral):
        #    raise ValueError("Retrieving/Setting elements in a sparse matrix"
        #                     " must only be performed at one row-element at a time.\n"
        #                     "However, multiple columns at a time are allowed.")
        # Ensure flattened array...
        j = asarrayi(j).ravel()
        if len(j) == 0:
            return arrayi([])
        if np_any(j < 0) or np_any(j >= self.shape[1]):
            raise IndexError('column index is out-of-bounds')

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
            new_j = j[in1d(j, col[ptr_i:ptr_i+ncol_i],
                           invert=True, assume_unique=True)]
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
            self._D = insert(self._D, ptr[i1],
                             zeros([ns, self.shape[2]], self._D.dtype), axis=0)

            # Lastly, shift all pointers above this row to account for the
            # new non-zero elements
            ptr[i1:] += int32(ns)

        if new_n > 0:
            # Ensure that we write the new elements to the matrix...

            # assign the column indices for the new entries
            # NOTE that this may not assign them in the order
            # of entry as new_j is sorted and thus new_j != j
            col[ncol_ptr_i:ncol_ptr_i+new_n] = new_j

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
        """ Extends the sparsity pattern to retain elements `j` in row `i`

        Parameters
        ----------
        i : int
           the row of the matrix
        n : int
           number of elements to add in the space for row `i`

        Raises
        ------
        IndexError for indices out of bounds
        """
        if i < 0 or i >= self.shape[0]:
            raise IndexError('row index is out-of-bounds')

        # fast reference
        i1 = int(i) + 1

        # Ensure that it is not-set as finalized
        # There is no need to set it all the time.
        # Simply because the first call to finalize
        # will reduce the sparsity pattern, which
        # on first expansion calls this part.
        self._finalized = False

        # Insert new empty elements in the column index
        # after the column
        self.col = insert(self.col, self.ptr[i] + self.ncol[i], full(n, -1, self.col.dtype))

        # Insert zero data in the data array
        # We use `zeros` as then one may set each dimension
        # individually...
        self._D = insert(self._D, self.ptr[i1],
                         zeros([n, self.shape[2]], self._D.dtype), axis=0)

        # Lastly, shift all pointers above this row to account for the
        # new non-zero elements
        self.ptr[i1:] += int32(n)

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
        numpy.ndarray
            indices of the existing elements
        """
        # Ensure flattened array...
        j = asarrayi(j).ravel()

        # Make it a little easier
        ptr = self.ptr[i]

        return indices(self.col[ptr:ptr+self.ncol[i]], j, ptr)

    def _get_only(self, i, j):
        """ Retrieves the data pointer arrays of the elements, only return elements in the sparse array

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
        # Ensure flattened array...
        j = asarrayi(j).ravel()

        # Make it a little easier
        ptr = self.ptr[i]

        return indices_only(self.col[ptr:ptr+self.ncol[i]], j) + ptr

    def __delitem__(self, key):
        """ Remove items from the sparse patterns """
        # Get indices of sparse data (-1 if non-existing)
        key = list(key)
        key[0] = self._slice2list(key[0], 0)
        if isiterable(key[0]):
            if len(key) == 2:
                for i in key[0]:
                    del self[i, key[1]]
            elif len(key) == 3:
                for i in key[0]:
                    del self[i, key[1], key[2]]
            return

        i = key[0]
        key[1] = self._slice2list(key[1], 1)
        index = self._get_only(i, key[1])
        index.sort()

        if len(index) == 0:
            # There are no elements to delete...
            return

        # Get short-hand
        ptr = self.ptr
        ncol = self.ncol

        # Get original values
        sl = slice(ptr[i], ptr[i] + ncol[i], None)
        oC = self.col[sl]
        oD = self._D[sl, :]

        # Now create the compressed data...
        index -= ptr[i]
        keep = in1d(_a.arangei(ncol[i]), index, invert=True)

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

        return r

    def __setitem__(self, key, data):
        """ Intrinsic sparse matrix assignment of the item.

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
        i = key[0]
        j = key[1]
        if isinstance(i, (list, ndarray)) and isinstance(j, (list, ndarray)):

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
                        raise ValueError("could not broadcast input array from shape {} into shape {}".format(data.shape, ij.shape + (self.dim,)))
                    if len(key) == 3:
                        if atleast_1d(key[2]).size > 1:
                            raise ValueError("could not broadcast input array from shape {} into shape {}".format(data.shape, ij.shape + (atleast_1d(key[2]).size,)))
                    # flatten data
                    data.shape = (-1,)
                # ij.ndim == 1
                # this should correspond to the diagonal specification case
                # and we don't need to do anything
                # if ij.size != data.shape[0] an error should occur down below
            elif data.ndim == 3:
                if ij.ndim != 2:
                    raise ValueError("could not broadcast input array from 3 dimensions into 2")
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
        """ Check whether a sparse index is non-zero """
        # Get indices of sparse data (-1 if non-existing)
        return np_all(self._get(key[0], key[1]) >= 0)

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
            row = asarrayi(row).ravel()
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

    def eliminate_zeros(self, atol=1e-16):
        """ Remove all zero elememts from the sparse matrix

        This is an *in-place* operation

        Parameters
        ----------
        atol : float, optional
            absolute tolerance below this value will be considered 0.
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

    def copy(self, dims=None, dtype=None):
        """ A deepcopy of the sparse matrix

        Parameters
        ----------
        dims : array-like, optional
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
            new._D = self._D.astype(dtype, copy=True)
        else:
            new._D = empty([len(self.col), dim], dtype)
            for i, dim in enumerate(dims):
                new._D[:, i] = self._D[:, dim]

        # Mark it as the same state as the other one
        new._finalized = self._finalized

        return new

    def tocsr(self, dim=0, **kwargs):
        """ Convert dimension `dim` into a :class:`~scipy.sparse.csr_matrix` format

        Parameters
        ----------
        dim : int, optional
           dimension of the data returned in a scipy sparse matrix format
        **kwargs:
           arguments passed to the :class:`~scipy.sparse.csr_matrix` routine
        """
        shape = self.shape[:2]
        if self.finalized:
            # Easy case...
            return csr_matrix((self._D[:, dim].copy(),
                               self.col.astype(int32, copy=True), self.ptr.astype(int32, copy=True)),
                              shape=shape, **kwargs)

        # Use array_arange
        idx = array_arange(self.ptr[:-1], n=self.ncol)
        # create new pointer
        ptr = insert(_a.cumsumi(self.ncol), 0, 0)

        return csr_matrix((self._D[idx, dim].copy(), self.col[idx], ptr.astype(int32, copy=False)),
                          shape=shape, **kwargs)

    def remove(self, indices):
        """ Return a new sparse CSR matrix with all the indices removed

        Parameters
        ----------
        indices : array_like
           the indices of the rows *and* columns that are removed in the sparse pattern
        """
        indices = asarrayi(indices)

        # Check if we have a square matrix or a rectangular one
        if self.shape[0] >= self.shape[1]:
            rindices = delete(_a.arangei(self.shape[0]), indices)

        else:
            rindices = delete(_a.arangei(self.shape[1]), indices)

        return self.sub(rindices)

    def sub(self, indices):
        """ Create a new sparse CSR matrix with the data only for the given rows and columns

        All rows and columns in `indices` are retained, everything else is removed.

        Parameters
        ----------
        indices : array_like
           the indices of the rows *and* columns that are retained in the sparse pattern
        """
        indices = asarrayi(indices).ravel()

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
        ncol1[:] = _a.fromiteri(map(count_nonzero,
                                    split(idx_take, ptr1[1:-1]))).ravel()

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
            return self._D.sum()
        if axis == -1 or axis == 2:
            # We simply create a new sparse matrix with only one entry
            ret = self.copy([0])
            ret._D[:, 0] = self._D.sum(1)
        elif axis == -2 or axis == 1:
            ret = zeros(self.shape[1], dtype=self.dtype)
            raise NotImplementedError('Currently performing a sum on the columns is not implemented')
        elif axis == 0:
            ret = empty([self.shape[0], self.shape[2]], dtype=self.dtype)
            ptr = self.ptr
            ncol = self.ncol
            for r in range(self.shape[0]):
                ret[r, :] = self._D[ptr[r]:ptr[r]+ncol[r], :].sum(0)

        return ret

    def __str__(self):
        """ Representation of the sparse matrix model """
        ints = self.shape[:] + (self.nnz,)
        return self.__class__.__name__ + '{{dim={2}, kind={kind},\n  rows: {0}, columns: {1},\n  non-zero: {3}\n}}'.format(*ints, kind=self.dkind)

    ###############################
    # Overload of math operations #
    ###############################
    def __add__(self, other):
        c = self.copy(dtype=get_dtype(other, other=self.dtype))
        c += other
        return c
    __radd__ = __add__

    def __iadd__(self, other):
        if isinstance(other, SparseCSR):
            if self.shape != other.shape:
                raise ValueError('Adding two sparse matrices requires the same shape')

            # loop and add elements
            for r in range(self.shape[0]):
                # pointers
                bptr = other.ptr[r]
                bn = other.ncol[r]
                sl = slice(bptr, bptr+bn, None)

                # Get positions of b-elements in a:
                in_a = self._extend(r, other.col[sl])
                self._D[in_a, :] += other._D[sl, :]

        elif isspmatrix(other):
            tmp = SparseCSR(other, shape=self.shape[:2])
            self += tmp

        else:
            self._D += other
        return self

    def __sub__(self, other):
        c = self.copy(dtype=get_dtype(other, other=self.dtype))
        c -= other
        return c

    def __rsub__(self, other):
        if isinstance(other, SparseCSR):
            c = other.copy(dtype=get_dtype(self, other=other.dtype))
            c += -1 * self
        else:
            c = other + (-1) * self
        return c

    def __isub__(self, other):
        if isinstance(other, SparseCSR):
            if self.shape != other.shape:
                raise ValueError('Subtracting two sparse matrices requires the same shape')

            # loop and add elements
            for r in range(self.shape[0]):
                # pointers
                bptr = other.ptr[r]
                bn = other.ncol[r]
                sl = slice(bptr, bptr+bn, None)

                # Get positions of b-elements in a:
                in_a = self._extend(r, other.col[sl])
                self._D[in_a, :] -= other._D[sl, :]

        elif isspmatrix(other):
            tmp = SparseCSR(other, shape=self.shape[:2])
            self -= tmp

        else:
            self._D -= other
        return self

    def __mul__(self, other):
        c = self.copy(dtype=get_dtype(other, other=self.dtype))
        c *= other
        return c
    __rmul__ = __mul__

    def __imul__(self, other):
        if isinstance(other, SparseCSR):
            if self.shape != other.shape:
                raise ValueError('Multiplication of two sparse matrices requires the same shape')

            # Note that for multiplication of these two matrices
            # it is not required that they are aligned...
            # 0 * float == 0
            # Hence aligning is superfluous

            # loop and add elements
            for r in range(self.shape[0]):
                # pointers
                aptr = self.ptr[r]
                an = self.ncol[r]
                bptr = other.ptr[r]
                bn = other.ncol[r]

                acol = self.col[aptr:aptr+an]
                bcol = other.col[bptr:bptr+bn]

                # Get positions of b-elements in a:
                in_a = self._get_only(r, bcol)
                # Everything else *must* be zeroes! :)
                self._D[in_a, :] *= other._D[bptr:bptr+bn, :]

                # Now set everything *not* in b but in a, to zero
                not_in_b = in1d(acol, bcol, invert=True).nonzero()[0]
                self._D[aptr+not_in_b, :] = 0

        elif isspmatrix(other):
            tmp = SparseCSR(other, shape=self.shape[:2])
            self *= tmp

        else:
            self._D *= other
        return self

    def __div__(self, other):
        c = self.copy(dtype=get_dtype(other, other=self.dtype))
        c /= other
        return c

    def __rdiv__(self, other):
        c = other.copy(dtype=get_dtype(self, other=other.dtype))
        c /= self
        return c

    def __idiv__(self, other):
        if isinstance(other, SparseCSR):
            if self.shape != other.shape:
                raise ValueError('Division of two sparse matrices requires the same shape')

            # loop and add elements
            for r in range(self.shape[0]):
                # pointers
                bptr = other.ptr[r]
                bn = other.ncol[r]
                sl = slice(bptr, bptr+bn, None)

                # Get positions of b-elements in a:
                in_a = self._extend(r, other.col[sl])
                self._D[in_a, :] /= other._D[sl, :]

        elif isspmatrix(other):
            tmp = SparseCSR(other, shape=self.shape[:2])
            self /= tmp

        else:
            self._D /= other
        return self

    def __floordiv__(self, other):
        c = self.copy(dtype=get_dtype(other, other=self.dtype))
        c //= other
        return c

    def __ifloordiv__(self, other):
        if isinstance(other, SparseCSR):
            if self.shape != other.shape:
                raise ValueError('Floor-division of two sparse matrices requires the same shape')

            # loop and add elements
            for r in range(self.shape[0]):
                # pointers
                bptr = other.ptr[r]
                bn = other.ncol[r]
                sl = slice(bptr, bptr+bn, None)

                # Get positions of b-elements in a:
                in_a = self._extend(r, other.col[sl])
                self._D[in_a, :] //= other._D[sl, :]

        elif isspmatrix(other):
            tmp = SparseCSR(other, shape=self.shape[:2])
            self //= tmp

        else:
            self._D //= other
        return self

    def __truediv__(self, other):
        c = self.copy(dtype=get_dtype(other, other=self.dtype))
        c /= other
        return c

    def __itruediv__(self, other):
        if isinstance(other, SparseCSR):
            if self.shape != other.shape:
                raise ValueError('True-division of two sparse matrices requires the same shape')

            # loop and add elements
            for r in range(self.shape[0]):
                # pointers
                bptr = other.ptr[r]
                bn = other.ncol[r]
                sl = slice(bptr, bptr+bn, None)

                # Get positions of b-elements in a:
                in_a = self._get(r, other.col[sl])
                self._D[in_a, :].__itruediv__(other._D[sl, :])

        elif isspmatrix(other):
            tmp = SparseCSR(other, shape=self.shape[:2])
            self /= tmp

        else:
            self._D /= other
        return self

    def __pow__(self, other):
        c = self.copy(dtype=get_dtype(other, other=self.dtype))
        c **= other
        return c

    def __rpow__(self, other):
        if isinstance(other, SparseCSR):
            raise NotImplementedError
        c = self.copy(dtype=get_dtype(other, other=self.dtype))
        c._D[...] = other ** c._D[...]
        return c

    def __ipow__(self, other):
        if isinstance(other, SparseCSR):
            if self.shape != other.shape:
                raise ValueError('True-division of two sparse matrices requires the same shape')
            # Ensure that a is aligned with b
            # 0 ** float == 1.

            # loop and add elements
            for r in range(self.shape[0]):
                # pointers
                aptr = self.ptr[r]
                an = self.ncol[r]
                bptr = other.ptr[r]
                bn = other.ncol[r]

                acol = self.col[aptr:aptr+an]
                bcol = other.col[bptr:bptr+bn]

                # Get positions of b-elements in a:
                in_a = self._extend(r, bcol)
                self._D[in_a, :] **= other._D[bptr:bptr+bn, :]

                # Now set everything *not* in b but in a, to 1
                #  float ** 0 == 1
                not_in_b = in1d(acol, bcol, invert=True).nonzero()[0]
                self._D[aptr+not_in_b, :] = 1

        elif isspmatrix(other):
            tmp = SparseCSR(other, shape=self.shape[:2])
            self **= tmp

        else:
            self._D **= other
        return self

    def __getstate__(self):
        """ Return dictionary with the current state """
        d = {}
        # Reduce array sizes
        self.finalize()
        return {
            'shape': self._shape[:],
            'ncol': self.ncol.copy(),
            'col': self.col.copy(),
            'D': self._D.copy()
        }

    def __setstate__(self, state):
        """ Reset state of the object """
        self._shape = tuple(state['shape'][:])
        self.ncol = state['ncol']
        self.ptr = insert(_a.cumsumi(self.ncol), 0, 0)
        self.col = state['col']
        self._D = state['D']


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

    nrow = len(unique(map_row(arange(matrix.shape[0], dtype=int32))))
    ncol = len(unique(map_col(arange(matrix.shape[1], dtype=int32))))
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
