"""
Sparsity pattern used to express matrices in concise manners.
"""
from __future__ import print_function, division

import warnings
from numbers import Integral, Real, Complex

from scipy.sparse import isspmatrix

# Other functions may be retrieved from np.
import numpy as np

# To speed up the extension algorithm we limit
# the lookup table
from numpy import where, insert, diff
from numpy import array, asarray, empty, zeros, arange
from numpy import intersect1d, setdiff1d
from numpy import argsort, unique, in1d

from sisl._help import ensure_array, get_dtype

# Although this re-implements the CSR in scipy.sparse.csr_matrix
# we use it slightly differently and thus require this new sparse pattern.

__all__ = ['SparseCSR']


class SparseCSR(object):
    """
    A compressed sparse row matrix, slightly different than ``scipy.sparse.csr_matrix``.

    This class holds all required information regarding the CSR matrix format.

    Note that this sparse matrix of data does not retain the number of columns
    in the matrix, i.e. it has no way of determining whether the input is 
    correct.
    """

    def __init__(self, arg1, dim=1, dtype=None,
                 nnzpr=20, nnz=None,
                 **kwargs):
        """ Initialize a new sparse CSR matrix

        This sparse matrix class tries to resemble the
        ``scipy.sparse.csr_matrix`` as much as possible with
        the difference of this class being multi-dimensional.

        Creating a new sparse matrix is much similar to the 
        ``scipy`` equivalent.

        `nzs` is only used if `nzs > nr * nzsr`.

        This class may be instantiated by verious means.

        - `SparseCSR(S)`
          where `S` is a ``scipy.sparse`` matrix
        - `SparseCSR((M,N)[, dtype])`
          the shape of the sparse matrix (equivalent
          to `SparseCSR((M,N,K)[, dtype])`.
        - `SparseCSR((M,N,K)[, dtype])`
          creating a sparse matrix with `M` rows, `N` columns
          and `K` elements per sparse element.

        Additionally these parameters control the
        creation of the sparse matrix

        Parameters
        ----------
        nnzpr : int, 20
           initial number of non-zero elements per row.
           Only used if `nnz` is not supplied
        nnz : int
           initial total number of non-zero elements
           This quantity has precedence over `nnzpr`
        dim : int, 1
           number of elements stored per sparse element, only used if (M,N) is passed
        dtype : numpy data type, `numpy.float64`
           data type of the matrix

        Attributes
        ----------
        ncol: int-array, `self.shape[0]`
           number of entries per row
        ptr: int-array, `self.shape[0]+1`
           pointer index in the 1D column indices of the corresponding row
        col: int-array, 
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
                                  nnz=1,
                                  **kwargs)

                # Copy data to the arrays
                self.ptr = arg1[2]
                self.ncol = diff(self.ptr)
                self.col = arg1[1]
                self._nnz = len(self.col)
                self._D = np.empty([len(arg1[1]), self.shape[-1]], dtype=self.dtype)
                self._D[:, 0] = arg1[0]

                self.finalize()

    def __init_shape(self, arg1, dim=1, dtype=None,
                     nnzpr=20, nnz=None,
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
        self.ncol = np.zeros([M], np.int32)
        # Create pointer array
        self.ptr = np.cumsum(np.array([nnzpr] * (M+1), np.int32)) - nnzpr
        # Create column array
        self.col = np.empty(nnz, np.int32)
        # Store current number of non-zero elements
        self._nnz = 0

        # Important that this is zero
        # For instance one may set one dimension at a time
        # thus automatically zeroing the other dimensions.
        self._D = np.zeros([nnz, K], dtype)

        # Denote that this sparsity pattern hasn't been finalized
        self._finalized = False

    def empty(self, keep=False):
        """ Delete all sparse information from the sparsity pattern

        Essentially this deletes all entries.

        Parameters
        ----------
        keep: boolean, False
           if `True` it will keep the sparse elements _as is_.
           I.e. it will merely set the stored sparse elements as zero.
           This may be advantagegous when re-constructing a new sparse 
           matrix from an old sparse matrix
        """
        self._D[:, :] = 0.

        if not keep:
            self._finalized = False
            # The user does not wish to retain the
            # sparse pattern
            self.ncol[:] = 0
            self._nnz = 0
            # We do not mess with the other arrays
            # they may be obscure data any-way.

    @property
    def shape(self):
        """ Return shape of the sparse matrix """
        return self._shape

    @property
    def dim(self):
        """ Return extra dimensionality of the sparse matrix """
        return self.shape[2]

    @property
    def data(self):
        """ Return data contained in the sparse matrix """
        return self._D

    @property
    def dtype(self):
        """ Return the data-type in the sparse matrix """
        return self._D.dtype

    @property
    def nnz(self):
        """ Return number of non-zero elements in the sparsity pattern """
        return self._nnz

    def __len__(self):
        """ Return number of non-zero elements in the sparse pattern """
        return self.nnz

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
        sort: bool, True
           sort the column indices for each row
        """
        if self.finalized:
            return

        # Fast reference
        ptr = self.ptr
        ncol = self.ncol
        col = self.col
        D = self._D

        # We truncate all the connections
        iptr = 0
        for r in range(self.shape[0]):

            # number of elements in this row
            nor = ncol[r]

            # Starting pointer index
            sptr = ptr[r]
            eptr = sptr + nor

            # update current row pointer
            ptr[r] = iptr

            if nor == 0:
                continue

            # assert no two connections
            if unique(col[sptr:eptr]).shape[0] != nor:
                raise ValueError(
                    'You cannot have two hoppings between the same ' +
                    'orbitals ({}), something has went terribly wrong.'.format(r))

            # update the colunm vector and data
            col[iptr:iptr + nor] = col[sptr:eptr]
            D[iptr:iptr + nor, :] = D[sptr:eptr, :]

            # Simultaneausly we sort the entries
            if sort:
                si = argsort(col[iptr:iptr + nor])
                col[iptr:iptr + nor] = col[iptr + si]
                D[iptr:iptr + nor, :] = D[iptr + si, :]

            # update front of row
            iptr += nor

        # Correcting the size of the pointer array
        ptr[self.shape[0]] = iptr

        if iptr != self.nnz:
            print(iptr, self.nnz)
            raise ValueError('Final size in the sparse matrix finalization '
                             'went wrong.')

        # Truncate values to correct size
        self._D = self._D[:self.nnz, :]
        self.col = self.col[:self.nnz]

        # Check that all column indices are within the expected shape
        if np.any(self.shape[1] <= self.col):
            warnings.warn("Sparse matrix contains column indices outside the shape "
                          "of the matrix. Data may not represent what you had expected")

        # Signal that we indeed have finalized the data
        self._finalized = True

    def _extend(self, i, j):
        """ Extends the sparsity pattern to retain elements `j` in row `i`

        Parameters
        ----------
        i : int
           the row of the matrix
        j : int, array-like
           columns belonging to row ``i`` where a non-zero element is stored.

        Returns
        -------
        index : array-like
           the indicies of the existing/added elements. 
        """

        # We skip this check and let sisl die if wrong input is given...
        #if not isinstance(i, Integral):
        #    raise ValueError("Retrieving/Setting elements in a sparse matrix"
        #                     " must only be performed at one row-element at a time.\n"
        #                     "However, multiple columns at a time are allowed.")

        # fast reference
        ptr = self.ptr
        ncol = self.ncol
        col = self.col

        # Ensure flattened array...
        j = ensure_array(j)
        if len(j) == 0:
            return np.array([], np.int32)

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
            exists = []

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

            # First shift all pointers above this row
            ptr[i + 1:] += ns

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

        # lets retrieve the indices...

        # preallocate the indices
        index = empty(len(j), np.int32)

        # we may use further dereference
        col = col[ptr[i]:ptr[i] + ncol[i]]

        # We probably should change this to something faster...
        for ii, jj in enumerate(j):
            index[ii] = where(col == jj)[0]

        # Correct index by the pointer offset and return
        return ptr[i] + index

    def _get(self, i, j):
        """ Retrieves the data pointer arrays of the elements, if it is non-existing, it will return -1

        Parameters
        ----------
        i : int
           the row of the matrix
        j : int, array-like
           columns belonging to row ``i`` where a non-zero element is stored.

        Returns
        -------
        index : array-like
           the indicies of the existing elements. 
        """

        # Ensure flattened array...
        j = asarray(j, np.int32).flatten()

        # we use dereference
        ptr = self.ptr[i]
        col = self.col[ptr:ptr+self.ncol[i]]

        # preallocate the indices (with pointer offset)
        index = ptr + zeros([len(j)], np.int32)

        # We probably should change this to something faster...
        for ii, jj in enumerate(j):
            w = where(col == jj)[0]
            if len(w) == 0:
                index[ii] = -1
            else:
                index[ii] += w

        return index

    def __delitem__(self, key):
        """ Remove items from the sparse patterns """
        # Get indices of sparse data (-1 if non-existing)
        i = key[0]
        index = self._get(i, key[1])

        # First remove all negative indices.
        # The element isn't there anyway...
        index.sort()
        index = index[index >= 0]

        if len(index) == 0:
            # There are no elements to delete...
            return

        # Get short-hand
        ptr = self.ptr
        ncol = self.ncol

        # Get original values
        oC = self.col[ptr[i]:ptr[i]+ncol[i]]
        oD = self._D[ptr[i]:ptr[i]+ncol[i], :]

        # Now create the compressed data...
        index -= ptr[i]
        keep = in1d(arange(ncol[i]), index, invert=True)

        # Update new count of the number of
        # non-zero elements
        self.ncol[i] -= len(index)

        # Now update the column indices and the data
        self.col[ptr[i]:ptr[i]+self.ncol[i]] = oC[keep]
        self._D[ptr[i]:ptr[i]+self.ncol[i], :] = oD[keep, :]

        # Once we remove some things, it is NOT
        # finalized...
        self._finalized = False
        self._nnz -= len(index)

    def __getitem__(self, key):
        """ Intrinsic sparse matrix retrieval of a non-zero element
        """

        # Get indices of sparse data (-1 if non-existing)
        index = self._get(key[0], key[1])

        # Check which data to retrieve
        if len(key) > 2:

            # user requests a specific element

            data = empty(len(index), self._D.dtype)

            # get dimension retrieved
            d = key[2]

            # Copy data over
            for i, j in enumerate(index):
                if j < 0:
                    data[i] = 0.
                else:
                    data[i] = self._D[j, d]

        else:

            # user request all stored data

            data = empty([len(index), self.shape[2]], self._D.dtype)

            # Copy data over
            for i, j in enumerate(index):
                if j < 0:
                    data[i, :] = 0.
                else:
                    data[i, :] = self._D[j, :]

        # Return data
        return data

    def __setitem__(self, key, data):
        """ Intrinsic sparse matrix assignment of the item. 

        It will only allow to set the data in the sparse
        matrix if the dimensions match.

        If the `data` parameter is `None` or an array
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
            data.shape = (-1, self.shape[2])

            # Now there are two cases
            if data.shape[0] == 1:
                # we copy all elements
                self._D[index, :] = data[None, :]

            else:
                # each element have different data
                self._D[index, :] = data[:, :]

    def copy(self, dims=None, dtype=None):
        """ Returns an exact copy of the sparse matrix

        Parameters
        ----------
        dims: array-like, (all)
           which dimensions to store in the copy
        dtype : `numpy.dtype`
           this defaults to the dtype of the object, 
           but one may change it if supplied.
        """
        # Create sparse matrix (with only one entry per
        # row, we overwrite it immediately afterward)
        if dims is None:
            dims = range(self.dim)

        # Create correct input
        dim = len(dims)
        shape = list(self.shape[:])
        shape[2] = dim

        if dtype is None:
            dtype = self.dtype

        new = self.__class__(shape, nnz=self.nnz,
                             dtype=dtype)

        # The default sizes are not passed
        # Hence we *must* copy the arrays
        # directly
        new.ptr = np.array(self.ptr, np.int32)
        new.ncol = np.array(self.ncol, np.int32)
        new.col = np.array(self.col, np.int32)
        new._nnz = self.nnz

        new._D = np.array(self._D, dtype=dtype)
        for dim in dims:
            new._D[:, dims] = self._D[:, dims]

        return new

    def tocsr(self, dim=0, **kwargs):
        """ Return the data in ``scipy.sparse.csr_matrix`` format

        Parameters
        ----------
        dim: int
           the dimension of the data to create the sparse matrix
        """
        self.finalize()
        from scipy.sparse import csr_matrix

        shape = self.shape[:2]
        return csr_matrix((self._D[:, dim], self.col, self.ptr), shape=shape, **kwargs)

    ###############################
    # Overload of math operations #
    ###############################
    def __add__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._D += b
        return c
    __radd__ = __add__

    def __iadd__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        a._D += b
        return a

    def __sub__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._D -= b
        return c

    def __rsub__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = b + (-1) * a
        return c

    def __isub__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        a._D -= b
        return a

    def __mul__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._D *= b
        return c
    __rmul__ = __mul__

    def __imul__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        a._D *= b
        return a

    def __div__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._D /= b
        return c

    def __idiv__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        a._D /= b
        return a

    def __floordiv__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._D //= b
        return c

    def __ifloordiv__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        a._D //= b
        return a

    def __truediv__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._D /= b
        return c

    def __itruediv__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        a._D /= b
        return a

    def __pow__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._D **= b
        return c

    def __rpow__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._D[...] = b ** c._D[...]
        return c

    def __ipow__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        a._D **= b
        return a

    @classmethod
    def register_math(cls, var, routines=None):
        """ Register math operators on the `cls` class using `var` as attribute `getattr(cls, var)` 

        Parameters
        ----------
        cls : class
           class which gets registered overloaded math operators
        var : `str`
           name of attribute that is `SparseCSR` object in `cls`
        routines : list of str
           names of routines that gets overloaded, defaults to:
            ['__sub__', '__add__', '__mul__', '__div__', 
             '__truediv__', '__pow__']
        """

        if routines is None:
            routines = ['__sub__', '__add__', '__mul__', '__div__',
                        '__truediv__', '__pow__']

        # What we want is something like this:
        # def func(a, b):
        #   if isinstance(a, cls):
        #       setattr(a, var, getattr(a, var) OP b)
        #   if isinstance(b, cls):
        #       setattr(b, var, a OP getattr(b, var))
        # setattr(cls,__ROUTINE__, func):

        # Now register all things
        for r in routines:
            pass
