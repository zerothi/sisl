"""
Sparsity pattern used to express matrices in concise manners.
"""
from __future__ import print_function, division

import warnings
from numbers import Integral

from scipy.sparse import isspmatrix

# Other functions may be retrieved from np.
import numpy as np

# To speed up the extension algorithm we limit
# the lookup table
from numpy import where, insert, diff
from numpy import array, asarray, empty, zeros
from numpy import intersect1d, setdiff1d
from numpy import argsort, unique

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
                 nnzpr=20,nnz=None,
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
           number of elements stored per sparse element
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

        if isspmatrix(arg1):
            # This is a sparse matrix
            # The data-type is infered from the
            # input sparse matrix.
            arg1 = arg1.tocsr()

            # Create sparse matrix (with only one entry per
            # row, we overwrite it immediately afterward)
            self = self.__class__(arg1.shape, nnz=1, 
                                  dim=dim, dtype=arg1.dtype)
            
            self.ptr = arg1.indptr
            self.ncol = diff(self.ptr)
            self.col = arg1.indices
            # total number of sparse elements
            self._nnz = arg1.getnnz()
            self._D = arg1.data
            self._D.shape = (-1, 1)

            # This should also sort the entries...
            self.finalize()

        elif isinstance(arg1, tuple):

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
            
        # step size in sparse elements
        # If there isn't enough room for adding
        # a non-zero element, the # of elements
        # for the insert row is increased at least by this number
        self._ns = 10

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
        self._D[:,:] = 0.
        
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
        j = asarray(j, np.int32).flatten()

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
        """ Set items to zero
        """
        
        # Get indices of sparse data (-1 if non-existing)
        index = self._get(key[0], key[1])

        # When deleting, we should remove them from the sparse matrix
        raise NotImplementedError("Deletion of a sparse element is not implemented yet.")


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
                    data[i,:] = 0.
                else:
                    data[i,:] = self._D[j, :]
        
        # Return data
        return data            

            
    def __setitem__(self, key, data):
        """ Intrinsic sparse matrix assignment of the item. 

        It will only allow to set the data in the sparse
        matrix if the dimensions match.
        """
        index = self._extend(key[0], key[1])

        # Ensure data type... possible casting...
        data = asarray(data, self._D.dtype)

        if len(key) > 2:
            # Explicit data of certain dimension
            self._D[index, key[2]] = data

        else:
            # Ensure correct shape
            data.shape = (-1,self.shape[2])

            # Now there are two cases
            if data.shape[0] == 1:
                # we copy all elements
                self._D[index, :] = data[None,:]
                
            else:
                # each element have different data
                self._D[index, :] = data[:,:]


    def copy(self, dims=None):
        """ Returns an exact copy of the sparse matrix

        Parameters
        ----------
        dims: array-like, (all)
           which dimensions to store in the copy
        """
        # Create sparse matrix (with only one entry per
        # row, we overwrite it immediately afterward)
        if dims is None:
            dims = range(self.dim)

        # Create correct input
        dim = len(dims)
        shape = list(self.shape[:])
        shape[2] = dim

        new = self.__class__(shape, nnz=self.nnz, 
                             dim=dim, dtype=self.dtype)

        new.ptr[:] = self.ptr[:]
        new.ncol[:] = self.ncol[:]
        new.col[:] = self.col[:]
        new._nnz = self.nnz

        for dim in dims:
            new._D[:,dims] = self._D[:,dims]
        
        return new

                
    @staticmethod
    def fromcsr(self, csr, dim=1):
        """ Return a new ``SparseCSR`` object from a csr matrix """

        # Initialize a new matrix...
        sd = self.__class__(csr.shape, nnzpr=1, nnz=csr.getnnz(), dim=dim, dtype=csr.dtype)

        # Copy data...
        sd.ptr[:] = csr.indptr[:]
        # Create our own additional information in the CSR format
        sd.ncol[:] = diff(sd.ptr)
        sd.col[:] = csr.indices[:]
        
        # The data is only 1D
        sd._D[:, 0] = csr.data[:]

        return sd


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
