from __future__ import print_function, division

import warnings
import functools as ftool
from numbers import Integral
import numpy as np
from numpy import int32
from numpy import insert, unique, take, delete, argsort
from numpy import tile, repeat, concatenate

from . import _array as _a
from .atom import Atom
from .orbital import Orbital
from .geometry import Geometry
from .messages import warn, SislError, SislWarning, tqdm_eta
from ._indices import indices_only
from ._help import get_dtype
from ._help import _zip as zip, _range as range, _map as map
from .utils.ranges import array_arange
from .sparse import SparseCSR

__all__ = ['SparseAtom', 'SparseOrbital']


class _SparseGeometry(object):
    """ Sparse object containing sparse elements for a given geometry.

    This is a base class intended to be sub-classed because the sparsity information
    needs to be extracted from the ``_size`` attribute.

    The sub-classed object _must_ implement the ``_size`` attribute.
    The sub-classed object may re-implement the ``_cls_kwargs`` routine
    to pass down keyword arguments when a new class is instantiated.

    This object contains information regarding the
     - geometry

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

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        """ Create sparse object with element between orbitals """
        self._geometry = geometry

        # Initialize the sparsity pattern
        self.reset(dim, dtype, nnzpr)

    @property
    def geometry(self):
        """ Associated geometry """
        return self._geometry
    geom = geometry

    @property
    def _size(self):
        """ The size of the sparse object """
        return self.geometry.na

    def __len__(self):
        """ Number of rows in the basis """
        return self._size

    def _cls_kwargs(self):
        """ Custom keyword arguments when creating a new instance """
        return {}

    def reset(self, dim=None, dtype=np.float64, nnzpr=None):
        """ The sparsity pattern has all elements removed and everything is reset.

        The object will be the same as if it had been
        initialized with the same geometry as it were
        created with.

        Parameters
        ----------
        dim : int, optional
           number of dimensions per element, default to the current number of
           elements per matrix element.
        dtype : numpy.dtype, optional
           the datatype of the sparse elements
        nnzpr : int, optional
           number of non-zero elements per row
        """
        # I know that this is not the most efficient way to
        # access a C-array, however, for constructing a
        # sparse pattern, it should be faster if memory elements
        # are closer...
        if dim is None:
            dim = self.dim

        # We check the first atom and its neighbours, we then
        # select max(5,len(nc) * 4)
        if nnzpr is None:
            nnzpr = self.geometry.close(0)
            if nnzpr is None:
                nnzpr = 8
            else:
                nnzpr = max(5, len(nnzpr) * 4)

        # query dimension of sparse matrix
        s = self._size
        self._csr = SparseCSR((s, s * self.geometry.n_s, dim), nnzpr=nnzpr, dtype=dtype)

        # Denote that one *must* specify all details of the elements
        self._def_dim = -1

    def empty(self, keep_nnz=False):
        """ See :meth:`~sparse.SparseCSR.empty` for details """
        self._csr.empty(keep_nnz)

    def copy(self, dtype=None):
        """ A copy of this object

        Parameters
        ----------
        dtype : numpy.dtype, optional
           it is possible to convert the data to a different data-type
           If not specified, it will use ``self.dtype``
        """
        if dtype is None:
            dtype = self.dtype
        new = self.__class__(self.geometry.copy(), self.dim, dtype, 1, **self._cls_kwargs())
        # Be sure to copy the content of the SparseCSR object
        new._csr = self._csr.copy(dtype=dtype)
        return new

    @property
    def dim(self):
        """ Number of components per element """
        return self._csr.shape[-1]

    @property
    def shape(self):
        """ Shape of sparse matrix """
        return self._csr.shape

    @property
    def dtype(self):
        """ Data type of sparse elements """
        return self._csr.dtype

    @property
    def dkind(self):
        """ Data type of sparse elements (in str) """
        return self._csr.dkind

    @property
    def nnz(self):
        """ Number of non-zero elements """
        return self._csr.nnz

    def _translate_cells(self, old, new):
        """ Translates all columns in the `old` cell indices to the `new` cell indices

        Since the physical matrices are stored in a CSR form, with shape ``(no, no * n_s)`` each
        block of ``(no, no)`` refers to supercell matrices with an offset according to the internal
        supercell index.
        This routine may be used to translate from one sorting of the columns to another sorting of the columns.

        Parameters
        ----------
        old : list of int
           integer list of supercell indices (all smaller than `n_s`) that the current blocks of matrices
           belong to.
        new : list of int
           integer list of supercell indices (all smaller than `n_s`) that the current blocks of matrices
           are being transferred to. Must have same length as `old`.
        """
        old = _a.asarrayi(old).ravel()
        new = _a.asarrayi(new).ravel()

        if len(old) != len(new):
            raise ValueError(self.__class__.__name__+".translate_cells requires input and output indices with "
                             "equal length")

        no = self.no
        # Number of elements per matrix
        n = _a.emptyi(len(old))
        n.fill(no)
        old = array_arange(old * no, n=n)
        new = array_arange(new * no, n=n)
        self._csr.translate_columns(old, new)

    def edges(self, atom, exclude=None):
        """ Retrieve edges (connections) of a given `atom` or list of `atom`'s

        The returned edges are unique and sorted (see `numpy.unique`) and are returned
        in supercell indices (i.e. ``0 <= edge < self.geometry.na_s``).

        Parameters
        ----------
        atom : int or list of int
            the edges are returned only for the given atom
        exclude : int or list of int, optional
           remove edges which are in the `exclude` list.
           Default to `atom`.

        See Also
        --------
        SparseCSR.edges: the underlying routine used for extracting the edges
        """
        return self._csr.edges(atom, exclude)

    def __str__(self):
        """ Representation of the sparse model """
        s = self.__class__.__name__ + '{{dim: {0}, non-zero: {1}, kind={2}\n '.format(self.dim, self.nnz, self.dkind)
        s += str(self.geometry).replace('\n', '\n ')
        return s + '\n}'

    def __getattr__(self, attr):
        """ Overload attributes from the hosting geometry

        Any attribute not found in the sparse class will
        be looked up in the hosting geometry.
        """
        return getattr(self.geometry, attr)

    # Make the indicis behave on the contained sparse matrix
    def __delitem__(self, key):
        """ Delete elements of the sparse elements """
        del self._csr[key]

    def __contains__(self, key):
        """ Check whether a sparse index is non-zero """
        return key in self._csr

    def set_nsc(self, size, *args, **kwargs):
        """ Reset the number of allowed supercells in the sparse geometry

        If one reduces the number of supercells, *any* sparse element
        that references the supercell will be deleted.

        See `SuperCell.set_nsc` for allowed parameters.

        See Also
        --------
        SuperCell.set_nsc : the underlying called method
        """
        sc = self.sc.copy()
        # Try first in the new one, then we figure out what to do
        sc.set_nsc(*args, **kwargs)
        if np.all(sc.nsc == self.sc.nsc):
            return

        # Create an array of all things that should be translated
        old = []
        new = []
        deleted = np.empty(self.n_s, np.bool_)
        deleted[:] = True
        for i, sc_off in sc:
            try:
                # Luckily there are *only* one time wrap-arounds
                j = self.sc.sc_index(sc_off)
                # Now do translation
                old.append(j)
                new.append(i)
                deleted[j] = False
            except:
                # Not found, i.e. new, so no need to translate
                pass

        if len(old) not in [self.n_s, sc.n_s]:
            raise SislError("Not all supercells are accounted for")

        # 1. Ensure that any one of the *old* supercells that
        #    are now deleted are put in the end
        for i, j in enumerate(deleted.nonzero()[0]):
            # Old index (j)
            old.append(j)
            # Move to the end (*HAS* to be higher than the number of
            # cells in the new supercell structure)
            new.append(sc.n_s + i)

        old = _a.arrayi(old)
        new = _a.arrayi(new)

        # Assert that there are only unique values
        if len(unique(old)) != len(old):
            raise SislError("non-unique values in old set_nsc")
        if len(unique(new)) != len(new):
            raise SislError("non-unique values in new set_nsc")
        if self.n_s != len(old):
            raise SislError("non-valid size of in old set_nsc")

        # Figure out if we need to do any work
        keep = (old != new).nonzero()[0]
        if len(keep) > 0:
            # Reduce pivoting work
            old = old[keep]
            new = new[keep]

            # Create the translation tables
            n = tile([size], len(old))

            old = array_arange(old * size, n=n)
            new = array_arange(new * size, n=n)

            # Move data to new positions
            self._csr.translate_columns(old, new)

            max_n = new.max() + 1
        else:
            max_n = 0

        # Make sure we delete all column values where we have put fake values
        delete = _a.arangei(sc.n_s * size, max(max_n, self.shape[1]))
        if len(delete) > 0:
            self._csr.delete_columns(delete, keep_shape=True)

        # Ensure the shape is correct
        shape = list(self._csr.shape)
        shape[1] = size * sc.n_s
        self._csr._shape = tuple(shape)

        self.geometry.set_nsc(*args, **kwargs)

    def transpose(self):
        """ Create the transposed sparse geometry by interchanging supercell indices

        Sparse geometries are (typically) relying on symmetry in the supercell picture.
        Thus when one transposes a sparse geometry one should *ideally* get the same
        matrix. This is true for the Hamiltonian, density matrix, etc.

        This routine transposes all rows and columns such that any interaction between
        row, `r`, and column `c` in a given supercell `(i,j,k)` will be transposed
        into row `c`, column `r` in the supercell `(-i,-j,-k)`.

        Notes
        -----
        For Hamiltonians with non-collinear or spin-orbit there is no transposing of the
        sub-spin matrix box. This needs to be done *manually*.

        Examples
        --------

        Force a sparse geometry to be Hermitian:

        >>> sp = SparseOrbital(...)
        >>> sp = (sp + sp.transpose()) * 0.5

        Returns
        -------
        object
            an equivalent sparse geometry with transposed matrix elements
        """
        # Create a temporary copy to put data into
        T = self.copy()
        T._csr.ptr = None
        T._csr.col = None
        T._csr.ncol = None
        T._csr._D = None

        # Short-linkes
        sc = self.geometry.sc

        # Create "DOK" format indices
        csr = self._csr
        # Number of rows (used for converting to supercell indices)
        size = csr.shape[0]

        # First extract the actual data
        ncol = csr.ncol.view()
        if csr.finalized:
            ptr = csr.ptr.view()
            col = csr.col.copy()
        else:
            idx = array_arange(csr.ptr[:-1], n=ncol, dtype=int32)
            ptr = insert(_a.cumsumi(ncol), 0, 0)
            col = csr.col[idx]
            del idx

        # Create an array ready for holding all transposed columns
        row = _a.zerosi(len(col))
        row[ptr[1:-1]] = 1
        _a.cumsumi(row, out=row)
        D = csr._D.copy()

        # Now we have the DOK format
        #  row, col, _D

        # Retrieve all sc-indices in the new transposed array
        new_sc_off = sc.sc_index(- sc.sc_off)

        # Calculate the row-offsets in the new sparse geometry
        row += new_sc_off[sc.sc_index(sc.sc_off[col // size, :])] * size

        # Now convert columns into unit-cell
        col %= size

        # Now we can re-create the sparse matrix
        # All we need is to count the number of non-zeros per column.
        _, nrow = unique(col, return_counts=True)

        # Now we have everything ready...
        # Simply figure out how to sort the columns
        # such that we have them unified.
        idx = argsort(col)

        # Our new data will then be
        T._csr.col = take(row, idx, out=row).astype(int32, copy=False)
        del row
        T._csr._D = take(D, idx, axis=0)
        del D
        T._csr.ncol = nrow.astype(int32, copy=False)
        T._csr.ptr = insert(_a.cumsumi(nrow), 0, 0)

        # For-sure we haven't sorted the columns.
        # We haven't changed the number of non-zeros
        T._csr._finalized = False

        return T

    def spalign(self, other):
        """ See :meth:`~sisl.sparse.SparseCSR.align` for details """
        if isinstance(other, SparseCSR):
            self._csr.align(other)
        else:
            self._csr.align(other._csr)

    def eliminate_zeros(self, atol=0.):
        """ Removes all zero elements from the sparse matrix

        This is an *in-place* operation.

        Parameters
        ----------
        atol : float, optional
            absolute tolerance equal or below this value will be considered 0.
        """
        self._csr.eliminate_zeros(atol)

    # Create iterations on the non-zero elements
    def iter_nnz(self):
        """ Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz():
        ...    self[i, j] # is then the non-zero value
        """
        for i, j in self._csr:
            yield i, j

    __iter__ = iter_nnz

    def create_construct(self, R, param):
        """ Create a simple function for passing to the `construct` function.

        This is simply to leviate the creation of simplistic
        functions needed for setting up the sparse elements.

        Basically this returns a function:

        >>> def func(self, ia, idxs, idxs_xyz=None):
        ...     idx = self.geometry.close(ia, R=R, idx=idxs)
        ...     for ix, p in zip(idx, param):
        ...         self[ia, ix] = p

        Notes
        -----
        This function only works for geometry sparse matrices (i.e. one
        element per atom). If you have more than one element per atom
        you have to implement the function your-self.

        Parameters
        ----------
        R : array_like
           radii parameters for different shells.
           Must have same length as `param` or one less.
           If one less it will be extended with ``R[0]/100``
        param : array_like
           coupling constants corresponding to the `R`
           ranges. ``param[0,:]`` are the elements
           for the all atoms within ``R[0]`` of each atom.

        See Also
        --------
        construct : routine to create the sparse matrix from a generic function (as returned from `create_construct`)
        """

        def func(self, ia, idxs, idxs_xyz=None):
            idx = self.geometry.close(ia, R=R, idx=idxs, idx_xyz=idxs_xyz)
            for ix, p in zip(idx, param):
                self[ia, ix] = p

        return func

    def construct(self, func, na_iR=1000, method='rand', eta=False):
        """ Automatically construct the sparse model based on a function that does the setting up of the elements

        This may be called in two variants.

        1. Pass a function (`func`), see e.g. ``create_construct``
           which does the setting up.
        2. Pass a tuple/list in `func` which consists of two
           elements, one is ``R`` the radii parameters for
           the corresponding parameters.
           The second is the parameters
           corresponding to the ``R[i]`` elements.
           In this second case all atoms must only have
           one orbital.

        Parameters
        ----------
        func : callable or array_like
           this function *must* take 4 arguments.
           1. Is this object (``self``)
           2. Is the currently examined atom (``ia``)
           3. Is the currently bounded indices (``idxs``)
           4. Is the currently bounded indices atomic coordinates (``idxs_xyz``)
           An example `func` could be:

           >>> def func(self, ia, idxs, idxs_xyz=None):
           ...     idx = self.geometry.close(ia, R=[0.1, 1.44], idx=idxs, idx_xyz=idxs_xyz)
           ...     self[ia, idx[0]] = 0
           ...     self[ia, idx[1]] = -2.7

        na_iR : int, optional
           number of atoms within the sphere for speeding
           up the `iter_block` loop.
        method : {'rand', str}
           method used in `Geometry.iter_block`, see there for details
        eta : bool, optional
           whether an ETA will be printed

        See Also
        --------
        create_construct : a generic function used to create a generic function which this routine requires
        tile : tiling *after* construct is much faster for very large systems
        repeat : repeating *after* construct is much faster for very large systems
        """

        if not callable(func):
            if not isinstance(func, (tuple, list)):
                raise ValueError('Passed `func` which is not a function, nor tuple/list of `R, param`')

            if np.any(np.diff(self.geometry.lasto) > 1):
                raise ValueError("Automatically setting a sparse model "
                              "for systems with atoms having more than 1 "
                              "orbital *must* be done by your-self. You have to define a corresponding `func`.")

            # Convert to a proper function
            func = self.create_construct(func[0], func[1])

        iR = self.geometry.iR(na_iR)

        # Create eta-object
        eta = tqdm_eta(self.na, self.__class__.__name__ + '.construct', 'atom', eta)

        # Do the loop
        for ias, idxs in self.geometry.iter_block(iR=iR, method=method):

            # Get all the indexed atoms...
            # This speeds up the searching for coordinates...
            idxs_xyz = self.geometry[idxs, :]

            # Loop the atoms inside
            for ia in ias:
                func(self, ia, idxs, idxs_xyz)

            eta.update(len(ias))

        eta.close()

    @property
    def finalized(self):
        """ Whether the contained data is finalized and non-used elements have been removed """
        return self._csr.finalized

    def remove(self, atom):
        """ Create a subset of this sparse matrix by removing the atoms corresponding to `atom`

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom : array_like of int
            indices of removed atoms

        See Also
        --------
        Geometry.remove : equivalent to the resulting `Geometry` from this routine
        Geometry.sub : the negative of `Geometry.remove`
        sub : the opposite of `remove`, i.e. retain a subset of atoms
        """
        atom = self.sc2uc(atom)
        atom = delete(_a.arangei(self.na), atom)
        return self.sub(atom)

    def sub(self, atom):
        """ Create a subset of this sparse matrix by retaining the atoms corresponding to `atom`

        Indices passed must be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom : array_like of int
            indices of removed atoms

        See Also
        --------
        Geometry.remove : equivalent to the resulting `Geometry` from this routine
        Geometry.sub : the negative of `Geometry.remove`
        remove : the negative of `sub`, i.e. remove a subset of atoms
        """
        pass

    def swap(self, a, b):
        """ Swaps atoms in the sparse geometry to obtain a new order of atoms

        This can be used to reorder elements of a geometry.

        Parameters
        ----------
        a : array_like
             the first list of atomic coordinates
        b : array_like
             the second list of atomic coordinates
        """
        a = _a.asarrayi(a)
        b = _a.asarrayi(b)
        # Create full index list
        full = _a.arangei(len(self.geometry))
        # Regardless of whether swapping or new indices are requested
        # this should work.
        full[a] = b
        full[b] = a
        return self.sub(full)

    def finalize(self):
        """ Finalizes the model

        Finalizes the model so that all non-used elements are removed. I.e. this simply reduces the memory requirement for the sparse matrix.

        Note that adding more elements to the sparse matrix is more time-consuming than for a non-finalized sparse matrix due to the
        internal data-representation.
        """
        self._csr.finalize()

    def tocsr(self, dim=0, isc=None, **kwargs):
        """ Return a :class:`~scipy.sparse.csr_matrix` for the specified dimension

        Parameters
        ----------
        dim : int, optional
           the dimension in the sparse matrix (for non-orthogonal cases the last
           dimension is the overlap matrix)
        isc : int, optional
           the supercell index, or all (if ``isc=None``)
        """
        if isc is not None:
            raise NotImplementedError("Requesting sub-sparse has not been implemented yet")
        return self._csr.tocsr(dim, **kwargs)

    def spsame(self, other):
        """ Compare two sparse objects and check whether they have the same entries.

        This does not necessarily mean that the elements are the same
        """
        return self._csr.spsame(other._csr)

    @classmethod
    def fromsp(cls, geom, *sp):
        """ Create a sparse model from a preset Geometry and a list of sparse matrices """
        # Ensure it is a list (no tuples can be used)
        sp = list(sp)
        for i, s in enumerate(sp):
            if isinstance(s, (tuple, list)):
                # Downcast to a single list of sparse matrices
                if len(sp) > 1:
                    raise ValueError("Argument should be a single list or a sequence of arguments, not both.")
                sp = s
                break

        # Number of dimensions
        dim = len(sp)
        nnzpr = 1
        # Sort all indices for the passed sparse matrices
        for i in range(dim):
            sp[i] = sp[i].tocsr()
            sp[i].sort_indices()
            sp[i].sum_duplicates()

            # Figure out the maximum connections per
            # row to reduce number of re-allocations to 0
            nnzpr = max(nnzpr, sp[i].nnz // sp[i].shape[0])

        # Create the sparse object
        S = cls(geom, dim, sp[0].dtype, nnzpr)

        if S._size != sp[0].shape[0]:
            raise ValueError(cls.__name__ + '.fromsp cannot create a new class, the geometry ' + \
                             'and sparse matrices does not have coinciding dimensions size != sp.shape[0]')

        for i in range(dim):
            ptr = sp[i].indptr
            col = sp[i].indices
            D = sp[i].data

            # loop and add elements
            for r in range(S.shape[0]):
                sl = slice(ptr[r], ptr[r+1], None)
                S[r, col[sl], i] = D[sl]

        return S

    ###############################
    # Overload of math operations #
    ###############################
    def __add__(self, b):
        c = self.copy(dtype=get_dtype(b, other=self.dtype))
        c += b
        return c
    __radd__ = __add__

    def __iadd__(self, b):
        if isinstance(b, _SparseGeometry):
            self._csr += b._csr
        else:
            self._csr += b
        return self

    def __sub__(self, b):
        c = self.copy(dtype=get_dtype(b, other=self.dtype))
        c -= b
        return c

    def __rsub__(self, b):
        if isinstance(b, _SparseGeometry):
            c = b.copy(dtype=get_dtype(self, other=b.dtype))
            c._csr += -1 * self._csr
        else:
            c = b + (-1) * self
        return c

    def __isub__(self, b):
        if isinstance(b, _SparseGeometry):
            self._csr -= b._csr
        else:
            self._csr -= b
        return self

    def __mul__(self, b):
        c = self.copy(dtype=get_dtype(b, other=self.dtype))
        c *= b
        return c
    __rmul__ = __mul__

    def __imul__(self, b):
        if isinstance(b, _SparseGeometry):
            self._csr *= b._csr
        else:
            self._csr *= b
        return self

    def __div__(self, b):
        c = self.copy(dtype=get_dtype(b, other=self.dtype))
        c /= b
        return c

    def __rdiv__(self, b):
        c = b.copy(dtype=get_dtype(self, other=b.dtype))
        c /= self
        return c

    def __idiv__(self, b):
        if isinstance(b, _SparseGeometry):
            self._csr /= b._csr
        else:
            self._csr /= b
        return self

    def __floordiv__(self, b):
        if isinstance(b, _SparseGeometry):
            raise NotImplementedError
        c = self.copy(dtype=get_dtype(b, other=self.dtype))
        c //= b
        return c

    def __ifloordiv__(self, b):
        if isinstance(b, _SparseGeometry):
            raise NotImplementedError
        self._csr //= b
        return self

    def __truediv__(self, b):
        if isinstance(b, _SparseGeometry):
            raise NotImplementedError
        c = self.copy(dtype=get_dtype(b, other=self.dtype))
        c /= b
        return c

    def __itruediv__(self, b):
        if isinstance(b, _SparseGeometry):
            raise NotImplementedError
        self._csr /= b
        return self

    def __pow__(self, b):
        c = self.copy(dtype=get_dtype(b, other=self.dtype))
        c **= b
        return c

    def __rpow__(self, b):
        c = self.copy(dtype=get_dtype(b, other=self.dtype))
        c._csr = b ** c._csr
        return c

    def __ipow__(self, b):
        if isinstance(b, _SparseGeometry):
            self._csr **= b._csr
        else:
            self._csr **= b
        return self

    def __getstate__(self):
        """ Return dictionary with the current state """
        return {
            'geometry': self.geometry.__getstate__(),
            'csr': self._csr.__getstate__()
        }

    def __setstate__(self, state):
        """ Return dictionary with the current state """
        geom = Geometry([0] * 3, Atom(1))
        geom.__setstate__(state['geometry'])
        self._geometry = geom
        csr = SparseCSR((2, 2, 2))
        csr.__setstate__(state['csr'])
        self._csr = csr
        self._def_dim = -1


class SparseAtom(_SparseGeometry):
    """ Sparse object with number of rows equal to the total number of atoms in the `Geometry` """

    def __getitem__(self, key):
        """ Elements for the index(s) """
        dd = self._def_dim
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geometry.sc_index(key[-1]) * self.na
                key = [el for el in key[:-1]]
                key[1] = self.geometry.sc2uc(key[1]) + off
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        d = self._csr[key]
        return d

    def __setitem__(self, key, val):
        """ Set or create elements in the sparse data

        Override set item for slicing operations and enables easy
        setting of parameters in a sparse matrix
        """
        dd = self._def_dim
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geometry.sc_index(key[-1]) * self.na
                key = [el for el in key[:-1]]
                key[1] = self.geometry.sc2uc(key[1]) + off
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        self._csr[key] = val

    @property
    def _size(self):
        return self.geometry.na

    def nonzero(self, atom=None, only_col=False):
        """ Indices row and column indices where non-zero elements exists

        Parameters
        ----------
        atom : int or array_like of int, optional
           only return the tuples for the requested atoms, default is all atoms
        only_col : bool, optional
           only return then non-zero columns

        See Also
        --------
        SparseCSR.nonzero : the equivalent function call
        """
        return self._csr.nonzero(row=atom, only_col=only_col)

    def iter_nnz(self, atom=None):
        """ Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz():
        ...    self[i, j] # is then the non-zero value

        Parameters
        ----------
        atom : int or array_like
            only loop on the non-zero elements coinciding with the atoms
        """
        if not atom is None:
            atom = _a.asarrayi(atom).ravel()
            for i, j in self._csr.iter_nnz(atom):
                yield i, j
        else:
            for i, j in self._csr.iter_nnz():
                yield i, j

    def set_nsc(self, *args, **kwargs):
        """ Reset the number of allowed supercells in the sparse atom

        If one reduces the number of supercells *any* sparse element
        that references the supercell will be deleted.

        See `SuperCell.set_nsc` for allowed parameters.

        See Also
        --------
        SuperCell.set_nsc : the underlying called method
        """
        super(SparseAtom, self).set_nsc(self.na, *args, **kwargs)

    def cut(self, seps, axis, *args, **kwargs):
        """ Cuts the sparse atom model into different parts.

        Recreates a new sparse atom object with only the cutted
        atoms in the structure.

        Cutting is the opposite of tiling.

        Parameters
        ----------
        seps : int
           number of times the structure will be cut
        axis : int
           the axis that will be cut
        """
        new_w = None
        # Create new geometry
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Create new cut geometry
            geom = self.geometry.cut(seps, axis, *args, **kwargs)
            # Check whether the warning exists
            if len(w) > 0:
                if issubclass(w[-1].category, SislWarning):
                    new_w = str(w[-1].message)
                    new_w += ("\n---\n"
                              "The sparse atom cannot be cut as the structure "
                              "cannot be tiled accordingly. ANY use of the model has been "
                              "relieved from sisl.")
                    warn(new_w)

        # Now we need to re-create number of supercells
        na = self.na
        S = self.tocsr(0)

        # First we need to figure out how long the interaction range is
        # in the cut-direction
        # We initialize to be the same as the parent direction
        nsc = _a.arrayi(self.nsc // 2)
        nsc[axis] = 0  # we count the new direction
        isc = _a.zerosi([3])
        isc[axis] -= 1
        out = False
        while not out:
            # Get supercell index
            isc[axis] += 1
            try:
                idx = self.sc_index(isc)
            except:
                break

            sub = S[0:geom.na, idx * na:(idx + 1) * na].indices[:]

            if len(sub) == 0:
                break

            c_max = np.amax(sub)
            # Count the number of cells it interacts with
            i = (c_max % na) // geom.na
            ic = idx * na
            for j in range(i):
                idx = ic + geom.na * j
                # We need to ensure that every "in between" index exists
                # if it does not we discard those indices
                if len(np.logical_and(idx <= sub,
                                      sub < idx + geom.na).nonzero()[0]) == 0:
                    i = j - 1
                    out = True
                    break
            nsc[axis] = isc[axis] * seps + i

            if out:
                warn('Cut the connection at nsc={0} in direction {1}.'.format(nsc[axis], axis))

        # Update number of super-cells
        nsc[:] = nsc[:] * 2 + 1
        geom.sc.set_nsc(nsc)

        # Now we have a correct geometry, and
        # we are now ready to create the sparsity pattern
        # Reduce the sparsity pattern, first create the new one
        S = self.__class__(geom, self.dim, self.dtype, np.amax(self._csr.ncol), **self._cls_kwargs())

        def _sca2sca(M, a, m, seps, axis):
            # Converts an o from M to m
            isc = _a.arrayi(M.a2isc(a))
            isc[axis] = isc[axis] * seps
            # Correct for cell-offset
            isc[axis] = isc[axis] + (a % M.na) // m.na
            # find the equivalent cell in m
            try:
                # If a fail happens it is due to a discarded
                # interaction across a non-interacting region
                return (a % m.na,
                        m.sc_index(isc) * m.na,
                        m.sc_index(-isc) * m.na)
            except:
                return None, None, None

        # only loop on the atoms remaining in the cutted structure
        for ja, ia in self.iter_nnz(range(geom.na)):

            # Get the equivalent orbital in the smaller cell
            a, afp, afm = _sca2sca(self.geometry, ia, S.geom, seps, axis)
            if a is None:
                continue
            S[ja, a + afp] = self[ja, ia]
            # TODO check that we indeed have Hermiticity for non-collinear and spin-orbit
            S[a, ja + afm] = self[ja, ia]

        return S

    def sub(self, atom):
        """ Create a subset of this sparse matrix by only retaining the elements corresponding to the ``atom``

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom : array_like of int
            indices of retained atoms

        See Also
        --------
        Geometry.remove : the negative of `Geometry.sub`
        Geometry.sub : equivalent to the resulting `Geometry` from this routine
        remove : the negative of `sub`, i.e. remove a subset of atoms
        """
        atom = self.sc2uc(atom)
        geom = self.geometry.sub(atom)

        idx = tile(atom, self.n_s)
        # Use broadcasting rules
        idx.shape = (self.n_s, -1)
        idx += (_a.arangei(self.n_s) * self.na).reshape(-1, 1)
        idx.shape = (-1,)

        # Now create the new sparse orbital class
        S = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())
        S._csr = self._csr.sub(idx)

        return S

    def tile(self, reps, axis):
        """ Create a tiled sparse atom object, equivalent to `Geometry.tile`

        The already existing sparse elements are extrapolated
        to the new supercell by repeating them in blocks like the coordinates.

        Notes
        -----
        Calling this routine will automatically `finalize` the `SparseAtom`. This
        is required to greatly increase performance.

        Parameters
        ----------
        reps : int
            number of repetitions along cell-vector `axis`
        axis : int
            0, 1, 2 according to the cell-direction

        See Also
        --------
        Geometry.tile: the same ordering as the final geometry
        Geometry.repeat: a different ordering of the final geometry
        repeat: a different ordering of the final geometry
        """
        # Create the new sparse object
        g = self.geometry.tile(reps, axis)
        S = self.__class__(g, self.dim, self.dtype, 1, **self._cls_kwargs())

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geometry
        na = int32(self.na)
        csr = self._csr
        ncol = csr.ncol
        if self.finalized or csr.nnz == csr.ptr[-1]:
            col = csr.col
            D = csr._D
        else:
            ptr = csr.ptr
            idx = array_arange(ptr[:-1], n=ncol)
            col = csr.col[idx]
            D = csr._D[idx, :]
            del ptr, idx

        # Information for the new Hamiltonian sparse matrix
        na_n = int32(S.na)
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = tile(ncol, reps)
        # Now indptr is complete
        indptr = insert(_a.cumsumi(ncol), 0, 0)
        del ncol
        indices = _a.emptyi([indptr[-1]])
        indices.shape = (reps, -1)

        # Now we should fill the data
        isc = geom.a2isc(col)
        # resulting atom in the new geometry (without wrapping
        # for correct supercell, that will happen below)
        JA = col % na + na * isc[:, axis]

        # Create repetitions
        for rep in range(reps):
            # Correct the supercell information
            isc[:, axis] = JA // na_n

            indices[rep, :] = JA % na_n + sc_index(isc) * na_n

            # Step atoms
            JA += na

        # Clean-up
        del isc, JA

        S._csr = SparseCSR((tile(D, (reps, 1)), indices.ravel(), indptr),
                           shape=(geom_n.na, geom_n.na_s))

        return S

    def repeat(self, reps, axis):
        """ Create a repeated sparse atom object, equivalent to `Geometry.repeat`

        The already existing sparse elements are extrapolated
        to the new supercell by repeating them in blocks like the coordinates.

        Parameters
        ----------
        reps : int
            number of repetitions along cell-vector `axis`
        axis : int
            0, 1, 2 according to the cell-direction

        See Also
        --------
        Geometry.repeat: the same ordering as the final geometry
        Geometry.tile: a different ordering of the final geometry
        tile: a different ordering of the final geometry
        """
        # Create the new sparse object
        g = self.geometry.repeat(reps, axis)
        S = self.__class__(g, self.dim, self.dtype, 1, **self._cls_kwargs())

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geometry
        na = int32(self.na)
        csr = self._csr
        ncol = csr.ncol
        if self.finalized or csr.nnz == csr.ptr[-1]:
            col = csr.col
            D = csr._D
        else:
            ptr = csr.ptr
            idx = array_arange(ptr[:-1], n=ncol)
            col = csr.col[idx]
            D = csr._D[idx, :]
            del ptr, idx

        # Information for the new Hamiltonian sparse matrix
        na_n = int32(S.na)
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = repeat(ncol, reps)
        # Now indptr is complete
        indptr = insert(_a.cumsumi(ncol), 0, 0)
        del ncol
        indices = _a.emptyi([indptr[-1]])

        # Now we should fill the data
        isc = geom.a2isc(col)
        # resulting atom in the new geometry (without wrapping
        # for correct supercell, that will happen below)
        JA = (col % na) * reps
        # Get the offset atoms
        A = isc[:, axis] - 1

        for rep in range(reps):

            # Update the offset
            A += 1
            # Correct supercell information
            isc[:, axis] = A // reps

            # Create the indices for the repetition
            idx = array_arange(indptr[rep:-1:reps], n=csr.ncol)
            indices[idx] = JA + A % reps + sc_index(isc) * na_n

        # Clean-up
        del isc, JA, A, idx

        # In the repeat we have to tile individual atomic couplings
        # So we should split the arrays and tile them individually
        # Now D is made up of D values, per atom
        if geom.na == 1:
            D = tile(D, (reps, 1))
        else:
            ntile = ftool.partial(tile, reps=(reps, 1))
            D = np.vstack(tuple(map(ntile, np.split(D, _a.cumsumi(csr.ncol[:-1]), axis=0))))

        S._csr = SparseCSR((D, indices, indptr),
                           shape=(geom_n.na, geom_n.na_s))

        return S

    def rij(self, dtype=np.float64):
        r""" Create a sparse matrix with the distance between atoms

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type of the sparse matrix.

        Notes
        -----
        The returned sparse matrix with distances are taken from the current sparse pattern.
        I.e. a subsequent addition of sparse elements will make them inequivalent.
        It is thus important to *only* create the sparse distance when the sparse
        structure is completed.
        """
        R = self.Rij(dtype)
        R._csr = (R._csr ** 2).sum(-1) ** 0.5
        return R

    def Rij(self, dtype=np.float64):
        r""" Create a sparse matrix with the vectors between atoms

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type of the sparse matrix.

        Notes
        -----
        The returned sparse matrix with vectors are taken from the current sparse pattern.
        I.e. a subsequent addition of sparse elements will make them inequivalent.
        It is thus important to *only* create the sparse vector matrix when the sparse
        structure is completed.
        """
        geom = self.geometry
        Rij = geom.Rij

        # Pointers
        ncol = self._csr.ncol
        ptr = self._csr.ptr
        col = self._csr.col

        # Create the output class
        R = SparseAtom(geom, 3, dtype, nnzpr=1)

        # Re-create the sparse matrix data
        R._csr.ptr = ptr.copy()
        R._csr.ncol = ncol.copy()
        R._csr.col = col.copy()
        R._csr._nnz = self._csr.nnz
        R._csr._D = np.zeros([self._csr._D.shape[0], 3], dtype=dtype)
        R._csr._finalized = self.finalized
        for ia in range(self.shape[0]):
            sl = slice(ptr[ia], ptr[ia] + ncol[ia])
            R._csr._D[sl, :] = Rij(ia, col[sl])

        return R


class SparseOrbital(_SparseGeometry):
    """ Sparse object with number of rows equal to the total number of orbitals in the `Geometry` """

    def __getitem__(self, key):
        """ Elements for the index(s) """
        dd = self._def_dim
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geometry.sc_index(key[-1]) * self.no
                key = [el for el in key[:-1]]
                key[1] = self.geometry.osc2uc(key[1]) + off
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        d = self._csr[key]
        return d

    def __setitem__(self, key, val):
        """ Set or create elements in the sparse data

        Override set item for slicing operations and enables easy
        setting of parameters in a sparse matrix
        """
        dd = self._def_dim
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geometry.sc_index(key[-1]) * self.no
                key = [el for el in key[:-1]]
                key[1] = self.geometry.osc2uc(key[1]) + off
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        self._csr[key] = val

    @property
    def _size(self):
        return self.geometry.no

    def edges(self, atom=None, exclude=None, orbital=None):
        """ Retrieve edges (connections) of a given `atom` or list of `atom`'s

        The returned edges are unique and sorted (see `numpy.unique`) and are returned
        in supercell indices (i.e. ``0 <= edge < self.geometry.no_s``).

        Parameters
        ----------
        atom : int or list of int
            the edges are returned only for the given atom (but by using  all orbitals of the
            requested atom). The returned edges are also atoms.
        exclude : int or list of int, optional
           remove edges which are in the `exclude` list.
           Default to `atom`.
        orbital : int or list of int
            the edges are returned only for the given orbital. The returned edges are orbitals.

        See Also
        --------
        SparseCSR.edges: the underlying routine used for extracting the edges
        """
        if atom is None and orbital is None:
            raise ValueError(self.__class__.__name__ + '.edges must have either "atom" or "orbital" keyword defined.')
        if orbital is None:
            return unique(self.geometry.o2a(self._csr.edges(self.geometry.a2o(atom, True), exclude)))
        return self._csr.edges(orbital, exclude)

    def nonzero(self, atom=None, only_col=False):
        """ Indices row and column indices where non-zero elements exists

        Parameters
        ----------
        atom : int or array_like of int, optional
           only return the tuples for the requested atoms, default is all atoms
           But for *all* orbitals.
        only_col : bool, optional
           only return then non-zero columns

        See Also
        --------
        SparseCSR.nonzero : the equivalent function call
        """
        if atom is None:
            return self._csr.nonzero(only_col=only_col)
        row = self.geometry.a2o(atom, all=True)
        return self._csr.nonzero(row=row, only_col=only_col)

    def iter_nnz(self, atom=None, orbital=None):
        """ Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz():
        ...    self[i, j] # is then the non-zero value

        Parameters
        ----------
        atom : int or array_like
            only loop on the non-zero elements coinciding with the orbitals
            on these atoms (not compatible with the ``orbital`` keyword)
        orbital : int or array_like
            only loop on the non-zero elements coinciding with the orbital
            (not compatible with the ``atom`` keyword)
        """
        if not atom is None:
            orbital = self.geometry.a2o(atom)
        elif not orbital is None:
            orbital = _a.asarrayi(orbital)
        if not orbital is None:
            for i, j in self._csr.iter_nnz(orbital):
                yield i, j
        else:
            for i, j in self._csr.iter_nnz():
                yield i, j

    def set_nsc(self, *args, **kwargs):
        """ Reset the number of allowed supercells in the sparse orbital

        If one reduces the number of supercells *any* sparse element
        that references the supercell will be deleted.

        See `SuperCell.set_nsc` for allowed parameters.

        See Also
        --------
        SuperCell.set_nsc : the underlying called method
        """
        super(SparseOrbital, self).set_nsc(self.no, *args, **kwargs)

    def cut(self, seps, axis, *args, **kwargs):
        """ Cuts the sparse orbital model into different parts.

        Recreates a new sparse orbital object with only the cutted
        atoms in the structure.

        Cutting is the opposite of tiling.

        Parameters
        ----------
        seps : int
           number of times the structure will be cut
        axis : int
           the axis that will be cut
        """
        new_w = None
        # Create new geometry
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Create new cut geometry
            geom = self.geometry.cut(seps, axis, *args, **kwargs)
            # Check whether the warning exists
            if len(w) > 0:
                if issubclass(w[-1].category, SislWarning):
                    new_w = str(w[-1].message)
                    new_w += ("\n---\n"
                              "The sparse orbital cannot be cut as the structure "
                              "cannot be tiled accordingly. ANY use of the model has been "
                              "relieved from sisl.")
                    warn(new_w)

        # Now we need to re-create number of supercells
        no = self.no
        S = self.tocsr(0)

        # First we need to figure out how long the interaction range is
        # in the cut-direction
        # We initialize to be the same as the parent direction
        nsc = self.nsc // 2
        nsc[axis] = 0  # we count the new direction
        isc = _a.zerosi([3])
        isc[axis] -= 1
        out = False
        while not out:
            # Get supercell index
            isc[axis] += 1
            try:
                idx = self.sc_index(isc)
            except:
                break

            sub = S[0:geom.no, idx * no:(idx + 1) * no].indices[:]

            if len(sub) == 0:
                break

            c_max = np.amax(sub)
            # Count the number of cells it interacts with
            i = (c_max % no) // geom.no
            ic = idx * no
            for j in range(i):
                idx = ic + geom.no * j
                # We need to ensure that every "in between" index exists
                # if it does not we discard those indices
                if len(np.logical_and(idx <= sub,
                                      sub < idx + geom.no).nonzero()[0]) == 0:
                    i = j - 1
                    out = True
                    break
            nsc[axis] = isc[axis] * seps + i

            if out:
                warn('Cut the connection at nsc={0} in direction {1}.'.format(nsc[axis], axis))

        # Update number of super-cells
        nsc[:] = nsc[:] * 2 + 1
        geom.sc.set_nsc(nsc)

        # Now we have a correct geometry, and
        # we are now ready to create the sparsity pattern
        # Reduce the sparsity pattern, first create the new one
        S = self.__class__(geom, self.dim, self.dtype, np.amax(self._csr.ncol), **self._cls_kwargs())

        def _sco2sco(M, o, m, seps, axis):
            # Converts an o from M to m
            isc = _a.arrayi(M.o2isc(o), copy=True)
            isc[axis] = isc[axis] * seps
            # Correct for cell-offset
            isc[axis] = isc[axis] + (o % M.no) // m.no
            # find the equivalent cell in m
            try:
                # If a fail happens it is due to a discarded
                # interaction across a non-interacting region
                return (o % m.no,
                        m.sc_index(isc) * m.no,
                        m.sc_index(-isc) * m.no)
            except:
                return None, None, None

        # only loop on the orbitals remaining in the cutted structure
        for jo, io in self.iter_nnz(orbital=range(geom.no)):

            # Get the equivalent orbital in the smaller cell
            o, ofp, ofm = _sco2sco(self.geometry, io, S.geom, seps, axis)
            if o is None:
                continue
            d = self[jo, io]
            S[jo, o + ofp] = d
            S[o, jo + ofm] = d

        return S

    def remove(self, atom, orb_index=None):
        """ Remove a subset of this sparse matrix by only retaining the atoms corresponding to `atom`

        Parameters
        ----------
        atom : array_like of int or Atom
            indices of removed atoms or Atom for direct removal of all atoms

        See Also
        --------
        Geometry.remove : equivalent to the resulting `Geometry` from this routine
        Geometry.sub : the negative of `Geometry.remove`
        sub : the opposite of `remove`, i.e. retain a subset of atoms
        """
        if isinstance(atom, Atom):
            atom = self.geometry.atoms.index(atom)
            atom = (self.geometry.atoms.specie == atom).nonzero()[0]
        # This will digress to call .sub
        return super(SparseOrbital, self).remove(atom)

    def remove_orbital(self, atom, orbital):
        """ Remove a subset of orbitals on `atom` according to `orbital`

        Parameters
        ----------
        atom : array_like of int or Atom
            indices of atoms or `Atom` that will be reduced in size according to `orbital`
        orbital : array_like of int or Orbital
            indices of the orbitals on `atom` that are removed from the sparse matrix.

        Examples
        --------

        >>> obj = SparseOrbital(...)
        >>> # remove the second orbital on the 2nd atom
        >>> # all other orbitals are retained
        >>> obj.remove_orbital(1, 1)
        """
        # Get specie index of the atom
        if isinstance(atom, Atom):
            # All atoms with this specie
            atom = self.geometry.atoms.index(atom)
            atom = (self.geometry.atoms.specie == atom).nonzero()[0]
        atom = np.asarray(atom).ravel()

        # Figure out if all atoms have the same species
        specie = self.geometry.atoms.specie[atom]
        uniq_specie, indices = unique(specie, return_inverse=True)
        if len(uniq_specie) > 1:
            # In case there are multiple different species but one wishes to
            # retain the same orbital index, then we loop on the unique species
            new = self
            for i in range(uniq_specie.size):
                idx = (indices == i).nonzero()[0]
                new = new.remove_orbital(atom[idx], orbital)
            return new

        # Get the atom object we wish to reduce
        # We know np.all(geom.atoms[atom] == old_atom)
        old_atom = self.geometry.atoms[atom[0]]

        # Retrieve index of orbital
        if isinstance(orbital, Orbital):
            orbital = old_atom.index(orbital)
        # Create the reverse index-table to delete those not required
        orbital = delete(_a.arangei(len(old_atom)), np.asarray(orbital).ravel())
        return self.sub_orbital(atom, orbital)

    def sub(self, atom):
        """ Create a subset of this sparse matrix by only retaining the atoms corresponding to `atom`

        Negative indices are wrapped and thus works, supercell atoms are also wrapped to the unit-cell.

        Parameters
        ----------
        atom : array_like of int or Atom
            indices of retained atoms or `Atom` for retaining only *that* atom

        Examples
        --------

        >>> obj = SparseOrbital(...)
        >>> obj.sub(1) # only retain the second atom in the SparseGeometry
        >>> obj.sub(obj.atoms.atom[0]) # retain all atoms which is equivalent to
        >>>                            # the first atomic specie

        See Also
        --------
        Geometry.remove : the negative of `Geometry.sub`
        Geometry.sub : equivalent to the resulting `Geometry` from this routine
        remove : the negative of `sub`, i.e. remove a subset of atoms
        """
        if isinstance(atom, Atom):
            idx = self.geometry.atoms.index(atom)
            atom = (self.geometry.atoms.specie == idx).nonzero()[0]

        atom = self.sc2uc(atom)
        orbs = self.a2o(atom, all=True)
        geom = self.geometry.sub(atom)

        idx = tile(orbs, self.n_s)
        # Use broadcasting rules
        idx.shape = (self.n_s, -1)
        idx += (_a.arangei(self.n_s) * self.no).reshape(-1, 1)
        idx.shape = (-1,)

        # Now create the new sparse orbital class
        S = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())
        S._csr = self._csr.sub(idx)

        return S

    def sub_orbital(self, atom, orbital):
        """ Retain only a subset of the orbitals on `atom` according to `orbital`

        This allows one to retain only a given subset of the sparse matrix elements. 

        Parameters
        ----------
        atom : array_like of int or Atom
            indices of atoms or `Atom` that will be reduced in size according to `orbital`
        orbital : array_like of int or Orbital
            indices of the orbitals on `atom` that are retained in the sparse matrix, the list of
            orbitals will be sorted. One cannot re-arrange matrix elements currently.

        Notes
        -----
        Future implementations may allow one to re-arange orbitals using this method.

        Examples
        --------

        >>> obj = SparseOrbital(...)
        >>> # only retain the second orbital on the 2nd atom
        >>> # all other orbitals are retained
        >>> obj.sub_orbital(1, 1)
        """
        # Get specie index of the atom
        if isinstance(atom, (tuple, list)):
            if isinstance(atom[0], Atom):
                spg = self
                for a in atom:
                    spg = spg.sub_orbital(a, orbital)
                return spg
        if isinstance(atom, Atom):
            # All atoms with this specie
            atom = self.geometry.atoms.index(atom)
            atom = (self.geometry.atoms.specie == atom).nonzero()[0]
        atom = np.asarray(atom).ravel()

        # Figure out if all atoms have the same species
        specie = self.geometry.atoms.specie[atom]
        uniq_specie, indices = unique(specie, return_inverse=True)
        if len(uniq_specie) > 1:
            # In case there are multiple different species but one wishes to
            # retain the same orbital index, then we loop on the unique species
            new = self
            for i in range(uniq_specie.size):
                idx = (indices == i).nonzero()[0]
                new = new.sub_orbital(atom[idx], orbital)
            return new

        # At this point we are sure that uniq_specie is *only* one specie!
        geom = self.geometry.copy()

        # Get the atom object we wish to reduce
        old_atom = geom.atoms[atom[0]]

        # Retrieve index of orbital
        if isinstance(orbital, Orbital):
            orbital = old_atom.index(orbital)
        orbital = np.sort(np.asarray(orbital).ravel())
        if len(orbital) == 0:
            raise ValueError('trying to retain 0 orbitals on a given atom. This is not allowed!')

        new_atom = old_atom.sub(orbital)
        # Rename the new-atom to <>_1_2 for orbital == [1, 2]
        new_atom.tag += '_' + '_'.join(map(str, orbital))

        # We catch the warning about reducing the number of orbitals!
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            geom.atoms.replace_atom(old_atom, new_atom)

        # Now create the new sparse orbital class
        SG = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())

        rem_orbs = delete(_a.arangei(old_atom.no), orbital)
        # Find orbitals to remove (note this HAS to be from the original array)
        rem_orbs = np.add.outer(self.geometry.a2o(atom), rem_orbs).ravel()

        # Generate a list of orbitals to retain
        sub_idx = delete(_a.arangei(self.no), rem_orbs)

        # Generate full supercell indices
        n_s = self.geometry.n_s
        sc_off = _a.arangei(n_s) * self.no
        sub_idx = tile(sub_idx, n_s).reshape(n_s, -1) + sc_off.reshape(-1, 1)
        SG._csr = self._csr.sub(sub_idx)

        return SG

    def tile(self, reps, axis):
        """ Create a tiled sparse orbital object, equivalent to `Geometry.tile`

        The already existing sparse elements are extrapolated
        to the new supercell by repeating them in blocks like the coordinates.

        Parameters
        ----------
        reps : int
            number of repetitions along cell-vector `axis`
        axis : int
            0, 1, 2 according to the cell-direction

        See Also
        --------
        Geometry.tile: the same ordering as the final geometry
        Geometry.repeat: a different ordering of the final geometry
        repeat: a different ordering of the final geometry
        """
        # Create the new sparse object
        g = self.geometry.tile(reps, axis)
        S = self.__class__(g, self.dim, self.dtype, 1, **self._cls_kwargs())

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geometry
        no = int32(self.no)
        csr = self._csr
        ncol = csr.ncol
        if self.finalized or csr.nnz == csr.ptr[-1]:
            col = csr.col
            D = csr._D
        else:
            ptr = csr.ptr
            idx = array_arange(ptr[:-1], n=ncol)
            col = csr.col[idx]
            D = csr._D[idx, :]
            del ptr, idx

        # Information for the new Hamiltonian sparse matrix
        no_n = int32(S.no)
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = tile(ncol, reps)
        # Now indptr is complete
        indptr = insert(_a.cumsumi(ncol), 0, 0)
        del ncol
        indices = _a.emptyi([indptr[-1]])
        indices.shape = (reps, -1)

        # Now we should fill the data
        isc = geom.o2isc(col)
        # resulting atom in the new geometry (without wrapping
        # for correct supercell, that will happen below)
        JO = col % no + no * isc[:, axis]

        # Create repetitions
        for rep in range(reps):
            # Correct the supercell information
            isc[:, axis] = JO // no_n

            indices[rep, :] = JO % no_n + sc_index(isc) * no_n

            # Step orbitals
            JO += no

        # Clean-up
        del isc, JO

        S._csr = SparseCSR((tile(D, (reps, 1)), indices.ravel(), indptr),
                           shape=(geom_n.no, geom_n.no_s))

        return S

    def repeat(self, reps, axis):
        """ Create a repeated sparse orbital object, equivalent to `Geometry.repeat`

        The already existing sparse elements are extrapolated
        to the new supercell by repeating them in blocks like the coordinates.

        Parameters
        ----------
        reps : int
            number of repetitions along cell-vector `axis`
        axis : int
            0, 1, 2 according to the cell-direction

        See Also
        --------
        Geometry.repeat: the same ordering as the final geometry
        Geometry.tile: a different ordering of the final geometry
        tile: a different ordering of the final geometry
        """
        # Create the new sparse object
        g = self.geometry.repeat(reps, axis)
        S = self.__class__(g, self.dim, self.dtype, 1, **self._cls_kwargs())

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geometry
        no = int32(self.no)
        csr = self._csr
        ncol = csr.ncol
        if self.finalized or csr.nnz == csr.ptr[-1]:
            col = csr.col
            D = csr._D
        else:
            ptr = csr.ptr
            idx = array_arange(ptr[:-1], n=ncol)
            col = csr.col[idx]
            D = csr._D[idx, :]
            del ptr, idx

        # Information for the new Hamiltonian sparse matrix
        no_n = int32(S.no)
        geom_n = S.geom

        # First loop on axis tiling and local
        # orbitals in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = repeat(ncol, reps)
        # Now indptr is complete
        indptr = insert(_a.cumsumi(ncol), 0, 0)
        del ncol
        indices = _a.emptyi([indptr[-1]])

        # Now we should fill the data
        isc = geom.o2isc(col)
        # resulting orbital in the new geometry (without wrapping
        # for correct supercell, that will happen below)
        JO = col % no
        # Get number of orbitals per atom (lasto - firsto + 1)
        # This is faster than the direct call

        ja = geom.o2a(JO)
        oJ = geom.firsto[ja]
        oA = geom.lasto[ja] + 1 - oJ
        # Shift the orbitals corresponding to the
        # repetitions of all previous atoms
        JO += oJ * (reps - 1)
        # Get the offset orbitals
        O = isc[:, axis] - 1
        # We need to create and indexable atomic array
        # This is required for multi-orbital cases where
        # we should tile atomic orbitals, and repeat the atoms (only).
        # 'A' is now the first (non-repeated) atom in the new structure
        A = _a.arangei(geom.na) * reps
        AO = geom_n.lasto[A] - geom_n.firsto[A] + 1
        # subtract AO for first iteration in repetition loop
        OA = geom_n.firsto[A] - AO

        # Clean
        del ja, oJ, A

        # Get view of ncol
        ncol = self._csr.ncol

        # Create repetitions
        for _ in range(reps):

            # Update atomic offset
            OA += AO
            # Update the offset
            O += 1
            # Correct supercell information
            isc[:, axis] = O // reps

            # Create the indices for the repetition
            idx = array_arange(indptr[array_arange(OA, n=AO)], n=ncol)
            indices[idx] = JO + oA * (O % reps) + sc_index(isc) * no_n

        # Clean-up
        del isc, JO, O, OA, AO, idx

        # In the repeat we have to tile individual atomic couplings
        # So we should split the arrays and tile them individually
        # Now D is made up of D values, per atom
        if geom.na == 1:
            D = tile(D, (reps, 1))
        else:
            ntile = ftool.partial(tile, reps=(reps, 1))
            D = np.vstack(tuple(map(ntile, np.split(D, _a.cumsumi(ncol)[geom.lasto[:geom.na-1]], axis=0))))
        S._csr = SparseCSR((D, indices, indptr),
                           shape=(geom_n.no, geom_n.no_s))

        return S

    def rij(self, what='orbital', dtype=np.float64):
        r""" Create a sparse matrix with the distance between atoms/orbitals

        Parameters
        ----------
        what : {'orbital', 'atom'}
            which kind of sparse distance matrix to return, either an atomic distance matrix
            or an orbital distance matrix. The orbital matrix is equivalent to the atomic
            one with the same distance repeated for the same atomic orbitals.
            The default is the same type as the parent class.
        dtype : numpy.dtype, optional
            the data-type of the sparse matrix.

        Notes
        -----
        The returned sparse matrix with distances are taken from the current sparse pattern.
        I.e. a subsequent addition of sparse elements will make them inequivalent.
        It is thus important to *only* create the sparse distance when the sparse
        structure is completed.
        """
        R = self.Rij(what, dtype)
        R._csr = (R._csr ** 2).sum(-1) ** 0.5
        return R

    def Rij(self, what='orbital', dtype=np.float64):
        r""" Create a sparse matrix with the vectors between atoms/orbitals

        Parameters
        ----------
        what : {'orbital', 'atom'}
            which kind of sparse vector matrix to return, either an atomic vector matrix
            or an orbital vector matrix. The orbital matrix is equivalent to the atomic
            one with the same vectors repeated for the same atomic orbitals.
            The default is the same type as the parent class.
        dtype : numpy.dtype, optional
            the data-type of the sparse matrix.

        Notes
        -----
        The returned sparse matrix with vectors are taken from the current sparse pattern.
        I.e. a subsequent addition of sparse elements will make them inequivalent.
        It is thus important to *only* create the sparse vector matrix when the sparse
        structure is completed.
        """
        geom = self.geometry

        # Pointers
        ncol = self._csr.ncol
        ptr = self._csr.ptr
        col = self._csr.col

        if what == 'atom':
            R = SparseAtom(geom, 3, dtype, nnzpr=np.amax(ncol))
            Rij = geom.Rij
            o2a = geom.o2a

            # Orbitals
            orow = _a.arangei(self.shape[0])
            # Loop on orbitals and atoms
            for io, ia in zip(orow, o2a(orow)):
                coln = unique(o2a(col[ptr[io]:ptr[io]+ncol[io]]))
                R[ia, coln] = Rij(ia, coln)

        elif what in ['orbital', 'orb']:
            # We create an *exact* copy of the Rij
            R = SparseOrbital(geom, 3, dtype, nnzpr=1)
            Rij = geom.oRij

            # Re-create the sparse matrix data
            R._csr.ptr = ptr.copy()
            R._csr.ncol = ncol.copy()
            R._csr.col = col.copy()
            R._csr._nnz = self._csr.nnz
            R._csr._D = np.zeros([self._csr._D.shape[0], 3], dtype=dtype)
            R._csr._finalized = self.finalized

            for io in range(self.shape[0]):
                sl = slice(ptr[io], ptr[io] + ncol[io])
                R._csr._D[sl, :] = Rij(io, col[sl])

        else:
            raise ValueError(self.__class__.__name__ + '.Rij "what" is not one of [atom, orbital].')

        return R

    def prepend(self, other, axis, eps=0.01):
        """ See `append` for details

        This is currently equivalent to:

        >>> other.append(self, axis, eps)
        """
        return other.append(self, axis, eps)

    def append(self, other, axis, eps=0.01):
        """ Append `other` along `axis` to construct a new connected sparse matrix

        This method tries to append two sparse geometry objects together by
        the following these steps:

        1. Create the new extended geometry
        2. Use neighbor cell couplings from `self` as the couplings to `other`
           This *may* cause problems if the coupling atoms are not exactly equi-positioned.
           If the coupling coordinates and the coordinates in `other` differ by more than
           0.001 Ang, a warning will be issued.
           If this difference is above `eps` the couplings will be removed.

        When appending sparse matrices made up of atoms, this method assumes that
        the orbitals on the overlapping atoms have the same orbitals, as well as the
        same orbital ordering.

        Examples
        --------
        >>> sporb = SparseOrbital(....)
        >>> forced_hermitian = (sporb + sporb.transpose()) * 0.5

        Notes
        -----
        This routine and how it is functioning may change in future releases.
        There are many design choices in how to assign the matrix elements when
        combining two models and it is not clear what is the best procedure.

        The current implentation does not preserve the hermiticity of the matrix.

        Parameters
        ----------
        other : object
            must be an object of the same type as `self`
        axis : int
            axis to append the two sparse geometries along
        eps : float, optional
            tolerance that all coordinates *must* be within to allow an append.
            It is important that this value is smaller than half the distance between
            the two closests atoms such that there is no ambiguity in selecting
            equivalent atoms. An internal stricter eps is used as a baseline, see above.

        See Also
        --------
        prepend : equivalent scheme as this method
        transpose : ensure hermiticity by using this routine
        Geometry.append
        Geometry.prepend

        Raises
        ------
        ValueError if atomic coordinates does not overlap within `eps`

        Returns
        -------
        object
            a new instance with two sparse matrices joined and appended together
        """
        if not (type(self) is type(other)):
            raise ValueError(self.__class__.__name__ + '.append requires other to be of same type: {}'.format(other.__class__.__name__))

        if self.geometry.nsc[axis] > 3 or other.geometry.nsc[axis] > 3:
            raise ValueError(self.__class__.__name__ + '.append requires sparse-geometries to maximally '
                             'have 3 supercells along appending axis.')

        if np.any(self.geometry.nsc != other.geometry.nsc):
            raise ValueError(self.__class__.__name__ + '.append requires sparse-geometries to have the same '
                             'number of supercells along all directions.')

        if self.dim != other.dim:
            raise ValueError(self.__class__.__name__ + '.append requires the same number of dimensions in the matrix')

        # Create sparsity pattern in the atomic picture.
        # This makes it easier to find the coupling elements along axis.
        # It could also be done in the orbital space...

        def _sep_connections(spO, direction):
            """ Finds atoms that has connections crossing the `axis` along `direction`

            Returns
            -------
            self_connect
                atoms in `spO` which connects across `direction`
            other_connect
                atoms in `spO` which self_connect connects to along `direction`
            """
            geom = spO.geometry
            # We need to copy since we are deleting elements below
            csr = spO._csr.copy([0])

            # We will retain all connections crossing along the given direction
            n = spO.shape[0]

            # Figure out the matrix columns we should retain
            nsc = [None] * 3
            nsc[axis] = direction

            # Get all supercell indices that we should delete from the column specifications
            idx = delete(_a.arangei(geom.sc.n_s), geom.sc.sc_index(nsc)) * n

            # Calculate columns to delete
            cols = array_arange(idx, n=_a.fulli(idx.shape, n))

            # Delete all values in columns, but keep them to retain the supercell information
            csr.delete_columns(cols, keep_shape=True)
            # Now we are in a position to find the indices along the append direction
            self_connect = geom.sc2uc(geom.o2a((csr.ncol > 0).nonzero()[0], True), True)

            # Retrieve the connected atoms in the other structure
            other_connect = geom.sc2uc(geom.o2a(csr.col[array_arange(csr.ptr[:-1], n=csr.ncol)], True), True)

            return self_connect, other_connect

        # Naming convention:
        #  P_01 -> [0] -> [1]
        #  P_10 -> [1] -> [0]
        #  M_01 -> [0] -> [-1]
        #  M_10 -> [-1] -> [0]
        self_P_01, self_P_10 = _sep_connections(self, +1)
        self_M_01, self_M_10 = _sep_connections(self, -1)
        other_P_01, other_P_10 = _sep_connections(other, +1)
        other_M_01, other_M_10 = _sep_connections(other, -1)

        # I.e. the connections in the supercell picture will be:
        # Note that all indices are not in any supercell (which is why we need to
        # translate them anyhow).

        def _find_overlap(g1, g1_idx, isc1, g2, g2_idx, isc2, R):
            """ Finds `g1_idx` atoms in `g2_idx` """
            xyz1 = g1.axyz(g1_idx, isc=isc1)
            g1_g2 = []
            warn_atoms = []
            for ia, xyz in zip(g1_idx, xyz1):
                # Only search in the index
                idx = g2.close_sc(xyz, isc2, R=R, idx=g2_idx)
                g1_g2.append(_check(idx, ia, warn_atoms))
            return _a.arrayi(g1_g2).ravel(), warn_atoms

        def _check(idx, atom, warn_atoms):
            if len(idx[0]) == 0:
                warn_atoms.append(atom)
                if len(idx[1]) == 0:
                    raise ValueError(self.__class__.__name__ + '.append found incompatible self/other within the given eps value.')
                idx = idx[1]
            else:
                idx = idx[0]
            if len(idx) != 1:
                raise ValueError(self.__class__.__name__ + '.append found two atoms close to a mirror atom, a too high eps value was given.')
            return idx

        # Radius to use as precision array
        R = _a.arrayd([0.001, eps])

        # Initialize arrays for checking
        self_isc = [0] * 3
        other_isc = [0] * 3

        def _2(spg1, spg1_idx, spg1_isc, spg2, spg2_idx, spg2_isc, name):
            _error = self.__class__.__name__ + '.append({}) '.format(name)
            idx, warn_atoms = _find_overlap(spg1.geometry, spg1_idx, spg1_isc,
                                            spg2.geometry, spg2_idx, spg2_isc, R)
            if len(spg1_idx) != len(spg2_idx):
                raise ValueError(_error + 'did not find an equivalent overlap atoms between the two geometries.')
            if len(idx) != len(spg2_idx):
                raise ValueError(_error + 'did not find all overlapping atoms.')

            if len(warn_atoms) > 0:
                # Sort them and ensure they are a list
                warn_atoms = str(np.sort(warn_atoms).tolist())
                warn(_error + 'atoms farther than 0.001 Ang: {}.'.format(warn_atoms))

            # Now we have the atomic indices that we know are "dublicated"
            # Ensure the number of orbitals are the same in both geometries
            # (we don't check explicitly names etc. since this should be the users
            #  responsibility)
            s1 = spg1.geometry.atoms.sub(spg1_idx).reorder().firsto
            s2 = spg2.geometry.atoms.sub(idx).reorder().firsto
            if not np.all(s1 == s2):
                raise ValueError(_error + 'requires geometries to have the same '
                                 'number of orbitals in the overlapping region.')

            return idx

        # in the full sparse geometry:
        # [0] <-> [0]
        self_isc[axis] = 0
        other_isc[axis] = 0
        self_P_10_to_other_M_01 = _2(self, self_P_10, self_isc,
                                     other, other_M_01, other_isc, 'self[0] -> other[0]')

        self_isc[axis] = -1
        other_isc[axis] = -1
        other_M_10_to_self_P_01 = _2(other, other_M_10, other_isc,
                                     self, self_P_01, self_isc, 'other[0] -> self[0]')

        # [0] -> [-1]
        self_isc[axis] = -1
        other_isc[axis] = -1
        self_M_10_to_other_P_01 = _2(self, self_M_10, self_isc,
                                     other, other_P_01, other_isc, 'self[0] -> other[-1]')

        # [0] -> [1]
        self_isc[axis] = 0
        other_isc[axis] = 0
        other_P_10_to_self_M_01 = _2(other, other_P_10, other_isc,
                                     self, self_M_01, self_isc, 'other[0] -> self[1]')

        # Clean-up
        del self_isc, other_isc

        # Now we have the following operations to perform
        self_no = self.geometry.no
        other_no = other.geometry.no
        total_no = self_no + other_no

        # Now create the combined geometry + sparse matrix
        total_geom = self.geometry.append(other, axis)
        sc = total_geom.sc

        # 1. create a copy of the sparse-geometries
        # 2. translate old columns to new columns
        # 3. merge the two
        # 4. insert the overlapping stuff (both +/-)
        self_o2n = _a.arangei(self_no)
        self_o2n.shape = (1, -1)
        self_o2n = self_o2n + sc.sc_index(self.geometry.sc.sc_off).reshape(-1, 1) * total_no
        self_o2n.shape = (-1,)

        other_o2n = _a.arangei(other_no) + self_no
        other_o2n.shape = (1, -1)
        other_o2n = other_o2n + sc.sc_index(other.geometry.sc.sc_off).reshape(-1, 1) * total_no
        other_o2n.shape = (-1,)

        # Create a template new sparse matrix
        self_csr = self._csr
        other_csr = other._csr

        total = self.copy()
        # Overwrite geometry
        total._geometry = total_geom
        n_s = sc.n_s

        # Correct the new csr shape
        csr = total._csr
        csr._shape = (total_no, total_no * n_s, csr.dim)

        # Fix columns in the self part
        idx = array_arange(csr.ptr[:-1], n=csr.ncol)
        csr.col[idx] = self_o2n[csr.col[idx]]

        # Now add the `other` sparse data while fixing the supercell indices
        csr.ptr = concatenate((csr.ptr[:-1], csr.ptr[-1] + other_csr.ptr)).astype(int32, copy=False)
        csr.ncol = concatenate((csr.ncol, other_csr.ncol)).astype(int32, copy=False)
        # We use take since other_csr.col may contain non-finalized elements (i.e. too large values)
        # In this case we use take to *clip* the indices to the largest available one.
        # This may be done since col elements not touched by the .ptr + .ncol will never
        # be used.
        csr.col = concatenate((csr.col, take(other_o2n, other_csr.col, mode='clip'))).astype(int32, copy=False)
        csr._D = concatenate((csr._D, other_csr._D), axis=0)

        # Small clean-up
        del self_o2n, other_o2n, idx

        # At this point `csr` contains all data.
        # but the columns are incorrect. I.e. self -> self along the append axis
        # where it should connect to `other`.

        # Below we are correcting the column indices such that they
        # connect to the proper things.
        # Since some systems has crossings over diagonal supercells we need
        # all supercells with a non-zero component along the axis
        isc = [None] * 3

        def _transfer_indices(csr, rows, old_col, new_col):
            " Transfer indices in col to the equivalent column indices "
            if old_col.size != new_col.size:
                raise ValueError(self.__class__.__name__ + '.append requires the overlapping basis to '
                                 'be equivalent. We found different number of hopping elements between '
                                 'the two regions.')

            col_idx = array_arange(csr.ptr[rows], n=csr.ncol[rows], dtype=int32)
            col_idx = col_idx[indices_only(csr.col[col_idx], old_col)]

            # Indices are now the indices in csr.col such that
            #   col[col_idx] in old_col
            # Now we need to find the indices (in order)
            # such that
            #   col[col_idx] == old_col[old_idx]
            # This will let us do:
            #   col[col_idx] = new_col[old_idx]
            # since old_col and new_col have the same order
            # Since indices does not return a sorted index list
            # but only indices of elements in other list
            # we need to sort them correctly
            # Create the linear index that transfers from old -> new
            # Since old_col/new_col does not contain the full supercell picture
            # we need to create a fake indexing converter
            min_col = (old_col[0] // csr.shape[0]) * csr.shape[0]
            max_col = (old_col[-1] // csr.shape[0] + 1) * csr.shape[0]
            new_col_idx = _a.arangei(max_col - min_col)
            new_col_idx[old_col - min_col] = new_col
            csr.col[col_idx] = new_col_idx[csr.col[col_idx] - min_col]

        ## nomenclature in new supercell
        # self[0] -> other[0]

        # Now we have the two matrices merged.
        # We now need to fix connections crossing the border
        isc[axis] = 1
        rows = self.geometry.a2o(self_P_01, True)
        # We have to store isc_off since we require a one2one correspondance
        # of the new supercells. Also we require the supercell indices to be
        # sorted, and hence we sort the sc-indices (just in case)
        isc_off = np.sort(sc.sc_index(isc))
        sc_off = isc_off.reshape(-1, 1) * total_no
        # These columns should have a one-to-one correspondance
        old_col = (self.geometry.a2o(self_P_10, True).reshape(1, -1) + sc_off).ravel().astype(int32)
        # Since we are appending we actually move it into the primary cell (this is where the
        # requirement of nsc == 3 comes from...)
        # Shift all supercell indices to the primary one (along the append axis)
        sc_off = sc.sc_off[isc_off, :]
        sc_off[:, axis] = 0
        sc_off = sc.sc_index(sc_off).reshape(-1, 1) * total_no + self_no
        new_col = (other.geometry.a2o(self_P_10_to_other_M_01, True).reshape(1, -1) + sc_off).ravel().astype(int32)

        # Find columns in `rows` and transfer
        # the elements with values `old_col` -> `new_col`
        # Since this should catch *all* elements that cross the
        # boundary we will only have elements that are actually used
        # So we need simply to reduce idx to the indices that contain the elements
        # in `old_col`
        _transfer_indices(csr, rows, old_col, new_col)

        ##
        # other[0] -> self[0]
        isc[axis] = -1
        rows = other.geometry.a2o(other_M_01, True) + self_no
        isc_off = np.sort(sc.sc_index(isc))
        sc_off = isc_off.reshape(-1, 1) * total_no + self_no
        old_col = (other.geometry.a2o(other_M_10, True).reshape(1, -1) + sc_off).ravel().astype(int32)
        sc_off = sc.sc_off[isc_off, :]
        sc_off[:, axis] = 0
        sc_off = sc.sc_index(sc_off).reshape(-1, 1) * total_no
        new_col = (self.geometry.a2o(other_M_10_to_self_P_01, True).reshape(1, -1) + sc_off).ravel().astype(int32)
        _transfer_indices(csr, rows, old_col, new_col)

        ##
        # self[0] -> other[-1]
        isc[axis] = -1
        rows = self.geometry.a2o(self_M_01, True)
        isc_off = np.sort(sc.sc_index(isc))
        sc_off = isc_off.reshape(-1, 1) * total_no
        old_col = (self.geometry.a2o(self_M_10, True).reshape(1, -1) + sc_off).ravel().astype(int32)
        #sc_off = sc.sc_off[isc_off, :]
        #sc_off[:, axis] = -1
        #sc_off = sc.sc_index(sc_off).reshape(-1, 1) * total_no
        new_col = (other.geometry.a2o(self_M_10_to_other_P_01, True).reshape(1, -1) + self_no + sc_off).ravel().astype(int32)
        _transfer_indices(csr, rows, old_col, new_col)

        ##
        # other[0] -> self[1]
        isc[axis] = 1
        rows = other.geometry.a2o(other_P_01, True) + self_no
        isc_off = np.sort(sc.sc_index(isc))
        sc_off = isc_off.reshape(-1, 1) * total_no
        old_col = (other.geometry.a2o(other_P_10, True).reshape(1, -1) + self_no + sc_off).ravel().astype(int32)
        #sc_off = sc.sc_off[isc_off, :]
        #sc_off[:, axis] = 1
        #sc_off = sc.sc_index(sc_off).reshape(-1, 1) * total_no
        new_col = (self.geometry.a2o(other_P_10_to_self_M_01, True).reshape(1, -1) + sc_off).ravel().astype(int32)
        _transfer_indices(csr, rows, old_col, new_col)

        # Finally figure out the number of non-zero elements
        csr._nnz = csr.ncol.sum()
        csr._finalized = False

        return total

    def toSparseAtom(self, dim=None, dtype=None):
        """ Convert the sparse object (without data) to a new sparse object with equivalent but reduced sparse pattern

        This converts the orbital sparse pattern to an atomic sparse pattern.

        Parameters
        ----------
        dim : int, optional
           number of dimensions allocated in the SparseAtom object, default to the same
        dtype : numpy.dtype, optional
           used data-type for the sparse object. Defaults to the same.
        """
        if dim is None:
            dim = self.shape[-1]
        if dtype is None:
            dtype = self.dtype

        geom = self.geometry

        # Create a conversion vector
        orb2atom = tile(geom.o2a(_a.arangei(geom.no)), geom.n_s)
        orb2atom.shape = (-1, geom.no)
        orb2atom += _a.arangei(geom.n_s).reshape(-1, 1) * geom.na
        orb2atom.shape = (-1,)

        # First convert all rows to the same
        csr = self._csr

        # Now build the new sparse pattern
        ptr = _a.emptyi(geom.na+1)
        ptr[0] = 0
        col = [None] * geom.na
        for ia in range(geom.na):

            o1, o2 = geom.a2o([ia, ia + 1])
            # Get current atomic elements
            idx = array_arange(csr.ptr[o1:o2], n=csr.ncol[o1:o2])

            # These are now the atomic columns
            # Immediately reduce to unique elements
            acol = unique(orb2atom[csr.col[idx]])

            # Step counters
            col[ia] = acol
            ptr[ia+1] = ptr[ia] + len(acol)

        # Now we can create the sparse atomic
        col = np.concatenate(col, axis=0).astype(int32, copy=False)
        spAtom = SparseAtom(geom, dim=dim, dtype=dtype, nnzpr=0)
        spAtom._csr.ptr[:] = ptr[:]
        spAtom._csr.ncol[:] = np.diff(ptr)
        spAtom._csr.col = col
        spAtom._csr._D = np.zeros([len(col), dim], dtype=dtype)
        spAtom._csr._nnz = len(col)
        spAtom._csr._finalized = True # unique returns sorted elements
        return spAtom
