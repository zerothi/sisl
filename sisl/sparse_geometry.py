from __future__ import print_function, division

import warnings

import functools as ftool
import numpy as np

import sisl._array as _a
from ._help import get_dtype, ensure_array
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

    def __init__(self, geom, dim=1, dtype=None, nnzpr=None, **kwargs):
        """ Create sparse object with element between orbitals """
        self._geom = geom

        # Initialize the sparsity pattern
        self.reset(dim, dtype, nnzpr)

    @property
    def _size(self):
        """ The size of the sparse object """
        return self.geom.na

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
        dim: int, optional
           number of dimensions per element, default to the current number of
           elements per matrix element.
        dtype: numpy.dtype, optional
           the datatype of the sparse elements
        nnzpr: int, optional
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
            nnzpr = self.geom.close(0)
            if nnzpr is None:
                nnzpr = 8
            else:
                nnzpr = max(5, len(nnzpr) * 4)

        # query dimension of sparse matrix
        s = self._size
        self._csr = SparseCSR((s, s * self.geom.n_s, dim), nnzpr=nnzpr, dtype=dtype)

        # Denote that one *must* specify all details of the elements
        self._def_dim = -1

    def empty(self, keep_nnz=False):
        """ See `SparseCSR.empty` for details """
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
        new = self.__class__(self.geom.copy(), self.dim, dtype, 1, **self._cls_kwargs())
        # Be sure to copy the content of the SparseCSR object
        new._csr = self._csr.copy(dtype=dtype)
        return new

    @property
    def geometry(self):
        """ Associated geometry """
        return self._geom
    geom = geometry

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

    def edges(self, atom, exclude=None):
        """ Retrieve edges (connections) of a given `atom` or list of `atom`'s

        The returned edges are unique and sorted (see `numpy.unique`) and are returned
        in supercell indices (i.e. ``0 <= edge < self.geom.na_s``).

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

    def rij(self, what=None, dtype=np.float64):
        r""" Create a sparse matrix with the distance between atoms/orbitals

        Parameters
        ----------
        what : {None, 'atom', 'orbital'}
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
        R._csr = (R._csr ** 2).sum(-1)
        return R

    def Rij(self, what=None, dtype=np.float64):
        r""" Create a sparse matrix with the vectors between atoms/orbitals

        Parameters
        ----------
        what : {None, 'atom', 'orbital'}
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
        # Define default of what
        if what is None:
            if isinstance(self, SparseOrbital):
                what = 'orbital'
            else:
                what = 'atom'

        geom = self.geom

        if isinstance(self, SparseAtom):
            Rij = geom.Rij
        elif isinstance(self, SparseOrbital):
            Rij = geom.oRij
        else:
            raise ValueError(self.__class__.__name__ + ' is an unknown class. Perhaps the inheritance has been broken.')

        # Conversion before doing Rij on geometry
        # We default to expect atoms
        conv = lambda val: val
        # Always only keep unique entries
        col_conv = np.unique

        if what == 'atom':
            cls = SparseAtom
            if isinstance(self, SparseOrbital):
                conv = geom.o2a

        elif what in ['orbital', 'orb']:
            cls = SparseOrbital
            if isinstance(self, SparseAtom):
                raise NotImplementedError(self.__class__.__name__ + ' cannot create Rij in SparseAtom from SparseOrbital')

        else:
            raise ValueError(self.__class__.__name__ + '.Rij what= must be "atom", "orbital" or "orb".')

        ncol = self._csr.ncol.view()
        ptr = self._csr.ptr.view()
        col = self._csr.col.view()

        # Create the output class
        R = cls(geom, 3, dtype, nnzpr=np.amax(ncol))

        # Old rows
        orow = _a.arangei(self.shape[0])

        for ro, rn in zip(orow, conv(orow)):
            # Reduce to the unique columns
            coln = np.unique(conv(col[ptr[ro]:ptr[ro]+ncol[ro]]))
            R[rn, coln] = Rij(rn, coln)

        return R

    def __repr__(self):
        """ Representation of the sparse model """
        s = self.__class__.__name__ + '{{dim: {0}, non-zero: {1}, kind={2}\n '.format(self.dim, self.nnz, self.dkind)
        s += repr(self.geom).replace('\n', '\n ')
        return s + '\n}'

    def __getattr__(self, attr):
        """ Overload attributes from the hosting geometry

        Any attribute not found in the sparse class will
        be looked up in the hosting geometry.
        """
        return getattr(self.geom, attr)

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

        assert len(old) in [self.n_s, sc.n_s], "Not all supercells are accounted for"

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
        assert len(np.unique(old)) == len(old), "non-unique values in old set_nsc"
        assert len(np.unique(new)) == len(new), "non-unique values in new set_nsc"
        assert self.n_s == len(old), "non-valid size of in old set_nsc"

        # Figure out if we need to do any work
        keep = (old != new).nonzero()[0]
        if len(keep) > 0:
            # Reduce pivoting work
            old = old[keep]
            new = new[keep]

            # Create the translation tables
            n = np.tile([size], len(old))

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
            self._csr.delete_columns(delete)

        self.geom.set_nsc(*args, **kwargs)

    def spalign(self, other):
        """ See `SparseCSR.align` for details """
        if isinstance(other, SparseCSR):
            self._csr.align(other)
        else:
            self._csr.align(other._csr)

    def eliminate_zeros(self):
        """ Removes all zero elements from the sparse matrix

        This is an *in-place* operation.
        """
        self._csr.eliminate_zeros()

    # Create iterations on the non-zero elements
    def iter_nnz(self):
        """ Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz(): # doctest: +SKIP
        ...    self[i, j] # is then the non-zero value # doctest: +SKIP
        """
        for i, j in self._csr:
            yield i, j

    __iter__ = iter_nnz

    def create_construct(self, R, param):
        """ Create a simple function for passing to the `construct` function.

        This is simply to leviate the creation of simplistic
        functions needed for setting up the sparse elements.

        Basically this returns a function:

        >>> def func(self, ia, idxs, idxs_xyz=None): # doctest: +SKIP
        ...     idx = self.geom.close(ia, R=R, idx=idxs) # doctest: +SKIP
        ...     for ix, p in zip(idx, param): # doctest: +SKIP
        ...         self[ia, ix] = p # doctest: +SKIP

        Notes
        -----
        This function only works for geometry sparse matrices (i.e. one
        element per atom). If you have more than one element per atom
        you have to implement the function your-self.

        Parameters
        ----------
        R : array_like
           radii parameters for different shells.
           Must have same length as ``param`` or one less.
           If one less it will be extended with ``R[0]/100``
        param : array_like
           coupling constants corresponding to the ``R``
           ranges. ``param[0,:]`` are the elements
           for the all atoms within ``R[0]`` of each atom.

        See Also
        --------
        construct : routine to create the sparse matrix from a generic function (as returned from `create_construct`)
        """

        def func(self, ia, idxs, idxs_xyz=None):
            idx = self.geom.close(ia, R=R, idx=idxs, idx_xyz=idxs_xyz)
            for ix, p in zip(idx, param):
                self[ia, ix] = p

        return func

    def construct(self, func, na_iR=1000, method='rand', eta=False):
        """ Automatically construct the sparse model based on a function that does the setting up of the elements

        This may be called in two variants.

        1. Pass a function (``func``), see e.g. ``create_construct``
           which does the setting up.
        2. Pass a tuple/list in ``func`` which consists of two
           elements, one is ``R`` the radii parameters for
           the corresponding parameters.
           The second is the parameters
           corresponding to the ``R[i]`` elements.
           In this second case all atoms must only have
           one orbital.

        Parameters
        ----------
        func: callable or array_like
           this function *must* take 4 arguments.
           1. Is this object (``self``)
           2. Is the currently examined atom (``ia``)
           3. Is the currently bounded indices (``idxs``)
           4. Is the currently bounded indices atomic coordinates (``idxs_xyz``)
           An example `func` could be:

           >>> def func(self, ia, idxs, idxs_xyz=None): # doctest: +SKIP
           ...     idx = self.geom.close(ia, R=[0.1, 1.44], idx=idxs, idx_xyz=idxs_xyz) # doctest: +SKIP
           ...     self[ia, idx[0]] = 0 # doctest: +SKIP
           ...     self[ia, idx[1]] = -2.7 # doctest: +SKIP

        na_iR : int, optional
           number of atoms within the sphere for speeding
           up the `iter_block` loop.
        method : {'rand', str}
           method used in `Geometry.iter_block`, see there for details
        eta: bool, optional
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

            if np.any(np.diff(self.geom.lasto) > 1):
                raise ValueError("Automatically setting a sparse model "
                              "for systems with atoms having more than 1 "
                              "orbital *must* be done by your-self. You have to define a corresponding `func`.")

            # Convert to a proper function
            func = self.create_construct(func[0], func[1])

        iR = self.geom.iR(na_iR)

        # Get number of atoms
        na = self.na
        na_run = 0

        from time import time
        from sys import stdout
        t0 = time()
        name = self.__class__.__name__

        # Do the loop
        for ias, idxs in self.geom.iter_block(iR=iR, method=method):

            # Get all the indexed atoms...
            # This speeds up the searching for coordinates...
            idxs_xyz = self.geom[idxs, :]

            # Loop the atoms inside
            for ia in ias:
                func(self, ia, idxs, idxs_xyz)

            if eta:
                # calculate the remaining atoms to process
                na_run += len(ias)
                na -= len(ias)
                # calculate hours, minutes, seconds
                m, s = divmod((time()-t0)/na_run * na, 60)
                h, m = divmod(m, 60)
                stdout.write(name + ".construct() ETA = {0:5d}h {1:2d}m {2:5.2f}s\r".format(int(h), int(m), s))
                stdout.flush()

        if eta:
            # calculate hours, minutes, seconds spend on the computation
            m, s = divmod((time()-t0), 60)
            h, m = divmod(m, 60)
            stdout.write(name + ".construct() finished after {0:d}h {1:d}m {2:.1f}s\n".format(int(h), int(m), s))
            stdout.flush()

    @property
    def finalized(self):
        """ Whether the contained data is finalized and non-used elements have been removed """
        return self._csr.finalized

    def remove(self, atom):
        """ Create a subset of this sparse matrix by removing the atoms corresponding to `atom`

        Indices passed must be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : array_like of int
            indices of removed atoms

        See Also
        --------
        Geometry.remove : equivalent to the resulting `Geometry` from this routine
        Geometry.sub : the negative of `Geometry.remove`
        sub : the negative of `remove`, i.e. retain a subset of atoms
        """
        atom = self.sc2uc(atom)
        atom = np.delete(_a.arangei(self.na), atom)
        return self.sub(atom)

    def sub(self, atom):
        """ Create a subset of this sparse matrix by retaining the atoms corresponding to `atom`

        Indices passed must be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : array_like of int
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
        a = ensure_array(a)
        b = ensure_array(b)
        # Create full index list
        full = _a.arangei(len(self.geom))
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

    def tocsr(self, index, isc=None, **kwargs):
        """ Return a ``scipy.sparse.csr_matrix`` of the specified index

        Parameters
        ----------
        index : int
           the index in the sparse matrix (for non-orthogonal cases the last
           dimension is the overlap matrix)
        isc : int, optional
           the supercell index, or all (if ``isc=None``)
        """
        if isc is not None:
            raise NotImplementedError("Requesting sub-sparse has not been implemented yet")
        return self._csr.tocsr(index, **kwargs)

    def spsame(self, other):
        """ Compare two sparse objects and check whether they have the same entries.

        This does not necessarily mean that the elements are the same
        """
        return self._csr.spsame(other._csr)

    @classmethod
    def fromsp(cls, geom, *sp):
        """ Returns a sparse model from a preset Geometry and a list of sparse matrices """
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
    def __add__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c += b
        return c
    __radd__ = __add__

    def __iadd__(a, b):
        if isinstance(b, _SparseGeometry):
            a._csr += b._csr
        else:
            a._csr += b
        return a

    def __sub__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c -= b
        return c

    def __rsub__(a, b):
        if isinstance(b, _SparseGeometry):
            c = b.copy(dtype=get_dtype(a, other=b.dtype))
            c._csr += -1 * a._csr
        else:
            c = b + (-1) * a
        return c

    def __isub__(a, b):
        if isinstance(b, _SparseGeometry):
            a._csr -= b._csr
        else:
            a._csr -= b
        return a

    def __mul__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c *= b
        return c
    __rmul__ = __mul__

    def __imul__(a, b):
        if isinstance(b, _SparseGeometry):
            a._csr *= b._csr
        else:
            a._csr *= b
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
        if isinstance(b, _SparseGeometry):
            a._csr /= b._csr
        else:
            a._csr /= b
        return a

    def __floordiv__(a, b):
        if isinstance(b, _SparseGeometry):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c //= b
        return c

    def __ifloordiv__(a, b):
        if isinstance(b, _SparseGeometry):
            raise NotImplementedError
        a._csr //= b
        return a

    def __truediv__(a, b):
        if isinstance(b, _SparseGeometry):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c /= b
        return c

    def __itruediv__(a, b):
        if isinstance(b, _SparseGeometry):
            raise NotImplementedError
        a._csr /= b
        return a

    def __pow__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c **= b
        return c

    def __rpow__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._csr = b ** c._csr
        return c

    def __ipow__(a, b):
        if isinstance(b, _SparseGeometry):
            a._csr **= b._csr
        else:
            a._csr **= b
        return a


class SparseAtom(_SparseGeometry):
    """ Sparse object with number of rows equal to the total number of atoms in the `Geometry` """

    def __getitem__(self, key):
        """ Elements for the index(s) """
        dd = self._def_dim
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geom.sc_index(key[-1]) * self.na
                key = [el for el in key[:-1]]
                key[1] = self.geom.sc2uc(key[1]) + off
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
                off = self.geom.sc_index(key[-1]) * self.na
                key = [el for el in key[:-1]]
                key[1] = self.geom.sc2uc(key[1]) + off
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        self._csr[key] = val

    @property
    def _size(self):
        return self.geom.na

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
        >>> for i, j in self.iter_nnz(): # doctest: +SKIP
        ...    self[i, j] # is then the non-zero value # doctest: +SKIP

        Parameters
        ----------
        atom : int or array_like
            only loop on the non-zero elements coinciding with the atoms
        """
        if not atom is None:
            atom = ensure_array(atom)
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
            geom = self.geom.cut(seps, axis, *args, **kwargs)
            # Check whether the warning exists
            if len(w) > 0:
                if issubclass(w[-1].category, UserWarning):
                    new_w = str(w[-1].message)
                    new_w += ("\n---\n"
                              "The sparse atom cannot be cut as the structure "
                              "cannot be tiled accordingly. ANY use of the model has been "
                              "relieved from sisl.")
        if new_w:
            warnings.warn(new_w, UserWarning)

        # Now we need to re-create number of supercells
        na = self.na
        S = self.tocsr(0)

        # First we need to figure out how long the interaction range is
        # in the cut-direction
        # We initialize to be the same as the parent direction
        nsc = np.array(self.nsc, np.int32, copy=True) // 2
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
                warnings.warn(
                    'Cut the connection at nsc={0} in direction {1}.'.format(
                        nsc[axis], axis), UserWarning)

        # Update number of super-cells
        nsc[:] = nsc[:] * 2 + 1
        geom.sc.set_nsc(nsc)

        # Now we have a correct geometry, and
        # we are now ready to create the sparsity pattern
        # Reduce the sparsity pattern, first create the new one
        S = self.__class__(geom, self.dim, self.dtype, np.amax(self._csr.ncol), **self._cls_kwargs())

        def sca2sca(M, a, m, seps, axis):
            # Converts an o from M to m
            isc = np.array(M.a2isc(a), np.int32, copy=True)
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
            a, afp, afm = sca2sca(self.geom, ia, S.geom, seps, axis)
            if a is None:
                continue
            S[ja, a + afp] = self[ja, ia]
            # TODO check that we indeed have Hermiticity for non-colinear and spin-orbit
            S[a, ja + afm] = self[ja, ia]

        return S

    def sub(self, atom):
        """ Create a subset of this sparse matrix by only retaining the elements corresponding to the ``atom``

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : array_like of int
            indices of retained atoms

        See Also
        --------
        Geometry.remove : the negative of `Geometry.sub`
        Geometry.sub : equivalent to the resulting `Geometry` from this routine
        remove : the negative of `sub`, i.e. remove a subset of atoms
        """
        atom = self.sc2uc(atom)
        geom = self.geom.sub(atom)

        idx = np.tile(atom, self.n_s)
        # Use broadcasting rules
        idx.shape = (self.n_s, -1)
        tmp = _a.arangei(self.n_s) * self.na
        tmp.shape = (-1, 1)
        idx += tmp
        del tmp
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
        g = self.geom.tile(reps, axis)
        S = self.__class__(g, self.dim, self.dtype, 1, **self._cls_kwargs())

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geom
        na = self.na
        ncol = self._csr.ncol
        if self.finalized:
            col = self._csr.col
            D = self._csr._D
        else:
            ptr = self._csr.ptr
            idx = array_arange(ptr[:-1], n=ncol)
            col = np.take(self._csr.col, idx)
            D = np.take(self._csr._D, idx, 0)
            del ptr, idx

        # Information for the new Hamiltonian sparse matrix
        na_n = S.na
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = np.tile(ncol, reps)
        # Now indptr is complete
        indptr = np.insert(_a.cumsumi(ncol), 0, 0)
        del ncol
        indices = _a.emptyi([indptr[-1]])
        indices.shape = (reps, -1)

        # Now we should fill the data
        isc = geom.a2isc(col)
        # resulting atom in the new geometry (without wrapping
        # for correct supercell, that will happen below)
        JA = col % na + na * isc[:, axis] - na

        # Create repetitions
        for rep in range(reps):
            # Figure out the JA atoms
            JA += na
            # Correct the supercell information
            isc[:, axis] = JA // na_n

            indices[rep, :] = JA % na_n + sc_index(isc) * na_n

        # Clean-up
        del isc, JA

        indices.shape = (-1,)
        S._csr = SparseCSR((np.tile(D, (reps, 1)), indices, indptr),
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
        g = self.geom.repeat(reps, axis)
        S = self.__class__(g, self.dim, self.dtype, 1, **self._cls_kwargs())

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geom
        na = self.na
        ncol = self._csr.ncol
        if self.finalized:
            col = self._csr.col
            D = self._csr._D
        else:
            ptr = self._csr.ptr
            idx = array_arange(ptr[:-1], n=ncol)
            col = np.take(self._csr.col, idx)
            D = np.take(self._csr._D, idx, 0)
            del ptr, idx

        # Information for the new Hamiltonian sparse matrix
        na_n = S.na
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = np.repeat(ncol, reps)
        # Now indptr is complete
        indptr = np.insert(_a.cumsumi(ncol), 0, 0)
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
            idx = array_arange(indptr[rep:-1:reps], n=self._csr.ncol)
            indices[idx] = JA + A % reps + sc_index(isc) * na_n

        # Clean-up
        del isc, JA, A, idx

        # In the repeat we have to tile individual atomic couplings
        # So we should split the arrays and tile them individually
        # Now D is made up of D values, per atom
        if geom.na == 1:
            D = np.tile(D, (reps, 1))
        else:
            ntile = ftool.partial(np.tile, reps=(reps, 1))
            D = np.vstack(map(ntile, np.split(D, _a.cumsumi(self._csr.ncol[:-1]), axis=0)))

        S._csr = SparseCSR((D, indices, indptr),
                           shape=(geom_n.na, geom_n.na_s))

        return S


class SparseOrbital(_SparseGeometry):
    """ Sparse object with number of rows equal to the total number of orbitals in the `Geometry` """

    def __getitem__(self, key):
        """ Elements for the index(s) """
        dd = self._def_dim
        if len(key) > 2:
            # This may be a specification of supercell indices
            if isinstance(key[-1], tuple):
                # We guess it is the supercell index
                off = self.geom.sc_index(key[-1]) * self.no
                key = [el for el in key[:-1]]
                key[1] = self.geom.osc2uc(key[1]) + off
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
                off = self.geom.sc_index(key[-1]) * self.no
                key = [el for el in key[:-1]]
                key[1] = self.geom.osc2uc(key[1]) + off
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        self._csr[key] = val

    @property
    def _size(self):
        return self.geom.no

    def edges(self, atom=None, exclude=None, orbital=None):
        """ Retrieve edges (connections) of a given `atom` or list of `atom`'s

        The returned edges are unique and sorted (see `numpy.unique`) and are returned
        in supercell indices (i.e. ``0 <= edge < self.geom.no_s``).

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
            return np.unique(self.geom.o2a(self._csr.edges(self.geom.a2o(atom, True), exclude)))
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
        row = self.geom.a2o(atom, all=True)
        return self._csr.nonzero(row=row, only_col=only_col)

    def iter_nnz(self, atom=None, orbital=None):
        """ Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz(): # doctest: +SKIP
        ...    self[i, j] # is then the non-zero value # doctest: +SKIP

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
            orbital = self.geom.a2o(atom)
        elif not orbital is None:
            orbital = ensure_array(orbital)
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
            geom = self.geom.cut(seps, axis, *args, **kwargs)
            # Check whether the warning exists
            if len(w) > 0:
                if issubclass(w[-1].category, UserWarning):
                    new_w = str(w[-1].message)
                    new_w += ("\n---\n"
                              "The sparse orbital cannot be cut as the structure "
                              "cannot be tiled accordingly. ANY use of the model has been "
                              "relieved from sisl.")
        if new_w:
            warnings.warn(new_w, UserWarning)

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
                warnings.warn(
                    'Cut the connection at nsc={0} in direction {1}.'.format(
                        nsc[axis], axis), UserWarning)

        # Update number of super-cells
        nsc[:] = nsc[:] * 2 + 1
        geom.sc.set_nsc(nsc)

        # Now we have a correct geometry, and
        # we are now ready to create the sparsity pattern
        # Reduce the sparsity pattern, first create the new one
        S = self.__class__(geom, self.dim, self.dtype, np.amax(self._csr.ncol), **self._cls_kwargs())

        def sco2sco(M, o, m, seps, axis):
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
            o, ofp, ofm = sco2sco(self.geom, io, S.geom, seps, axis)
            if o is None:
                continue
            d = self[jo, io]
            S[jo, o + ofp] = d
            S[o, jo + ofm] = d

        return S

    def sub(self, atom):
        """ Create a subset of this sparse matrix by only retaining the atoms corresponding to `atom`

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : array_like of int
            indices of retained atoms

        See Also
        --------
        Geometry.remove : the negative of `Geometry.remove`
        Geometry.sub : equivalent to the resulting `Geometry` from this routine
        remove : the negative of `sub`, i.e. remove a subset of atoms
        """
        atom = self.sc2uc(atom)
        orbs = self.a2o(atom, all=True)
        geom = self.geom.sub(atom)

        idx = np.tile(orbs, self.n_s)
        # Use broadcasting rules
        idx.shape = (self.n_s, -1)
        tmp = _a.arangei(self.n_s) * self.no
        tmp.shape = (-1, 1)
        idx += tmp
        del tmp
        idx.shape = (-1,)

        # Now create the new sparse orbital class
        S = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())
        S._csr = self._csr.sub(idx)

        return S

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
        g = self.geom.tile(reps, axis)
        S = self.__class__(g, self.dim, self.dtype, 1, **self._cls_kwargs())

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geom
        no = self.no
        ncol = self._csr.ncol
        if self.finalized:
            col = self._csr.col
            D = self._csr._D
        else:
            ptr = self._csr.ptr
            idx = array_arange(ptr[:-1], n=ncol)
            col = np.take(self._csr.col, idx)
            D = np.take(self._csr._D, idx, 0)
            del ptr, idx

        # Information for the new Hamiltonian sparse matrix
        no_n = S.no
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = np.tile(ncol, reps)
        # Now indptr is complete
        indptr = np.insert(_a.cumsumi(ncol), 0, 0)
        del ncol
        indices = _a.emptyi([indptr[-1]])
        indices.shape = (reps, -1)

        # Now we should fill the data
        isc = geom.o2isc(col)
        # resulting atom in the new geometry (without wrapping
        # for correct supercell, that will happen below)
        JO = col % no + no * isc[:, axis] - no

        # Create repetitions
        for rep in range(reps):
            # Figure out the JO orbitals
            JO += no
            # Correct the supercell information
            isc[:, axis] = JO // no_n

            indices[rep, :] = JO % no_n + sc_index(isc) * no_n

        # Clean-up
        del isc, JO

        indices.shape = (-1,)
        S._csr = SparseCSR((np.tile(D, (reps, 1)), indices, indptr),
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
        g = self.geom.repeat(reps, axis)
        S = self.__class__(g, self.dim, self.dtype, 1, **self._cls_kwargs())

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geom
        no = self.no
        ncol = self._csr.ncol
        if self.finalized:
            col = self._csr.col
            D = self._csr._D
        else:
            ptr = self._csr.ptr
            idx = array_arange(ptr[:-1], n=ncol)
            col = np.take(self._csr.col, idx)
            D = np.take(self._csr._D, idx, 0)
            del ptr, idx

        # Information for the new Hamiltonian sparse matrix
        no_n = S.no
        geom_n = S.geom

        # First loop on axis tiling and local
        # orbitals in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = np.repeat(ncol, reps)
        # Now indptr is complete
        indptr = np.insert(_a.cumsumi(ncol), 0, 0)
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
        ncol = self._csr.ncol.view()

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
            D = np.tile(D, (reps, 1))
        else:
            ntile = ftool.partial(np.tile, reps=(reps, 1))
            D = np.vstack(map(ntile, np.split(D, _a.cumsumi(ncol)[geom.lasto[:geom.na-1]], axis=0)))
        S._csr = SparseCSR((D, indices, indptr),
                           shape=(geom_n.no, geom_n.no_s))

        return S
