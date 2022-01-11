# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from numbers import Integral
import warnings
import functools as ftool
import itertools
import operator
from collections import namedtuple
from collections.abc import Sequence
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy import (
    int32, intersect1d,
    take, delete, argsort, lexsort,
    insert, unique, diff, allclose,
    searchsorted,
    tile, repeat, concatenate
)

from ._internal import set_module
from . import _array as _a
from ._array import array_arange
from .atom import Atom
from .orbital import Orbital
from .geometry import Geometry
from .messages import warn, SislError, SislWarning, progressbar, deprecate_method
from ._indices import indices_only
from ._help import get_dtype
from .utils.ranges import list2str
from .sparse import SparseCSR, isspmatrix, _ncol_to_indptr


__all__ = ['SparseAtom', 'SparseOrbital']


class _SparseGeometry(NDArrayOperatorsMixin):
    """ Sparse object containing sparse elements for a given geometry.

    This is a base class intended to be sub-classed because the sparsity information
    needs to be extracted from the ``_size`` attribute.

    The sub-classed object _must_ implement the ``_size`` attribute.
    The sub-classed object may re-implement the ``_cls_kwargs`` routine
    to pass down keyword arguments when a new class is instantiated.

    This object contains information regarding the
     - geometry

    """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        """ Create sparse object with element between orbitals """
        self._geometry = geometry

        # Initialize the sparsity pattern
        self.reset(dim, dtype, nnzpr)

    @property
    def geometry(self):
        """ Associated geometry """
        return self._geometry

    @property
    @deprecate_method(f"*.geom is deprecated, use *.geometry instead")
    def geom(self):
        """ deprecated geometry """
        return self._geometry

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

    def edges(self, atoms, exclude=None):
        """ Retrieve edges (connections) for all `atoms`

        The returned edges are unique and sorted (see `numpy.unique`) and are returned
        in supercell indices (i.e. ``0 <= edge < self.geometry.na_s``).

        Parameters
        ----------
        atoms : int or list of int
            the edges are returned only for the given atom
        exclude : int or list of int or None, optional
           remove edges which are in the `exclude` list.

        See Also
        --------
        SparseCSR.edges: the underlying routine used for extracting the edges
        """
        return self._csr.edges(atoms, exclude)

    def __str__(self):
        """ Representation of the sparse model """
        s = self.__class__.__name__ + f'{{dim: {self.dim}, non-zero: {self.nnz}, kind={self.dkind}\n '
        s += str(self.geometry).replace('\n', '\n ')
        return s + '\n}'

    def __repr__(self):
        g = self.geometry
        return f"<{self.__module__}.{self.__class__.__name__} shape={self._csr.shape[:-1]}, dim={self.dim}, nnz={self.nnz}, kind={self.dkind}>"

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
        if allclose(sc.nsc, self.sc.nsc):
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
            except ValueError:
                # Not found, i.e. new, so no need to translate
                pass

        # 1. Ensure that any one of the *old* supercells that
        #    are now deleted are put in the end
        for i, j in enumerate(deleted.nonzero()[0]):
            # Old index (j)
            old.append(j)
            # Move to the end (*HAS* to be higher than the number of
            # cells in the new supercell structure)
            new.append(max(self.n_s, sc.n_s) + i)

        # Check that we will translate all indices in the old
        # sparsity pattern to the new one
        if len(old) not in [self.n_s, sc.n_s]:
            raise SislError("Not all supercells are accounted for")

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
            self._csr.translate_columns(old, new, clean=False)

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
        self._csr._clean_columns()

        self.geometry.set_nsc(*args, **kwargs)

    def transpose(self, sort=True):
        """ Create the transposed sparse geometry by interchanging supercell indices

        Sparse geometries are (typically) relying on symmetry in the supercell picture.
        Thus when one transposes a sparse geometry one should *ideally* get the same
        matrix. This is true for the Hamiltonian, density matrix, etc.

        This routine transposes all rows and columns such that any interaction between
        row, `r`, and column `c` in a given supercell `(i,j,k)` will be transposed
        into row `c`, column `r` in the supercell `(-i,-j,-k)`.

        Parameters
        ----------
        sort : bool, optional
           the returned columns for the transposed structure will be sorted
           if this is true, default

        Notes
        -----
        The components for each sparse element are not changed in this method.

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
        # clean memory to not crowd memory too much
        T._csr.ptr = None
        T._csr.col = None
        T._csr.ncol = None
        T._csr._D = None

        # Short-links
        sc = self.geometry.sc

        # Create "DOK" format indices
        csr = self._csr
        # Number of rows (used for converting to supercell indices)
        # With this we don't need to figure out if we are dealing with
        # atoms or orbitals
        size = csr.shape[0]

        # First extract the actual data
        ncol = csr.ncol.view()
        if csr.finalized:
            ptr = csr.ptr.view()
            col = csr.col.copy()
            D = csr._D.copy()
        else:
            idx = array_arange(csr.ptr[:-1], n=ncol, dtype=int32)
            ptr = _ncol_to_indptr(ncol)
            col = csr.col[idx]
            D = csr._D[idx, :].copy()
            del idx

        # figure out rows where ncol is > 0
        # we skip the first column
        row_nonzero = (ncol > 0).nonzero()[0]
        row = repeat(row_nonzero.astype(np.int32, copy=False), ncol[row_nonzero])

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
        rows, nrow = unique(col, return_counts=True)
        T._csr.ncol = _a.zerosi(size)
        T._csr.ncol[rows] = nrow
        del rows

        if sort:
            # also sort individual rows for each column
            idx = lexsort((row, col))
        else:
            # sort columns to get transposed values.
            # This will randomize the rows
            idx = argsort(col)

        # Our new data will then be
        T._csr.col = row[idx]
        del row
        T._csr._D = D[idx]
        del D
        T._csr.ptr = _ncol_to_indptr(T._csr.ncol)

        # If `sort` we have everything sorted, otherwise it
        # is not ensured
        T._csr._finalized = sort

        return T

    def spalign(self, other):
        """ See :meth:`~sisl.sparse.SparseCSR.align` for details """
        if isinstance(other, SparseCSR):
            self._csr.align(other)
        else:
            self._csr.align(other._csr)

    def eliminate_zeros(self, *args, **kwargs):
        """ Removes all zero elements from the sparse matrix

        This is an *in-place* operation.

        See Also
        --------
        SparseCSR.eliminate_zeros : method called, see there for parameters
        """
        self._csr.eliminate_zeros(*args, **kwargs)

    # Create iterations on the non-zero elements
    def iter_nnz(self):
        """ Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz():
        ...    self[i, j] # is then the non-zero value
        """
        yield from self._csr

    __iter__ = iter_nnz

    def create_construct(self, R, param):
        """ Create a simple function for passing to the `construct` function.

        This is simply to leviate the creation of simplistic
        functions needed for setting up the sparse elements.

        Basically this returns a function:

        >>> def func(self, ia, atoms, atoms_xyz=None):
        ...     idx = self.geometry.close(ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz)
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
        if len(R) != len(param):
            raise ValueError(f"{self.__class__.__name__}.create_construct got different lengths of `R` and `param`")

        def func(self, ia, atoms, atoms_xyz=None):
            idx = self.geometry.close(ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz)
            for ix, p in zip(idx, param):
                self[ia, ix] = p

        return func

    def construct(self, func, na_iR=1000, method='rand', eta=None):
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

           >>> def func(self, ia, atoms, atoms_xyz=None):
           ...     idx = self.geometry.close(ia, R=[0.1, 1.44], atoms=atoms, atoms_xyz=atoms_xyz)
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

            if np.any(diff(self.geometry.lasto) > 1):
                raise ValueError("Automatically setting a sparse model "
                              "for systems with atoms having more than 1 "
                              "orbital *must* be done by your-self. You have to define a corresponding `func`.")

            # Convert to a proper function
            func = self.create_construct(func[0], func[1])

        iR = self.geometry.iR(na_iR)

        # Create eta-object
        eta = progressbar(self.na, self.__class__.__name__ + '.construct', 'atom', eta)

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

    def remove(self, atoms):
        """ Create a subset of this sparse matrix by removing the atoms corresponding to `atoms`

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atoms : array_like of int
            indices of removed atoms

        See Also
        --------
        Geometry.remove : equivalent to the resulting `Geometry` from this routine
        Geometry.sub : the negative of `Geometry.remove`
        sub : the opposite of `remove`, i.e. retain a subset of atoms
        """
        atoms = self.sc2uc(atoms)
        atoms = delete(_a.arangei(self.na), atoms)
        return self.sub(atoms)

    def sub(self, atoms):
        """ Create a subset of this sparse matrix by retaining the atoms corresponding to `atoms`

        Indices passed must be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atoms : array_like of int
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
    def fromsp(cls, geometry, P, **kwargs):
        r""" Create a sparse model from a preset `Geometry` and a list of sparse matrices

        The passed sparse matrices are in one of `scipy.sparse` formats.

        Parameters
        ----------
        geometry : Geometry
           geometry to describe the new sparse geometry
        P : list of scipy.sparse or scipy.sparse
           the new sparse matrices that are to be populated in the sparse
           matrix
        **kwargs : optional
           any arguments that are directly passed to the ``__init__`` method
           of the class.

        Returns
        -------
        SparseGeometry
             a new sparse matrix that holds the passed geometry and the elements of `P`
        """
        # Ensure list of * format (to get dimensions)
        if isspmatrix(P):
            P = [P]
        if isinstance(P, tuple):
            P = list(P)

        p = cls(geometry, len(P), P[0].dtype, 1, **kwargs)
        p._csr = p._csr.fromsp(*P, dtype=kwargs.get("dtype"))

        if p._size != P[0].shape[0]:
            raise ValueError(f"{cls.__name__}.fromsp cannot create a new class, the geometry "
                             "and sparse matrices does not have coinciding dimensions size != P[0].shape[0]")

        return p

    # numpy dispatch methods (same priority as SparseCSR!)
    __array_priority__ = 14

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # grab the inputs and convert to the respective csr matrices
        # such that we can defer the call to that function
        # while converting, also grab the first _SparseGeometry
        # object such that we may create the output matrix
        sp_inputs = []
        obj = None
        for inp in inputs:
            if isinstance(inp, _SparseGeometry):
                # simply store a reference
                # if needed we will copy it later
                obj = inp
                sp_inputs.append(inp._csr)
            else:
                sp_inputs.append(inp)

        out = kwargs.get("out", None)
        if out is not None:
            (out,) = out
            kwargs["out"] = (out._csr,)

        result = self._csr.__array_ufunc__(ufunc, method, *sp_inputs, **kwargs)

        if out is None:
            out = obj.copy()
            out._csr = result
        return out

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


@set_module("sisl")
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

    def nonzero(self, atoms=None, only_col=False):
        """ Indices row and column indices where non-zero elements exists

        Parameters
        ----------
        atoms : int or array_like of int, optional
           only return the tuples for the requested atoms, default is all atoms
        only_col : bool, optional
           only return then non-zero columns

        See Also
        --------
        SparseCSR.nonzero : the equivalent function call
        """
        return self._csr.nonzero(row=atoms, only_col=only_col)

    def iter_nnz(self, atoms=None):
        """ Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz():
        ...    self[i, j] # is then the non-zero value

        Parameters
        ----------
        atoms : int or array_like
            only loop on the non-zero elements coinciding with the atoms
        """
        if atoms is None:
            yield from self._csr
        else:
            atoms = self.geometry._sanitize_atoms(atoms)
            yield from self._csr.iter_nnz(atoms)

    def set_nsc(self, *args, **kwargs):
        """ Reset the number of allowed supercells in the sparse atom

        If one reduces the number of supercells *any* sparse element
        that references the supercell will be deleted.

        See `SuperCell.set_nsc` for allowed parameters.

        See Also
        --------
        SuperCell.set_nsc : the underlying called method
        """
        super().set_nsc(self.na, *args, **kwargs)

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

            # TODO this is inconsistent if seg= is used as argument
            sub = S[0:geom.na, idx * na:(idx + 1) * na].indices[:]

            if len(sub) == 0:
                break

            # figure out how many cells it is connecting to
            ncell = np.amax(sub % na) // geom.na
            ic = idx * na
            for icell in range(ncell):
                idx = ic + geom.na * icell
                # We need to ensure that every "in between" index exists
                # if it does not we discard those indices
                if len(np.logical_and(idx <= sub,
                                      sub < idx + geom.na).nonzero()[0]) == 0:
                    ncell = icell - 1
                    out = True
                    break
            nsc[axis] = isc[axis] * seps + ncell

            if out:
                warn('Cut the connection at nsc={} in direction {}.'.format(nsc[axis], axis))

        # Update number of super-cells
        nsc[:] = nsc[:] * 2 + 1
        geom.set_nsc(nsc)

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
            a, afp, afm = _sca2sca(self.geometry, ia, S.geometry, seps, axis)
            if a is None:
                continue
            d = self[ja, ia]
            S[ja, a + afp] = d
            # TODO check that we indeed have Hermiticity for non-collinear and spin-orbit
            S[a, ja + afm] = d

        return S

    def sub(self, atoms):
        """ Create a subset of this sparse matrix by only retaining the elements corresponding to the `atoms`

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atoms : array_like of int
            indices of retained atoms

        See Also
        --------
        Geometry.remove : the negative of `Geometry.sub`
        Geometry.sub : equivalent to the resulting `Geometry` from this routine
        remove : the negative of `sub`, i.e. remove a subset of atoms
        """
        atoms = self.sc2uc(atoms)
        geom = self.geometry.sub(atoms)

        idx = tile(atoms, self.n_s)
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
        geom_n = S.geometry

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = tile(ncol, reps)
        # Now indptr is complete
        indptr = _ncol_to_indptr(ncol)
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
            isc[:, axis], mod = np.divmod(JA, na_n)

            indices[rep, :] = mod + sc_index(isc) * na_n

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
        geom_n = S.geometry

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = repeat(ncol, reps)
        # Now indptr is complete
        indptr = _ncol_to_indptr(ncol)
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
            isc[:, axis], mod = np.divmod(A, reps)

            # Create the indices for the repetition
            idx = array_arange(indptr[rep:-1:reps], n=csr.ncol)
            indices[idx] = JA + mod + sc_index(isc) * na_n

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
        R._csr = np.sum(R._csr ** 2, axis=-1) ** 0.5
        return R

    def Rij(self, dtype=np.float64):
        r""" Create a sparse matrix with vectors between atoms

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


@set_module("sisl")
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

    def edges(self, atoms=None, exclude=None, orbitals=None):
        """ Retrieve edges (connections) for all `atoms`

        The returned edges are unique and sorted (see `numpy.unique`) and are returned
        in supercell indices (i.e. ``0 <= edge < self.geometry.no_s``).

        Parameters
        ----------
        atoms : int or list of int
            the edges are returned only for the given atom (but by using  all orbitals of the
            requested atom). The returned edges are also atoms.
        exclude : int or list of int or None, optional
           remove edges which are in the `exclude` list, this list refers to orbitals.
        orbital : int or list of int
            the edges are returned only for the given orbital. The returned edges are orbitals.

        See Also
        --------
        SparseCSR.edges: the underlying routine used for extracting the edges
        """
        if atoms is None and orbitals is None:
            raise ValueError(f"{self.__class__.__name__}.edges must have either 'atom' or 'orbital' keyword defined.")
        if orbitals is None:
            return unique(self.geometry.o2a(self._csr.edges(self.geometry.a2o(atoms, True), exclude)))
        return self._csr.edges(orbitals, exclude)

    def nonzero(self, atoms=None, only_col=False):
        """ Indices row and column indices where non-zero elements exists

        Parameters
        ----------
        atoms : int or array_like of int, optional
           only return the tuples for the requested atoms, default is all atoms
           But for *all* orbitals.
        only_col : bool, optional
           only return then non-zero columns

        See Also
        --------
        SparseCSR.nonzero : the equivalent function call
        """
        if atoms is None:
            return self._csr.nonzero(only_col=only_col)
        row = self.geometry.a2o(atoms, all=True)
        return self._csr.nonzero(row=row, only_col=only_col)

    def iter_nnz(self, atoms=None, orbitals=None):
        """ Iterations of the non-zero elements

        An iterator on the sparse matrix with, row and column

        Examples
        --------
        >>> for i, j in self.iter_nnz():
        ...    self[i, j] # is then the non-zero value

        Parameters
        ----------
        atoms : int or array_like
            only loop on the non-zero elements coinciding with the orbitals
            on these atoms (not compatible with the `orbitals` keyword)
        orbitals : int or array_like
            only loop on the non-zero elements coinciding with the orbital
            (not compatible with the `atoms` keyword)
        """
        if not atoms is None:
            orbitals = self.geometry.a2o(atoms, True)
        elif not orbitals is None:
            orbitals = _a.asarrayi(orbitals)
        if orbitals is None:
            yield from self._csr
        else:
            yield from self._csr.iter_nnz(orbitals)

    def set_nsc(self, *args, **kwargs):
        """ Reset the number of allowed supercells in the sparse orbital

        If one reduces the number of supercells *any* sparse element
        that references the supercell will be deleted.

        See `SuperCell.set_nsc` for allowed parameters.

        See Also
        --------
        SuperCell.set_nsc : the underlying called method
        """
        super().set_nsc(self.no, *args, **kwargs)

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

            # TODO this is inconsistent if seg= is used as argument
            sub = S[0:geom.no, idx * no:(idx + 1) * no].indices[:]

            if len(sub) == 0:
                break

            # figure out how many cells it is connecting to
            ncell = np.amax(sub % no) // geom.no
            ic = idx * no
            for icell in range(ncell):
                idx = ic + geom.no * icell
                # We need to ensure that every "in between" index exists
                # if it does not we discard those indices
                if len(np.logical_and(idx <= sub,
                                      sub < idx + geom.no).nonzero()[0]) == 0:
                    ncell = icell - 1
                    out = True
                    break
            nsc[axis] = isc[axis] * seps + ncell

            if out:
                warn('Cut the connection at nsc={} in direction {}.'.format(nsc[axis], axis))

        # Update number of super-cells
        nsc[:] = nsc[:] * 2 + 1
        geom.set_nsc(nsc)

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
        for jo, io in self.iter_nnz(orbitals=range(geom.no)):

            # Get the equivalent orbital in the smaller cell
            o, ofp, ofm = _sco2sco(self.geometry, io, S.geometry, seps, axis)
            if o is None:
                continue
            d = self[jo, io]
            S[jo, o + ofp] = d
            # TODO check that we indeed have Hermiticity for non-collinear and spin-orbit
            S[o, jo + ofm] = d

        return S

    def remove(self, atoms):
        """ Remove a subset of this sparse matrix by only retaining the atoms corresponding to `atoms`

        Parameters
        ----------
        atoms : array_like of int or Atom
            indices of removed atoms or Atom for direct removal of all atoms

        See Also
        --------
        Geometry.remove : equivalent to the resulting `Geometry` from this routine
        Geometry.sub : the negative of `Geometry.remove`
        sub : the opposite of `remove`, i.e. retain a subset of atoms
        """
        # This will digress to call .sub
        return super().remove(atoms)

    def remove_orbital(self, atoms, orbitals):
        """ Remove a subset of orbitals on `atoms` according to `orbitals`

        For more detailed examples, please see the equivalent (but opposite) method
        `sub_orbital`.

        Parameters
        ----------
        atoms : array_like of int or Atom
            indices of atoms or `Atom` that will be reduced in size according to `orbitals`
        orbitals : array_like of int or Orbital
            indices of the orbitals on `atoms` that are removed from the sparse matrix.

        See Also
        --------
        sub_orbital : retaining a set of orbitals (see here for examples)
        """
        # Get specie index of the atom (convert to list of indices)
        atoms = self.geometry._sanitize_atoms(atoms).ravel()

        # Figure out if all atoms have the same species
        specie = self.geometry.atoms.specie[atoms]
        uniq_specie, indices = unique(specie, return_inverse=True)
        if len(uniq_specie) > 1:
            # In case there are multiple different species but one wishes to
            # retain the same orbital index, then we loop on the unique species
            new = self
            for i in range(uniq_specie.size):
                idx = (indices == i).nonzero()[0]
                # now determine whether it is the whole atom
                # or only part of the geometry
                new = new.remove_orbital(atoms[idx], orbitals)
            return new

        # Get the atom object we wish to reduce
        # We know np.all(geom.atoms[atom] == old_atom)
        old_atom = self.geometry.atoms[atoms[0]]

        if isinstance(orbitals, (Orbital, Integral)):
            orbitals = [orbitals]
        if isinstance(orbitals[0], Orbital):
            orbitals = [old_atom.index(orb) for orb in orbitals]
        orbitals = delete(_a.arangei(len(old_atom)), np.asarray(orbitals).ravel())

        # now call sub_orbital
        return self.sub_orbital(atoms, orbitals)

    def sub(self, atoms):
        """ Create a subset of this sparse matrix by only retaining the atoms corresponding to `atoms`

        Negative indices are wrapped and thus works, supercell atoms are also wrapped to the unit-cell.

        Parameters
        ----------
        atoms : array_like of int or Atom
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
        atoms = self.sc2uc(atoms)
        orbs = self.a2o(atoms, all=True)
        geom = self.geometry.sub(atoms)

        idx = tile(orbs, self.n_s)
        # Use broadcasting rules
        idx.shape = (self.n_s, -1)
        idx += (_a.arangei(self.n_s) * self.no).reshape(-1, 1)
        idx.shape = (-1,)

        # Now create the new sparse orbital class
        S = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())
        S._csr = self._csr.sub(idx)

        return S

    def sub_orbital(self, atoms, orbitals):
        r""" Retain only a subset of the orbitals on `atoms` according to `orbitals`

        This allows one to retain only a given subset of the sparse matrix elements.

        Parameters
        ----------
        atoms : array_like of int or Atom
            indices of atoms or `Atom` that will be reduced in size according to `orbitals`
        orbitals : array_like of int or Orbital
            indices of the orbitals on `atoms` that are retained in the sparse matrix, the list of
            orbitals will be sorted. One cannot re-arrange matrix elements currently.

        Notes
        -----
        Future implementations may allow one to re-arange orbitals using this method.

        When using this method the internal species list will be populated by another specie
        that is named after the orbitals removed. This is to distinguish different atoms.

        Examples
        --------

        >>> # a Carbon atom with 2 orbitals
        >>> C = sisl.Atom('C', [1., 2.])
        >>> # an oxygen atom with 3 orbitals
        >>> O = sisl.Atom('O', [1., 2., 2.4])
        >>> geometry = sisl.Geometry([[0] * 3, [1] * 3]], 2, [C, O])
        >>> obj = SparseOrbital(geometry).tile(3, 0)
        >>> # fill in obj data...

        Now ``obj`` is a sparse geometry with 2 different species and 6 atoms (3 of each).
        They are ordered ``[C, O, C, O, C, O]``. In the following we
        will note species that are different from the original by a ``'`` in the list.

        Retain 2nd orbital on the 2nd atom: ``[C, O', C, O, C, O]``

        >>> new_obj = obj.sub_orbital(1, 1)

        Retain 2nd orbital on 1st and 2nd atom: ``[C', O', C, O, C, O]``

        >>> new_obj = obj.sub_orbital([0, 1], 1)

        Retain 2nd orbital on the 1st atom and 3rd orbital on 4th atom: ``[C', O, C, O', C, O]``

        >>> new_obj = obj.sub_orbital(0, 1).sub_orbital(3, 2)

        Retain 2nd orbital on all atoms equivalent to the first atom: ``[C', O, C', O, C', O]``

        >>> new_obj = obj.sub_orbital(obj.geometry.atoms[0], 1)

        Retain 1st orbital on 1st atom, and 2nd orbital on 3rd and 5th atom: ``[C', O, C'', O, C'', O]``

        >>> new_obj = obj.sub_orbital(0, 0).sub_orbital([2, 4], 1)

        See Also
        --------
        remove_orbital : removing a set of orbitals (opposite of this)
        """
        atoms = self.geometry._sanitize_atoms(atoms).ravel()

        # Figure out if all atoms have the same species
        specie = self.geometry.atoms.specie[atoms]
        uniq_specie, indices = unique(specie, return_inverse=True)
        if len(uniq_specie) > 1:
            # In case there are multiple different species but one wishes to
            # retain the same orbital index, then we loop on the unique species
            new = self
            for i in range(uniq_specie.size):
                idx = (indices == i).nonzero()[0]
                # now determine whether it is the whole atom
                # or only part of the geometry
                new = new.sub_orbital(atoms[idx], orbitals)
            return new

        # Get the atom object we wish to reduce
        old_atom = self.geometry.atoms[atoms[0]]

        if isinstance(orbitals, (Orbital, Integral)):
            orbitals = [orbitals]
        if isinstance(orbitals[0], Orbital):
            orbitals = [old_atom.index(orb) for orb in orbitals]
        orbitals = np.sort(orbitals)

        # At this point we are sure that uniq_specie is *only* one specie!
        geom = self.geometry.sub_orbital(atoms, orbitals)

        # Now create the new sparse orbital class
        SG = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())

        rem_orbs = delete(_a.arangei(old_atom.no), orbitals)
        # Find orbitals to remove (note this HAS to be from the original array)
        rem_orbs = np.add.outer(self.geometry.a2o(atoms), rem_orbs).ravel()

        # Generate a list of orbitals to retain
        sub_idx = delete(_a.arangei(self.no), rem_orbs)

        # Generate full supercell indices
        n_s = self.geometry.n_s
        sc_off = _a.arangei(n_s) * self.no
        sub_idx = tile(sub_idx, n_s).reshape(n_s, -1) + sc_off.reshape(-1, 1)
        SG._csr = self._csr.sub(sub_idx)

        # just ensure we are doing the correct thing
        assert SG._csr.shape[0] == SG.geometry.no

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
            idx = array_arange(csr.ptr[:-1], n=ncol)
            col = csr.col[idx]
            D = csr._D[idx, :]
            del idx

        # Information for the new Hamiltonian sparse matrix
        no_n = int32(S.no)
        geom_n = S.geometry

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        ncol = tile(ncol, reps)
        # Now indptr is complete
        indptr = _ncol_to_indptr(ncol)
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
            isc[:, axis], mod = np.divmod(JO, no_n)

            indices[rep, :] = mod + sc_index(isc) * no_n

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
        if self.finalized or csr.nnz == csr.ptr[-1]:
            col = csr.col
            D = csr._D
        else:
            idx = array_arange(csr.ptr[:-1], n=csr.ncol)
            col = csr.col[idx]
            D = csr._D[idx, :]
            del idx

        # Information for the new Hamiltonian sparse matrix
        no_n = int32(S.no)
        geom_n = S.geometry

        # First loop on axis tiling and local
        # orbitals in the geometry
        sc_index = geom_n.sc_index

        # Create new indptr, indices and D
        idx = array_arange(repeat(geom.firsto[:-1], reps),
                           repeat(geom.firsto[1:], reps))
        ncol = csr.ncol[idx]
        # Now indptr is complete
        indptr = _ncol_to_indptr(ncol)
        # Note that D above is already reduced to a *finalized* state
        # So we have to re-create the reduced index pointer
        # Then we take repeat the data by smart indexing
        D = D[array_arange(_ncol_to_indptr(csr.ncol)[idx], n=ncol), :]
        del ncol, idx
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
            isc[:, axis], mod = np.divmod(O, reps)

            # Create the indices for the repetition
            idx = array_arange(indptr[array_arange(OA, n=AO)], n=ncol)
            indices[idx] = JO + oA * mod + sc_index(isc) * no_n

        # Clean-up
        del isc, JO, O, OA, AO, idx

        # In the repeat we have to tile individual atomic couplings
        # So we should split the arrays and tile them individually
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
        R._csr = np.sum(R._csr ** 2, axis=-1) ** 0.5
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

    def add(self, other, axis=None, offset=(0, 0, 0)):
        r""" Add two sparse matrices by adding the parameters to one set. The final matrix will have no couplings between `self` and `other`

        The final sparse matrix will not have any couplings between `self` and `other`. Not even if they
        have commensurate overlapping regions. If you want to create couplings you have to use `append` but that
        requires the structures are commensurate in the coupling region.

        Parameters
        ----------
        other : SparseGeometry
            the other sparse matrix to be added, all atoms will be appended
        axis : int or None, optional
            whether a specific axis of the cell will be added to the final geometry.
            For ``None`` the final cell will be that of `self`, otherwise the lattice
            vector corresponding to `axis` will be appended.
        offset : (3,), optional
            offset in geometry of `other` when adding the atoms.

        See Also
        --------
        append : append two matrices by also adding overlap couplings
        prepend : see `append`
        """
        # Check that the sparse matrices are compatible
        if not (type(self) is type(other)):
            raise ValueError(self.__class__.__name__ + f'.add requires other to be of same type: {other.__class__.__name__}')

        if self.dtype != other.dtype:
            raise ValueError(self.__class__.__name__ + '.add requires the same datatypes in the two matrices.')

        if self.dim != other.dim:
            raise ValueError(self.__class__.__name__ + '.add requires the same number of dimensions in the matrix.')

        if axis is None:
            geom = self.geometry.add(other.geometry, offset=offset)
        else:
            # Same effect but also adds the lattice vectors
            geom = self.geometry.append(other.geometry, axis, offset=offset)

        # Now we have the correct geometry, then create the correct
        # class
        # New indices and data (the constructor for SparseCSR copies)
        full = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())
        full._csr.ptr = concatenate((self._csr.ptr[:-1], other._csr.ptr))
        full._csr.ptr[self.no:] += self._csr.ptr[-1]
        full._csr.ncol = concatenate((self._csr.ncol, other._csr.ncol))
        full._csr._D = concatenate((self._csr._D, other._csr._D))
        full._csr._nnz = full._csr.ncol.sum()
        full._csr._finalized = False

        # Retrieve the maximum number of orbitals (in the supercell)
        # This may be used to remove couplings
        full_no_s = geom.no_s

        # Now we have to transfer the indices to the new sparse pattern

        # First create a local copy of the columns, then we transfer, and then we collect.
        s_col = self._csr.col.copy()
        transfer_idx = _a.arangei(self.geometry.no_s).reshape(-1, self.geometry.no)
        transfer_idx += _a.arangei(self.geometry.n_s).reshape(-1, 1) * other.geometry.no
        # Remove couplings along axis
        if not axis is None:
            idx = (self.geometry.sc.sc_off[:, axis] != 0).nonzero()[0]
            # Tell the routine to delete these indices
            transfer_idx[idx, :] = full_no_s + 1
        idx = array_arange(self._csr.ptr[:-1], n=self._csr.ncol)
        s_col[idx] = transfer_idx.ravel()[s_col[idx]]

        # Same for the other, but correct for deleted supercells and supercells along
        # disconnected auxiliary cells.
        o_col = other._csr.col.copy()
        transfer_idx = _a.arangei(other.geometry.no_s).reshape(-1, other.geometry.no)

        # Transfer the correct supercells
        o_idx = []
        s_idx = []
        idx_delete = []
        for isc, sc in enumerate(other.geometry.sc.sc_off):
            try:
                s_idx.append(self.geometry.sc.sc_index(sc))
                o_idx.append(isc)
            except ValueError:
                idx_delete.append(isc)
        # o_idx are transferred to s_idx
        transfer_idx[o_idx, :] += _a.arangei(1, other.geometry.n_s + 1)[s_idx].reshape(-1, 1) * self.geometry.no
        # Remove some columns
        transfer_idx[idx_delete, :] = full_no_s + 1
        # Clean-up to not confuse the rest of the algorithm
        del o_idx, s_idx, idx_delete

        # Now figure out if the supercells can be kept, at all...
        # find SC indices in other corresponding to self
        o_idx_uc = other.geometry.sc.sc_index([0] * 3)
        o_idx_sc = _a.arangei(other.geometry.sc.n_s)
        # Remove couplings along axis
        for i in range(3):
            if i == axis:
                idx = (other.geometry.sc.sc_off[:, axis] != 0).nonzero()[0]
            elif not allclose(geom.cell[i, :], other.cell[i, :]):
                # This will happen in case `axis` is None
                idx = (other.geometry.sc.sc_off[:, i] != 0).nonzero()[0]
            else:
                # When axis is not specified and cell parameters
                # are commensurate, then we will not change couplings
                continue
            # Tell the routine to delete these indices
            transfer_idx[idx, :] = full_no_s + 1

        idx = array_arange(other._csr.ptr[:-1], n=other._csr.ncol)
        o_col[idx] = transfer_idx.ravel()[o_col[idx]]

        # Now we need to decide whether the
        del transfer_idx, idx
        full._csr.col = concatenate([s_col, o_col])

        # Clean up (they could potentially be very large arrays)
        del s_col, o_col

        # Ensure we remove the elements
        full._csr._clean_columns()

        return full

    def prepend(self, other, axis, eps=0.005, scale=1):
        r""" See `append` for details

        This is currently equivalent to:

        >>> other.append(self, axis, eps, scale)
        """
        return other.append(self, axis, eps, scale)

    def append(self, other, axis, eps=0.005, scale=1):
        r""" Append `other` along `axis` to construct a new connected sparse matrix

        This method tries to append two sparse geometry objects together by
        the following these steps:

        1. Create the new extended geometry
        2. Use neighbor cell couplings from `self` as the couplings to `other`
           This *may* cause problems if the coupling atoms are not exactly equi-positioned.
           If the coupling coordinates and the coordinates in `other` differ by more than
           0.01 Ang, a warning will be issued.
           If this difference is above `eps` the couplings will be removed.

        When appending sparse matrices made up of atoms, this method assumes that
        the orbitals on the overlapping atoms have the same orbitals, as well as the
        same orbital ordering.

        Examples
        --------
        >>> sporb = SparseOrbital(....)
        >>> sporb2 = sporb.append(sporb, 0)
        >>> sporbt = sporb.tile(2, 0)
        >>> sporb2.spsame(sporbt)
        True

        To retain couplings only from the *left* sparse matrix, do:

        >>> sporb = left.append(right, 0, scale=(2, 0))
        >>> sporb = (sporb + sporb.transpose()) * 0.5

        To retain couplings only from the *right* sparse matrix, do:

        >>> sporb = left.append(right, 0, scale=(0, 2.))
        >>> sporb = (sporb + sporb.transpose()) * 0.5

        Notes
        -----
        The current implementation does not preserve the hermiticity of the matrix.
        If you want to preserve hermiticity of the matrix you have to do the
        following:

        >>> sm = (sm + sm.transpose()) / 2

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
        scale : float or array_like, optional
            the scale used for the overlapping region. For scalar values it corresponds
            to passing: ``(scale, scale)``.
            For array-like input ``scale[0]`` refers to the scale of the matrix elements
            coupling from `self`, while ``scale[1]`` is the scale of the matrix elements
            in `other`.

        See Also
        --------
        prepend : equivalent scheme as this method
        add : merge two matrices without considering overlap or commensurability
        transpose : ensure hermiticity by using this routine
        replace : replace a sub-set of atoms with another sparse matrix
        Geometry.append
        Geometry.prepend
        SparseCSR.scale_columns : method used to scale the two matrix elements values

        Raises
        ------
        ValueError
            if the two geometries are not compatible for either coordinate, orbital or supercell errors

        Returns
        -------
        object
            a new instance with two sparse matrices joined and appended together
        """
        if not (type(self) is type(other)):
            raise ValueError(f"{self.__class__.__name__}.append requires other to be of same type: {other.__class__.__name__}")

        if self.geometry.nsc[axis] > 3 or other.geometry.nsc[axis] > 3:
            raise ValueError(f"{self.__class__.__name__}.append requires sparse-geometries to maximally "
                             "have 3 supercells along appending axis.")

        if not allclose(self.geometry.nsc, other.geometry.nsc):
            raise ValueError(f"{self.__class__.__name__}.append requires sparse-geometries to have the same "
                             "number of supercells along all directions.")

        if not allclose(self.geometry.sc._isc_off, other.geometry.sc._isc_off):
            raise ValueError(f"{self.__class__.__name__}.append requires supercell offsets to be the same.")

        if self.dtype != other.dtype:
            raise ValueError(f"{self.__class__.__name__}.append requires the same datatypes in the two matrices.")

        if self.dim != other.dim:
            raise ValueError(f"{self.__class__.__name__}.append requires the same number of dimensions in the matrix.")

        if np.asarray(scale).size == 1:
            scale = np.array([scale, scale])
        scale = np.asarray(scale)

        # Our procedure will be to separate the sparsity patterns into separate chunks
        # First generate the full geometry
        geom = self.geometry.append(other.geometry, axis)

        # create the new sparsity patterns with offset

        # New indices and data (the constructor for SparseCSR copies)
        full = self.__class__(geom, self.dim, self.dtype, 1, **self._cls_kwargs())
        full._csr.ptr = concatenate((self._csr.ptr[:-1], other._csr.ptr))
        full._csr.ptr[self.no:] += self._csr.ptr[-1]
        full._csr.ncol = concatenate((self._csr.ncol, other._csr.ncol))
        full._csr._D = concatenate((self._csr._D, other._csr._D))
        full._csr._nnz = full._csr.ncol.sum()
        full._csr._finalized = False

        # First create a local copy of the columns, then we transfer, and then we collect.
        s_col = self._csr.col.copy()
        # transfer
        transfer_idx = _a.arangei(self.geometry.no_s).reshape(-1, self.geometry.no)
        transfer_idx += _a.arangei(self.geometry.n_s).reshape(-1, 1) * other.geometry.no
        idx = array_arange(self._csr.ptr[:-1], n=self._csr.ncol)
        s_col[idx] = transfer_idx.ravel()[s_col[idx]]

        o_col = other._csr.col.copy()
        # transfer
        transfer_idx = _a.arangei(other.geometry.no_s).reshape(-1, other.geometry.no)
        transfer_idx += _a.arangei(1, other.geometry.n_s + 1).reshape(-1, 1) * self.geometry.no
        idx = array_arange(other._csr.ptr[:-1], n=other._csr.ncol)
        o_col[idx] = transfer_idx.ravel()[o_col[idx]]

        # Store all column indices
        del transfer_idx, idx
        full._csr.col = concatenate((s_col, o_col))

        # Clean up (they could potentially be very large arrays)
        del s_col, o_col

        # Now everything is contained in 1 sparse matrix.
        # All matrix elements are as though they are in their own

        # What needs to be done is to find the overlapping atoms and transfer indices in
        # both these sparsity patterns to the correct elements.

        # 1. find overlapping atoms along axis
        idx_s_first, idx_o_first = self.geometry.overlap(other.geometry, eps=eps)
        idx_s_last, idx_o_last = self.geometry.overlap(other.geometry, eps=eps,
                                                       offset=-self.geometry.sc.cell[axis, :],
                                                       offset_other=-other.geometry.sc.cell[axis, :])
        # IFF idx_s_* contains duplicates, then we have multiple overlapping atoms which is not
        # allowed
        def _test(diff):
            if diff.size != diff.nonzero()[0].size:
                raise ValueError(f"{self.__class__.__name__}.append requires that there is maximally one "
                                 "atom overlapping one other atom in the other structure.")
        _test(diff(idx_s_first))
        _test(diff(idx_s_last))
        # Also ensure that atoms have the same number of orbitals in the two cases
        if (not allclose(self.geometry.orbitals[idx_s_first], other.geometry.orbitals[idx_o_first])) or \
           (not allclose(self.geometry.orbitals[idx_s_last], other.geometry.orbitals[idx_o_last])):
            raise ValueError(f"{self.__class__.__name__}.append requires the overlapping geometries "
                             "to have the same number of orbitals per atom that is to be replaced.")

        def _check_edges_and_coordinates(spgeom, atoms, isc, err_help):
            # Figure out if we have found all couplings
            geom = spgeom.geometry
            # Find orbitals that we wish to exclude from the orbital connections
            # This ensures that we only find couplings crossing the supercell boundaries
            irrelevant_sc = delete(_a.arangei(geom.sc.n_s), geom.sc.sc_index(isc))
            sc_orbitals = _a.arangei(geom.no_s).reshape(geom.sc.n_s, -1)
            exclude = sc_orbitals[irrelevant_sc, :].ravel()
            # get connections and transfer them to the unit-cell
            edges_sc = geom.o2a(spgeom.edges(orbitals=_a.arangei(geom.no), exclude=exclude), True)
            edges_uc = geom.sc2uc(edges_sc, True)
            edges_valid = np.isin(edges_uc, atoms, assume_unique=True)
            if not np.all(edges_valid):
                edges_uc = edges_sc % geom.na
                # Reduce edges to those that are faulty
                edges_valid = np.isin(edges_uc, atoms, assume_unique=False)
                errors = edges_sc[~edges_valid]
                # Get supercell offset and unit-cell atom
                isc_off, uca = np.divmod(errors, geom.na)
                # group atoms for each supercell index
                # find unique supercell offsets
                sc_off_atoms = []
                # This will be much faster
                for isc in unique(isc_off):
                    idx = (isc_off == isc).nonzero()[0]
                    sc_off_atoms.append("{k}: {v}".format(
                        k=str(geom.sc.sc_off[isc]),
                        v=list2str(np.sort(uca[idx]))))
                sc_off_atoms = "\n   ".join(sc_off_atoms)
                raise ValueError(f"{self.__class__.__name__}.append requires matching coupling elements.\n\n"
                                 f"The following atoms in a {err_help[1]} connection of `{err_help[0]}` super-cell "
                                 "are connected from the unit cell, but are not found in matches:\n\n"
                                 f"[sc-offset]: atoms\n   {sc_off_atoms}")

        # setup supercells to look up
        isc_inplace = [None] * 3
        isc_inplace[axis] = 0
        isc_forward = isc_inplace.copy()
        isc_forward[axis] = 1
        isc_back = isc_inplace.copy()
        isc_back[axis] = -1

        # Check that edges and overlapping atoms are the same (or at least that the
        # edges are all in the overlapping region)
        # [self|other]: self sc-connections forward must be on left-aligned matching atoms
        _check_edges_and_coordinates(self, idx_s_first, isc_forward, err_help=("self", "forward"))
        # [other|self]: other sc-connections forward must be on left-aligned matching atoms
        _check_edges_and_coordinates(other, idx_o_first, isc_forward, err_help=("other", "forward"))
        # [other|self]: self sc-connections backward must be on right-aligned matching atoms
        _check_edges_and_coordinates(self, idx_s_last, isc_back, err_help=("self", "backward"))
        # [self|other]: other sc-connections backward must be on right-aligned matching atoms
        _check_edges_and_coordinates(other, idx_o_last, isc_back, err_help=("other", "backward"))

        # Now we have ensured that the overlapping coordinates and the connectivity graph
        # co-incide and that we can actually perform the merge.
        idx = _a.arangei(geom.n_s).reshape(-1, 1) * geom.no

        def _sc_index_sort(isc):
            idx = geom.sc.sc_index(isc)
            # Now sort so that all indices are corresponding one2one
            # This is important since two different supercell indices
            # need not be sorted in the same manner.
            # This ensures that there is a correspondance between
            # two different sparse elements
            off = delete(geom.sc.sc_off[idx].T, axis, axis=0)
            return idx[np.lexsort(off)]

        idx_iscP = idx[_sc_index_sort(isc_forward)]
        idx_isc0 = idx[_sc_index_sort(isc_inplace)]
        idx_iscM = idx[_sc_index_sort(isc_back)]
        # Clean (for me to know what to do in this code)
        del idx, _sc_index_sort

        # First scale all values
        idx_s_first = self.geometry.a2o(idx_s_first, all=True).reshape(1, -1)
        idx_s_last = self.geometry.a2o(idx_s_last, all=True).reshape(1, -1)
        col = concatenate(((idx_s_first + idx_iscP).ravel(),
                           (idx_s_last + idx_iscM).ravel()))
        full._csr.scale_columns(col, scale[0])

        idx_o_first = other.geometry.a2o(idx_o_first, all=True).reshape(1, -1) + self.geometry.no
        idx_o_last = other.geometry.a2o(idx_o_last, all=True).reshape(1, -1) + self.geometry.no
        col = concatenate(((idx_o_first + idx_iscP).ravel(),
                           (idx_o_last + idx_iscM).ravel()))
        full._csr.scale_columns(col, scale[1])

        # Clean up (they may be very large)
        del col

        # Now we can easily build from->to arrays

        # other[0] -> other[1] changes to other[0] -> full_G[1] | self[1]
        # self[0] -> self[1] changes to self[0] -> full_G[0] | other[0]
        # self[0] -> self[-1] changes to self[0] -> full_G[-1] | other[-1]
        # other[0] -> other[-1] changes to other[0] -> full_G[0] | self[0]
        col_from = concatenate(((idx_o_first + idx_iscP).ravel(),
                                (idx_s_first + idx_iscP).ravel(),
                                (idx_s_last + idx_iscM).ravel(),
                                (idx_o_last + idx_iscM).ravel()))
        col_to = concatenate(((idx_s_first + idx_iscP).ravel(),
                              (idx_o_first + idx_isc0).ravel(),
                              (idx_o_last + idx_iscM).ravel(),
                              (idx_s_last + idx_isc0).ravel()))

        full._csr.translate_columns(col_from, col_to)
        return full

    def replace(self, atoms, other, other_atoms=None, eps=0.005, scale=1.):
        r""" Replace `atoms` in `self` with `other_atoms` in `other` and retain couplings between them

        This method replaces a subset of atoms in `self` with
        another sparse geometry retaining any couplings between them.
        The algorithm checks whether the coupling atoms have the same number of
        orbitals. Meaning that atoms in the overlapping region should have the same
        connections and number of orbitals per atom.
        It will _not_ check whether the orbitals or atoms _are_ the same, nor the order
        of the orbitals.

        Examples
        --------
        >>> minimal = SparseOrbital(....)
        >>> big = minimal.tile(2, 0)
        >>> big2 = big.replace(np.arange(big.na), minimal)
        >>> big.spsame(big2)
        True

        To retain couplings only from the ``big`` sparse matrix, one should
        do the following (note the subsequent transposing which ensures hermiticy
        and is effectively copying couplings from ``big`` to the replaced region.

        >>> big2 = big.replace(np.arange(big.na), minimal, scale=(2, 0))
        >>> big2 = (big2 + big2.transpose()) * 0.5

        To only retain couplings from the ``minimal`` sparse matrix:

        >>> big2 = big.replace(np.arange(big.na), minimal, scale=(0, 2))
        >>> big2 = (big2 + big2.transpose()) * 0.5

        Notes
        -----
        The current implementation does not preserve the hermiticity of the matrix.
        If you want to preserve hermiticity of the matrix you have to do the
        following:

        >>> sm = (sm + sm.transpose()) / 2

        Also note that the ordering of the atoms will be ``range(atoms.min()), range(len(other_atoms)), <rest>``.
        So algorithms using atomic indices should be careful.

        Parameters
        ----------
        atoms : array_like
            which atoms in `self` that are removed and replaced with ``other.sub(other_atoms)``
        other : object
            must be an object of the same type as `self`, a subset is taken from this
            sparse matrix and combined with `self` to create a new sparse matrix
        other_atoms : array_like, optional
            to select a subset of atoms in `other` that are taken out.
            Defaults to all atoms.
        eps : float, optional
            coordinate tolerance to allow a replacement.
            It is important that this value is smaller than half the distance between
            the two closests atoms such that there is no ambiguity in selecting
            equivalent atoms.
        scale : float or array_like, optional
            the scale used for the overlapping region. For scalar values it corresponds
            to passing: ``(scale, scale)``.
            For array-like input ``scale[0]`` refers to the scale of the matrix elements
            coupling from `self`, while ``scale[1]`` is the scale of the matrix elements
            in `other`.

        See Also
        --------
        prepend : equivalent scheme as this method
        add : merge two matrices without considering overlap or commensurability
        transpose : ensure hermiticity by using this routine
        append : append two sparse matrices
        Geometry.append
        Geometry.prepend
        SparseCSR.scale_columns : method used to scale the two matrix elements values

        Raises
        ------
        ValueError
           if the two geometries are not compatible for either coordinate, orbital or supercell errors
        AssertionError
           if the two geometries are not compatible for either coordinate, orbital or supercell errors


        Warns
        -----
        SislWarning
           in case the overlapping atoms are not comprising the same atomic specie. In some cases this may not be a problem. However, care must be taken by the user if this warning is issued.

        Returns
        -------
        object
            a new instance with two sparse matrices merged together by removing and adding
        """
        if np.asarray(scale).size == 1:
            scale = np.array([scale, scale])
        scale = np.asarray(scale)

        # here our connection is defined as what is connected to "in"
        # and what is connected to "out"
        # Say 0 -> 1
        # And `atoms` is [0].
        # Then in = [0], out = [1]
        # since atoms connect out to [1]

        # figure out the atoms that needs replacement
        def get_reduced_system(sp, atoms):
            """ convert the geometry in `sp` to only atoms `atoms` and return the following:

            1. atoms (sanitized and no order change)
            2. orbitals (ordered as `atoms`
            3. the atoms that are connected to OUT and IN
            4. the orbitals that are connected to OUT and IN
            """
            geom = sp.geometry
            atoms = _a.asarrayi(geom._sanitize_atoms(atoms)).ravel()
            if unique(atoms).size != atoms.size:
                raise ValueError(f"{self.__class__.__name__}.replace requires a unique set of atoms")
            orbs = geom.a2o(atoms, all=True)
            other_orbs = geom.ouc2sc(np.delete(_a.arangei(geom.no), orbs))

            # Find the orbitals that these atoms connect to such that we can compare
            # atomic coordinates
            out_connect_orb_sc = sp.edges(orbitals=orbs, exclude=orbs)
            out_connect_orb = geom.osc2uc(out_connect_orb_sc, True)
            out_connect_atom_sc = geom.o2a(out_connect_orb_sc, True)
            out_connect_atom = geom.asc2uc(out_connect_atom_sc, True)

            # figure out connecting back
            atoms_orbs = list(map(_a.arangei, geom.firsto[atoms], geom.firsto[atoms+1]))
            in_connect_atom = []
            in_connect_orb = []

            for atom, atom_orbs in zip(atoms, atoms_orbs):
                edges = sp.edges(orbitals=atom_orbs, exclude=orbs)
                if len(intersect1d(edges, out_connect_orb_sc)) > 0:
                    in_connect_atom.append(atom)
                    in_connect_orb.append(atom_orbs)

            in_connect_atom = _a.arrayi(in_connect_atom)
            in_connect_orb = concatenate(in_connect_orb)

            # create the connection tables
            atom_uc = Connect(in_connect_atom, out_connect_atom)
            atom_sc = Connect(in_connect_atom, out_connect_atom_sc)
            orb_uc = Connect(in_connect_orb, out_connect_orb)
            orb_sc = Connect(in_connect_orb, out_connect_orb_sc)
            atom_connect = UCSC(atom_uc, atom_sc)
            orb_connect = UCSC(orb_uc, orb_sc)

            return Info(atoms, orbs, atom_connect, orb_connect)

        UCSC = namedtuple("UCSC", ["uc", "sc"])
        Connect = namedtuple("Connect", ["IN", "OUT"])
        Info = namedtuple("Info", ["atoms", "orbitals", "atom_connect", "orb_connect"])

        sgeom = self.geometry
        s_info = get_reduced_system(self, atoms)
        atoms = s_info.atoms # sanitized (no order change)

        ogeom = other.geometry
        o_info = get_reduced_system(other, other_atoms)
        other_atoms = o_info.atoms # sanitized (no order change)

        # Get overlapping atoms by their offset
        # We need to get a 1-1 correspondance between the two connecting geometries
        # For instance `self` may be ordered differently than `other`.
        # So we need to figure out how the atoms are arranged in *both* regions.
        # This is where `eps` comes into play since we have to ensure that the
        # connecting regions are within some given tolerance.

        def create_geometry(geom, atoms):
            """ Create the supercell geometry with coordinates as given """
            xyz = geom.axyz(atoms)
            uc_atoms = geom.sc2uc(atoms)
            return Geometry(xyz, atoms=geom.atoms[uc_atoms])

        # We know that the *IN* connections are in the primary unit-cell
        # so we don't need to handle supercell information
        # Atoms *inside* the replacement region that couples out
        sgeom_in = sgeom.sub(s_info.atom_connect.uc.IN)
        ogeom_in = ogeom.sub(o_info.atom_connect.uc.IN)
        soverlap_in, ooverlap_in = sgeom_in.overlap(ogeom_in, eps=eps,
                                                    offset=-sgeom_in.xyz.min(0),
                                                    offset_other=-ogeom_in.xyz.min(0))

        # Not replacement region, i.e. the IN (above) atoms are connecting to
        # these atoms:
        sgeom_out = create_geometry(sgeom, s_info.atom_connect.sc.OUT)
        ogeom_out = create_geometry(ogeom, o_info.atom_connect.sc.OUT)
        soverlap_out, ooverlap_out = sgeom_out.overlap(ogeom_out, eps=eps,
                                                       offset=-sgeom_out.xyz.min(0),
                                                       offset_other=-ogeom_out.xyz.min(0))

        # trigger for errors
        err_msg = ""

        # Now we have the different geometries around to handle how the merging
        # process.
        # Before proceeding we will check whether the dimensions match.
        # I.e. checking that the orbitals connecting in/out are the same is important.

        #print("in:")
        #print(s_info.atom_connect.uc.IN)
        #print(soverlap_in)
        #print(o_info.atom_connect.uc.IN)
        #print(ooverlap_in)
        if not (len(sgeom_in) == len(soverlap_in) and
                len(ogeom_in) == len(ooverlap_in)):

            # figure out which atoms are not connecting
            s_diff = np.setdiff1d(np.arange(s_info.atom_connect.uc.IN.size),
                                     soverlap_in)
            o_diff = np.setdiff1d(np.arange(o_info.atom_connect.uc.IN.size),
                                     ooverlap_in)
            if len(s_diff) > 0 or len(o_diff) > 0:
                err_msg = f"""{err_msg}

The number of atoms in the replacement region that connects to the surrounding
atoms are not the same in 'self' and 'other'.
This means that the number of connections is not the same. Please ensure this."""

            if len(s_diff) > 0:
                err_msg = f"""{err_msg}

self: atoms not matched in 'other': {s_info.atom_connect.uc.IN[s_diff]}."""
            if len(o_diff) > 0:
                err_msg = f"""{err_msg}

other: atoms not matched in 'self': {o_info.atom_connect.uc.IN[o_diff]}."""

        elif not np.allclose(sgeom_in.orbitals[soverlap_in],
                             ogeom_in.orbitals[ooverlap_in]):
            err_msg = f"""{err_msg}

Atoms in the replacement region have different number of orbitals on the atoms
that lie at the border.

self orbitals:
   {sgeom_in.orbitals[soverlap_in]}
other orbitals:
   {ogeom_in.orbitals[ooverlap_in]}"""

        #print("out:")
        #print(s_info.atom_connect.uc.OUT)
        #print(soverlap_out)
        #print(o_info.atom_connect.uc.OUT)
        #print(ooverlap_out)

        # [so]overlap_out are now in the order of [so]_info.atom_connect.out
        # so we still have to convert them to proper indices if used
        # We cannot really check the soverlap_out == len(sgeom_out)
        # in case we have a replaced sparse matrix in the middle of another bigger
        # sparse matrix.
        if not (len(sgeom_out) == len(soverlap_out) and
                len(ogeom_out) == len(ooverlap_out)):

            # figure out which atoms are not connecting
            s_diff = np.setdiff1d(np.arange(s_info.atom_connect.sc.OUT.size),
                                     soverlap_out)
            o_diff = np.setdiff1d(np.arange(o_info.atom_connect.sc.OUT.size),
                                     ooverlap_out)
            if len(s_diff) > 0 or len(o_diff) > 0:
                err_msg = f"""{err_msg}

Number of atoms connecting to the replacement region are not the same in 'self' and 'other'.
Please ensure this."""

            if len(s_diff) > 0:
                err_msg = f"""{err_msg}

self: atoms (in supercell) connecting to 'atoms' not matched in 'other': {s_info.atom_connect.sc.OUT[s_diff]}."""
            if len(o_diff) > 0:
                err_msg = f"""{err_msg}

other: atoms (in supercell) connecting to 'other_atoms' not matched in 'self': {o_info.atom_connect.sc.OUT[o_diff]}."""

        elif not np.allclose(sgeom_out.orbitals[soverlap_out],
                             ogeom_out.orbitals[ooverlap_out]):
            err_msg = f"""{err_msg}

Atoms in the connection region have different number of orbitals on the atoms.

self orbitals:
   {sgeom_out.orbitals[soverlap_out]}
other orbitals:
   {ogeom_out.orbitals[ooverlap_out]}"""

        # we can only ensure the orbitals that connect *out* have the same count
        # For supercell connections hopping *IN* might be different due to the supercell
        if len(s_info.orb_connect.sc.OUT) != len(o_info.orb_connect.sc.OUT) and not err_msg:
            err_msg = f"""{err_msg}

Number of orbitals connecting to replacement region is not consistent
between 'self' and 'other'."""

        if err_msg:
            raise ValueError(err_msg[1:])

        warn_msg = ""
        S_ = s_info.atom_connect.uc.IN
        O_ = o_info.atom_connect.uc.IN
        for s_, o_ in zip(soverlap_in, ooverlap_in):
            if sgeom_in.atoms[s_] != ogeom_in.atoms[o_]:
                warn_msg = f"""{warn_msg}
Atom 'self[{S_[s_]}]' is not equivalent to 'other[{O_[o_]}]':
  {sgeom_in.atoms[s_]}  !=  {ogeom_in.atoms[o_]}"""

        if warn_msg:
            warn(f"""Inequivalent atoms found in replacement region, this may or may not be a problem
depending on your use case. Please be careful though.{warn_msg}""")

        warn_msg = ""
        S_ = s_info.atom_connect.sc.OUT
        O_ = o_info.atom_connect.sc.OUT
        checked1d = _a.zerosi([self.geometry.na])
        for s_, o_ in zip(soverlap_out, ooverlap_out):
            uc_s_ = S_[s_] % self.geometry.na
            if sgeom_out.atoms[s_] != ogeom_out.atoms[o_] and checked1d[uc_s_] == 0:
                checked1d[uc_s_] = 1
                warn_msg = f"""{warn_msg}
Atom 'self[{S_[s_]}]' is not equivalent to 'other[{O_[o_]}]':
  {sgeom_out.atoms[s_]}  !=  {ogeom_out.atoms[o_]}"""

        if warn_msg:
            warn(f"""Inequivalent atoms found in connection region, this may or may not be a problem
depending on your use case. Note indices in the following are supercell indices. Please be careful though.{warn_msg}""")

        # clean-up to make it clear that we are not going to use them.
        del sgeom_out, ogeom_out

        # this is where other.sub(other_atoms) gets inserted
        ainsert_idx = atoms.min()
        oinsert_idx = sgeom.a2o(ainsert_idx)
        # this is the indices of the new atoms in the new geometry
        self_other_atoms = _a.arangei(ainsert_idx, ainsert_idx + len(other_atoms))

        # We need to do the replacement in two steps
        # A. the geometry
        #    This will insert other at ainsert_idx
        #    Note that sub(other_atoms) re-arranges the atoms correctly
        idx = np.argmin((sgeom_in.xyz[soverlap_in] ** 2).sum(1))
        offset = sgeom_in.xyz[soverlap_in[idx]] - ogeom_in.xyz[ooverlap_in[idx]]
        # this will perhaps re-order atoms from other_atoms
        geom = sgeom.replace(atoms, other.geometry.sub(other_atoms), offset=offset)
        del sgeom_in, ogeom_in
        # A. DONE

        # B. Merge the two sparse patterns
        scsr = self._csr
        ncol = scsr.ncol
        col = scsr.col
        D = scsr._D
        # helper function

        def a2o(geom, atoms, sc=True):
            if sc:
                return geom.ouc2sc(geom.a2o(atoms, all=True))
            return geom.a2o(atoms, all=True)

        # Our first task is to merge the two sparse patterns.
        # Delete the *old* values
        # To ensure that inserting will not leave *empty* values
        # we first reduce arrays so that the ptr array is not needed
        ncol = delete(ncol, s_info.orbitals)
        ptr = delete(scsr.ptr, s_info.orbitals)
        idx = array_arange(ptr[:-1], n=ncol)
        col = col[idx]
        D = D[idx]

        # Do the same reduction for the inserted values
        ocsr = other._csr
        idx = array_arange(ocsr.ptr[o_info.orbitals], n=ocsr.ncol[o_info.orbitals])
        # we offset the new columns by self.shape[1], in this way we know
        # which couplings belong to the inserted and the original csr
        col = insert(col, ncol[:oinsert_idx].sum(), ocsr.col[idx] + self.shape[1])
        D = insert(D, ncol[:oinsert_idx].sum(), ocsr._D[idx], axis=0)
        ncol = insert(ncol, oinsert_idx, ocsr.ncol[o_info.orbitals])

        # Create the sparse pattern
        csr = SparseCSR((D, col, _ncol_to_indptr(ncol)),
                        shape=(geom.no, sgeom.no_s + ogeom.no_s, D.shape[1]))
        del D, col, ncol

        # Now we have merged the two sparse patterns
        # But we need to correct the orbital couplings
        # : *outside* refers to the original sparse pattern (without `atoms`)
        # : *inside* refers to the inserted sparse pattern (other.sub(other_atoms))
        # We have to do 1 and 2 simultaneously.
        # We have to do 3 and 4 simultaneously.
        # This is because they may have overlapping columns

        # 1: couplings from *outside* to *outside* (no scale)
        # 2: couplings from *outside* to *inside* (scaled)
        # 3: couplings from *inside* to *inside* (no scale)
        # 4: couplings from *inside* to *outside* (scaled)
        convert = [[], []]
        conc = np.concatenate

        def assert_unique(old, new):
            old = conc(old)
            new = conc(new)
            assert len(unique(old)) == len(old)
            assert len(unique(new)) == len(new)
            return old, new

        # 1:
        #print("1:")
        old = delete(_a.arangei(len(sgeom)), atoms)
        new = _a.arangei(len(old))
        new[ainsert_idx:] += len(other_atoms)
        old = a2o(sgeom, old)
        convert[0].append(old)
        new = a2o(geom, new)
        convert[1].append(new)
        rows = geom.osc2uc(new, unique=True)

        # 2:
        #print("2:")
        old = s_info.atom_connect.uc.IN[soverlap_in]
        # algorithm to get indices in other_atoms
        new = o_info.atom_connect.uc.IN[ooverlap_in]
        tmp = argsort(other_atoms)
        new = tmp[searchsorted(other_atoms, new, sorter=tmp)] + ainsert_idx
        old = a2o(sgeom, old)
        convert[0].append(old)
        new = a2o(geom, new)
        convert[1].append(new)

        # translate columns
        csr.translate_columns(*assert_unique(convert[0], convert[1]), rows=rows)
        # scale columns that connects inside
        csr.scale_columns(convert[1][1], scale=scale[0], rows=rows)

        # on to the *inside* 3, 4
        convert = [[], []]

        # 3:
        #print("3:")
        # we have all the *inside* column indices offset by self.shape[1]
        old = a2o(ogeom, other_atoms, False) + self.shape[1]
        new = ainsert_idx + _a.arangei(len(other_atoms))
        #print("old: ", old)
        #print("new: ", new)
        new = a2o(geom, new, False)
        convert[0].append(old)
        convert[1].append(new)
        rows = geom.osc2uc(new, unique=True)

        # 4:
        #print("4:")
        old = o_info.atom_connect.sc.OUT
        new = _a.emptyi(len(old))
        for i, atom in enumerate(old):
            idx = geom.close(ogeom.axyz(atom) + offset, R=eps)
            assert len(idx) == 1, f"More than 1 atom {idx} for atom {atom} = {ogeom.axyz(atom)}, {geom.axyz(idx)}"
            new[i] = idx[0]
        #print("old: ", old)
        #print("new: ", new)
        old = a2o(ogeom, old, False) + self.shape[1]
        new = a2o(geom, new, False)

        convert[0].append(old)
        convert[1].append(new)

        # translate columns
        csr.translate_columns(*assert_unique(convert[0], convert[1]), rows=rows)
        # scale columns that connects inside
        csr.scale_columns(convert[1][1], scale=scale[1], rows=rows)

        # ensure we have translated all columns correctly
        assert len((csr.col >= geom.no_s).nonzero()[0]) == 0
        # correct shape of column matrix
        csr._shape = (csr.shape[0], geom.no_s, csr.shape[2])
        out = self.copy()
        out._csr = csr
        out._geometry = geom
        return out

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
        spAtom._csr.ncol[:] = diff(ptr)
        spAtom._csr.col = col
        spAtom._csr._D = np.zeros([len(col), dim], dtype=dtype)
        spAtom._csr._nnz = len(col)
        spAtom._csr._finalized = True # unique returns sorted elements
        return spAtom
