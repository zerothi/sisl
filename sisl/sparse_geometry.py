"""
A generic sparse matrix which is based on a hosting `Geometry`.

The sparse matrix can in this case represent _any_ data and should be
sub-classed for specific uses.
"""
from __future__ import print_function, division

import warnings
from numbers import Integral
import itertools as itools

import numpy as np
import scipy.linalg as sli
from scipy.sparse import isspmatrix, csr_matrix
import scipy.sparse.linalg as ssli

from sisl._help import get_dtype, ensure_array
from sisl._help import _zip as zip, _range as range
from sisl.sparse import SparseCSR, ispmatrix, ispmatrixd

__all__ = ['SparseGeometry', 'SparseAtom', 'SparseOrbital']


class SparseGeometry(object):
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
        raise NotImplementedError

    def _cls_kwargs(self):
        """ Custom keyword arguments when creating a new instance """
        return {}

    def reset(self, dim=1, dtype=np.float64, nnzpr=None):
        """
        The sparsity pattern is cleaned and every thing
        is reset.

        The object will be the same as if it had been
        initialized with the same geometry as it were
        created with.

        Parameters
        ----------
        dim: int, optional
           number of dimensions per element
        dtype: numpy.dtype, optional
           the datatype of the sparse elements
        nnzpr: int, optional
           number of non-zero elements per row
        """
        # I know that this is not the most efficient way to
        # access a C-array, however, for constructing a
        # sparse pattern, it should be faster if memory elements
        # are closer...

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

    def empty(self, keep=False):
        """ See `SparseCSR.empty` for details """
        self._csr.empty(keep)

    def copy(self, dtype=None):
        """ A copy of this object 

        Parameters
        ----------
        dtype : numpy.dtype, optional
           it is possible to convert the data to a different data-type
           If not specified, it will use `self.dtype`
        """
        if dtype is None:
            dtype = self.dtype
        new = self.__class__(self.geom, self.dim, dtype, 1, **self._cls_kwargs())
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

    def __repr__(self):
        """ Representation of the sparse model """
        s = self.__class__.__name__ + '{{dim: {0}, non-zero: {1}\n '.format(self.dim, self.nnz)
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

    def __getitem__(self, key):
        """ Elements for the index(s) """
        dd = self._def_dim
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
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        self._csr[key] = val

    def __contains__(self, key):
        """ Check whether a sparse index is non-zero """
        return key in self._csr

    def align(self, other):
        """ See ``SparseCSR.align`` for details """
        if isinstance(other, SparseCSR):
            self._csr.align(other)
        else:
            self._csr.align(other._csr)

    def eliminate_zeros(self):
        """ Removes all zero elements from the sparse matrix

        This is an *in-place* operation
        """
        self._csr.eliminate_zeros()

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
        >>>     idx = self.geom.close(ia, R=R, idx=idxs)
        >>>     for ix, p in zip(idx, param):
        >>>         self[ia, ix] = p

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
           1. Is this object (`self`)
           2. Is the currently examined atom (`ia`)
           3. Is the currently bounded indices (`idxs`)
           4. Is the currently bounded indices atomic coordinates (`idxs_xyz`)
           An example `func` could be:

           >>> def func(self, ia, idxs, idxs_xyz=None):
           ...     idx = self.geom.close(ia, R=[0.1, 1.44], idx=idxs, idx_xyz=idxs_xyz)
           ...     self[ia, idx[0]] = 0
           ...     self[ia, idx[1]] = -2.7

        na_iR : int, optional
           number of atoms within the sphere for speeding
           up the `iter_block` loop.
        method : {'rand', str, 'rand'
           method used in `Geometry.iter_block`, see there for details
        eta: bool, False
           whether an ETA will be printed
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
                m, s = divmod(float(time()-t0)/na_run * na, 60)
                h, m = divmod(m, 60)
                stdout.write(name + ".construct() ETA = {0:5d}h {1:2d}m {2:5.2f}s\r".format(int(h), int(m), s))
                stdout.flush()

        if eta:
            # calculate hours, minutes, seconds spend on the computation
            m, s = divmod(float(time()-t0), 60)
            h, m = divmod(m, 60)
            stdout.write(name + ".construct() finished after {0:d}h {1:d}m {2:.1f}s\n".format(int(h), int(m), s))
            stdout.flush()

    @property
    def finalized(self):
        """ Whether the contained data is finalized and non-used elements have been removed """
        return self._csr.finalized

    def finalize(self):
        """ Finalizes the model

        Finalizes the model so that all non-used elements are removed. I.e. this simply reduces the memory requirement for the sparse matrix.

        Note that adding more elements to the sparse matrix is more time-consuming than for an non-finalized sparse matrix due to the
        internal data-representation.
        """
        self._csr.finalize()

    def tocsr(self, index, isc=None):
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
        return self._csr.tocsr(index)

    def spsame(self, other):
        """ Compare two sparse objects and check whether they have the same entries.

        This does not necessarily mean that the elements are the same
        """
        return self._csr.spsame(other._csr)

    def _init_larger(self, method, size, axis):
        """ Internal routine to start a bigger sparse model """
        # Create the new geometry
        g = getattr(self.geom, method)(size, axis)

        # Now create the new Hamiltonian
        # First figure out the initialization parameters
        nnzpr = np.amax(self._csr.ncol)
        dim = self.dim
        dtype = self.dtype
        return self.__class__(g, dim, dtype, nnzpr, **self._cls_kwargs())

    @classmethod
    def fromsp(cls, geom, sp):
        """ Returns a sparse model from a preset Geometry and a list of sparse matrices """
        # Calculate maximum number of connections per row
        nc = 0

        # Ensure list of csr format (to get dimensions)
        if isspmatrix(sp):
            sp = [sp]

        # Number of dimensions
        dim = len(sp)
        # Sort all indices for the passed sparse matrices
        for i in range(dim):
            sp[i] = sp[i].tocsr()
            sp[i].sort_indices()

        # Figure out the maximum connections per
        # row to reduce number of re-allocations to 0
        for i in range(sp[0].shape[0]):
            nc = max(nc, sp[0][i, :].getnnz())

        # Create the sparse object
        S = cls(geom, dim, sp[0].dtype, nc, **self._cls_kwargs())

        for i in range(dim):
            for jo, io, v in ispmatrixd(sp[i]):
                S[jo, io, i] = v

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
        if isinstance(b, SparseGeometry):
            a._csr += b._csr
        else:
            a._csr += b
        return a

    def __sub__(a, b):
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c -= b
        return c

    def __rsub__(a, b):
        if isinstance(b, SparseGeometry):
            c = b.copy(dtype=get_dtype(a, other=b.dtype))
            c._csr += -1 * a._csr
        else:
            c = b + (-1) * a
        return c

    def __isub__(a, b):
        if isinstance(b, SparseGeometry):
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
        if isinstance(b, SparseGeometry):
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
        if isinstance(b, SparseGeometry):
            a._csr /= b._csr
        else:
            a._csr /= b
        return a

    def __floordiv__(a, b):
        if isinstance(b, SparseGeometry):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c //= b
        return c

    def __ifloordiv__(a, b):
        if isinstance(b, SparseGeometry):
            raise NotImplementedError
        a._csr //= b
        return a

    def __truediv__(a, b):
        if isinstance(b, SparseGeometry):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c /= b
        return c

    def __itruediv__(a, b):
        if isinstance(b, SparseGeometry):
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
        if isinstance(b, SparseGeometry):
            a._csr **= b._csr
        else:
            a._csr **= b
        return a


class SparseAtom(SparseGeometry):
    """ Sparse object with number of rows equal to the total number of atoms in the `Geometry`
    """

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
        return self.geometry.na

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
            atom = ensure_array(atom)
            for i, j in self._csr.iter_nnz(atom):
                yield i, j
        else:
            for i, j in self._csr.iter_nnz():
                yield i, j

    def cut(self, seps, axis, *args, **kwargs):
        """ Cuts the sparse atom model into different parts.

        Recreates a new sparse atom object with only the cutted 
        atoms in the structure.

        Cutting is the opposite of tiling.

        Parameters
        ----------
        seps : int, optional
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
        isc = np.zeros([3], np.int32)
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
                if len(np.where(
                        np.logical_and(idx <= sub,
                                       sub < idx + geom.na))[0]) == 0:
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

    def remove(self, atom):
        """ Create a subset of this sparse matrix by removing the elements corresponding to ``atom``

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : array_like of int
            indices of removed atoms
        """
        atom = self.sc2uc(atom)
        atom = np.setdiff1d(np.arange(self.na), atom, assume_unique=True)
        return self.sub(atom)

    def sub(self, atom):
        """ Create a subset of this sparse matrix by only retaining the elements corresponding to the ``atom``

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : array_like of int
            indices of retained atoms
        """
        atom = self.sc2uc(atom)
        geom = self.geom.sub(atom)

        # Now create the new sparse orbital class
        # First figure out the initialization parameters
        nnzpr = np.amax(self._csr.ncol)
        S = self.__class__(geom, self.dim, self.dtype, nnzpr, **self._cls_kwargs())

        # Retrieve pointers to local data
        na = self.na
        D = self._csr

        # Create atomic pivot table
        pvt = np.zeros([self.na_s], np.int32) - 1
        idx = np.tile(atom, self.n_s) + \
              np.repeat(np.arange(self.n_s) * self.na, len(atom))
        pvt[idx] = np.arange(geom.na_s)

        col = D.col
        ptr = D.ptr
        ncol = D.ncol

        for IA, ia in enumerate(atom):
            ccol = col[ptr[ia]:ptr[ia]+ncol[ia]]
            jas = ccol[pvt[ccol] >= 0]
            S[IA, pvt[jas]] = self[ia, jas]
        S.finalize()

        return S

    def tile(self, reps, axis):
        """ Create a tiled sparse atom object, equivalent to `Geometry.tile`

        The already existing sparse elements are extrapolated
        to the new supercell by repeating them in blocks like the coordinates.

        Parameters
        ----------
        reps : int
            number of repetitions along cell-vector ``axis``
        axis : int
            0, 1, 2 according to the cell-direction
        """
        # Create the new sparse object
        S = self._init_larger('tile', reps, axis)

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geom
        na = self.na
        ptr = self._csr.ptr
        ncol = self._csr.ncol
        col = self._csr.col

        # Information for the new Hamiltonian sparse matrix
        na_n = S.na
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index
        rngreps = range(0, na*reps, na)
        for ia in range(geom.na):

            # Loop on the connection orbitals
            if ncol[ia] == 0:
                continue
            ccol = col[ptr[ia]:ptr[ia]+ncol[ia]]

            # supercells in the old geometry
            isc = geom.a2isc(ccol)
            # resulting atom in the new geometry (without wrapping
            # for correct supercell, that will happen below)
            A = ccol % na + na * isc[:, axis]

            # For recalculating stuff in the repetition loop
            ISC = np.copy(isc)

            # Create repetitions
            for rep in rngreps:

                # Figure out the JA atom
                JA = A + rep
                # Correct the supercell information
                ISC[:, axis] = JA // na_n

                S[ia + rep, JA % na_n + sc_index(ISC) * na_n] = self[ia, ccol]
        S.finalize()

        return S

    def repeat(self, reps, axis):
        """ Create a repeated sparse atom object, equivalent to `Geometry.repeat`

        The already existing sparse elements are extrapolated
        to the new supercell by repeating them in blocks like the coordinates.

        Parameters
        ----------
        reps : int
            number of repetitions along cell-vector ``axis``
        axis : int
            0, 1, 2 according to the cell-direction
        """
        # Create the new sparse object
        S = self._init_larger('repeat', reps, axis)

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geom
        na = self.na
        ptr = self._csr.ptr
        ncol = self._csr.ncol
        col = self._csr.col

        # Information for the new Hamiltonian sparse matrix
        na_n = S.na
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index
        rngreps = range(reps)
        for ia in range(geom.na):

            # ia * reps = the offset for all previous atoms
            IA = ia * reps

            # Loop on the connection orbitals
            if ncol[ia] == 0:
                continue
            ccol = col[ptr[ia]:ptr[ia]+ncol[ia]]

            # supercells in the old geometry
            isc = geom.a2isc(ccol)
            # resulting atom in the new geometry (without wrapping
            # for correct supercell, that will happen below)
            JA = (ccol % na) * reps

            # For recalculating stuff in the repetition loop
            ISC = np.copy(isc)

            for rep in rngreps:

                A = isc[:, axis] + rep
                ISC[:, axis] = A // reps

                S[IA + rep, JA + A % reps + sc_index(ISC) * na_n] = self[ia, ccol]
        S.finalize()

        return S


class SparseOrbital(SparseGeometry):
    """ Sparse object with number of rows equal to the total number of orbitals in the `Geometry`
    """

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
            orbital = self.geom.a2o(atom)
        elif not orbital is None:
            orbital = ensure_array(orbital)
        if not orbital is None:
            for i, j in self._csr.iter_nnz(orbital):
                yield i, j
        else:
            for i, j in self._csr.iter_nnz():
                yield i, j

    def cut(self, seps, axis, *args, **kwargs):
        """ Cuts the sparse orbital model into different parts.

        Recreates a new sparse orbital object with only the cutted 
        atoms in the structure.

        Cutting is the opposite of tiling.

        Parameters
        ----------
        seps : int, optional
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
        isc = np.zeros([3], np.int32)
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
                if len(np.where(
                        np.logical_and(idx <= sub,
                                       sub < idx + geom.no))[0]) == 0:
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
            isc = np.array(M.o2isc(o), np.int32, copy=True)
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
        """ Create a subset of this sparse matrix by only retaining the elements corresponding to the ``atom``

        Indices passed *MUST* be unique.

        Negative indices are wrapped and thus works.

        Parameters
        ----------
        atom  : array_like of int
            indices of retained atoms
        """
        atom = self.sc2uc(atom)
        otom = self.geom.a2o(atom, all=True)
        geom = self.geom.sub(atom)
        Otom = geom.a2o(np.arange(len(geom)), all=True)

        # Now create the new sparse orbital class
        # First figure out the initialization parameters
        nnzpr = np.max(self._csr.ncol)
        S = self.__class__(geom, self.dim, self.dtype, nnzpr, **self._cls_kwargs())

        # Retrieve pointers to local data
        no = self.no
        D = self._csr

        # Create orbital pivot table
        pvt = np.zeros([self.no_s], np.int32) - 1
        idx = np.tile(otom, self.n_s) + \
              np.repeat(np.arange(self.n_s) * self.no, len(otom))
        pvt[idx] = np.arange(geom.no_s)

        col = D.col
        ptr = D.ptr
        ncol = D.ncol

        for IO, io in zip(Otom[:geom.no], otom[:geom.no]):
            ccol = col[ptr[io]:ptr[io]+ncol[io]]
            jos = ccol[pvt[ccol] >= 0]
            S[IO, pvt[jos]] = self[io, jos]
        S.finalize()

        return S

    def tile(self, reps, axis):
        """ Create a tiled sparse orbital object, equivalent to `Geometry.tile`

        The already existing sparse elements are extrapolated
        to the new supercell by repeating them in blocks like the coordinates.

        Parameters
        ----------
        reps : int
            number of repetitions along cell-vector ``axis``
        axis : int
            0, 1, 2 according to the cell-direction
        """

        # Create the new sparse object
        S = self._init_larger('tile', reps, axis)

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geom
        no = self.no
        ptr = self._csr.ptr
        ncol = self._csr.ncol
        col = self._csr.col

        # Information for the new Hamiltonian sparse matrix
        no_n = S.no
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index
        rngreps = range(0, no*reps, no)
        for io in range(geom.no):

            # Loop on the connection orbitals
            if ncol[io] == 0:
                continue
            ccol = col[ptr[io]:ptr[io]+ncol[io]]

            # supercells in the old geometry
            isc = geom.o2isc(ccol)
            # resulting orbital in the new geometry (without wrapping
            # for correct supercell, that will happen below)
            O = ccol % no + no * isc[:, axis]

            # For recalculating stuff in the repetition loop
            ISC = np.copy(isc)

            # Create repetitions
            for rep in rngreps:

                # Figure out the JO orbital
                JO = O + rep
                # Correct the supercell information
                ISC[:, axis] = JO // no_n

                S[io + rep, JO % no_n + sc_index(ISC) * no_n] = self[io, ccol]
        S.finalize()

        return S

    def repeat(self, reps, axis):
        """ Create a repeated sparse orbital object, equivalent to `Geometry.repeat`

        The already existing sparse elements are extrapolated
        to the new supercell by repeating them in blocks like the coordinates.

        Parameters
        ----------
        reps : int
            number of repetitions along cell-vector ``axis``
        axis : int
            0, 1, 2 according to the cell-direction
        """
        # Create the new sparse object
        S = self._init_larger('repeat', reps, axis)

        # Now begin to populate it accordingly
        # Retrieve local pointers to the information
        # regarding the current Hamiltonian sparse matrix
        geom = self.geom
        no = self.no
        ptr = self._csr.ptr
        ncol = self._csr.ncol
        col = self._csr.col

        # Information for the new Hamiltonian sparse matrix
        no_n = S.no
        geom_n = S.geom

        # First loop on axis tiling and local
        # atoms in the geometry
        sc_index = geom_n.sc_index
        rngreps = range(reps)
        for io in range(geom.no):

            ia = geom.o2a(io)
            # firsto * reps = the offset for all previous atoms
            # io - firsto = the orbital on the atom
            IO = geom.firsto[ia] * (reps-1) + io
            oa = geom.atom[ia].orbs

            # Loop on the connection orbitals
            if ncol[io] == 0:
                continue
            ccol = col[ptr[io]:ptr[io]+ncol[io]]

            isc = geom.o2isc(ccol)

            # Unit-cell orbitals
            JO = ccol % no

            # Get the number of orbitals of the residing atoms
            ja = geom.o2a(JO)
            oJ = geom.firsto[ja]
            oA = geom.lasto[ja] - oJ + 1
            ISC = np.copy(isc)
            JO = oJ * (reps - 1) + JO

            for rep in rngreps:

                A = isc[:, axis] + rep
                ISC[:, axis] = A // reps

                S[IO + oa * rep, JO + oA * (A % reps) + sc_index(ISC) * no_n] = self[io, ccol]
        S.finalize()

        return S
