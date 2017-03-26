"""
Tight-binding class to create tight-binding models.
"""
from __future__ import print_function, division

import warnings
from numbers import Integral

import numpy as np
import scipy.linalg as sli
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as ssli

from sisl._help import get_dtype
from sisl._help import _zip as zip
from sisl.sparse import SparseCSR, ispmatrix, ispmatrixd

__all__ = ['Hamiltonian', 'TightBinding']


class Hamiltonian(object):
    """ Hamiltonian object containing the coupling constants between orbitals.

    The Hamiltonian object contains information regarding the 
     - geometry
     - coupling constants between orbitals

    It contains an intrinsic sparse matrix of the Hamiltonian elements.

    Assigning or changing Hamiltonian elements is as easy as with
    standard ``numpy`` assignments:

    >>> ham = Hamiltonian(...)
    >>> ham.H[1,2] = 0.1

    which assigns 0.1 as the coupling constant between orbital 2 and 3.
    (remember that Python is 0-based elements).
    """

    # The order of the Energy
    # I.e. whether energy should be in other units than Ry
    # This conversion is made: [eV] ** _E_order
    _E_order = 1

    def __init__(self, geom, nnzpr=None, orthogonal=True, spin=1,
                 dtype=None, *args, **kwargs):
        """Create tight-binding model from geometry

        Initializes a tight-binding model using the :code:`geom` object
        as the underlying geometry for the tight-binding parameters.
        """
        self._geom = geom

        # Initialize the sparsity pattern
        self.reset(nnzpr=nnzpr, orthogonal=orthogonal, spin=spin, dtype=dtype)

    def reset(self, nnzpr=None, orthogonal=True, spin=1, dtype=None):
        """
        The sparsity pattern is cleaned and every thing
        is reset.

        The object will be the same as if it had been
        initialized with the same geometry as it were
        created with.

        Parameters
        ----------
        nnzpr: int
           number of non-zero elements per row
        orthogonal: boolean, True
           if there is an overlap matrix associated with the
           Hamiltonian
        spin: int, 1
           number of spin-components
        dtype: ``numpy.dtype``, `numpy.float64`
           the datatype of the Hamiltonian
        """
        # I know that this is not the most efficient way to
        # access a C-array, however, for constructing a
        # sparse pattern, it should be faster if memory elements
        # are closer...
        # Hence, this choice of having H and S like this

        # We check the first atom and its neighbours, we then
        # select max(5,len(nc) * 4)
        if nnzpr is None:
            nnzpr = self.geom.close(0)
            if nnzpr is None:
                nnzpr = 8
            else:
                nnzpr = max(5, len(nnzpr) * 4)

        self._orthogonal = orthogonal

        # Reset the sparsity pattern
        if not orthogonal:
            self._data = SparseCSR((self.no, self.no_s, spin+1), nnzpr=nnzpr, dtype=dtype)
        else:
            self._data = SparseCSR((self.no, self.no_s, spin), nnzpr=nnzpr, dtype=dtype)

        self._spin = spin

        if spin == 1:
            self.UP = 0
            self.DOWN = 0
            self.S_idx = 1
            self.Hk = self._Hk_unpolarized
            self.Sk = self._Sk
        elif spin == 2:
            self.UP = 0
            self.DOWN = 1
            self.S_idx = 2
            self.Hk = self._Hk_polarized
            self.Sk = self._Sk
        elif spin == 4:
            self.Hk = self._Hk_non_collinear
            self.Sk = self._Sk_non_collinear
            self.S_idx = 4
        elif spin == 8:
            self.Hk = self._Hk_spin_orbit
            self.Sk = self._Sk_non_collinear
            self.S_idx = 8
            raise ValueError("Currently the Hamiltonian has only been implemented with up to non-collinear spin.")

        if orthogonal:
            # There is no overlap matrix
            self.S_idx = -1

            def diagonal_Sk(self, k, dtype=None):
                """ For an orthogonal case we always return the identity matrix """
                if dtype is None:
                    dtype = np.float64
                no = self.no
                S = csr_matrix((no, no), dtype=dtype)
                S.setdiag(1.)
                return S
            self.Sk = diagonal_Sk

        # Denote that one *must* specify all details of the elements
        self._def_dim = -1

    def empty(self, keep=False):
        """ See `SparseCSR.empty` for details """
        self._data.empty(keep)

    def copy(self, dtype=None):
        """ Return a copy of the ``Hamiltonian`` object """
        if dtype is None:
            dtype = self.dtype
        H = self.__class__(self.geom, orthogonal=self.orthogonal,
                           spin=self.spin, dtype=dtype)
        # Be sure to copy the content of the SparseCSR object
        H._data = self._data.copy(dtype=dtype)
        return H

    ######### Definitions of overrides ############
    @property
    def geometry(self):
        """ Return the attached geometry """
        return self._geom
    geom = geometry

    @property
    def spin(self):
        """ Return number of spin-components in Hamiltonian """
        return self._spin

    @property
    def dtype(self):
        """ Return data type of Hamiltonian (and overlap matrix) """
        return self._data.dtype

    @property
    def orthogonal(self):
        """ Return whether the Hamiltonian is orthogonal """
        return self._orthogonal

    def __len__(self):
        """ Returns number of rows in the Hamiltonian """
        return self.geom.no

    def __repr__(self):
        """ Representation of the tight-binding model """
        s = self.geom.__repr__()
        return s + '\nNumber of spin / non-zero elements {0} / {1} '.format(self.spin, self.nnz)

    def __getattr__(self, attr):
        """ Returns the attributes from the underlying geometry

        Any attribute not found in the Hamiltonian class will
        be looked up in the underlying geometry.
        """
        return getattr(self.geom, attr)

    def __getitem__(self, key):
        """ Return Hamiltonian coupling elements for the index(s) """
        dd = self._def_dim
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        d = self._data[key]
        return d

    def __setitem__(self, key, val):
        """ Set or create couplings between orbitals in the Hamiltonian

        Override set item for slicing operations and enables easy
        setting of tight-binding parameters in a sparse matrix
        """
        dd = self._def_dim
        if dd >= 0:
            key = tuple(key) + (dd,)
            self._def_dim = -1
        self._data[key] = val

        if dd < 0 and not self.orthogonal:
            warnings.warn(('Hamiltonian specification of both H and S simultaneously is deprecated. '
                           'This functionality will be removed in a future release.'))

    def __get_H(self):
        self._def_dim = self.UP
        return self
    _get_H = __get_H

    def __set_H(self, key, value):
        if len(key) == 2:
            self._def_dim = self.UP
        self[key] = value
    _set_H = __set_H

    H = property(__get_H, __set_H)

    def __get_S(self):
        if self.orthogonal:
            return None
        self._def_dim = self.S_idx
        return self
    _get_S = __get_S

    def __set_S(self, key, value):
        if self.orthogonal:
            return None
        self._def_dim = self.S_idx
        self[key] = value
    _set_S = __set_S

    S = property(__get_S, __set_S)

    # Create iterations on entire set of orbitals
    def iter(self, local=False):
        """ Iterations of the orbital space in the geometry, two indices from loop

        An iterator returning the current atomic index and the corresponding
        orbital index.

        >>> for ia, io in self:

        In the above case `io` always belongs to atom `ia` and `ia` may be
        repeated according to the number of orbitals associated with
        the atom `ia`.

        Parameters
        ----------
        local : `bool=False`
           whether the orbital index is the global index, or the local index relative to 
           the atom it resides on.
        """
        for ia, io in self.geom.iter_orbitals(local=local):
            yield ia, io

    __iter__ = iter

    # Create iterations on the non-zero elements
    def iter_nnz(self, atom=None, orbital=None):
        """ Iterations of the non-zero elements, returns a tuple of orbital and coupling orbital

        An iterator returning the current orbital index and the corresponding
        connected orbital where a non-zero is defined

        >>> for io, jo in self.iter_nnz():

        In the above case `io` and `jo` are orbitals such that:

        >>> self.H[io,jo] 

        returns the non-zero element of the Hamiltonian.

        One may reduce the iterated space by either requesting a specific set of atoms,
        or orbitals, _not_ both simultaneously.

        Examples
        --------
        Looping only on one or more atoms:

        >>> for io, jo in self.iter_nnz(atom=[2, 3]):
        >>>     # loop on all orbitals on atom 3 and 4 (0 indexing)

        >>> for io, jo in self.iter_nnz(orbital=[2, 3]):
        >>>     # loop on orbitals 3 and 4 (0 indexing)


        Parameters
        ----------
        atom : ``int``/``array_like``
           iterate on couplings to the set of atoms (not compatible with `orbital`)
        orbital : ``int``/``array_like``
           iterate on couplings to the set of orbitals (not compatible with `atom`)
        """
        if atom is not None and orbital is not None:
            raise ValueError("iter_nnz: both atom and orbital has been passed, only one allowed.")

        if atom is not None:
            orbs = self.geom.a2o(atom, all=True)
            for io, jo in self._data.iter_nnz(orbs):
                yield io, jo
        elif orbital is not None:
            for io, jo in self._data.iter_nnz(orbital):
                yield io, jo
        else:
            for io, jo in self._data:
                yield io, jo

    def create_construct(self, dR, param):
        """ Returns a simple function for passing to the `construct` function.

        This is simply to leviate the creation of simplistic
        functions needed for setting up the Hamiltonian.

        Basically this returns a function:
        >>> def func(self, ia, idxs, idxs_xyz=None):
        >>>     idx = self.geom.close(ia, dR=dR, idx=idxs)
        >>>     for ix, p in zip(idx, param):
        >>>         self[ia, ix] = p

        Note
        ----
        This function only works for geometries with one orbital
        per atom.
        If you have more than one orbital on any atom, you should 
        define your own function.

        Parameters
        ----------
        dR : array_like
           radii parameters for tight-binding parameters.
           Must have same length as ``param`` or one less.
           If one less it will be extended with ``dR[0]/100``
        param : array_like
           coupling constants corresponding to the ``dR``
           ranges. ``param[0,:]`` are the tight-binding parameter
           for the all atoms within ``dR[0]`` of each atom.
        """

        if self.orthogonal:
            def func(self, ia, idxs, idxs_xyz=None):
                idx = self.geom.close(ia, dR=dR, idx=idxs, idx_xyz=idxs_xyz)
                for ix, p in zip(idx, param):
                    self[ia, ix] = p
        else:
            def func(self, ia, idxs, idxs_xyz=None):
                idx = self.geom.close(ia, dR=dR, idx=idxs, idx_xyz=idxs_xyz)
                for ix, p in zip(idx, param):
                    self.H[ia, ix] = p[:-1]
                    self.S[ia, ix] = p[-1]

        return func

    def construct(self, func, na_iR=1000, method='rand', eta=False):
        """ Automatically construct the Hamiltonian model based on a function that does the setting up of the Hamiltonian

        This may be called in two variants.

        1. Pass a function (``func``), see e.g. ``create_construct`` 
           which does the setting up.
        2. Pass a tuple/list in ``func`` which consists of two 
           elements, one is ``dR`` the radii parameters for
           the corresponding tight-binding parameters.
           The second is the tight-binding parameters
           corresponding to the ``dR[i]`` elements.
           In this second case all atoms must only have
           one orbital.

        Parameters
        ----------
        func: `callable` or `tuple/list`
           this function *must* take 4 arguments.
           1. Is the Hamiltonian object it-self (`self`)
           2. Is the currently examined atom (`ia`)
           3. Is the currently bounded indices (`idxs`)
           4. Is the currently bounded indices atomic coordinates (`idxs_xyz`)
           An example `func` could be:

           >>> def func(self, ia, idxs, idxs_xyz=None):
           >>>     idx = self.geom.close(ia, dR=[0.1, 1.44], idx=idxs, idx_xyz=idxs_xyz)
           >>>     self.H[ia, idx[0]] = 0.   # on-site
           >>>     self.H[ia, idx[1]] = -2.7 # nearest-neighbour
        na_iR : `int=1000`
           number of atoms within the sphere for speeding
           up the `iter_block` loop.
        eta: `bool=False`
           whether an ETA will be printed
        """

        if not callable(func):
            if not isinstance(func, (tuple, list)):
                raise ValueError('Passed `func` which is not a function, nor tuple/list of `dR, param`')

            if np.any(np.diff(self.geom.lasto) > 1):
                raise ValueError("Automatically setting a tight-binding model "
                              "for systems with atoms having more than 1 "
                              "orbital *must* be done by your-self. You have to define a corresponding `func`.")

            # Convert to a proper function
            func = self.create_construct(func[0], func[1])

        iR = self.geom.iR(na_iR)

        # Get number of atoms
        na = len(self.geom)
        na_run = 0

        from time import time
        from sys import stdout
        t0 = time()

        # Do the loop
        for ias, idxs in self.geom.iter_block(iR=iR, method=method):

            # Get all the indexed atoms...
            # This speeds up the searching for
            # coordinates...
            idxs_xyz = self.geom[idxs, :]

            # Loop the atoms inside
            for ia in ias:
                func(self, ia, idxs, idxs_xyz)

            if eta:
                # calculate the remaining atoms to process
                na_run += len(ias)
                na -= len(ias)
                t1 = time()
                # calculate hours, minutes, seconds
                m, s = divmod(float(t1-t0)/na_run * na, 60)
                h, m = divmod(m, 60)
                stdout.write("Hamiltonian.construct() ETA = {0:5d}h {1:2d}m {2:5.2f}s\r".format(int(h), int(m), s))
                stdout.flush()

        if eta:
            stdout.write("Hamiltonian.construct() {0:23s}\n".format('DONE'))
            stdout.flush()

    @property
    def finalized(self):
        """ Whether the contained data is finalized and non-used elements have been removed """
        return self._data.finalized

    def finalize(self):
        """ Finalizes the tight-binding model

        Finalizes the tight-binding model so that no new sparse
        elements can be added.

        Sparse elements can still be changed.
        """
        self._data.finalize()

        # Get the folded Hamiltonian at the Gamma point
        Hk = self.Hk()

        nzs = Hk.nnz

        if nzs != (Hk + Hk.T).nnz:
            warnings.warn(
                'Hamiltonian does not retain symmetric couplings, this might be problematic.')

    @property
    def nnz(self):
        """ Returns number of non-zero elements in the tight-binding model """
        return self._data.nnz

    @property
    def no(self):
        """ Returns number of orbitals as used when the object was created """
        return self._data.nr

    def tocsr(self, index, isc=None):
        """ Return a ``scipy.sparse.csr_matrix`` from the specified index

        Parameters
        ----------
        index : ``int``
           the index in the sparse matrix (for non-orthogonal cases the last
           dimension is the overlap matrix)
        isc : ``int``, `None`
           the supercell index (or all)
        """
        if isc is not None:
            raise NotImplementedError("Requesting sub-Hamiltonian has not been implemented yet")
        return self._data.tocsr(index)

    def _Hk_unpolarized(self, k=(0, 0, 0), dtype=None):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        """
        return self._Hk_polarized(k, dtype=dtype)

    def _Hk_polarized(self, k=(0, 0, 0), spin=0, dtype=None):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k` for a polarized calculation

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        spin: ``int``, `0`
           the spin-index of the Hamiltonian
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        """
        if dtype is None:
            dtype = np.complex128

        exp = np.exp
        dot = np.dot

        k = np.asarray(k, np.float64)
        k.shape = (-1,)

        if not np.allclose(k, 0.):
            if np.dtype(dtype).kind != 'c':
                raise ValueError("Hamiltonian setup at k different from Gamma requires a complex matrix")

        # Setup the Hamiltonian for this k-point
        Hf = self.tocsr(spin)

        no = self.no
        s = (no, no)
        H = csr_matrix(s, dtype=dtype)

        # Get the reciprocal lattice vectors dotted with k
        kr = dot(self.rcell, k)
        for si in range(self.sc.n_s):
            isc = self.sc_off[si, :]
            phase = exp(-1j * dot(kr, dot(self.cell, isc)))
            H += Hf[:, si * no:(si + 1) * no] * phase

        del Hf

        return H

    def _Hk_non_collinear(self, k=(0, 0, 0), dtype=None):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k` for a non-collinear
        Hamiltonian.

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        """
        if dtype is None:
            dtype = np.complex128

        if np.dtype(dtype).kind != 'c':
            raise ValueError("Non-collinear Hamiltonian setup requires a complex matrix")

        exp = np.exp
        dot = np.dot

        k = np.asarray(k, np.float64)
        k.shape = (-1,)

        no = self.no * 2
        s = (no, no)
        H = csr_matrix(s, dtype=dtype)

        # get back-dimension of the intrinsic sparse matrix
        no = self.no

        # Get the reciprocal lattice vectors dotted with k
        kr = dot(self.rcell, k)
        for si in range(self.sc.n_s):
            isc = self.sc_off[si, :]
            phase = exp(-1j * dot(kr, dot(self.cell, isc)))

            # diagonal elements
            Hf1 = self.tocsr(0)[:, si*no:(si+1)*no] * phase
            for i, j, h in ispmatrixd(Hf1):
                H[i*2, j*2] += h
            Hf1 = self.tocsr(1)[:, si*no:(si+1)*no] * phase
            for i, j, h in ispmatrixd(Hf1):
                H[1+i*2, 1+j*2] += h

            # off-diagonal elements
            Hf1 = self.tocsr(2)[:, si*no:(si+1)*no]
            Hf2 = self.tocsr(3)[:, si*no:(si+1)*no]
            # We expect Hf1 and Hf2 to be aligned equivalently!
            # TODO CHECK
            for i, j, hr in ispmatrixd(Hf1):
                # get value for the imaginary part
                hi = Hf2[i, j]
                H[i*2, 1+j*2] += (hr - 1j * hi) * phase
                H[1+i*2, j*2] += (hr + 1j * hi) * phase

        del Hf1, Hf2

        return H

    def _Sk(self, k=(0, 0, 0), dtype=None):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        """
        # we forward it to Hk_polarized (same thing for S)
        return self._Hk_polarized(k, spin=self.S_idx, dtype=dtype)

    def _Sk_non_collinear(self, k=(0, 0, 0), dtype=None):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        """
        if dtype is None:
            dtype = np.complex128

        if not np.allclose(k, 0.):
            if np.dtype(dtype).kind != 'c':
                raise ValueError("Hamiltonian setup at k different from Gamma requires a complex matrix")

        exp = np.exp
        dot = np.dot

        k = np.asarray(k, np.float64)
        k.shape = (-1,)

        # Get the overlap matrix
        Sf = self.tocsr(self.S_idx)

        no = self.no * 2
        s = (no, no)
        S = csr_matrix(s, dtype=dtype)

        # Get back dimensionality of the intrinsic orbitals
        no = self.no

        # Get the reciprocal lattice vectors dotted with k
        kr = dot(self.rcell, k)
        for si in range(self.sc.n_s):
            isc = self.sc_off[si, :]
            phase = exp(-1j * dot(kr, dot(self.cell, isc)))
            # Setup the overlap for this k-point
            sf = Sf[:, si*no:(si+1)*no]
            for i, j, s in ispmatrixd(sf):
                S[i*2,   j*2] += s
                S[1+i*2, 1+j*2] += s

        del Sf

        return S

    def eigh(self, k=(0, 0, 0),
            atoms=None, eigvals_only=True,
            overwrite_a=True, overwrite_b=True,
            *args,
            **kwargs):
        """ Returns the eigenvalues of the Hamiltonian

        Setup the Hamiltonian and overlap matrix with respect to
        the given k-point, then reduce the space to the specified atoms
        and calculate the eigenvalues.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigh`
        """
        H = self.Hk(k=k)
        if not self.orthogonal:
            S = self.Sk(k=k)
        # Reduce sparsity pattern
        if not atoms is None:
            orbs = self.a2o(atoms)
            # Reduce space
            H = H[orbs, orbs]
            if not self.orthogonal:
                S = S[orbs, orbs]
        if self.orthogonal:
            return sli.eigh(H.todense(),
                *args,
                eigvals_only=eigvals_only,
                overwrite_a=overwrite_a,
                **kwargs)

        return sli.eigh(H.todense(), S.todense(),
            *args,
            eigvals_only=eigvals_only,
            overwrite_a=overwrite_a,
            overwrite_b=overwrite_b,
            **kwargs)

    def eigsh(self, k=(0, 0, 0), n=10,
            atoms=None, eigvals_only=True,
            *args,
            **kwargs):
        """ Returns the eigenvalues of the Hamiltonian

        Setup the Hamiltonian and overlap matrix with respect to
        the given k-point, then reduce the space to the specified atoms
        and calculate the eigenvalues.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigh`
        """

        # We always request the smallest eigenvalues...
        kwargs.update({'which': kwargs.get('which', 'SM')})

        H = self.Hk(k=k)
        if not self.orthogonal:
            raise ValueError("The sparsity pattern is non-orthogonal, you cannot use the Arnoldi procedure with scipy")

        # Reduce sparsity pattern
        if not atoms is None:
            orbs = self.a2o(atoms)
            # Reduce space
            H = H[orbs, orbs]

        return ssli.eigsh(H, k=n,
                          *args,
                          return_eigenvectors=not eigvals_only,
                          **kwargs)

    def cut(self, seps, axis, *args, **kwargs):
        """ Cuts the tight-binding model into different parts.

        Creates a tight-binding model by retaining the parameters
        for the cut-out region, possibly creating a super-cell.

        Parameters
        ----------
        seps  : integer, optional
           number of times the structure will be cut.
        axis  : integer
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
                              "The tight-binding model cannot be cut as the structure "
                              "cannot be tiled accordingly. ANY use of the model has been "
                              "relieved from sisl.")
        if new_w:
            warnings.warn(new_w, UserWarning)

        # Now we need to re-create the tight-binding model
        H = self.tocsr(0)
        has_S = self.S_idx > 0
        if has_S:
            S = self.tocsr(self.S_idx)
        # they are created similarly, hence the following
        # should keep their order

        # First we need to figure out how long the interaction range is
        # in the cut-direction
        # We initialize to be the same as the parent direction
        nsc = np.copy(self.nsc) // 2
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

            # Figure out if the Hamiltonian has interactions
            # to 'isc'
            sub = H[0:geom.no, idx * self.no:(idx + 1) * self.no].indices[:]
            if has_S:
                sub = np.unique(np.concatenate(
                    (sub, S[0:geom.no, idx * self.no:(idx + 1) * self.no].indices[:]), axis=0))
            if len(sub) == 0:
                break

            c_max = np.amax(sub)
            # Count the number of cells it interacts with
            i = (c_max % self.no) // geom.no
            ic = idx * self.no
            for j in range(i):
                idx = ic + geom.no * j
                # We need to ensure that every "in between" index exists
                # if it does not we discard those indices
                if len(np.where(
                        np.logical_and(idx <= sub,
                                       sub < idx + geom.no)
                )[0]) == 0:
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
        ham = self.__class__(geom, nnzpr=np.amax(self._data.ncol), spin=self.spin, orthogonal=self.orthogonal)

        def sco2sco(M, o, m, seps, axis):
            # Converts an o from M to m
            isc = np.copy(M.o2isc(o))
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

        # Copy elements
        if has_S:
            for jo in range(geom.no):

                # make smaller cut
                sH = H[jo, :]
                sS = S[jo, :]

                for io, iH in zip(sH.indices, sH.data):
                    # Get the equivalent orbital in the smaller cell
                    o, ofp, ofm = sco2sco(self.geom, io, ham.geom, seps, axis)
                    if o is None:
                        continue
                    ham.H[jo, o + ofp] = iH
                    ham.S[jo, o + ofp] = S[jo, io]
                    ham.H[o, jo + ofm] = iH
                    ham.S[o, jo + ofm] = S[jo, io]

                if np.any(sH.indices != sS.indices):

                    # Ensure that S is also cut
                    for io, iS in zip(sS.indices, sS.data):
                        # Get the equivalent orbital in the smaller cell
                        o, ofp, ofm = sco2sco(self.geom, io, ham.geom, seps, axis)
                        if o is None:
                            continue
                        ham.H[jo, o + ofp] = H[jo, io]
                        ham.S[jo, o + ofp] = iS
                        ham.H[o, jo + ofm] = H[jo, io]
                        ham.S[o, jo + ofm] = iS

        else:
            for jo in range(geom.no):
                sH = H[jo, :]

                for io, iH in zip(sH.indices, sH.data):
                    # Get the equivalent orbital in the smaller cell
                    o, ofp, ofm = sco2sco(self.geom, io, ham.geom, seps, axis)
                    if o is None:
                        continue
                    ham[jo, o + ofp] = iH
                    ham[o, jo + ofm] = iH

        return ham

    def tile(self, reps, axis):
        """ Returns a repeated tight-binding model for this, much like the `Geometry`

        The already existing tight-binding parameters are extrapolated
        to the new supercell by repeating them in blocks like the coordinates.

        Parameters
        ----------
        reps : number of tiles (repetitions)
        axis : direction of tiling
            0, 1, 2 according to the cell-direction
        """

        # Create the new geometry
        g = self.geom.tile(reps, axis)

        raise NotImplementedError(('tiling a Hamiltonian model has not been '
                              'fully implemented yet.'))

    def repeat(self, reps, axis):
        """ Refer to `tile` instead """
        # Create the new geometry
        g = self.geom.repeat(reps, axis)

        raise NotImplementedError(('repeating a Hamiltonian model has not been '
                              'fully implemented yet, use tile instead.'))

    @classmethod
    def sp2HS(cls, geom, H, S=None):
        """ Returns a tight-binding model from a preset H, S and Geometry
        """
        # Calculate number of connections
        nc = 0

        has_S = not S is None

        # Ensure csr format
        H = H.tocsr()
        if has_S:
            S = S.tocsr()
        for i in range(geom.no):
            nc = max(nc, H[i, :].getnnz())
            if has_S:
                nc = max(nc, S[i, :].getnnz())

        # Create the Hamiltonian
        ham = cls(geom, nnzpr=nc,
                  orthogonal=not has_S, dtype=H.dtype)

        # Copy data to the model
        if has_S:
            for jo, io in ispmatrix(H):
                ham.S[jo, io] = S[jo, io]

            # If the Hamiltonian for one reason or the other
            # is zero in the diagonal, then we *must* account for
            # this as it isn't captured in the above loop.
            skip_S = np.all(H.row == S.row)
            skip_S = skip_S and np.all(H.col == S.col)
            skip_S = False
            if not skip_S:
                # Re-convert back to allow index retrieval
                H = H.tocsr()
                for jo, io, s in ispmatrixd(S):
                    ham[jo, io] = (H[jo, io], s)

        else:
            for jo, io, h in ispmatrixd(H):
                ham[jo, io] = h

        return ham

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads Hamiltonian from `Sile` using `read_H`.

        Parameters
        ----------
        sile : `Sile`, str
            a `Sile` object which will be used to read the Hamiltonian
            and the overlap matrix (if any)
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_es(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_es(*args, **kwargs)
        else:
            return get_sile(sile).read_es(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes a tight-binding model to the `Sile` as implemented in the :code:`Sile.write_es` method """
        self.finalize()

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_es(self, *args, **kwargs)
        else:
            get_sile(sile, 'w').write_es(self, *args, **kwargs)

    ###############################
    # Overload of math operations #
    ###############################
    def __add__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c += b
        return c
    __radd__ = __add__

    def __iadd__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        a._data += b
        return a

    def __sub__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c -= b
        return c

    def __rsub__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        c = b + (-1) * a
        return c

    def __isub__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        a._data -= b
        return a

    def __mul__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c *= b
        return c
    __rmul__ = __mul__

    def __imul__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        a._data *= b
        return a

    def __div__(a, b):
        if isinstance(a, Hamiltonian):
            if isinstance(b, Hamiltonian):
                raise NotImplementedError
            c = a.copy(dtype=get_dtype(b, other=a.dtype))
            c /= b
        elif isinstance(b, Hamiltonian):
            c = b.copy(dtype=get_dtype(a, other=b.dtype))
            c._data = a / c._data
        return c

    def __idiv__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        a._data /= b
        return a

    def __floordiv__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c //= b
        return c

    def __ifloordiv__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        a._data //= b
        return a

    def __truediv__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c /= b
        return c

    def __itruediv__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        a._data /= b
        return a

    def __pow__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c **= b
        return c

    def __rpow__(a, b):
        if isinstance(b, SparseCSR):
            raise NotImplementedError
        c = a.copy(dtype=get_dtype(b, other=a.dtype))
        c._data = b ** c._data
        return c

    def __ipow__(a, b):
        if isinstance(b, Hamiltonian):
            raise NotImplementedError
        a._data **= b
        return a

# For backwards compatibility we also use TightBinding
# NOTE: that this is not sub-classed...
TightBinding = Hamiltonian
