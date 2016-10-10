"""
Tight-binding class to create tight-binding models.
"""
from __future__ import print_function, division

import warnings
from numbers import Integral

import numpy as np
import scipy.linalg as sli
import scipy.sparse.linalg as ssli

from sisl.sparse import SparseCSR


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

    def __init__(self, geom, nnzpr=None, ortho=True, spin=1,
                 dtype=None, *args, **kwargs):
        """Create tight-binding model from geometry

        Initializes a tight-binding model using the :code:`geom` object
        as the underlying geometry for the tight-binding parameters.
        """
        self.geom = geom

        # Initialize the sparsity pattern
        self.reset(nnzpr=nnzpr, ortho=ortho, spin=spin, dtype=dtype)

    def reset(self, nnzpr=None, ortho=True, spin=1, dtype=None):
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
        ortho: boolean, True
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
            if nnzpr is None: nnzpr = [0,0]
            nnzpr = max(5, len(nnzpr) * 4)

        self._ortho = ortho

        # Reset the sparsity pattern
        if not ortho:
            self._data = SparseCSR((self.no, self.no_s, spin+1), nnzpr=nnzpr, dtype=dtype)
        else:
            self._data = SparseCSR((self.no, self.no_s, spin), nnzpr=nnzpr, dtype=dtype)


        self._spin = spin

        if spin == 1:
            self.UP = 0
            self.DOWN = 0
            self.S_idx = 1
        elif spin == 2:
            self.UP = 0
            self.DOWN = 1
            self.S_idx = 2
        else:
            raise ValueError("Currently the Hamiltonian has only been implemented with up to collinear spin.")

        if ortho:
            # There is no overlap matrix
            self.S_idx = -1

        # Denote that one *must* specify all details of the elements
        self._def_dim = -1


    def empty(self, keep=False):
        """ See `SparseCSR.empty` for specifics """
        self._data.empty(keep)


    ######### Definitions of overrides ############
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
        return self._ortho

    def __len__(self):
        """ Returns number of rows in the Hamiltonian """
        return self.geom.no

    def __repr__(self):
        """ Representation of the tight-binding model """
        s = self.geom.__repr__()
        return s + '\nNumber of non-zero elements {0}'.format(len(self))

    def __getattr__(self, attr):
        """ Returns the attributes from the underlying geometry

        Any attribute not found in the tight-binding model will
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


    def __get_H(self):
        self._def_dim = self.UP
        return self
    _get_H = __get_H

    def __set_H(self, key, value):
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


    # Create iterations module
    def iter_linear(self):
        """ Iterations of the orbital space, two indices from loop

        An iterator returning the current atomic index and the corresponding
        orbital index.

        >>> for ia, io in self:

        In the above case `io` always belongs to atom `ia` and `ia` may be
        repeated according to the number of orbitals associated with
        the atom `ia`.
        """
        for ia in self.geom:
            ia1, ia2 = self.geom.lasto[ia], self.geom.lasto[ia + 1]
            for io in range(ia1, ia2):
                yield ia, io

    __iter__ = iter_linear


    def construct(self, dR, param):
        """ Automatically construct the Hamiltonian model based on ``dR`` and associated hopping integrals ``param``.

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
        # Ensure that we are dealing with a numpy array
        param = np.array(param)

        if len(dR) + 1 == len(param):
            R = np.hstack((dR[0] / 100, np.asarray(dR)))
        elif len(dR) == len(param):
            R = np.asarray(dR).copy()
        else:
            raise ValueError("Length of dR and param must be the same "
                             "or dR one shorter than param. "
                             "One tight-binding parameter for each radii.")

        if not self.orthogonal:
            if len(param[0]) != 2:
                raise ValueError("Number of parameters "
                                 "for each element is not 2. "
                                 "You must make len(param[0] == 2) for non-orthogonal Hamiltonians.")

        if np.any(np.diff(self.geom.lasto) > 1):
            warnings.warn("Automatically setting a tight-binding model "
                          "for systems with atoms having more than 1 "
                          "orbital is not adviced. Please do it your-self.")

        eq_atoms = []
        def print_equal(eq_atoms):
            if len(eq_atoms) > 0:
                s = ("The geometry has one or more atoms having the same "
                     "atomic position. "
                     "The atoms are within {} Ang "
                     "of each other.\n".format(R[0]))
                
                for ia, ja in eq_atoms:
                    s += "  {0:7d} -- {1:7d}\n".format(ia, ja)
                warnings.warn(s)

        def append_equal(eq_atoms, ia, idx):
            # Append to the list of equal atoms the atomic indices
            if len(idx) > 1:
                tmp = list(idx)
                # only add in "one" direction
                for ja in tmp:
                    if ja > ia:
                        eq_atoms.append( (ia,ja) )


        if len(self.geom) < 1501:
            # there is no need to do anything complex
            # for small systems
            for ia in self.geom:
                # Find atoms close to 'ia'
                idx = self.geom.close(ia, dR=R)
                append_equal(eq_atoms, ia, idx[0])

                for ix, h in zip(idx, param):
                    # Set the tight-binding parameters
                    self[ia, ix] = h

            print_equal(eq_atoms)
            return self

        # check how many atoms are within the standard 10 dR
        # range of some random atom.
        ia = np.random.randint(len(self.geom) - 1)

        # default block iterator
        d = self.geom.dR
        na = len(self.geom.close(ia, dR=d * 10))

        # Convert to 1000 atoms spherical radii
        iR = int(4 / 3 * np.pi * d ** 3 / na * 1000)

        # Do the loop
        for ias, idxs in self.geom.iter_block(iR=iR):
            # Loop the atoms inside
            for ia in ias:
                # Find atoms close to 'ia'
                idx = self.geom.close(ia, dR=R, idx=idxs)
                append_equal(eq_atoms, ia, idx[0])

                for ix, h in zip(idx, param):
                    # Set the tight-binding parameters
                    self[ia, ix] = h

        print_equal(eq_atoms)


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


    def tocsr(self, index):
        """ Return a ``scipy.sparse.csr_matrix`` from the specified index
        """
        return self._data.tocsr(index)
        

    def Hk(self, k=(0, 0, 0), spin=0):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k: float*3
           k-point 
        spin: int, 0
           the spin-index of the Hamiltonian
        """
        # Create csr sparse formats.
        # We import here as the user might not want to
        # rely on this feature.
        from scipy.sparse import csr_matrix

        dot = np.dot

        k = np.asarray(k, np.float64)
        k.shape = (-1,)
        
        # Setup the Hamiltonian for this k-point
        Hf = self.tocsr(spin)

        no = self.no
        s = (no, no)
        H = csr_matrix(s, dtype=np.complex128)

        # Get the reciprocal lattice vectors dotted with k
        kr = dot(self.rcell, k)
        for si in range(self.sc.n_s):
            isc = self.sc_off[si, :]
            phase = np.exp(-1j * dot(kr, dot(self.cell, isc)))
            H += Hf[:, si * no:(si + 1) * no] * phase
            
        del Hf
        
        return H


    def Sk(self, k=(0, 0, 0), spin=0):
        """ Return the overlap matrix in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k: float*3
           k-point 
        """
        if self.orthogonal:
            return None

        # Create csr sparse formats.
        # We import here as the user might not want to
        # rely on this feature.
        from scipy.sparse import csr_matrix

        dot = np.dot

        k = np.asarray(k, np.float64)
        k.shape = (-1,)
        
        # Setup the Hamiltonian for this k-point
        Sf = self.tocsr(self.S_idx)

        no = self.no
        s = (no, no)
        S = csr_matrix(s, dtype=np.complex128)

        # Get the reciprocal lattice vectors dotted with k
        kr = dot(self.rcell, k)
        for si in range(self.sc.n_s):
            isc = self.sc_off[si, :]
            phase = np.exp(-1j * dot(kr, dot(self.cell, isc)))
            S += Sf[:, si * no:(si + 1) * no] * phase
            
        del Sf
        
        return S


    def eigh(self,k=(0,0,0),
            atoms=None, eigvals_only=True,
            overwrite_a=True, overwrite_b=True,
            *args,
            **kwargs):
        """ Returns the eigenvalues of the tight-binding model

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


    def eigsh(self, k=(0,0,0), n=10,
            atoms=None, eigvals_only=True,
            *args,
            **kwargs):
        """ Returns the eigenvalues of the tight-binding model

        Setup the Hamiltonian and overlap matrix with respect to
        the given k-point, then reduce the space to the specified atoms
        and calculate the eigenvalues.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigh`
        """
        
        # We always request the smallest eigenvalues... 
        kwargs.update({'which':kwargs.get('which', 'SM')})
        
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
        H = self.Hk()
        S = self.Sk()
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
            sub = np.unique(np.hstack(
                (sub, S[0:geom.no, idx * self.no:(idx + 1) * self.no].indices[:])))
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
        ham = self.__class__(geom, nc=np.amax(self.ncol), spin=self.spin)

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
        for jo in range(geom.no):

            # make smaller cut
            sH = H[jo, :]
            sS = S[jo, :]

            for io, iH in zip(sH.indices, sH.data):
                # Get the equivalent orbital in the smaller cell
                o, ofp, ofm = sco2sco(self.geom, io, ham.geom, seps, axis)
                if o is None:
                    continue
                ham[jo, o + ofp] = iH, S[jo, io]
                ham[o, jo + ofm] = iH, S[jo, io]

            if np.any(sH.indices != sS.indices):

                # Ensure that S is also cut
                for io, iS in zip(sS.indices, sS.data):
                    # Get the equivalent orbital in the smaller cell
                    o, ofp, ofm = sco2sco(self.geom, io, ham.geom, seps, axis)
                    if o is None:
                        continue
                    ham[jo, o + ofp] = H[jo, io], iS
                    ham[o, jo + ofm] = H[jo, io], iS

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

        if has_S:
            ham = cls(geom, nnzpr=nc,
                      ortho=False, dtype=H.dtype)
        else:
            ham = cls(geom, nnzpr=nc, dtype=H.dtype)

        # Copy data to the model
        H = H.tocoo()
        if has_S:
            for jo, io, h in zip(H.row, H.col, H.data):
                ham[jo, io] = (h, S[jo, io])

            # Convert S to coo matrix
            S = S.tocoo()
            # If the Hamiltonian for one reason or the other
            # is zero in the diagonal, then we *must* account for
            # this as it isn't captured in the above loop.
            skip_S = np.all(H.row == S.row)
            skip_S = skip_S and np.all(H.col == S.col)
            if not skip_S:
                # Re-convert back to allow index retrieval
                H = H.tocsr()
                for jo, io, s in zip(S.row, S.col, S.data):
                    ham[jo, io] = (H[jo, io], s)

        else:
            for jo, io, h in zip(H.row, H.col, H.data):
                ham[jo, io] = h

        return ham


    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads Hamiltonian from `Sile` using `read_H`.

        Parameters
        ----------
        sile : Sile, str
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
        """ Writes a tight-binding model to the `Sile` as implemented in the :code:`ObjSile.write_es` method """
        self.finalize()

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_es(self, *args, **kwargs)
        else:
            get_sile(sile, 'w').write_es(self, *args, **kwargs)


# For backwards compatibility we also use TightBinding
# NOTE: that this is not sub-classed...
TightBinding = Hamiltonian
