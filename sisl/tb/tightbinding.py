"""
Tight-binding class to create tight-binding models.
"""
from __future__ import print_function, division

import warnings
from numbers import Integral

import numpy as np
import scipy.linalg as sli


from sisl import Atom, Geometry, Quaternion

__all__ = ['TightBinding']


class TightBinding(object):
    """
    The defining tight-binding model for constructing models
    in python for easy manipulation in tbtrans.

    This class allows the creation of sparse tight-binding
    models on extreme scales.

    Any tight-binding model has an underlying geometry which
    is a requirement for the construction of a full tight-binding
    parameter set.
    """

    # The order of the Energy
    # I.e. whether energy should be in other units than Ry
    # This conversion is made: [eV] ** _E_order
    _E_order = 1

    def __init__(self, geom, *args, **kwargs):
        """Create tight-binding model from geometry

        Initializes a tight-binding model using the :code:`geom` object
        as the underlying geometry for the tight-binding parameters.
        """
        self.geom = geom

        self.reset(**kwargs)

    ######### Definitions of overrides ############
    def __len__(self):
        """ Returns number of non-zero elements in the model """
        return self._nnzs

    def __repr__(self):
        """ Representation of the tight-binding model """
        s = self.geom.__repr__()
        return s + '\nNumber of non-zero elements {0}'.format(len(self))

    def __getitem__(self, key):
        """ Return tight-binding parameters for the index

        Returns the tight-binding parameters for the index.
        """
        if isinstance(key, Integral):
            # We allow the retrieval of the full row
            i = key
            ind = self.ptr[i]
            n = self.ncol[i]
            return self.col[ind:ind + n], self._TB[ind:ind + n, :]

        # It must be both row and column
        i, j = key
        ind = np.where(
            self.col[
                self.ptr[i]:self.ptr[i] +
                self.ncol[i]] == j)[0]
        if len(ind) > 0:
            return self._TB[self.ptr[i] + ind[0], :]
        else:
            # this is a zero element
            return np.array([0., 0.], self._TB.dtype)

    def __getattr__(self, attr):
        """ Returns the attributes from the underlying geometry

        Any attribute not found in the tight-binding model will
        be looked up in the underlying geometry.
        """
        return getattr(self.geom, attr)

    def __setitem__(self, key, val):
        """ Sparse creation of the sparsity pattern

        Override set item for slicing operations and enables easy
        setting of tight-binding parameters in a sparse matrix

        It does allow fancy slicing in both dimensions with limited usability

        Ok, it is not pretty, it is not fast, but it works!
        """
        # unpack index
        i, j = key
        # Copy over the values of the tight-binding model
        v = np.array([val], self._TB.dtype).flatten()
        if not isinstance(i, Integral):
            # Recursively handle index,index = val
            # designators.
            if len(i) > 1:
                for ii in i:
                    self[ii, j] = val
                return
            i = int(i[0])

        # step pointer of all above this
        ptr = self.ptr[i]
        ncol = self.ncol[i]
        # Create the column indices in a strict manner
        jj = np.array([j], np.int32).flatten()
        lj = jj.shape[0]
        if ncol > 0:
            # Checks whether any values in either array exists
            # if so we remove those from the jj
            idx = np.intersect1d(
                jj,
                self.col[
                    ptr:ptr +
                    ncol],
                assume_unique=True)
        else:
            idx = []

        if len(idx) > 0:
            where = np.where

            # Here we truncate jj to the "new" values,
            # this allows us to both overwrite and add new values to the
            # sparsity pattern (simultaneously)
            jj = np.setdiff1d(jj, idx, assume_unique=True)
            lj = jj.shape[0]

            # the values corresponding to idx already exists,
            # we overwrite that value
            if isinstance(j, Integral):
                ix = where(j == self.col[ptr:ptr + ncol])[0][0]
                self._TB[ptr + ix, :] = v
            else:
                # remember that idx is the intersection values
                for ij in idx:
                    ix = where(ij == self.col[ptr:ptr + ncol])[0][0]
                    self._TB[ptr + ix, :] = v

            # if no new values are left we return immediately
            if lj == 0:
                return

        while self.ptr[i + 1] - ptr < ncol + lj:
            # Ensure that it is not-set as finalized
            # There is no need to set it all the time.
            # Simply because the first call to finalize
            # will reduce the sparsity pattern, which
            # on first expansion calls this part.
            self._finalized = False

            # Expand size of the sparsity pattern
            # We add 10 new elements on each extension
            # This may be too reductionists, however,
            # it should happen rarely
            self.ptr[i + 1:] += 10
            self.col = np.insert(
                self.col,
                ptr + ncol,
                np.empty(
                    [10],
                    self.col.dtype))
            self._TB = np.insert(self._TB, ptr +
                                 ncol, np.empty([10, 2], self._TB.dtype), axis=0)

        # Step to the placement of the new values
        ptr += ncol
        # set current value
        self.col[ptr:ptr + lj] = jj
        self._TB[ptr:ptr + lj, :] = v
        # Append the new columns
        self.ncol[i] += lj
        # Increment number of non-zero elements
        self._nnzs += lj

    ############# DONE creating easy overrides #################

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

    def reset(self, nc=None, dtype=np.float64):
        """
        The sparsity pattern is cleaned and every thing
        is reset.

        The object will be the same as if it had been
        initialized with the same geometry as it were
        created with.
        """
        # I know that this is not the most efficient way to
        # access a C-array, however, for constructing a
        # sparse pattern, it should be faster if memory elements
        # are closer...
        # Hence, this choice of having H and S like this

        # We check the first atom and its neighbours, we then
        # select max(5,len(nc) * 4)
        if nc is None:
            nc = self.geom.close(0)
            nc = max(5, len(nc) * 4)

        self._no = self.geom.no

        # Reset the sparsity pattern
        self.ncol = np.zeros([self.no], np.int32)
        # Create the interstitial pointer for each orbital
        self.ptr = np.cumsum(np.array([nc] * (self.no + 1), np.int32)) - nc
        self._nnzs = 0
        self.col = np.empty([self.ptr[-1]], np.int32)

        # Denote the tight-binding model as _not_ finalized
        # Before saving, or anything being done, it _has_ to be
        # finalized.
        self._finalized = False

        # Initialize TB size
        # NOTE, this is not zeroed!
        self._TB = np.empty([self.ptr[-1], 2], dtype=dtype)

    def construct(self, dR, param):
        """ Automatically construct the tight-binding model based on ``dR`` and associated hopping integrals ``param``.

        Parameters
        ----------
        dR : array_like
           radii parameters for tight-binding parameters.
           Must have same length as ``param`` or one less.
           If one less it will be extended with ``dR[0]/100``
        param : array_like
           tight-binding parameters corresponding to the ``dR``
           ranges. ``param[0,:]`` are the tight-binding parameter
           for the all atoms within ``dR[0]`` of each atom.
        """

        if len(dR) + 1 == len(param):
            R = np.hstack((dR[0] / 100, dR))
        elif len(dR) == len(param):
            R = dR.copy()
        else:
            raise ValueError(("Length of dR and param must be the same "
                              "or dR one shorter than param. "
                              "One tight-binding parameter for each radii."))

        if len(param[0]) != 2:
            raise ValueError(("Number of parameters "
                              "for each element is not 2. "
                              "You must make len(param[0] == 2."))

        if np.any(np.diff(self.geom.lasto) > 1):
            warnings.warn(("Automatically setting a tight-binding model "
                           "for systems with atoms having more than 1 "
                           "orbital is not adviced. Please do it your-self."))

        if len(self.geom) < 2000:
            # there is no need to do anything complex
            # for small systems
            for ia in self.geom:
                # Find atoms close to 'ia'
                idx = self.geom.close(ia, dR=R)
                for ix, h in zip(idx, param):
                    # Set the tight-binding parameters
                    self[ia, ix] = h

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
                for ix, h in zip(idx, param):
                    # Set the tight-binding parameters
                    self[ia, ix] = h

        return self

    def finalize(self):
        """ Finalizes the tight-binding model

        Finalizes the tight-binding model so that no new sparse
        elements can be added.

        Sparse elements can still be changed.
        """
        if self._finalized:
            return

        self._finalized = True
        ptr = self.ncol[0]
        if np.unique(self.col[:ptr]).shape[0] != ptr:
            raise ValueError('You cannot have two hoppings between the same ' +
                             'orbitals, something has went terribly wrong.')

        if self.no != self.geom.no:
            raise ValueError(
                ("You have changed the geometry in the TightBinding "
                 "object, this is not allowed."))

        if self.no > 1:
            # We truncate all the connections
            for io in range(1, self.no):
                cptr = self.ptr[io]
                # Update actual pointer position
                self.ptr[io] = ptr
                no = self.ncol[io]
                if no == 0:
                    continue  # no non-assigned elements
                self.col[ptr:ptr + no] = self.col[cptr:cptr + no]
                self._TB[ptr:ptr + no, :] = self._TB[cptr:cptr + no, :]
                # we also assert no two connections
                if np.unique(self.col[ptr:ptr + no]).shape[0] != no:
                    raise ValueError(
                        'You cannot have two hoppings between the same ' +
                        'orbitals ({}), something has went terribly wrong.'.format(io))
                ptr += no

        # Correcting the size of the pointer array
        self.ptr[self.no] = ptr
        if ptr != self._nnzs:
            print(ptr, self._nnzs)
            raise ValueError('Final size in the tight-binding finalization ' +
                             'went wrong.')

        # Truncate values to correct size
        self._TB = self._TB[:self._nnzs, :]
        self.col = self.col[:self._nnzs]

        # Sort the indices, this is not strictly required, but
        # it should speed up things.
        for io in range(self.no):
            ptr = self.ptr[io]
            no = self.ncol[io]
            if no == 0:
                continue
            # Sort the indices
            si = np.argsort(self.col[ptr:ptr + no])
            self.col[ptr:ptr + no] = self.col[ptr + si]
            self._TB[ptr:ptr + no, :] = self._TB[ptr + si, :]

        # Check that the couplings are symmetric
        TB = self.tocsr(k=[0, 0, 0])[0]
        t1 = TB.nnz
        t2 = (TB + TB.T).nnz
        if t1 != t2:
            warnings.warn(
                'Tight-binding model does not retain symmetric couplings, this might be problematic: nnz(H) = {}, nnz(H+H^T) = {}.'.format(
                    t1,
                    t2),
                UserWarning)
        del TB

    @property
    def nnzs(self):
        """ Returns number of non-zero elements in the tight-binding model """
        return self._nnzs

    @property
    def no(self):
        """ Returns number of orbitals as used when the object was created """
        return self._no

    def tocsr(self, k=None):
        """ Returns :code:`scipy.sparse` matrices for the tight-binding model

        Returns a CSR sparse matrix for both the Hamiltonian
        and the overlap matrix using the scipy package.

        This method depends on scipy.
        """
        # Ensure completeness
        self.finalize()

        # Create csr sparse formats.
        # We import here as the user might not want to
        # rely on this feature.
        from scipy.sparse import csr_matrix

        if k is None:
            kw = {'shape': (self.no, self.no_s),
                  'dtype': self._TB.dtype}
            if self._TB.shape[1] == 1:
                return csr_matrix((self._TB[:, 0], self.col, self.ptr), **kw)
            return (csr_matrix((self._TB[:, 0], self.col, self.ptr), **kw),
                    csr_matrix((self._TB[:, 1], self.col, self.ptr), **kw))
        else:
            k = np.asarray(k, np.float64)
            # Setup the Hamiltonian for this k-point
            Hfull, Sfull = self.tocsr()

            s = (self.no, self.no)

            # Create k-space Hamiltonian
            H = csr_matrix(s, dtype=np.complex128)
            S = csr_matrix(s, dtype=np.complex128)

            # Get the reciprocal lattice vectors dotted with k
            rcell = self.rcell
            kr = np.dot(rcell, np.asarray(k)) * np.pi * 2.
            for si in range(self.sc.n_s):
                isc = self.sc_off[si, :]
                phase = np.exp(-1j * np.dot(kr, np.dot(self.cell, isc)))
                H += Hfull[:, si * self.no:(si + 1) * self.no] * phase
                S += Sfull[:, si * self.no:(si + 1) * self.no] * phase
            del Hfull, Sfull
            return (H, S)

    def eigh(
            self,
            k=None,
            atoms=None,
            eigvals_only=True,
            overwrite_a=True,
            overwrite_b=True,
            *args,
            **kwargs):
        """ Returns the eigenvalues of the tight-binding model

        Setup the Hamiltonian and overlap matrix with respect to
        the given k-point, then reduce the space to the specified atoms
        and calculate the eigenvalues.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigh`
        """
        H, S = self.tocsr(k=k)
        # Reduce sparsity pattern
        if not atoms is None:
            orbs = self.a2o(atoms)
            # Reduce space
            H = H[orbs, orbs]
            S = S[orbs, orbs]
        return sli.eigh(
            H.todense(),
            S.todense(),
            *args,
            eigvals_only=eigvals_only,
            overwrite_a=overwrite_a,
            overwrite_b=overwrite_b,
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
        H, S = self.tocsr()
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
        tb = self.__class__(geom, nc=np.amax(self.ncol))

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
                o, ofp, ofm = sco2sco(self.geom, io, tb.geom, seps, axis)
                if o is None:
                    continue
                tb[jo, o + ofp] = iH, S[jo, io]
                tb[o, jo + ofm] = iH, S[jo, io]

            if np.any(sH.indices != sS.indices):

                # Ensure that S is also cut
                for io, iS in zip(sS.indices, sS.data):
                    # Get the equivalent orbital in the smaller cell
                    o, ofp, ofm = sco2sco(self.geom, io, tb.geom, seps, axis)
                    if o is None:
                        continue
                    tb[jo, o + ofp] = H[jo, io], iS
                    tb[o, jo + ofm] = H[jo, io], iS

        return tb

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

        # Get the Hamiltonian and overlap matrices
        H, S = self.tocsr()

        # Create new object
        TB = self.__class__(g)

        raise NotImplemented(('tiling a TightBinding model has not been '
                              'fully implemented yet.'))

    def repeat(self, reps, axis):
        """ Refer to `tile` instead """
        raise NotImplemented(('repeating a TightBinding model has not been '
                              'fully implemented yet, use tile instead.'))

    @classmethod
    def sp2tb(cls, geom, H, S):
        """ Returns a tight-binding model from a preset H, S and Geometry

        H and S are overwritten on exit
        """

        # Calculate number of connections
        nc = 0
        # Ensure csr format
        H = H.tocsr()
        S = S.tocsr()
        for i in range(geom.no):
            nc = max(nc, H[i, :].getnnz())
            nc = max(nc, S[i, :].getnnz())

        tb = cls(geom, nc=nc)

        # Copy data to the model
        H = H.tocoo()
        for jo, io, h in zip(H.row, H.col, H.data):
            tb[jo, io] = (h, S[jo, io])

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
                tb[jo, io] = (H[jo, io], s)

        return tb

    def write(self, sile, *args, **kwargs):
        """ Writes a tight-binding model to the `Sile` as implemented in the :code:`ObjSile.write_tb` method """
        self.finalize()

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_tb(self, *args, **kwargs)
        else:
            get_sile(sile, 'w').write_tb(self, *args, **kwargs)


if __name__ == "__main__":
    import datetime
    from sisl.geom import graphene

    # Create graphene unit-cell
    gr = graphene()
    gr.sc.set_nsc([3, 3, 1])

    # Create nearest-neighbour tight-binding
    # graphene lattice constant 1.42
    dR = (0.1, 1.5)
    on = (0., 1.)
    nn = (-0.5, 0.)

    tb = TightBinding(gr)
    for ia in tb.geom:
        idx_a = tb.geom.close(ia, dR=dR)
        tb[ia, idx_a[0]] = on
        tb[ia, idx_a[1]] = nn
    print(len(tb))
    tb.finalize()
    print('H\n', tb.tocsr()[0])
    print('H\n', tb.tocsr(k=[.25, .25, 0])[0])
    print('eig\n', tb.eigh(k=[3. / 4, .5, 0], eigvals_only=True))

    print('\nCheck expansion')
    tb = TightBinding(gr)
    tb.reset(1)  # force only one connection (this should force expansion)
    for ia in tb.geom:
        idx_a = tb.geom.close(ia, dR=dR)
        tb[ia, idx_a[0]] = on
        tb[ia, idx_a[1]] = nn
    print(len(tb))
    print('H\n', tb.tocsr()[0])
    print('H\n', tb.tocsr(k=[.25, .25, 0])[0])

    # Lets try and create a huge sample
    print('\n\nStarting time... ' + str(datetime.datetime.now().time()))
    tb = TightBinding(gr.tile(41, 0).tile(41, 1))
    for ias, idxs in tb.iter_block(13):
        for ia in ias:
            idx_a = tb.geom.close(ia, dR=dR)
            tb[ia, idx_a[0]] = on
            tb[ia, idx_a[1]] = nn
    print(len(tb))
    print('Ending time... ' + str(datetime.datetime.now().time()))
