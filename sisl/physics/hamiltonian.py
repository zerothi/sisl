from __future__ import print_function, division

import numpy as np
from numpy import int32, float64, pi
from numpy import take, ogrid
from numpy import add, subtract, multiply, divide
from numpy import cos, sin, arctan2, conj
from numpy import dot, sqrt, square, floor, ceil

from sisl.messages import warn
from sisl._help import _range as range, _str as str
from sisl._help import ensure_array
import sisl._array as _a
from sisl import Geometry
from sisl.eigensystem import EigenSystem
from .distribution_function import distribution as dist_func
from .spin import Spin
from .sparse import SparseOrbitalBZSpin

__all__ = ['Hamiltonian', 'TightBinding']
__all__ += ['EigenState']


class Hamiltonian(SparseOrbitalBZSpin):
    """ Object containing the coupling constants between orbitals.

    The Hamiltonian object contains information regarding the
     - geometry
     - coupling constants between orbitals

    It contains an intrinsic sparse matrix of the Hamiltonian elements.

    Assigning or changing Hamiltonian elements is as easy as with
    standard `numpy` assignments:

    >>> ham = Hamiltonian(...) # doctest: +SKIP
    >>> ham.H[1,2] = 0.1 # doctest: +SKIP

    which assigns 0.1 as the coupling constant between orbital 2 and 3.
    (remember that Python is 0-based elements).
    """

    def __init__(self, geom, dim=1, dtype=None, nnzpr=None, **kwargs):
        """Create Hamiltonian model from geometry

        Initializes a Hamiltonian using the ``geom`` object
        as the underlying geometry for the tight-binding parameters.
        """
        super(Hamiltonian, self).__init__(geom, dim, dtype, nnzpr, **kwargs)

        self.Hk = self.Pk

    def Hk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the Hamiltonian for a given k-point

        Creation and return of the Hamiltonian for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
          H(k) = H_{ij} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.

        Another possible gauge is the orbital distance which can be written as

        .. math::
          H(k) = H_{ij} e^{i k r}

        where :math:`r` is the distance between the orbitals :math:`i` and :math:`j`.
        Currently the second gauge is not implemented (yet).

        Parameters
        ----------
        k : array_like
           the k-point to setup the Hamiltonian at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for orbital distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`) or `numpy.matrix` (`'dense'`).
        spin : int, optional
           if the Hamiltonian is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the Hamiltonian is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        Sk : Overlap matrix at `k`
        """
        pass

    def _get_H(self):
        self._def_dim = self.UP
        return self

    def _set_H(self, key, value):
        if len(key) == 2:
            self._def_dim = self.UP
        self[key] = value

    H = property(_get_H, _set_H)

    def shift(self, E):
        """ Shift the electronic structure by a constant energy

        Parameters
        ----------
        E : float
           the energy (in eV) to shift the electronic structure
        """
        if not self.orthogonal:
            # For non-colinear and SO only the diagonal (real) components
            # should be shifted.
            for i in range(min(self.spin.spins, 2)):
                self._csr._D[:, i] += self._csr._D[:, self.S_idx] * E
        else:
            for i in range(self.shape[0]):
                for j in range(min(self.spin.spins, 2)):
                    self[i, i, j] = self[i, i, j] + E

    def eigenstate(self, k=(0, 0, 0), gauge='R', **kwargs):
        """ Calculate the eigenstates at `k` and return an `EigenState` object containing all eigenstates

        Parameters
        ----------
        k : array_like*3, optional
            the k-point at which to evaluate the eigenstates at
        gauge : str, optional
            the gauge used for calculating the eigenstates
        sparse : bool, optional
            if ``True``, `eigsh` will be called, else `eigh` will be
            called (default).
        **kwargs : dict, optional
            passed arguments to the `eigh` routine

        See Also
        --------
        eigh : the used eigenvalue routine
        eigsh : the used eigenvalue routine

        Returns
        -------
        EigenState
        """
        if kwargs.pop('sparse', False):
            e, v = self.eigsh(k, gauge=gauge, eigvals_only=False, **kwargs)
        else:
            e, v = self.eigh(k, gauge, eigvals_only=False, **kwargs)
        info = {'k': k,
                'gauge': gauge}
        if 'spin' in kwargs:
            info['spin'] = kwargs['spin']
        # Since eigh returns the eigenvectors [:, i] we have to transpose
        return EigenState(e, v.T, self, **info)

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads Hamiltonian from `Sile` using `read_hamiltonian`.

        Parameters
        ----------
        sile : `Sile`, str
            a `Sile` object which will be used to read the Hamiltonian
            and the overlap matrix (if any)
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_hamiltonian(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_hamiltonian(*args, **kwargs)
        else:
            with get_sile(sile) as fh:
                return fh.read_hamiltonian(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes a Hamiltonian to the `Sile` as implemented in the :code:`Sile.write_hamiltonian` method """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_hamiltonian(self, *args, **kwargs)
        else:
            with get_sile(sile, 'w') as fh:
                fh.write_hamiltonian(self, *args, **kwargs)

    def DOS(self, E, k=(0, 0, 0), distribution=None, **kwargs):
        r""" Calculate the DOS at the given energies for a specific `k` point

        Parameters
        ----------
        E : array_like
            energies to calculate the DOS at
        k : array_like, optional
            k-point at which the DOS is calculated
        distribution : func, optional
            a function that accepts :math:`E-\epsilon` as argument and calculates the
            distribution function.
            If ``None`` ``sisl.physics.distribution('gaussian')`` will be used.
        **kwargs: optional
            additional parameters passed to the `eigenstate` routine

        See Also
        --------
        sisl.physics.distribution : setup a distribution function, see details regarding the `distribution` argument
        eigenstate : method used to calculate the eigenstates
        PDOS : Calculate projected DOS
        EigenState.DOS : Underlying method used to calculate the DOS
        EigenState.PDOS : Underlying method used to calculate the projected DOS
        """
        # Calculate the eigenvalues to create the EigenState without the
        # eigenvectors
        e = self.eigh(k, **kwargs)
        return EigenState(e, e, self, k=k, **kwargs).DOS(E, distribution)

    def PDOS(self, E, k=(0, 0, 0), distribution=None, **kwargs):
        r""" Calculate the projected DOS at the given energies for a specific `k` point

        Parameters
        ----------
        E : array_like
            energies to calculate the projected DOS at
        k : array_like, optional
            k-point at which the projected DOS is calculated
        distribution : func, optional
            a function that accepts :math:`E-\epsilon` as argument and calculates the
            distribution function.
            If ``None`` ``sisl.physics.distribution('gaussian')`` will be used.
        **kwargs: optional
            additional parameters passed to the `eigenstate` routine

        See Also
        --------
        sisl.physics.distribution : setup a distribution function, see details regarding the `distribution` argument
        eigenstate : method used to calculate the eigenstates
        DOS : Calculate total DOS
        EigenState.DOS : Underlying method used to calculate the DOS
        EigenState.PDOS : Underlying method used to calculate the projected DOS
        """
        return self.eigenstate(k, **kwargs).PDOS(E, distribution)


class EigenState(EigenSystem):
    """ Eigenstates associated by a Hamiltonian object

    This object can be generated from a Hamiltonian via `Hamiltonian.eigenstate`.
    Subsequent DOS calculations and/or wavefunction calculations (`Grid.psi`) may be
    performed using this object.
    """

    def norm(self, idx=None):
        r""" Return the individual orbital norms for each eigenstate, possibly only for a subset of eigenstates

        The norm is calculated as:

        .. math::
            |\psi|_\nu = \psi^*_\nu [\mathbf S | \psi]_\nu

        while the sum :math:`\sum_\nu|\psi|_\nu\equiv1`.

        Parameters
        ----------
        idx : int or array_like
           only return for the selected indices
        """
        if idx is None:
            idx = range(len(self))
        idx = ensure_array(idx)

        # Now create the correct normalization for each
        opt = {'k': self.info.get('k', (0, 0, 0))}
        if 'gauge' in self.info:
            opt['gauge'] = self.info['gauge']

        is_nc = False
        if isinstance(self.parent, Hamiltonian):
            # Calculate the overlap matrix
            Sk = self.parent.Sk(**opt)
            is_nc = self.parent.spin > Spin('p')
        else:
            # Assume orthogonal basis set and Gamma-point
            # TODO raise warning, should we do this here?
            class _K(object):
                def dot(self, v):
                    return v
            Sk = _K()

        # A true normalization should only be real, hence we force this.
        # TODO, perhaps check that it is correct...
        if is_nc:
            return (conj(self.v[idx, :].T) * Sk.dot(self.v[idx, :].T)).real.T.reshape(len(idx), -1, 2).sum(-1)
        return (conj(self.v[idx, :].T) * Sk.dot(self.v[idx, :].T)).real.T

    def DOS(self, E, distribution=None):
        r""" Calculate the DOS for the provided energies (`E`), using the supplied distribution function

        The Density Of States at a specific energy is calculated via the broadening function:

        .. math::
            \mathrm{DOS}(E) = \sum_i D(E-\epsilon_i) \approx\delta(E-\epsilon_i)

        where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
        used may be a user-defined function. Alternatively a distribution function may
        be aquired from `sisl.physics.distribution`.

        Parameters
        ----------
        E : array_like
            energies to calculate the DOS from
        distribution : func, optional
            a function that accepts :math:`E-\epsilon` as argument and calculates the
            distribution function.
            If ``None`` ``sisl.physics.distribution('gaussian')`` will be used.

        See Also
        --------
        sisl.physics.distribution : a selected set of implemented distribution functions
        PDOS : the projected DOS

        Returns
        -------
        numpy.ndarray : DOS calculated at energies, has same length as `E`
        """
        if distribution is None:
            distribution = dist_func('gaussian')
        elif isinstance(distribution, str):
            distribution = dist_func(distribution)
        DOS = distribution(E - self.e[0])
        for i in range(1, len(self)):
            DOS += distribution(E - self.e[i])
        return DOS

    def PDOS(self, E, distribution=None):
        r""" Calculate the projected-DOS for the provided energies (`E`), using the supplied distribution function

        The projected DOS is calculated as:

        .. math::
             \mathrm{PDOS}_\nu(E) = \sum_i \psi^*_{i,\nu} [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)

        where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
        used may be a user-defined function. Alternatively a distribution function may
        be aquired from `sisl.physics.distribution`.

        In case of an orthogonal basis set :math:`\mathbf S` is equal to the identity matrix.
        Note that `DOS` is the sum of the orbital projected DOS:

        .. math::
            \mathrm{DOS}(E) = \sum_\nu\mathrm{PDOS}_\nu(E)

        Parameters
        ----------
        E : array_like
            energies to calculate the projected-DOS from
        distribution : func, optional
            a function that accepts :math:`E-\epsilon` as argument and calculates the
            distribution function.
            If ``None`` ``sisl.physics.distribution('gaussian')`` will be used.

        See Also
        --------
        sisl.physics.distribution : a selected set of implemented distribution functions
        DOS : the total DOS

        Returns
        -------
        numpy.ndarray : projected DOS calculated at energies, has dimension ``(self.size, len(E))``.
        """
        if distribution is None:
            distribution = dist_func('gaussian')
        elif isinstance(distribution, str):
            distribution = dist_func(distribution)

        # Retrieve options for the Sk calculation
        opt = {'k': self.info.get('k', (0, 0, 0))}
        if 'gauge' in self.info:
            opt['gauge'] = self.info['gauge']

        is_nc = False
        if isinstance(self.parent, Hamiltonian):
            # Calculate the overlap matrix
            Sk = self.parent.Sk(**opt)
            is_nc = self.parent.spin > Spin('p')
            if is_nc:
                Sk = Sk[::2, ::2]
        else:
            # Assume orthogonal basis set and Gamma-point
            # TODO raise warning, should we do this here?
            class _K(object):
                def dot(self, v):
                    return v
            Sk = _K()

        if is_nc:
            def W2M(WD, W12):
                """ Convert spin-box wave-function product to total Q and S(x), S(y), S(z) """
                W = np.empty([W12.size, 1, 4], WD.dtype)
                WD.shape = (-1, 2)
                add(WD[:, 0], WD[:, 1], out=W[:, 0, 0])
                multiply(W12.real, 2, out=W[:, 0, 1])
                multiply(W12.imag, 2, out=W[:, 0, 2])
                subtract(WD[:, 0], WD[:, 1], out=W[:, 0, 3])
                return W

            # Make short-hand
            V = self.v

            v = Sk.dot(V[0].reshape(-1, 2))
            PDOS = W2M((conj(V[0]) * v.ravel()).real, conj(V[0, 1::2]) * v[:, 0]) \
                   * distribution(E - self.e[0]).reshape(1, -1, 1)
            for i in range(1, len(self)):
                v = Sk.dot(V[i].reshape(-1, 2))
                add(PDOS, W2M((conj(V[i]) * v.ravel()).real, conj(V[i, 1::2]) * v[:, 0])
                    * distribution(E - self.e[i]).reshape(1, -1, 1),
                    out=PDOS)

            # Clean-up
            del v, V

        else:

            PDOS = (conj(self.v[0, :]) * Sk.dot(self.v[0, :])).real.reshape(-1, 1) \
                   * distribution(E - self.e[0]).reshape(1, -1)
            for i in range(1, len(self)):
                add(PDOS, (conj(self.v[i]) * Sk.dot(self.v[i])).real.reshape(-1, 1)
                    *distribution(E - self.e[i]).reshape(1, -1), out=PDOS)

        return PDOS

    def psi(self, grid, k=None, spinor=0):
        r""" Add the wave-function (`Orbital.psi`) component of each orbital to the grid

        This routine calculates the real-space wave-function components in the
        specified grid. 

        This is an *in-place* operation that *adds* to the current values in the grid.

        It may be instructive to check that an eigenstate is normalized:

        >>> grid = Grid(...) # doctest: +SKIP
        >>> es = EigenState(...) # doctest: +SKIP
        >>> es.sub(0).psi(grid) # doctest: +SKIP
        >>> (np.abs(grid.grid) ** 2).sum() * grid.dvolume == 1. # doctest: +SKIP

        Note: To calculate :math:`\psi(\mathbf r)` in a unit-cell different from the
        originating geometry, simply pass a grid with a unit-cell smaller than the originating
        supercell.

        The wavefunctions are calculated in real-space via:

        .. math::
            \psi(\mathbf r) = \sum_i\phi_i(\mathbf r) |\psi\rangle_i \exp(-i\mathbf k \mathbf R)

        While for non-colinear/spin-orbit calculations the wavefunctions are determined from the
        spinor component (`spinor`)

        .. math::
            \psi_{\alpha/\beta}(\mathbf r) = \sum_i\phi_i(\mathbf r) |\psi_{\alpha/\beta}\rangle_i \exp(-i\mathbf k \mathbf R)

        where ``spinor==0|1`` determines :math:`\alpha` or :math:`\beta`, respectively.

        Parameters
        ----------
        grid : Grid
           grid on which the wavefunction will be plotted.
           If multiple eigenstates are in this object, they will be summed.
        k : array_like, optional
           k-point associated with wavefunction, by default the inherent k-point used
           to calculate the eigenstate will be used (generally shouldn't be used unless the `EigenState` object
           has not been created via `Hamiltonian.eigenstate`).
        spinor : int, optional
           the spinor for non-colinear/spin-orbit calculations. This is only used if the
           eigenstate object has been created from a parent object with a `Spin` object
           contained, *and* if the spin-configuration is non-colinear or spin-orbit coupling.
           Default to the first spinor component.
        """
        geom = None
        is_nc = False
        if isinstance(self.parent, Geometry):
            geom = self.parent
        elif isinstance(self.parent, Hamiltonian):
            geom = self.parent.geom
            is_nc = self.parent.spin > Spin('p')
        else:
            try:
                if isinstance(self.parent.geom, Geometry):
                    geom = self.parent.geom
            except:
                pass
        if geom is None:
            geom = grid.geometry

        # Do the sum over all eigenstates
        v = self.v.sum(0)
        if is_nc:
            # Select spinor component
            v = v.reshape(-1, 2)[:, spinor]
        if len(v) != geom.no:
            raise ValueError(self.__class__.__name__ + ".psi "
                             "requires the coefficient to have length as the number of orbitals.")

        # Check for k-points
        if k is None:
            k = self.info.get('k', (0, 0, 0))
        k = ensure_array(k, float64)
        kl = (k ** 2).sum() ** 0.5
        has_k = kl > 0.000001

        # Check that input/grid makes sense.
        # If the coefficients are complex valued, then the grid *has* to be
        # complex valued.
        # Likewise if a k-point has been passed.
        is_complex = np.iscomplexobj(v) or has_k
        if is_complex and not np.iscomplexobj(grid.grid):
            raise ValueError(self.__class__.__name__ + ".psi "
                             "has input coefficients as complex values but the grid is real.")

        if is_complex:
            psi_init = _a.zerosz
        else:
            psi_init = _a.zerosd

        # Extract sub variables used throughout the loop
        shape = ensure_array(grid.shape)
        dcell = grid.dcell
        rc = grid.sc.rcell / (2. * pi) * shape.reshape(1, -1)

        # In the following we don't care about division
        # So 1) save error state, 2) turn off divide by 0, 3) calculate, 4) turn on old error state
        old_err = np.seterr(divide='ignore', invalid='ignore')

        addouter = add.outer
        def idx2spherical(ix, iy, iz, offset, dc, R):
            """ Calculate the spherical coordinates from indices """
            rx = addouter(addouter(ix * dc[0, 0], iy * dc[1, 0]), iz * dc[2, 0] - offset[0]).ravel()
            ry = addouter(addouter(ix * dc[0, 1], iy * dc[1, 1]), iz * dc[2, 1] - offset[1]).ravel()
            rz = addouter(addouter(ix * dc[0, 2], iy * dc[1, 2]), iz * dc[2, 2] - offset[2]).ravel()
            # Total size of the indices
            n = rx.size
            # Calculate radius ** 2
            rr = square(rx)
            add(rr, square(ry), out=rr)
            add(rr, square(rz), out=rr)
            # Reduce our arrays to where the radius is "fine"
            idx = (rr <= R ** 2).nonzero()[0]
            rx = take(rx, idx)
            ry = take(ry, idx)
            arctan2(ry, rx, out=rx) # theta == rx
            rz = take(rz, idx)
            sqrt(take(rr, idx), out=ry) # rr == ry
            divide(rz, ry, out=rz) # cos_phi == rz
            rz[ry == 0.] = 0
            return n, idx, ry, rx, rz

        # Figure out the max-min indices with a spacing of 1 radians
        rad1 = pi / 180
        theta, phi = ogrid[-pi:pi:rad1, 0:pi:rad1]
        cphi, sphi = cos(phi), sin(phi)
        ctheta_sphi = cos(theta) * sphi
        stheta_sphi = sin(theta) * sphi
        del sphi
        nrxyz = (theta.size, phi.size, 3)
        del theta, phi, rad1

        # First we calculate the min/max indices for all atoms
        idx_mm = _a.emptyi([geom.na, 2, 3])
        rxyz = _a.emptyd(nrxyz)
        origo = grid.sc.origo.reshape(1, -1)
        for atom, ia in geom.atom.iter(True):
            if len(ia) == 0:
                continue
            R = atom.maxR()

            # Reshape
            rxyz.shape = nrxyz
            rxyz[..., 0] = ctheta_sphi
            rxyz[..., 1] = stheta_sphi
            rxyz[..., 2] = cphi
            rxyz.shape = (-1, 3)

            idx = dot(rc, rxyz.T)
            idx_m = idx.min(1) * R
            idx_M = idx.max(1) * R

            # Now do it for all the atoms to get indices of the middle of
            # the atoms
            # The coordinates are relative to origo, so we need to shift (when writing a grid
            # it is with respect to origo)
            xyz = geom.xyz[ia, :] - origo
            idx = dot(rc, xyz.T).T

            # Get min-max for all atoms, note we should first do the floor here
            idx_mm[ia, 0, :] = floor(idx_m.reshape(1, -3) + idx).astype(int32)
            idx_mm[ia, 1, :] = ceil(idx_M.reshape(1, -3) + idx).astype(int32)

        # Now we have min-max for all atoms
        # When we run the below loop all indices can be retrieved by looking
        # up in the above table.

        # Before continuing, we can easily clean up the temporary arrays
        del ctheta_sphi, stheta_sphi, cphi, nrxyz, rxyz, origo, idx

        aranged = _a.aranged

        # Loop over all atoms in the full supercell structure
        for IA in range(geom.na_s):

            # Get atomic coordinate
            xyz = geom.axyz(IA) - grid.origo
            # Reduce to unit-cell atom
            ia = IA % geom.na
            # Get current atom
            atom = geom.atom[ia]

            if ia == 0:
                # start over for every new supercell
                io = -1
                isc = geom.a2isc(IA)
                if has_k:
                    phase = np.exp(-1j * dot(dot(dot(geom.rcell, k), geom.cell), isc))
                else:
                    # do not force complex values for Gamma only (user is responsible)
                    phase = 1

            # Extract maximum R
            R = atom.maxR()
            if R <= 0.:
                warn("Atom '{}' does not have a wave-function, skipping atom.".format(atom))
                # Skip this atom
                io += atom.no
                continue

            # Get indices in the supercell grid
            idxm = idx_mm[ia, 0, :] + shape * isc
            idxM = idx_mm[ia, 1, :] + shape * isc

            # Fast check whether we can skip this point
            if idxm[0] >= shape[0] or idxm[1] >= shape[1] or \
               idxm[2] >= shape[2] or \
               idxM[0] < 0 or idxM[1] < 0 or idxM[2] < 0:
                io += atom.no
                continue

            # Truncate values
            if idxm[0] < 0:
                idxm[0] = 0
            if idxM[0] >= shape[0]:
                idxM[0] = shape[0] - 1
            if idxm[1] < 0:
                idxm[1] = 0
            if idxM[1] >= shape[1]:
                idxM[1] = shape[1] - 1
            if idxm[2] < 0:
                idxm[2] = 0
            if idxM[2] >= shape[2]:
                idxM[2] = shape[2] - 1

            # Now idxm/M contains min/max indices used
            # Convert to xyz-coordinate
            sx = slice(idxm[0], idxM[0]+1)
            sy = slice(idxm[1], idxM[1]+1)
            sz = slice(idxm[2], idxM[2]+1)

            # Convert to spherical coordinates
            n, idx, r, theta, phi = idx2spherical(aranged(idxm[0], idxM[0] + 0.5),
                                                  aranged(idxm[1], idxM[1] + 0.5),
                                                  aranged(idxm[2], idxM[2] + 0.5), xyz, dcell, R)

            # Allocate a temporary array where we add the psi elements
            psi = psi_init(n)

            # Loop on orbitals on this atom, grouped by radius
            for os in atom.iter(True):

                # Get the radius of orbitals (os)
                oR = os[0].R

                if oR <= 0.:
                    warn("Orbital(s) '{}' does not have a wave-function, skipping orbital.".format(os))
                    # Skip these orbitals
                    io += len(os)
                    continue

                # Downsize to the correct indices
                if np.allclose(oR, R):
                    idx1 = idx.view()
                    r1 = r.view()
                    theta1 = theta.view()
                    phi1 = phi.view()
                else:
                    idx1 = (r <= oR).nonzero()[0]
                    # Reduce arrays
                    r1 = take(r, idx1)
                    theta1 = take(theta, idx1)
                    phi1 = take(phi, idx1)
                    idx1 = take(idx, idx1)

                # Loop orbitals with the same radius
                for o in os:
                    io += 1

                    # Evaluate psi component of the wavefunction and add it for this atom
                    psi[idx1] += o.psi_spher(r1, theta1, phi1, cos_phi=True) * (v[io] * phase)

            # Clean-up
            del idx1, r1, theta1, phi1, idx, r, theta, phi

            # Convert to correct shape and add the current atom contribution to the wavefunction
            psi.shape = idxM - idxm + 1
            grid.grid[sx, sy, sz] += psi

            # Clean-up
            del psi

        # Reset the error code for division
        np.seterr(**old_err)


# For backwards compatibility we also use TightBinding
# NOTE: that this is not sub-classed...
TightBinding = Hamiltonian
