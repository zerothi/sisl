"""
=====================================================================
Electron related functions and classes (:mod:`sisl.physics.electron`)
=====================================================================

.. module:: sisl.physics.electron
   :noindex:

In :mod:`sisl` electronic structure calculations are relying on routines
specific for electrons. For instance density of states calculations from
electronic eigenvalues and other quantities.

This module implements the necessary tools required for calculating
DOS, PDOS, spin moments of non-colinear calculations and plotting
real-space wavefunctions.

.. autosummary::
   :toctree:

   DOS
   PDOS
   spin_moment
   wavefunction
   CoefficientElectron
   StateElectron
   StateCElectron
   EigenvalueElectron
   EigenvectorElectron
   EigenstateElectron

"""
from __future__ import print_function, division

import numpy as np
from numpy import conj, dot, take, ogrid
from numpy import arctan2, cos, sin, pi, int32
from numpy import add, sqrt, divide

from sisl.geometry import Geometry
import sisl._array as _a
from sisl.messages import info, warn, tqdm_eta
from sisl._help import dtype_complex_to_real, _range as range
from .distributions import distribution as dist_func
from .spin import Spin
from .sparse import SparseOrbitalBZSpin
from .state import Coefficient, State, StateC


__all__ = ['DOS', 'PDOS', 'spin_moment', 'wavefunction']
__all__ += ['CoefficientElectron', 'StateElectron', 'StateCElectron']
__all__ += ['EigenvalueElectron', 'EigenvectorElectron', 'EigenstateElectron']


def DOS(E, eig, distribution='gaussian'):
    r""" Calculate the density of states (DOS) for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{DOS}(E)` is calculated as:
    .. math::

       \mathrm{DOS}(E) = \sum_i D(E-\epsilon_i) \approx\delta(E-\epsilon_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be retrieved from `sisl.physics.distribution`.

    Parameters
    ----------
    E : array_like
       energies to calculate the DOS at
    eig : array_like
       eigenvalues
    distribution : func or str, optional
       a function that accepts :math:`E-\epsilon` as argument and calculates the
       distribution function.

    See Also
    --------
    sisl.physics.distribution : a selected set of implemented distribution functions
    PDOS : projected DOS (same as this, but projected onto each orbital)
    spin_moment: spin moment of eigenvectors

    Returns
    -------
    numpy.ndarray : DOS calculated at energies, has same length as `E`
    """
    if isinstance(distribution, str):
        distribution = dist_func(distribution)

    DOS = distribution(E - eig[0])
    for i in range(1, len(eig)):
        DOS += distribution(E - eig[i])
    return DOS


def PDOS(E, eig, eig_v, S=None, distribution='gaussian', spin=None):
    r""" Calculate the projected density of states (PDOS) for a set of energies, `E`, with a distribution function

    The :math:`\mathrm{PDOS}(E)` is calculated as:
    .. math::

       \mathrm{PDOS}_\nu(E) = \sum_i \psi^*_{i,\nu} [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)

    where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
    used may be a user-defined function. Alternatively a distribution function may
    be aquired from `sisl.physics.distribution`.

    In case of an orthogonal basis set :math:`\mathbf S` is equal to the identity matrix.
    Note that `DOS` is the sum of the orbital projected DOS:

    .. math::

       \mathrm{DOS}(E) = \sum_\nu\mathrm{PDOS}_\nu(E)

    For non-colinear calculations (this includes spin-orbit calculations) the PDOS is additionally
    separated into 4 components (in this order):

    - Total projected DOS
    - Projected spin magnetic moment along :math:`x` direction
    - Projected spin magnetic moment along :math:`y` direction
    - Projected spin magnetic moment along :math:`z` direction

    These are calculated using the Pauli matrices :math:`\sigma_x`, :math:`\sigma_y` and :math:`\sigma_z`:

    .. math::

       \mathrm{PDOS}_\nu^\sigma(E) &= \sum_i \psi^*_{i,\nu} \sigma_z \sigma_z [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
       \\
       \mathrm{PDOS}_\nu^x(E) &= \sum_i \psi^*_{i,\nu} \sigma_x [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
       \\
       \mathrm{PDOS}_\nu^y(E) &= \sum_i \psi^*_{i,\nu} \sigma_y [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)
       \\
       \mathrm{PDOS}_\nu^z(E) &= \sum_i \psi^*_{i,\nu} \sigma_z [\mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)

    Note that the total PDOS may be calculated using :math:`\sigma_i\sigma_i` where :math:`i` may be either of :math:`x`,
    :math:`y` or :math:`z`.

    Parameters
    ----------
    E : array_like
       energies to calculate the projected-DOS from
    eig : array_like
       eigenvalues
    eig_v : array_like
       eigenvectors
    S : array_like, optional
       overlap matrix used in the :math:`\rangle\psi^*|\mathbf S|\psi\langle` calculation. If `None` the identity
       matrix is assumed. For non-colinear calculations this matrix may be halve the size of ``len(eig_v[0, :])`` to
       trigger the non-colinear calculation of PDOS.
    distribution : func or str, optional
       a function that accepts :math:`E-\epsilon` as argument and calculates the
       distribution function.
    spin : str or Spin, optional
       the spin configuration. This is generally only needed when the eigenvectors correspond to a non-colinear
       calculation.

    See Also
    --------
    sisl.physics.distribution : a selected set of implemented distribution functions
    DOS : total DOS (same as summing over orbitals)
    spin_moment: spin moment of eigenvectors

    Returns
    -------
    numpy.ndarray
        projected DOS calculated at energies, has dimension ``(eig_v.shape[1], len(E))``.
        For non-colinear calculations it will be ``(4, eig_v.shape[1] // 2, len(E))``, ordered as
        indicated in the above list.
    """
    if isinstance(distribution, str):
        distribution = dist_func(distribution)

    # Figure out whether we are dealing with a non-colinear calculation
    if S is None:
        class __S(object):
            __slots__ = []
            shape = (eig_v.shape[1], eig_v.shape[1])
            @staticmethod
            def dot(v):
                return v
        S = __S()

    if spin is None:
        if S.shape[1] == eig_v.shape[1] // 2:
            spin = Spin('nc')
            S = S[::2, ::2]
        else:
            spin = Spin()

    # check for non-colinear (or SO)
    if spin.kind > Spin.POLARIZED:
        # Non colinear eigenvectors
        if S.shape[1] == eig_v.shape[1]:
            # Since we are going to reshape the eigen-vectors
            # to more easily get the mixed states, we can reduce the overlap matrix
            S = S[::2, ::2]

        # Initialize data
        PDOS = np.empty([4, eig_v.shape[1] // 2, len(E)], dtype=dtype_complex_to_real(eig_v.dtype))

        d = distribution(E - eig[0]).reshape(1, -1)
        v = S.dot(eig_v[0].reshape(-1, 2))
        D = (conj(eig_v[0]) * v.ravel()).real.reshape(-1, 2) # diagonal PDOS
        PDOS[0, :, :] = D.sum(1).reshape(-1, 1) * d # total DOS
        PDOS[3, :, :] = (D[:, 0] - D[:, 1]).reshape(-1, 1) * d # z-dos
        D = (conj(eig_v[0, 1::2]) * 2 * v[:, 0]).reshape(-1, 1) # psi_down * psi_up * 2
        PDOS[1, :, :] = D.real * d # x-dos
        PDOS[2, :, :] = D.imag * d # y-dos
        for i in range(1, len(eig)):
            d = distribution(E - eig[i]).reshape(1, -1)
            v = S.dot(eig_v[i].reshape(-1, 2))
            D = (conj(eig_v[i]) * v.ravel()).real.reshape(-1, 2)
            PDOS[0, :, :] += D.sum(1).reshape(-1, 1) * d
            PDOS[3, :, :] += (D[:, 0] - D[:, 1]).reshape(-1, 1) * d
            D = (conj(eig_v[i, 1::2]) * 2 * v[:, 0]).reshape(-1, 1)
            PDOS[1, :, :] += D.real * d
            PDOS[2, :, :] += D.imag * d

    else:
        PDOS = (conj(eig_v[0]) * S.dot(eig_v[0])).real.reshape(-1, 1) \
               * distribution(E - eig[0]).reshape(1, -1)
        for i in range(1, len(eig)):
            PDOS[:, :] += (conj(eig_v[i]) * S.dot(eig_v[i])).real.reshape(-1, 1) \
                          * distribution(E - eig[i]).reshape(1, -1)

    return PDOS


def spin_moment(eig_v, S=None):
    r""" Calculate the spin magnetic moment (also known as spin texture)

    This calculation only makes sense for non-colinear calculations.

    The returned quantities are given in this order:

    - Total spin magnetic moment
    - Spin magnetic moment along :math:`x` direction
    - Spin magnetic moment along :math:`y` direction
    - Spin magnetic moment along :math:`z` direction

    These are calculated using the Pauli matrices :math:`\sigma_x`, :math:`\sigma_y` and :math:`\sigma_z`:

    .. math::

       \mathbf{S}_i^\sigma &= \sum_i \langle \psi^*_i |\sigma_z \mathbf S \sigma_z | \psi_i \rangle
       \\
       \mathbf{S}_i^x(E) &= \sum_i \langle \psi^*_i | \sigma_x \mathbf S | \psi_i \rangle
       \\
       \mathbf{S}_i^y(E) &= \sum_i \langle \psi^*_i | \sigma_y \mathbf S | \psi_i \rangle
       \\
       \mathbf{S}_i^z(E) &= \sum_i \langle \psi^*_i | \sigma_z \mathbf S | \psi_i \rangle

    Note that the total spin magnetic moment, S may be calculated using
    :math:`\sigma_i\sigma_i` where :math:`i` may be either of :math:`x`, :math:`y` or :math:`z`.

    Parameters
    ----------
    eig_v : array_like
       eigenvectors
    S : array_like, optional
       overlap matrix used in the :math:`\rangle\psi^*|\mathbf S|\psi\langle` calculation. If `None` the identity
       matrix is assumed. The overlap matrix should correspond to the system and :math:`k` point the eigenvectors
       have been evaluated at.

    Notes
    -----
    This routine cannot check whether the input eigenvectors originate from a non-colinear calculation.
    If a non-polarized eigenvector is passed to this routine, the output will have no physical meaning.

    See Also
    --------
    DOS : total DOS
    PDOS : projected DOS

    Returns
    -------
    numpy.ndarray
        spin moments per eigenvector with final dimension ``(4, eig_v.shape[0])``.
    """
    if S is None:
        class __S(object):
            __slots__ = []
            shape = (eig_v.shape[1] // 2, eig_v.shape[1] // 2)
            @staticmethod
            def dot(v):
                return v
        S = __S()

    if S.shape[1] == eig_v.shape[1]:
        S = S[::2, ::2]

    # Initialize
    s = np.empty([4, eig_v.shape[0]], dtype=dtype_complex_to_real(eig_v.dtype))

    # TODO consider doing this all in a few lines
    # TODO Since there are no energy dependencies here we can actually do all
    # TODO dot products in one go and then use b-casting rules. Should be much faster
    # TODO but also way more memory demanding!
    v = S.dot(eig_v[0].reshape(-1, 2))
    D = (conj(eig_v[0]) * v.ravel()).real.reshape(-1, 2) # diagonal elements
    s[0, 0] = D.sum() # total spin moment
    s[3, 0] = (D[:, 0] - D[:, 1]).sum() # S_z
    D = 2 * (conj(eig_v[0, 1::2]) * v[:, 0]).sum() # psi_down * psi_up * 2
    s[1, 0] = D.real # S_x
    s[2, 0] = D.imag # S_y
    for i in range(1, len(eig_v)):
        v = S.dot(eig_v[i].reshape(-1, 2))
        D = (conj(eig_v[i]) * v.ravel()).real.reshape(-1, 2)
        s[0, i] += D.sum()
        s[3, i] += (D[:, 0] - D[:, 1]).sum()
        D = 2 * (conj(eig_v[i, 1::2]) * v[:, 0]).sum()
        s[1, i] += D.real
        s[2, i] += D.imag

    return s


def wavefunction(v, grid, geometry=None, k=None, spinor=0, spin=None, eta=False):
    r""" Add the wave-function (`Orbital.psi`) component of each orbital to the grid

    This routine calculates the real-space wave-function components in the
    specified grid.

    This is an *in-place* operation that *adds* to the current values in the grid.

    It may be instructive to check that an eigenstate is normalized:

    >>> grid = Grid(...) # doctest: +SKIP
    >>> psi(state, grid) # doctest: +SKIP
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

    where ``spinor in [0, 1]`` determines :math:`\alpha` or :math:`\beta`, respectively.

    Notes
    -----
    Currently this method only works for :math:`\Gamma` states

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
    eta : bool, optional
       Display a console progressbar.
    """
    if geometry is None:
        geometry = grid.geometry
        warn('psi was not passed a geometry associated, will use the geometry associated with the Grid.')
    if geometry is None:
        raise SislError('psi did not find a usable Geometry through keywords or the Grid!')

    # In case the user has passed several vectors we sum them to plot the summed state
    if v.ndim == 2:
        v = v.sum(0)

    if spin is None:
        if len(v) // 2 == geometry.no:
            # We can see from the input that the vector *must* be a non-colinear calculation
            v = v.reshape(-1, 2)[:, spinor]
            info('psi assumes the input wavefunction coefficients to originate from a non-colinear calculation!')

    elif spin.kind > Spin.POLARIZED:
        # For non-colinear cases the user selects the spinor component.
        v = v.reshape(-1, 2)[:, spinor]

    if len(v) != geometry.no:
        raise ValueError("psi require wavefunction coefficients corresponding to number of orbitals in the geometry.")

    # Check for k-points
    k = _a.asarrayd(k)
    kl = (k ** 2).sum() ** 0.5
    has_k = kl > 0.000001
    if has_k:
        raise NotImplementedError('psi for k != Gamma does not produce correct wavefunctions!')

    # Check that input/grid makes sense.
    # If the coefficients are complex valued, then the grid *has* to be
    # complex valued.
    # Likewise if a k-point has been passed.
    is_complex = np.iscomplexobj(v) or has_k
    if is_complex and not np.iscomplexobj(grid.grid):
        raise SislError("psi input coefficients are complex, while grid only contains real.")

    if is_complex:
        psi_init = _a.zerosz
    else:
        psi_init = _a.zerosd

    # Extract sub variables used throughout the loop
    shape = _a.asarrayi(grid.shape)
    dcell = grid.dcell
    ic = grid.sc.icell * shape.reshape(1, -1)
    geom_shape = dot(ic, geometry.cell.T).T

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
        n = rx.shape[0]
        # Calculate radius ** 2
        rr = rx ** 2 + ry ** 2 + rz ** 2
        # Reduce our arrays to where the radius is "fine"
        idx = (rr <= R ** 2).nonzero()[0]
        rx = rx[idx]
        ry = ry[idx]
        rz = rz[idx]
        arctan2(ry, rx, out=rx) # theta == rx
        sqrt(rr[idx], out=ry) # rr == ry
        divide(rz, ry, out=rz) # cos_phi == rz
        rz[ry == 0.] = 0
        return n, idx, ry, rx, rz

    # Figure out the max-min indices with a spacing of 1 radian
    rad1 = pi / 180
    theta, phi = ogrid[-pi:pi:rad1, 0:pi:rad1]
    cphi, sphi = cos(phi), sin(phi)
    ctheta_sphi = cos(theta) * sphi
    stheta_sphi = sin(theta) * sphi
    del sphi
    nrxyz = (theta.size, phi.size, 3)
    del theta, phi, rad1

    # First we calculate the min/max indices for all atoms
    idx_mm = _a.emptyi([geometry.na, 2, 3])
    rxyz = _a.emptyd(nrxyz)
    rxyz[..., 0] = ctheta_sphi
    rxyz[..., 1] = stheta_sphi
    rxyz[..., 2] = cphi
    # Reshape
    rxyz.shape = (-1, 3)
    idx = dot(ic, rxyz.T)
    idxm = idx.min(1).reshape(1, 3)
    idxM = idx.max(1).reshape(1, 3)
    del ctheta_sphi, stheta_sphi, cphi, idx, rxyz, nrxyz

    origo = grid.sc.origo.reshape(1, -1)
    for atom, ia in geometry.atom.iter(True):
        if len(ia) == 0:
            continue
        R = atom.maxR()

        # Now do it for all the atoms to get indices of the middle of
        # the atoms
        # The coordinates are relative to origo, so we need to shift (when writing a grid
        # it is with respect to origo)
        xyz = geometry.xyz[ia, :] - origo
        idx = dot(ic, xyz.T).T

        # Get min-max for all atoms, note we should first do the floor here
        idx_mm[ia, 0, :] = idxm * R + idx
        idx_mm[ia, 1, :] = idxM * R + idx

    # Now we have min-max for all atoms
    # When we run the below loop all indices can be retrieved by looking
    # up in the above table.

    # Before continuing, we can easily clean up the temporary arrays
    del origo, idx

    aranged = _a.aranged
    allclose = np.allclose

    # In case this grid does not have a Geometry associated
    # We can *perhaps* easily attach a geometry with the given
    # atoms in the unit-cell
    sc = grid.sc.copy()
    if grid.geometry is None:
        # Create the actual geometry that encompass the grid
        ia, xyz, _ = geometry.inf_within(sc)
        if len(ia) > 0:
            grid.set_geometry(Geometry(xyz, geometry.atom[ia], sc=sc))

    # Instead of looping all atoms in the supercell we find the exact atoms
    # and their supercell indices.
    add_R = _a.zerosd(3) + geometry.maxR()
    sc = sc + np.diag(add_R * 2)
    sc.origo = sc.origo[:] - add_R

    # Retrieve all atoms within the grid supercell
    # (and the neighbours that connect into the cell)
    IA, XYZ, ISC = geometry.inf_within(sc)

    r_k = dot(geometry.rcell, k)
    r_k_cell = dot(r_k, geometry.cell)
    phase = 1

    # Retrieve progressbar
    eta = tqdm_eta(len(IA), 'psi', 'atom', eta)

    # Loop over all atoms in the full supercell structure
    for ia, xyz, isc in zip(IA, XYZ - grid.origo.reshape(1, 3), ISC):
        # Get current atom
        atom = geometry.atom[ia]

        # Extract maximum R
        R = atom.maxR()
        if R <= 0.:
            warn("Atom '{}' does not have a wave-function, skipping atom.".format(atom))
            eta.update()
            continue

        # Get indices in the supercell grid
        idx = (isc.reshape(3, 1) * geom_shape).sum(0)
        idxm = (idx_mm[ia, 0, :] + idx).astype(int32)
        idxM = (idx_mm[ia, 1, :] + idx).astype(int32) + 1

        # Fast check whether we can skip this point
        if idxm[0] >= shape[0] or idxm[1] >= shape[1] or idxm[2] >= shape[2] or \
           idxM[0] <= 0 or idxM[1] <= 0 or idxM[2] <= 0:
            eta.update()
            continue

        # Truncate values
        if idxm[0] < 0:
            idxm[0] = 0
        if idxM[0] > shape[0]:
            idxM[0] = shape[0]
        if idxm[1] < 0:
            idxm[1] = 0
        if idxM[1] > shape[1]:
            idxM[1] = shape[1]
        if idxm[2] < 0:
            idxm[2] = 0
        if idxM[2] > shape[2]:
            idxM[2] = shape[2]

        # Now idxm/M contains min/max indices used
        # Convert to spherical coordinates
        n, idx, r, theta, phi = idx2spherical(aranged(idxm[0], idxM[0]),
                                              aranged(idxm[1], idxM[1]),
                                              aranged(idxm[2], idxM[2]), xyz, dcell, R)

        # Get initial orbital
        io = geometry.a2o(ia)

        if has_k:
            phase = np.exp(-1j * (dot(r_k_cell, isc)))
            # TODO
            # Possibly the phase should be an additional
            # array for the position in the unit-cell!
            #   + np.exp(-1j * dot(r_k, spher2cart(r, theta, np.arccos(phi)).T) )

        # Allocate a temporary array where we add the psi elements
        psi = psi_init(n)

        # Loop on orbitals on this atom, grouped by radius
        for os in atom.iter(True):

            # Get the radius of orbitals (os)
            oR = os[0].R

            if oR <= 0.:
                warn("Orbital(s) '{}' does not have a wave-function, skipping orbital!".format(os))
                # Skip these orbitals
                io += len(os)
                continue

            # Downsize to the correct indices
            if allclose(oR, R):
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
                # Evaluate psi component of the wavefunction and add it for this atom
                psi[idx1] += o.psi_spher(r1, theta1, phi1, cos_phi=True) * (v[io] * phase)
                io += 1

        # Clean-up
        del idx1, r1, theta1, phi1, idx, r, theta, phi

        # Convert to correct shape and add the current atom contribution to the wavefunction
        psi.shape = idxM - idxm
        grid.grid[idxm[0]:idxM[0], idxm[1]:idxM[1], idxm[2]:idxM[2]] += psi

        # Clean-up
        del psi

        # Step progressbar
        eta.update()

    eta.close()

    # Reset the error code for division
    np.seterr(**old_err)


class _common_State(object):
    __slots__ = []

    def __is_nc(self):
        """ Internal routine to check whether this is a non-colinear calculation """
        try:
            spin = self.parent.spin
        except:
            spin = Spin()
        return spin.kind > Spin.POLARIZED

    def Sk(self, format='csr', spin=None):
        r""" Retrieve the overlap matrix corresponding to the originating parent structure.

        When ``self.parent`` is a Hamiltonian this will return :math:`\mathbf S(k)` for the
        :math:`k`-point these eigenstates originate from

        Parameters
        ----------
        format: str, optional
           the returned format of the overlap matrix. This only takes effect for
           non-orthogonal parents.
        spin : Spin, optional
           for non-colinear spin configurations the *fake* overlap matrix returned
           will have halve the size of the input matrix. If you want the *full* overlap
           matrix, simply do not specify the `spin` argument.
        """

        if isinstance(self.parent, SparseOrbitalBZSpin):
            # Calculate the overlap matrix
            if not self.parent.orthogonal:
                opt = {'k': self.info.get('k', (0, 0, 0)),
                       'format': format}
                gauge = self.info.get('gauge', None)
                if not gauge is None:
                    opt['gauge'] = gauge
                return self.parent.Sk(**opt)

        class __FakeSk(object):
            """ Replacement object which superseedes a matrix """
            __slots__ = []
            shape = (self.shape[1], self.shape[1])
            @staticmethod
            def dot(v):
                return v

        if spin is None:
            return __FakeSk()
        if spin.kind > Spin.POLARIZED:
            class __FakeSk(object):
                """ Replacement object which superseedes a matrix """
                __slots__ = []
                shape = (self.shape[1] // 2, self.shape[1] // 2)
                @staticmethod
                def dot(v):
                    return v
        return __FakeSk()

    def norm2(self, sum=True):
        r""" Return a vector with the norm of each state :math:`\langle\psi|\psi\rangle`

        Parameters
        ----------
        sum : bool, optional
           if true the summed orbital square is returned (a vector). For false a matrix
           with normalization squared per orbital is returned.

        Returns
        -------
        numpy.ndarray
            the normalization on each orbital for each state
        """
        # Retrieve the overlap matrix (FULL S is required for NC)
        S = self.Sk()

        # TODO, perhaps check that it is correct... and fix multiple transposes
        if sum:
            if self.__is_nc():
                return (conj(self.state) * S.dot(self.state.T).T).real.reshape(len(self), -1, 2).sum(-1).sum(0)
            return (conj(self.state) * S.dot(self.state.T).T).real.sum(0)
        if self.__is_nc():
            return (conj(self.state) * S.dot(self.state.T).T).real.reshape(len(self), -1, 2).sum(-1)
        return (conj(self.state) * S.dot(self.state.T).T).real

    def spin_moment(self):
        r""" Calculate spin moment

        This routine calls `sisl.physics.electrons.spin_moment` with appropriate arguments
        and returns the spin moment.

        See `sisl.physics.electrons.spin_moment` for argument details.
        """
        try:
            spin = self.parent.spin
        except:
            spin = None
        return spin_moment(self.state, self.Sk(spin=spin))

    def wavefunction(self, grid, spinor=0, eta=False):
        r""" Expand the coefficients as the wavefunction on `grid` *as-is*

        See `sisl.physics.electron.wavefunction` for argument details.
        """
        try:
            spin = self.parent.spin
        except:
            spin = None

        if isinstance(self.parent, Geometry):
            geometry = self.parent
        else:
            try:
                geometry = self.parent.geometry
            except:
                geometry = None

        # Retrieve k
        k = self.info.get('k', _a.zerosd(3))

        wavefunction(self.state, grid, geometry=geometry, k=k, spinor=spinor,
                     spin=spin, eta=eta)

    # TODO to be deprecated
    psi = wavefunction


class CoefficientElectron(Coefficient):
    __slots__ = []


class StateElectron(_common_State, State):
    __slots__ = []


class StateCElectron(_common_State, StateC):
    __slots__ = []


class EigenvalueElectron(CoefficientElectron):
    """ Eigenvalues of electronic states, no eigenvectors retained

    This holds routines that enable the calculation of density of states.
    """
    __slots__ = []

    @property
    def eig(self):
        return self.c

    def DOS(self, E, distribution='gaussian'):
        r""" Calculate DOS for provided energies, `E`.

        This routine calls `sisl.physics.electrons.DOS` with appropriate arguments
        and returns the DOS.

        See `sisl.physics.electrons.DOS` for argument details.
        """
        return DOS(E, self.eig, distribution)


class EigenvectorElectron(StateElectron):
    """ Eigenvectors of electronic states, no eigenvalues retained

    This holds routines that enable the calculation of spin moments.
    """
    __slots__ = []


class EigenstateElectron(StateCElectron):
    """ Eigen states of electrons with eigenvectors and eigenvalues.

    This holds routines that enable the calculation of (projected) density of states,
    spin moments (spin texture).
    """
    __slots__ = []

    @property
    def eig(self):
        return self.c

    def DOS(self, E, distribution='gaussian'):
        r""" Calculate DOS for provided energies, `E`.

        This routine calls `sisl.physics.electrons.DOS` with appropriate arguments
        and returns the DOS.

        See `sisl.physics.electrons.DOS` for argument details.
        """
        return DOS(E, self.c, distribution)

    def PDOS(self, E, distribution='gaussian'):
        r""" Calculate PDOS for provided energies, `E`.

        This routine calls `sisl.physics.electrons.PDOS` with appropriate arguments
        and returns the PDOS.

        See `sisl.physics.electrons.PDOS` for argument details.
        """
        try:
            spin = self.parent.spin
        except:
            spin = None
        return PDOS(E, self.c, self.state, self.Sk(spin=spin), distribution, spin)
