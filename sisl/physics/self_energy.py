import numpy as np
from numpy import dot, conjugate
from numpy import subtract
from numpy import empty, zeros, eye, delete
from numpy import zeros_like, empty_like
from numpy import complex128
from numpy import abs as _abs

from sisl.messages import warn, info
from sisl.utils.mathematics import fnorm
from sisl.utils.ranges import array_arange
from sisl._help import array_replace
import sisl._array as _a
from sisl.linalg import linalg_info, solve, inv
from sisl.linalg.base import _compute_lwork
from sisl.physics.brillouinzone import MonkhorstPack
from sisl.physics.bloch import Bloch


__all__ = ['SelfEnergy', 'SemiInfinite']
__all__ += ['RecursiveSI']
__all__ += ['RealSpaceSE', 'RealSpaceSI']


class SelfEnergy:
    r""" Self-energy object able to calculate the dense self-energy for a given sparse matrix

    The self-energy object contains a `SparseGeometry` object which, in it-self
    contains the geometry.

    This is the base class for self-energies.
    """

    def __init__(self, *args, **kwargs):
        r""" Self-energy class for constructing a self-energy. """
        pass

    def __len__(self):
        r"""Dimension of the self-energy"""
        raise NotImplementedError

    @staticmethod
    def se2scat(SE):
        r""" Calculate the scattering matrix from the self-energy

        .. math::
            \boldsymbol\Gamma = i(\boldsymbol\Sigma - \boldsymbol \Sigma ^\dagger)

        Parameters
        ----------
        SE : matrix
            self-energy matrix
        """
        return 1j * (SE - conjugate(SE.T))

    def _setup(self, *args, **kwargs):
        """ Class specific setup routine """
        pass

    def self_energy(self, *args, **kwargs):
        raise NotImplementedError

    def scattering_matrix(self, *args, **kwargs):
        r""" Calculate the scattering matrix by first calculating the self-energy

        Any arguments that is passed to this method is directly passed to `self_energy`.

        See `self_energy` for details.

        This corresponds to:

        .. math::
            \boldsymbol\Gamma = i(\boldsymbol\Sigma - \boldsymbol \Sigma ^\dagger)

        Examples
        --------

        Calculating both the self-energy and the scattering matrix.

        >>> SE = SelfEnergy(...)
        >>> self_energy = SE.self_energy(0.1)
        >>> gamma = SE.scattering_matrix(0.1)

        For a huge performance boost, please do:

        >>> SE = SelfEnergy(...)
        >>> self_energy = SE.self_energy(0.1)
        >>> gamma = SE.se2scat(self_energy)

        Notes
        -----
        When using *both* the self-energy *and* the scattering matrix please use `se2scat` after having
        calculated the self-energy, this will be *much*, *MUCH* faster!

        See Also
        --------
        se2scat : converting the self-energy to the scattering matrix
        self_energy : the used routine to calculate the self-energy before calculating the scattering matrix
        """
        return self.se2scat(self.self_energy(*args, **kwargs))

    def __getattr__(self, attr):
        r""" Overload attributes from the hosting object """
        pass


class SemiInfinite(SelfEnergy):
    r""" Self-energy object able to calculate the dense self-energy for a given `SparseGeometry` in a semi-infinite chain.

    Parameters
    ----------
    spgeom : SparseGeometry
       any sparse geometry matrix which may return matrices
    infinite : str
       axis specification for the semi-infinite direction (`+A`/`-A`/`+B`/`-B`/`+C`/`-C`)
    eta : float, optional
       the default imaginary part of the self-energy calculation
    """

    def __init__(self, spgeom, infinite, eta=1e-4):
        """ Create a `SelfEnergy` object from any `SparseGeometry` """
        self.eta = eta

        # Determine whether we are in plus/minus direction
        if infinite.startswith('+'):
            self.semi_inf_dir = 1
        elif infinite.startswith('-'):
            self.semi_inf_dir = -1
        else:
            raise ValueError(self.__class__.__name__ + ": infinite keyword does not start with `+` or `-`.")

        # Determine the direction
        INF = infinite.upper()
        if INF.endswith('A'):
            self.semi_inf = 0
        elif INF.endswith('B'):
            self.semi_inf = 1
        elif INF.endswith('C'):
            self.semi_inf = 2

        # Check that the Hamiltonian does have a non-zero V along the semi-infinite direction
        if spgeom.geometry.sc.nsc[self.semi_inf] == 1:
            warn('Creating a semi-infinite self-energy with no couplings along the semi-infinite direction')

        # Finalize the setup by calling the class specific routine
        self._setup(spgeom)

    def __str__(self):
        """ String representation of SemiInfinite """
        return  '{0}{{direction: {1}{2}}}'.format(self.__class__.__name__,
                                                  {-1: '-', 1: '+'}.get(self.semi_inf_dir),
                                                  {0: 'A', 1: 'B', 2: 'C'}.get(self.semi_inf))


class RecursiveSI(SemiInfinite):
    """ Self-energy object using the Lopez-Sancho Lopez-Sancho algorithm """

    def __getattr__(self, attr):
        """ Overload attributes from the hosting object """
        return getattr(self.spgeom0, attr)

    def __str__(self):
        """ Representation of the RecursiveSI model """
        direction = {-1: '-', 1: '+'}
        axis = {0: 'A', 1: 'B', 2: 'C'}
        return '{0}{{direction: {1}{2},\n {3}\n}}'.format(self.__class__.__name__,
                                                          direction[self.semi_inf_dir], axis[self.semi_inf],
                                                          str(self.spgeom0).replace('\n', '\n '),
        )

    def _setup(self, spgeom):
        """ Setup the Lopez-Sancho internals for easy axes """

        # Create spgeom0 and spgeom1
        self.spgeom0 = spgeom.copy()
        nsc = np.copy(spgeom.geometry.sc.nsc)
        nsc[self.semi_inf] = 1
        self.spgeom0.set_nsc(nsc)

        # For spgeom1 we have to do it slightly differently
        old_nnz = spgeom.nnz
        self.spgeom1 = spgeom.copy()
        nsc[self.semi_inf] = 3

        # Already now limit the sparse matrices
        self.spgeom1.set_nsc(nsc)
        if self.spgeom1.nnz < old_nnz:
            warn(self.__class__.__name__ + ": SparseGeometry has connections across the first neighbouring cell. "
                 "These values will be forced to 0 as the principal cell-interaction is a requirement")

        # I.e. we will delete all interactions that are un-important
        n_s = self.spgeom1.geometry.sc.n_s
        n = self.spgeom1.shape[0]
        # Figure out the matrix columns we should set to zero
        nsc = [None] * 3
        nsc[self.semi_inf] = self.semi_inf_dir
        # Get all supercell indices that we should delete
        idx = np.delete(_a.arangei(n_s),
                        _a.arrayi(self.spgeom1.geometry.sc.sc_index(nsc))) * n

        cols = array_arange(idx, idx + n)
        # Delete all values in columns, but keep them to retain the supercell information
        self.spgeom1._csr.delete_columns(cols, keep_shape=True)

    def __len__(self):
        r"""Dimension of the self-energy"""
        return len(self.spgeom0)

    def green(self, E, k=(0, 0, 0), dtype=None, eps=1e-14, **kwargs):
        r""" Return a dense matrix with the bulk Green function at energy `E` and k-point `k` (default Gamma).

        Parameters
        ----------
        E : float/complex
          energy at which the calculation will take place
        k : array_like, optional
          k-point at which the Green function should be evaluated.
          the k-point should be in units of the reciprocal lattice vectors.
        dtype : numpy.dtype
          the resulting data type
        eps : float, optional
          convergence criteria for the recursion
        **kwargs : dict, optional
           arguments passed directly to the ``self.parent.Pk`` method (not ``self.parent.Sk``), for instance ``spin``

        Returns
        -------
        self-energy : the self-energy corresponding to the semi-infinite direction
        """
        if E.imag == 0.:
            E = E.real + 1j * self.eta

        # Get k-point
        k = _a.asarrayd(k)

        if dtype is None:
            dtype = complex128

        sp0 = self.spgeom0
        sp1 = self.spgeom1

        # As the SparseGeometry inherently works for
        # orthogonal and non-orthogonal basis, there is no
        # need to have two algorithms.
        GB = sp0.Sk(k, dtype=dtype, format='array') * E - sp0.Pk(k, dtype=dtype, format='array', **kwargs)
        n = GB.shape[0]

        ab = empty([n, 2, n], dtype=dtype)
        shape = ab.shape

        # Get direct arrays
        alpha = ab[:, 0, :].view()
        beta = ab[:, 1, :].view()

        # Get solve step arary
        ab2 = ab.view()
        ab2.shape = (n, 2 * n)

        if sp1.orthogonal:
            alpha[:, :] = sp1.Pk(k, dtype=dtype, format='array', **kwargs)
            beta[:, :] = conjugate(alpha.T)
        else:
            P = sp1.Pk(k, dtype=dtype, format='array', **kwargs)
            S = sp1.Sk(k, dtype=dtype, format='array')
            alpha[:, :] = P - S * E
            beta[:, :] = conjugate(P.T) - conjugate(S.T) * E
            del P, S

        # Get faster methods since we don't want overhead of solve
        gesv = linalg_info('gesv', dtype)

        getrf = linalg_info('getrf', dtype)
        getri = linalg_info('getri', dtype)
        getri_lwork = linalg_info('getri_lwork', dtype)
        lwork = int(1.01 * _compute_lwork(getri_lwork, n))
        def inv(A):
            lu, piv, info = getrf(A, overwrite_a=True)
            if info == 0:
                x, info = getri(lu, piv, lwork=lwork, overwrite_lu=True)
            if info != 0:
                raise ValueError(self.__class__.__name__ + '.green could not compute the inverse.')
            return x

        while True:
            _, _, tab, info = gesv(GB, ab2, overwrite_a=False, overwrite_b=False)
            tab.shape = shape
            if info != 0:
                raise ValueError(self.__class__.__name__ + '.green could not solve G x = B system!')

            # Update bulk Green function
            subtract(GB, dot(alpha, tab[:, 1, :]), out=GB)
            subtract(GB, dot(beta, tab[:, 0, :]), out=GB)

            # Update forward/backward
            alpha[:, :] = dot(alpha, tab[:, 0, :])
            beta[:, :] = dot(beta, tab[:, 1, :])

            # Convergence criteria, it could be stricter
            if _abs(alpha).max() < eps:
                # Return the pristine Green function
                del ab, alpha, beta, ab2, tab
                return inv(GB)

        raise ValueError(self.__class__.__name__+'.green could not converge Green function calculation')

    def self_energy(self, E, k=(0, 0, 0), dtype=None, eps=1e-14, bulk=False, **kwargs):
        r""" Return a dense matrix with the self-energy at energy `E` and k-point `k` (default Gamma).

        Parameters
        ----------
        E : float/complex
          energy at which the calculation will take place
        k : array_like, optional
          k-point at which the self-energy should be evaluated.
          the k-point should be in units of the reciprocal lattice vectors.
        dtype : numpy.dtype
          the resulting data type
        eps : float, optional
          convergence criteria for the recursion
        bulk : bool, optional
          if true, :math:`E\cdot \mathbf S - \mathbf H -\boldsymbol\Sigma` is returned, else
          :math:`\boldsymbol\Sigma` is returned (default).
        **kwargs : dict, optional
           arguments passed directly to the ``self.parent.Pk`` method (not ``self.parent.Sk``), for instance ``spin``

        Returns
        -------
        self-energy : the self-energy corresponding to the semi-infinite direction
        """
        if E.imag == 0.:
            E = E.real + 1j * self.eta

        # Get k-point
        k = _a.asarrayd(k)

        if dtype is None:
            dtype = complex128

        sp0 = self.spgeom0
        sp1 = self.spgeom1

        # As the SparseGeometry inherently works for
        # orthogonal and non-orthogonal basis, there is no
        # need to have two algorithms.
        GB = sp0.Sk(k, dtype=dtype, format='array') * E - sp0.Pk(k, dtype=dtype, format='array', **kwargs)
        n = GB.shape[0]

        ab = empty([n, 2, n], dtype=dtype)
        shape = ab.shape

        # Get direct arrays
        alpha = ab[:, 0, :].view()
        beta = ab[:, 1, :].view()

        # Get solve step arary
        ab2 = ab.view()
        ab2.shape = (n, 2 * n)

        if sp1.orthogonal:
            alpha[:, :] = sp1.Pk(k, dtype=dtype, format='array', **kwargs)
            beta[:, :] = conjugate(alpha.T)
        else:
            P = sp1.Pk(k, dtype=dtype, format='array', **kwargs)
            S = sp1.Sk(k, dtype=dtype, format='array')
            alpha[:, :] = P - S * E
            beta[:, :] = conjugate(P.T) - conjugate(S.T) * E
            del P, S

        # Surface Green function (self-energy)
        if bulk:
            GS = GB.copy()
        else:
            GS = zeros_like(GB)

        # Get faster methods since we don't want overhead of solve
        gesv = linalg_info('gesv', dtype)

        # Specifying dot with 'out' argument should be faster
        tmp = empty_like(GS)
        while True:
            _, _, tab, info = gesv(GB, ab2, overwrite_a=False, overwrite_b=False)
            tab.shape = shape
            if info != 0:
                raise ValueError(self.__class__.__name__ + '.self_energy could not solve G x = B system!')

            dot(alpha, tab[:, 1, :], tmp)
            # Update bulk Green function
            subtract(GB, tmp, out=GB)
            subtract(GB, dot(beta, tab[:, 0, :]), out=GB)
            # Update surface self-energy
            subtract(GS, tmp, out=GS)

            # Update forward/backward
            alpha[:, :] = dot(alpha, tab[:, 0, :])
            beta[:, :] = dot(beta, tab[:, 1, :])

            # Convergence criteria, it could be stricter
            if _abs(alpha).max() < eps:
                # Return the pristine Green function
                del ab, alpha, beta, ab2, tab, GB
                if bulk:
                    return GS
                return - GS

        raise ValueError(self.__class__.__name__+': could not converge self-energy calculation')

    def self_energy_lr(self, E, k=(0, 0, 0), dtype=None, eps=1e-14, bulk=False, **kwargs):
        r""" Return two dense matrices with the left/right self-energy at energy `E` and k-point `k` (default Gamma).

        Note calculating the LR self-energies simultaneously requires that their chemical potentials are the same.
        I.e. only when the reference energy is equivalent in the left/right schemes does this make sense.

        Parameters
        ----------
        E : float/complex
          energy at which the calculation will take place, if complex, the hosting ``eta`` won't be used.
        k : array_like, optional
          k-point at which the self-energy should be evaluated.
          the k-point should be in units of the reciprocal lattice vectors.
        dtype : numpy.dtype, optional
          the resulting data type, default to ``np.complex128``
        eps : float, optional
          convergence criteria for the recursion
        bulk : bool, optional
          if true, :math:`E\cdot \mathbf S - \mathbf H -\boldsymbol\Sigma` is returned, else
          :math:`\boldsymbol\Sigma` is returned (default).
        **kwargs : dict, optional
           arguments passed directly to the ``self.parent.Pk`` method (not ``self.parent.Sk``), for instance ``spin``

        Returns
        -------
        left : the left self-energy
        right : the right self-energy
        """
        if E.imag == 0.:
            E = E.real + 1j * self.eta

        # Get k-point
        k = _a.asarrayd(k)

        if dtype is None:
            dtype = complex128

        sp0 = self.spgeom0
        sp1 = self.spgeom1

        # As the SparseGeometry inherently works for
        # orthogonal and non-orthogonal basis, there is no
        # need to have two algorithms.
        SmH0 = sp0.Sk(k, dtype=dtype, format='array') * E - sp0.Pk(k, dtype=dtype, format='array', **kwargs)
        GB = SmH0.copy()
        n = GB.shape[0]

        ab = empty([n, 2, n], dtype=dtype)
        shape = ab.shape

        # Get direct arrays
        alpha = ab[:, 0, :].view()
        beta = ab[:, 1, :].view()

        # Get solve step arary
        ab2 = ab.view()
        ab2.shape = (n, 2 * n)

        if sp1.orthogonal:
            alpha[:, :] = sp1.Pk(k, dtype=dtype, format='array', **kwargs)
            beta[:, :] = conjugate(alpha.T)
        else:
            P = sp1.Pk(k, dtype=dtype, format='array', **kwargs)
            S = sp1.Sk(k, dtype=dtype, format='array')
            alpha[:, :] = P - S * E
            beta[:, :] = conjugate(P.T) - conjugate(S.T) * E
            del P, S

        # Surface Green function (self-energy)
        if bulk:
            GS = GB.copy()
        else:
            GS = zeros_like(GB)

        # Get faster methods since we don't want overhead of solve
        gesv = linalg_info('gesv', dtype)

        # Specifying dot with 'out' argument should be faster
        tmp = empty_like(GS)
        while True:
            _, _, tab, info = gesv(GB, ab2, overwrite_a=False, overwrite_b=False)
            tab.shape = shape
            if info != 0:
                raise ValueError(self.__class__.__name__ + '.self_energy_lr could not solve G x = B system!')

            dot(alpha, tab[:, 1, :], tmp)
            # Update bulk Green function
            subtract(GB, tmp, out=GB)
            subtract(GB, dot(beta, tab[:, 0, :]), out=GB)
            # Update surface self-energy
            subtract(GS, tmp, out=GS)

            # Update forward/backward
            alpha[:, :] = dot(alpha, tab[:, 0, :])
            beta[:, :] = dot(beta, tab[:, 1, :])

            # Convergence criteria, it could be stricter
            if _abs(alpha).max() < eps:
                # Return the pristine Green function
                del ab, alpha, beta, ab2, tab
                if self.semi_inf_dir == 1:
                    # GS is the "right" self-energy
                    if bulk:
                        return GB - GS + SmH0, GS
                    return GS - GB + SmH0, - GS
                # GS is the "left" self-energy
                if bulk:
                    return GS, GB - GS + SmH0
                return - GS, GS - GB + SmH0

        raise ValueError(self.__class__.__name__+': could not converge self-energy (LR) calculation')


class RealSpaceSE(SelfEnergy):
    r""" Bulk real-space self-energy (or Green function) for a given physical object with periodicity

    The real-space self-energy is calculated via the k-averaged Green function:

    .. math::
        \boldsymbol\Sigma^\mathcal{R}(E) = \mathbf S^\mathcal{R} (E+i\eta) - \mathbf H^\mathcal{R}
             - \Big[\sum_{\mathbf k} \mathbf G_{\mathbf k}(E)\Big]^{-1}

    The method actually used is relying on `RecursiveSI` and `~sisl.physics.Bloch` objects.

    Parameters
    ----------
    parent : SparseOrbitalBZ
        a physical object from which to calculate the real-space self-energy.
        The parent object *must* have only 3 supercells along the direction where
        self-energies are used.
    semi_axis : int
        semi-infinite direction (where self-energies are used and thus *exact* precision)
    k_axes : array_like of int
        the axes where k-points are desired. 1 or 2 values are required and the `semi_axis`
        cannot be one of them
    unfold : (3,) of int
        number of times the `parent` structure is tiled along each direction
        The resulting Green function/self-energy ordering is always tiled along
        the semi-infinite direction first, and then the other directions in order.
    eta : float, optional
        imaginary part in the self-energy calculations (default 1e-4 eV)
    dk : float, optional
        fineness of the default integration grid, specified in units of Ang, default to 1000 which
        translates to 1000 k-points along reciprocal cells of length 1. Ang^-1.
    bz : BrillouinZone, optional
        integration k-points, if not passed the number of k-points will be determined using
        `dk` and time-reversal symmetry will be determined by `trs`, the number of points refers
        to the unfolded system.
    trs : bool, optional
        whether time-reversal symmetry is used in the BrillouinZone integration, default
        to true.

    Examples
    --------
    >>> graphene = geom.graphene()
    >>> H = Hamiltonian(graphene)
    >>> H.construct([(0.1, 1.44), (0, -2.7)])
    >>> rse = RealSpaceSE(H, 0, 1, (3, 4, 1))
    >>> rse.green(0.1)

    The Brillouin zone integration is determined naturally.

    >>> graphene = geom.graphene()
    >>> H = Hamiltonian(graphene)
    >>> H.construct([(0.1, 1.44), (0, -2.7)])
    >>> rse = RealSpaceSE(H, 0, 1, (3, 4, 1))
    >>> rse.set_options(eta=1e-3, bz=MonkhorstPack(H, [1, 1000, 1]))
    >>> rse.initialize()
    >>> rse.green(0.1) # eta = 1e-3
    >>> rse.green(0.1 + 1j * 1e-4) # eta = 1e-4

    Manually specify Brillouin zone integration and default :math:`\eta` value.
    """

    def __init__(self, parent, semi_axis, k_axes, unfold=(1, 1, 1), **options):
        """ Initialize real-space self-energy calculator """
        self.parent = parent

        # Store axes
        self._semi_axis = semi_axis
        self._k_axes = np.sort(_a.asarrayi(k_axes).ravel())

        # Check axis
        s_ax = self._semi_axis
        k_ax = self._k_axes
        if s_ax in k_ax:
            raise ValueError(self.__class__.__name__ + ' found the self-energy direction to be '
                             'the same as one of the k-axes, this is not allowed.')
        if np.any(self.parent.nsc[k_ax] < 3):
            raise ValueError(self.__class__.__name__ + ' found k-axes without periodicity. '
                             'Correct k_axes via .set_options.')
        if self.parent.nsc[s_ax] != 3:
            raise ValueError(self.__class__.__name__ + ' found the self-energy direction to be '
                             'incompatible with the parent object. It *must* have 3 supercells along the '
                             'semi-infinite direction.')

        # Local variables for the completion of the details
        self._unfold = _a.arrayi([max(1, un) for un in unfold])

        self._options = {
            # fineness of the integration k-grid [Ang]
            'dk': 1000,
            # whether TRS is used (G + G.T) * 0.5
            'trs': True,
            # imaginary part used in the Green function calculation (unless an imaginary energy is passed)
            'eta': 1e-4,
            # The BrillouinZone used for integration
            'bz': None,
        }
        self.set_options(**options)
        self.initialize()

    def __len__(self):
        r"""Dimension of the self-energy"""
        return len(self.parent) * np.prod(self._unfold)

    def __str__(self):
        """ String representation of RealSpaceSE """
        d = {'class': self.__class__.__name__}
        for i in range(3):
            d[f'u{i}'] = self._unfold[i]
        d['semi'] = self._semi_axis
        d['k'] = str(list(self._k_axes))
        d['parent'] = str(self.parent).replace('\n', '\n ')
        d['bz'] = str(self._options['bz']).replace('\n', '\n ')
        d['trs'] = str(self._options['trs'])
        return  ('{class}{{unfold: [{u0}, {u1}, {u2}],\n '
                 'semi-axis: {semi}, k-axes: {k}, trs: {trs},\n '
                 'bz: {bz},\n '
                 '{parent}\n}}').format(**d)

    def set_options(self, **options):
        r""" Update options in the real-space self-energy

        After updating options one should re-call `initialize` for consistency.

        Parameters
        ----------
        eta : float, optional
            imaginary part in the self-energy calculations (default 1e-4 eV)
        dk : float, optional
            fineness of the default integration grid, specified in units of Ang, default to 1000 which
            translates to 1000 k-points along reciprocal cells of length 1. Ang^-1.
        bz : BrillouinZone, optional
            integration k-points, if not passed the number of k-points will be determined using
            `dk` and time-reversal symmetry will be determined by `trs`, the number of points refers
            to the unfolded system.
        trs : bool, optional
            whether time-reversal symmetry is used in the BrillouinZone integration, default
            to true.
        """
        self._options.update(options)

    def real_space_parent(self):
        """ Return the parent object in the real-space unfolded region """
        s_ax = self._semi_axis
        k_ax = self._k_axes
        # Always start with the semi-infinite direction, since we
        # Bloch expand the other directions
        unfold = self._unfold.copy()
        P0 = self.parent.tile(unfold[s_ax], s_ax)
        unfold[s_ax] = 1
        for ax in range(3):
            if unfold[ax] == 1:
                continue
            P0 = P0.tile(unfold[ax], ax)
        # Only specify the used axis without periodicity
        # This will allow one to use the real-space self-energy
        # for *circles*
        nsc = array_replace(P0.nsc, (s_ax, 1), (k_ax, 1))
        P0.set_nsc(nsc)
        return P0

    def real_space_coupling(self, ret_indices=False):
        r""" Real-space coupling parent where sites fold into the parent real-space unit cell

        The resulting parent object only contains the inner-cell couplings for the elements that couple
        out of the real-space matrix.

        Parameters
        ----------
        ret_indices : bool, optional
           if true, also return the atomic indices (corresponding to `real_space_parent`) that encompass the coupling matrix

        Returns
        -------
        parent : parent object only retaining the elements of the atoms that couple out of the primary unit cell
        atom_index : indices for the atoms that couple out of the geometry (`ret_indices`)
        """
        s_ax = self._semi_axis
        k_ax = self._k_axes

        # If there are any axes that still has k-point sampling (for e.g. circles)
        # we should remove that periodicity before figuring out which atoms that connect out.
        # This is because the self-energy should *only* remain on the sites connecting
        # out of the self-energy used. The k-axis retains all atoms, per see.
        unfold = self._unfold.copy()
        PC = self.parent.tile(unfold[s_ax], s_ax)
        unfold[s_ax] = 1
        for ax in range(3):
            if unfold[ax] == 1:
                continue
            PC = PC.tile(unfold[ax], ax)

        # Reduce periodicity along non-semi/k axes
        nsc = array_replace(PC.nsc, (s_ax, None), (k_ax, None), other=1)
        PC.set_nsc(nsc)

        # Geometry short-hand
        g = PC.geometry
        # Remove all inner-cell couplings (0, 0, 0) to figure out the
        # elements that couple out of the real-space region
        n = PC.shape[0]
        idx = g.sc.sc_index([0, 0, 0])
        cols = _a.arangei(idx * n, (idx + 1) * n)
        csr = PC._csr.copy([0]) # we just want the sparse pattern, so forget about the other elements
        csr.delete_columns(cols, keep_shape=True)
        # Now PC only contains couplings along the k and semi-inf directions
        # Extract the connecting orbitals and reduce them to unique atomic indices
        orbs = g.osc2uc(csr.col[array_arange(csr.ptr[:-1], n=csr.ncol)], True)
        atom_idx = g.o2a(orbs, True)

        # Only retain coupling atoms
        # Remove all out-of-cell couplings such that we only have inner-cell couplings
        # Or, if we retain periodicity along a given direction, we will retain those
        # as well.
        unfold = self._unfold.copy()
        PC = self.parent.tile(unfold[s_ax], s_ax)
        unfold[s_ax] = 1
        for ax in range(3):
            if unfold[ax] == 1:
                continue
            PC = PC.tile(unfold[ax], ax)
        PC = PC.sub(atom_idx)

        # Truncate nsc along the repititions
        nsc = array_replace(PC.nsc, (s_ax, 1), (k_ax, 1))
        PC.set_nsc(nsc)
        if ret_indices:
            return PC, atom_idx
        return PC

    def initialize(self):
        r""" Initialize the internal data-arrays used for efficient calculation of the real-space quantities

        This method should first be called *after* all options has been specified.

        If the user hasn't specified the ``bz`` value as an option this method will update the internal
        integration Brillouin zone based on ``dk`` and ``trs`` options. The :math:`\mathbf k` point sampling corresponds
        to the number of points in the non-folded system and thus the final sampling is equivalent to the
        sampling times the unfolding (per :math:`\mathbf k` direction).
        """
        s_ax = self._semi_axis
        k_ax = self._k_axes

        # Create temporary access elements in the calculation dictionary
        # to be used in .green and .self_energy
        P0 = self.real_space_parent()
        V_atoms = self.real_space_coupling(True)[1]
        self._calc = {
            # The below algorithm requires the direction to be negative
            # if changed, B, C should be reversed below
            'SE': RecursiveSI(self.parent, '-' + 'ABC'[s_ax], eta=self._options['eta']),
            # Used to calculate the real-space self-energy
            'P0': P0.Pk,
            'S0': P0.Sk,
            # Orbitals in the coupling atoms
            'orbs': P0.a2o(V_atoms, True).reshape(-1, 1),
        }

        # Update the BrillouinZone integration grid in case it isn't specified
        if self._options['bz'] is None:
            # Update the integration grid
            # Note this integration grid is based on the big system.
            sc = self.parent.sc * self._unfold
            rcell = fnorm(sc.rcell)[k_ax]
            nk = _a.onesi(3)
            nk[k_ax] = np.ceil(self._options['dk'] * rcell).astype(np.int32)
            self._options['bz'] = MonkhorstPack(sc, nk, trs=self._options['trs'])

    def self_energy(self, E, k=(0, 0, 0), bulk=False, coupling=False, dtype=None, **kwargs):
        r""" Calculate the real-space self-energy

        The real space self-energy is calculated via:

        .. math::
            \boldsymbol\Sigma^{\mathcal{R}}(E) = \mathbf S^{\mathcal{R}} E - \mathbf H^{\mathcal{R}}
               - \Big[\sum_{\mathbf k} \mathbf G_{\mathbf k}(E)\Big]^{-1}

        Parameters
        ----------
        E : float/complex
           energy to evaluate the real-space self-energy at
        k : array_like, optional
           only viable for 3D bulk systems with real-space self-energies along 2 directions.
           I.e. this would correspond to circular self-energies.
        bulk : bool, optional
           if true, :math:`\mathbf S^{\mathcal{R}} E - \mathbf H^{\mathcal{R}} - \boldsymbol\Sigma^\mathcal{R}`
           is returned, otherwise :math:`\boldsymbol\Sigma^\mathcal{R}` is returned
        coupling : bool, optional
           if True, only the self-energy terms located on the coupling geometry (`coupling_geometry`)
           are returned
        dtype : numpy.dtype, optional
          the resulting data type, default to ``np.complex128``
        **kwargs : dict, optional
           arguments passed directly to the ``self.parent.Pk`` method (not ``self.parent.Sk``), for instance ``spin``
        """
        if dtype is None:
            dtype = complex128
        if E.imag == 0:
            E = E.real + 1j * self._options['eta']

        # Calculate the real-space Green function
        G = self.green(E, k, dtype=dtype)

        if coupling:
            orbs = self._calc['orbs']
            iorbs = delete(_a.arangei(len(G)), orbs).reshape(-1, 1)
            SeH = self._calc['S0'](k, dtype=dtype) * E - self._calc['P0'](k, dtype=dtype, **kwargs)
            if bulk:
                return solve(G[orbs, orbs.T], eye(orbs.size, dtype=dtype) - dot(G[orbs, iorbs.T], SeH[iorbs, orbs.T].toarray()), True, True)
            return SeH[orbs, orbs.T].toarray() - solve(G[orbs, orbs.T], eye(orbs.size, dtype=dtype) - dot(G[orbs, iorbs.T], SeH[iorbs, orbs.T].toarray()), True, True)

            # Another way to do the coupling calculation would be the *full* thing
            # which should always be slower.
            # However, I am not sure which is the most numerically accurate
            # since comparing the two yields numerical differences on the order 1e-8 eV depending
            # on the size of the full matrix G.

            #orbs = self._calc['orbs']
            #iorbs = _a.arangei(orbs.size).reshape(1, -1)
            #I = zeros([G.shape[0], orbs.size], dtype)
            ### Set diagonal
            #I[orbs.ravel(), iorbs.ravel()] = 1.
            #if bulk:
            #    return solve(G, I, True, True)[orbs, iorbs]
            #return (self._calc['S0'](k, dtype=dtype) * E - self._calc['P0'](k, dtype=dtype, **kwargs))[orbs, orbs.T].toarray() \
            #    - solve(G, I, True, True)[orbs, iorbs]

        if bulk:
            return inv(G, True)
        return (self._calc['S0'](k, dtype=dtype) * E - self._calc['P0'](k, dtype=dtype, **kwargs)).toarray() - inv(G, True)

    def green(self, E, k=(0, 0, 0), dtype=None, **kwargs):
        r""" Calculate the real-space Green function

        The real space Green function is calculated via:

        .. math::
            \mathbf G^\mathcal{R}(E) = \sum_{\mathbf k} \mathbf G_{\mathbf k}(E)

        Parameters
        ----------
        E : float/complex
           energy to evaluate the real-space Green function at
        k : array_like, optional
           only viable for 3D bulk systems with real-space Green functions along 2 directions.
           I.e. this would correspond to a circular real-space Green function
        dtype : numpy.dtype, optional
          the resulting data type, default to ``np.complex128``
        **kwargs : dict, optional
           arguments passed directly to the ``self.parent.Pk`` method (not ``self.parent.Sk``), for instance ``spin``
        """
        opt = self._options

        # Retrieve integration k-grid
        bz = opt['bz']
        try:
            # If the BZ implements TRS (MonkhorstPack) then force it
            trs = bz._trs >= 0
        except:
            trs = opt['trs']

        if dtype is None:
            dtype = complex128

        # Now we are to calculate the real-space self-energy
        if E.imag == 0:
            E = E.real + 1j * opt['eta']

        # Used axes
        s_ax = self._semi_axis
        k_ax = self._k_axes

        k = _a.asarrayd(k)
        is_k = np.any(k != 0.)
        if is_k:
            axes = [s_ax] + k_ax.tolist()
            if np.any(k[axes] != 0.):
                raise ValueError(f'{self.__class__.__name__}.green requires the k-point to be zero along the integrated axes.')
            if trs:
                raise ValueError(f'{self.__class__.__name__}.green requires a k-point sampled Green function to not use time reversal symmetry.')
            # Shift k-points to get the correct k-point in the larger one.
            bz._k += k.reshape(1, 3)

        # Calculate both left and right at the same time.
        SE = self._calc['SE'].self_energy_lr

        # Define Bloch unfolding routine and number of tiles along the semi-inf direction
        unfold = self._unfold.copy()
        tile = unfold[s_ax]
        unfold[s_ax] = 1
        bloch = Bloch(unfold)

        # We always need the inverse
        getrf = linalg_info('getrf', dtype)
        getri = linalg_info('getri', dtype)
        getri_lwork = linalg_info('getri_lwork', dtype)
        lwork = int(1.01 * _compute_lwork(getri_lwork, self._calc['SE'].spgeom0.shape[0]))
        def inv(A):
            lu, piv, info = getrf(A, overwrite_a=True)
            if info == 0:
                x, info = getri(lu, piv, lwork=lwork, overwrite_lu=True)
            if info != 0:
                raise ValueError(self.__class__.__name__ + '.green could not compute the inverse.')
            return x

        if tile == 1:
            # When not tiling, it can be simplified quite a bit
            M0 = self._calc['SE'].spgeom0
            M0Pk = M0.Pk
            if self.parent.orthogonal:
                # Orthogonal *always* identity
                S0E = eye(len(M0), dtype=dtype) * E
                def _calc_green(k, dtype, no, tile, idx0):
                    SL, SR = SE(E, k, dtype=dtype, **kwargs)
                    return inv(S0E - M0Pk(k, dtype=dtype, format='array', **kwargs) - SL - SR)
            else:
                M0Sk = M0.Sk
                def _calc_green(k, dtype, no, tile, idx0):
                    SL, SR = SE(E, k, dtype=dtype, **kwargs)
                    return inv(M0Sk(k, dtype=dtype, format='array') * E - M0Pk(k, dtype=dtype, format='array', **kwargs) - SL - SR)

        else:
            # Get faster methods since we don't want overhead of solve
            gesv = linalg_info('gesv', dtype)
            M1 = self._calc['SE'].spgeom1
            M1Pk = M1.Pk
            if self.parent.orthogonal:
                def _calc_green(k, dtype, no, tile, idx0):
                    # Calculate left/right self-energies
                    Gf, A2 = SE(E, k, dtype=dtype, bulk=True, **kwargs) # A1 == Gf, because of memory usage
                    # skip negation since we don't do negation on tY/tX
                    B = M1Pk(k, dtype=dtype, format='array', **kwargs)
                    # C = conjugate(B.T)

                    _, _, tY, info = gesv(Gf, conjugate(B.T), overwrite_a=True, overwrite_b=True)
                    if info != 0:
                        raise ValueError(self.__class__.__name__ + '.green could not solve tY x = B system!')
                    Gf[:, :] = inv(A2 - dot(B, tY))
                    _, _, tX, info = gesv(A2, B, overwrite_a=True, overwrite_b=True)
                    if info != 0:
                        raise ValueError(self.__class__.__name__ + '.green could not solve tX x = B system!')

                    # Since this is the pristine case, we know that
                    # G11 and G22 are the same:
                    #  G = [A1 + C.tX]^-1 == [A2 + B.tY]^-1

                    G = empty([tile, no, tile, no], dtype=dtype)
                    G[idx0, :, idx0, :] = Gf.reshape(1, no, no)
                    for i in range(1, tile):
                        G[idx0[i:], :, idx0[:-i], :] = dot(tX, G[i-1, :, 0, :]).reshape(1, no, no)
                        G[idx0[:-i], :, idx0[i:], :] = dot(tY, G[0, :, i-1, :]).reshape(1, no, no)
                    return G.reshape(tile * no, -1)

            else:
                M1Sk = M1.Sk
                def _calc_green(k, dtype, no, tile, idx0):
                    Gf, A2 = SE(E, k, dtype=dtype, bulk=True, **kwargs) # A1 == Gf, because of memory usage
                    tY = M1Sk(k, dtype=dtype, format='array') # S
                    tX = M1Pk(k, dtype=dtype, format='array', **kwargs) # H
                    # negate B to allow faster gesv method
                    B = tX - tY * E
                    # C = _conj(tY.T) * E - _conj(tX.T)

                    _, _, tY[:, :], info = gesv(Gf, conjugate(tX.T) - conjugate(tY.T) * E, overwrite_a=True, overwrite_b=True)
                    if info != 0:
                        raise ValueError(self.__class__.__name__ + '.green could not solve tY x = B system!')
                    Gf[:, :] = inv(A2 - dot(B, tY))
                    _, _, tX[:, :], info = gesv(A2, B, overwrite_a=True, overwrite_b=True)
                    if info != 0:
                        raise ValueError(self.__class__.__name__ + '.green could not solve tX x = B system!')

                    G = empty([tile, no, tile, no], dtype=dtype)
                    G[idx0, :, idx0, :] = Gf.reshape(1, no, no)
                    for i in range(1, tile):
                        G[idx0[i:], :, idx0[:-i], :] = dot(tX, G[i-1, :, 0, :]).reshape(1, no, no)
                        G[idx0[:-i], :, idx0[i:], :] = dot(tY, G[0, :, i-1, :]).reshape(1, no, no)
                    return G.reshape(tile * no, -1)

        # Create functions used to calculate the real-space Green function
        # For TRS we only-calculate +k and average by using G(k) = G(-k)^T
        # The extra arguments is because the internal decorator is actually pretty slow
        # to filter out unused arguments.

        # If using Bloch's theorem we need to wrap the Green function calculation
        # as the method call.
        if len(bloch) > 1:
            def _func_bloch(k, dtype, no, tile, idx0):
                return bloch(_calc_green, k, dtype=dtype, no=no, tile=tile, idx0=idx0)
        else:
            _func_bloch = _calc_green

        # Tiling indices
        idx0 = _a.arangei(tile)
        no = len(self.parent)

        # calculate the Green function
        G = bz.asaverage().call(_func_bloch, dtype=dtype, no=no, tile=tile, idx0=idx0)

        if is_k:
            # Revert k-points
            bz._k -= k.reshape(1, 3)

        if trs:
            # Faster to do it once, than per G
            return (G + G.T) * 0.5
        return G

    def clear(self):
        """ Clears the internal arrays created in `initialize` """
        del self._calc


class RealSpaceSI(SelfEnergy):
    r""" Surface real-space self-energy (or Green function) for a given physical object with limited periodicity

    The surface real-space self-energy is calculated via the k-averaged Green function:

    .. math::
        \boldsymbol\Sigma^\mathcal{R}(E) = \mathbf S^\mathcal{R} (E+i\eta) - \mathbf H^\mathcal{R}
             - \Big[\sum_{\mathbf k} \mathbf G_{\mathbf k}(E)\Big]^{-1}

    The method actually used is relying on `RecursiveSI` and `~sisl.physics.Bloch` objects.

    Parameters
    ----------
    semi : SemiInfinite
        physical object which contains the semi-infinite direction, it is from
        this object we calculate the self-energy to be put into the surface.
        a physical object from which to calculate the real-space self-energy.
        `semi` and `surface` must have parallel lattice vectors.
    surface : SparseOrbitalBZ
        parent object containing the surface of system. `semi` is attached into this
        object via the overlapping regions, the atoms that overlap `semi` and `surface`
        are determined in the `initialize` routine.
        `semi` and `surface` must have parallel lattice vectors.
    k_axes : array_like of int
        axes where k-points are desired. 1 or 2 values are required. The axis cannot be a direction
        along the `semi` semi-infinite direction.
    unfold : (3,) of int
        number of times the `surface` structure is tiled along each direction
        Since this is a surface there will maximally be 2 unfolds being non-unity.
    eta : float, optional
        imaginary part in the self-energy calculations (default 1e-4 eV)
    dk : float, optional
        fineness of the default integration grid, specified in units of Ang, default to 1000 which
        translates to 1000 k-points along reciprocal cells of length 1. Ang^-1.
    bz : BrillouinZone, optional
        integration k-points, if not passed the number of k-points will be determined using
        `dk` and time-reversal symmetry will be determined by `trs`, the number of points refers
        to the unfolded system.
    trs : bool, optional
        whether time-reversal symmetry is used in the BrillouinZone integration, default
        to true.

    Examples
    --------
    >>> graphene = geom.graphene()
    >>> H = Hamiltonian(graphene)
    >>> H.construct([(0.1, 1.44), (0, -2.7)])
    >>> se = RecursiveSI(H, '-A')
    >>> Hsurf = H.tile(3, 0)
    >>> Hsurf.set_nsc(a=1)
    >>> rsi = RealSpaceSI(se, Hsurf, 1, (1, 4, 1))
    >>> rsi.green(0.1)

    The Brillouin zone integration is determined naturally.

    >>> graphene = geom.graphene()
    >>> H = Hamiltonian(graphene)
    >>> H.construct([(0.1, 1.44), (0, -2.7)])
    >>> se = RecursiveSI(H, '-A')
    >>> Hsurf = H.tile(3, 0)
    >>> Hsurf.set_nsc(a=1)
    >>> rsi = RealSpaceSI(se, Hsurf, 1, (1, 4, 1))
    >>> rsi.set_options(eta=1e-3, bz=MonkhorstPack(H, [1, 1000, 1]))
    >>> rsi.initialize()
    >>> rsi.green(0.1) # eta = 1e-3
    >>> rsi.green(0.1 + 1j * 1e-4) # eta = 1e-4

    Manually specify Brillouin zone integration and default :math:`\eta` value.
    """

    def __init__(self, semi, surface, k_axes, unfold=(1, 1, 1), **options):
        """ Initialize real-space self-energy calculator """
        self.semi = semi
        self.surface = surface

        if not self.semi.sc.parallel(surface.sc):
            raise ValueError(self.__class__.__name__ + ' requires semi and surface to have parallel '
                             'lattice vectors.')

        self._k_axes = np.sort(_a.arrayi(k_axes).ravel())
        k_ax = self._k_axes

        if self.semi.semi_inf in k_ax:
            raise ValueError(self.__class__.__name__ + ' found the self-energy direction to be '
                             'the same as one of the k-axes, this is not allowed.')

        # Local variables for the completion of the details
        self._unfold = _a.arrayi([max(1, un) for un in unfold])

        if self.surface.nsc[semi.semi_inf] > 1:
            raise ValueError(self.__class__.__name__ + ' surface has periodicity along the semi-infinite '
                             'direction. This is not allowed.')
        if np.any(self.surface.nsc[k_ax] < 3):
            raise ValueError(self.__class__.__name__ + ' found k-axes without periodicity. '
                             'Correct k_axes via .set_option.')

        if self._unfold[semi.semi_inf] > 1:
            raise ValueError(self.__class__.__name__ + ' cannot unfold along the semi-infinite direction. '
                             'This is a surface real-space self-energy.')

        # Now we need to figure out the atoms in the surface that corresponds to the
        # semi-infinite direction.
        # Now figure out which atoms in `semi` intersects those in `surface`
        semi_inf = self.semi.semi_inf
        semi_na = self.semi.geometry.na
        semi_min = self.semi.geometry.xyz.min(0)

        surf_na = self.surface.geometry.na

        # Check the coordinates...
        if self.semi.semi_inf_dir == 1:
            # "right", last atoms
            atoms = np.arange(surf_na - semi_na, surf_na)
        else:
            # "left", first atoms
            atoms = np.arange(semi_na)

        # Semi-infinite atoms in surface
        surf_min = self.surface.geometry.xyz[atoms, :].min(0)

        g_surf = self.surface.geometry.xyz[atoms, :] - (surf_min - semi_min)

        # Check atomic coordinates are the same
        # Precision is 0.001 Ang
        if not np.allclose(self.semi.geometry.xyz, g_surf, rtol=0, atol=1e-3):
            print('Coordinate difference:')
            print(self.semi.geometry.xyz - g_surf)
            raise ValueError(self.__class__.__name__ + ' overlapping semi-infinite '
                             'and surface atoms does not coincide!')

        # Surface orbitals to put in the semi-infinite self-energy into.
        self._surface_orbs = self.surface.geometry.a2o(atoms, True).reshape(-1, 1)

        self._options = {
            # For true, the semi-infinite direction will use the bulk values for the
            # elements that overlap with the semi-infinito
            'semi_bulk': True,
            # fineness of the integration k-grid [Ang]
            'dk': 1000,
            # whether TRS is used (G + G.T) * 0.5
            'trs': True,
            # imaginary part used in the Green function calculation (unless an imaginary energy is passed)
            'eta': 1e-4,
            # The BrillouinZone used for integration
            'bz': None,
        }
        self.set_options(**options)
        self.initialize()

    def __str__(self):
        """ String representation of RealSpaceSI """
        d = {'class': self.__class__.__name__}
        for i in range(3):
            d[f'u{i}'] = self._unfold[i]
        d['k'] = str(list(self._k_axes))
        d['semi'] = str(self.semi).replace('\n', '\n  ')
        d['surface'] = str(self.surface).replace('\n', '\n  ')
        d['bz'] = str(self._options['bz']).replace('\n', '\n ')
        d['trs'] = str(self._options['trs'])
        return  ('{class}{{unfold: [{u0}, {u1}, {u2}],\n '
                 'k-axes: {k}, trs: {trs},\n '
                 'bz: {bz},\n '
                 'semi-infinite:\n  {semi},\n '
                 'surface:\n  {surface}\n}}').format(**d)

    def set_options(self, **options):
        r""" Update options in the real-space self-energy

        After updating options one should re-call `initialize` for consistency.

        Parameters
        ----------
        semi_bulk : bool, optional
            whether the semi-infinite matrix elements are used for in the surface. Default to true.
        eta : float, optional
            imaginary part in the self-energy calculations (default 1e-4 eV)
        dk : float, optional
            fineness of the default integration grid, specified in units of Ang, default to 1000 which
            translates to 1000 k-points along reciprocal cells of length 1. Ang^-1.
        bz : BrillouinZone, optional
            integration k-points, if not passed the number of k-points will be determined using
            `dk` and time-reversal symmetry will be determined by `trs`, the number of points refers
            to the unfolded system.
        trs : bool, optional
            whether time-reversal symmetry is used in the BrillouinZone integration, default
            to true.
        """
        self._options.update(options)

    def real_space_parent(self):
        r""" Fully expanded real-space surface parent

        Notes
        -----
        The returned object does *not* obey the ``semi_bulk`` option. I.e. the matrix elements
        correspond to the `self.surface` object, always!
        """
        P0 = self.surface
        for ax in range(3):
            if self._unfold[ax] == 1:
                continue
            P0 = P0.tile(self._unfold[ax], ax)
        nsc = array_replace(P0.nsc, (self._k_axes, 1))
        P0.set_nsc(nsc)
        return P0

    def real_space_coupling(self, ret_indices=False):
        r""" Real-space coupling surafec where the outside fold into the surface real-space unit cell

        The resulting parent object only contains the inner-cell couplings for the elements that couple
        out of the real-space matrix.

        Parameters
        ----------
        ret_indices : bool, optional
           if true, also return the atomic indices (corresponding to `real_space_parent`) that encompass the coupling matrix

        Returns
        -------
        parent : parent object only retaining the elements of the atoms that couple out of the primary unit cell
        atom_index : indices for the atoms that couple out of the geometry (`ret_indices`)
        """
        k_ax = self._k_axes
        n_unfold = np.prod(self._unfold)

        # There are 2 things to check:
        #  1. The semi-infinite system
        #  2. The full surface
        PC_k = self.semi.spgeom0
        PC_semi = self.semi.spgeom1
        PC = self.surface
        for ax in range(3):
            if self._unfold[ax] == 1:
                continue
            PC_k = PC_k.tile(self._unfold[ax], ax)
            PC_semi = PC_semi.tile(self._unfold[ax], ax)
            PC = PC.tile(self._unfold[ax], ax)

        # If there are any axes that still has k-point sampling (for e.g. circles)
        # we should remove that periodicity before figuring out which atoms that connect out.
        # This is because the self-energy should *only* remain on the sites connecting
        # out of the self-energy used. The k-axis retains all atoms, per see.
        nsc = array_replace(PC_k.nsc, (k_ax, None), (self.semi.semi_inf, None), other=1)
        PC_k.set_nsc(nsc)
        nsc = array_replace(PC_semi.nsc, (k_ax, None), (self.semi.semi_inf, None), other=1)
        PC_semi.set_nsc(nsc)
        nsc = array_replace(PC.nsc, (k_ax, None), other=1)
        PC.set_nsc(nsc)

        # Now we need to figure out the coupled elements
        # In all cases we remove the inner cell components
        def get_connections(PC, nrep=1, na=0, na_off=0):
            # Geometry short-hand
            g = PC.geometry
            # Remove all inner-cell couplings (0, 0, 0) to figure out the
            # elements that couple out of the real-space region
            n = PC.shape[0]
            idx = g.sc.sc_index([0, 0, 0])
            cols = _a.arangei(idx * n, (idx + 1) * n)
            csr = PC._csr.copy([0]) # we just want the sparse pattern, so forget about the other elements
            csr.delete_columns(cols, keep_shape=True)
            # Now PC only contains couplings along the k and semi-inf directions
            # Extract the connecting orbitals and reduce them to unique atomic indices
            orbs = g.osc2uc(csr.col[array_arange(csr.ptr[:-1], n=csr.ncol)], True)
            atom = g.o2a(orbs, True)
            expand(atom, nrep, na, na_off)
            return atom

        def expand(atom, nrep, na, na_off):
            if nrep > 1:
                la = np.logical_and
                off = na_off - na
                for rep in range(nrep - 1, 0, -1):
                    r_na = rep * na
                    atom[la(r_na + na > atom, atom >= r_na)] += rep * off

        # The semi-infinite direction is a bit different since now we want what couples out along the
        # semi-infinite direction
        atom_semi = []
        for atom in PC_semi.geometry:
            if len(PC_semi.edges(atom)) > 0:
                atom_semi.append(atom)
        atom_semi = _a.arrayi(atom_semi)
        expand(atom_semi, n_unfold, self.semi.spgeom1.geometry.na, self.surface.geometry.na)
        atom_k = get_connections(PC_k, n_unfold, self.semi.spgeom0.geometry.na, self.surface.geometry.na)
        atom = get_connections(PC)
        del PC_k, PC_semi, PC

        # Now join the lists and find the unique set of atoms
        atom_idx = np.unique(np.concatenate([atom_k, atom_semi, atom]))

        # Only retain coupling atoms
        # Remove all out-of-cell couplings such that we only have inner-cell couplings
        # Or, if we retain periodicity along a given direction, we will retain those
        # as well.
        PC = self.surface
        for ax in range(3):
            if self._unfold[ax] == 1:
                continue
            PC = PC.tile(self._unfold[ax], ax)
        PC = PC.sub(atom_idx)

        # Remove all out-of-cell couplings such that we only have inner-cell couplings.
        nsc = array_replace(PC.nsc, (k_ax, 1))
        PC.set_nsc(nsc)

        if ret_indices:
            return PC, atom_idx
        return PC

    def initialize(self):
        r""" Initialize the internal data-arrays used for efficient calculation of the real-space quantities

        This method should first be called *after* all options has been specified.

        If the user hasn't specified the ``bz`` value as an option this method will update the internal
        integration Brillouin zone based on the ``dk`` option. The :math:`\mathbf k` point sampling corresponds
        to the number of points in the non-folded system and thus the final sampling is equivalent to the
        sampling times the unfolding (per :math:`\mathbf k` direction).
        """
        P0 = self.real_space_parent()
        V_atoms = self.real_space_coupling(True)[1]
        self._calc = {
            # Used to calculate the real-space self-energy
            'P0': P0.Pk,
            'S0': P0.Sk,
            # Orbitals in the coupling atoms
            'orbs': P0.a2o(V_atoms, True).reshape(-1, 1),
        }

        # Update the BrillouinZone integration grid in case it isn't specified
        if self._options['bz'] is None:
            # Update the integration grid
            # Note this integration grid is based on the big system.
            sc = self.surface.sc * self._unfold
            rcell = fnorm(sc.rcell)[self._k_axes]
            nk = _a.onesi(3)
            nk[self._k_axes] = np.ceil(self._options['dk'] * rcell).astype(np.int32)
            self._options['bz'] = MonkhorstPack(sc, nk, trs=self._options['trs'])

    def self_energy(self, E, k=(0, 0, 0), bulk=False, coupling=False, dtype=None, **kwargs):
        r""" Calculate real-space surface self-energy

        The real space self-energy is calculated via:
        .. math::
            \boldsymbol\Sigma^{\mathcal{R}}(E) = \mathbf S^{\mathcal{R}} E - \mathbf H^{\mathcal{R}}
               - \Big[\sum_{\mathbf k} \mathbf G_{\mathbf k}(E)\Big]^{-1}

        Parameters
        ----------
        E : float/complex
           energy to evaluate the real-space self-energy at
        k : array_like, optional
           only viable for 3D bulk systems with real-space self-energies along 2 directions.
           I.e. this would correspond to circular self-energies.
        bulk : bool, optional
           if true, :math:`\mathbf S^{\mathcal{R}} E - \mathbf H^{\mathcal{R}} - \boldsymbol\Sigma^\mathcal{R}`
           is returned, otherwise :math:`\boldsymbol\Sigma^\mathcal{R}` is returned
        coupling : bool, optional
           if True, only the self-energy terms located on the coupling geometry (`coupling_geometry`)
           are returned
        dtype : numpy.dtype, optional
          the resulting data type, default to ``np.complex128``
        **kwargs : dict, optional
           arguments passed directly to the ``self.surface.Pk`` method (not ``self.surface.Sk``), for instance ``spin``
        """
        if dtype is None:
            dtype = complex128
        if E.imag == 0:
            E = E.real + 1j * self._options['eta']

        # Calculate the real-space Green function
        G = self.green(E, k, dtype=dtype)

        if coupling:
            orbs = self._calc['orbs']
            iorbs = delete(_a.arangei(len(G)), orbs).reshape(-1, 1)
            SeH = self._calc['S0'](k, dtype=dtype) * E - self._calc['P0'](k, dtype=dtype, **kwargs)
            if bulk:
                return solve(G[orbs, orbs.T], eye(orbs.size, dtype=dtype) - dot(G[orbs, iorbs.T], SeH[iorbs, orbs.T].toarray()), True, True)
            return SeH[orbs, orbs.T].toarray() - solve(G[orbs, orbs.T], eye(orbs.size, dtype=dtype) - dot(G[orbs, iorbs.T], SeH[iorbs, orbs.T].toarray()), True, True)

            # Another way to do the coupling calculation would be the *full* thing
            # which should always be slower.
            # However, I am not sure which is the most numerically accurate
            # since comparing the two yields numerical differences on the order 1e-8 eV depending
            # on the size of the full matrix G.

            #orbs = self._calc['orbs']
            #iorbs = _a.arangei(orbs.size).reshape(1, -1)
            #I = zeros([G.shape[0], orbs.size], dtype)
            # Set diagonal
            #I[orbs.ravel(), iorbs.ravel()] = 1.
            #if bulk:
            #    return solve(G, I, True, True)[orbs, iorbs]
            #return (self._calc['S0'](k, dtype=dtype) * E - self._calc['P0'](k, dtype=dtype, **kwargs))[orbs, orbs.T].toarray() \
            #    - solve(G, I, True, True)[orbs, iorbs]

        if bulk:
            return inv(G, True)
        return (self._calc['S0'](k, dtype=dtype) * E - self._calc['P0'](k, dtype=dtype, **kwargs)).toarray() - inv(G, True)

    def green(self, E, k=(0, 0, 0), dtype=None, **kwargs):
        r""" Calculate the real-space Green function

        The real space Green function is calculated via:

        .. math::
            \mathbf G^\mathcal{R}(E) = \sum_{\mathbf k} \mathbf G_{\mathbf k}(E)

        Parameters
        ----------
        E : float/complex
           energy to evaluate the real-space Green function at
        k : array_like, optional
           only viable for 3D bulk systems with real-space Green functions along 2 directions.
           I.e. this would correspond to a circular real-space Green function
        dtype : numpy.dtype, optional
          the resulting data type, default to ``np.complex128``
        **kwargs : dict, optional
           arguments passed directly to the ``self.surface.Pk`` method (not ``self.surface.Sk``), for instance ``spin``
        """
        opt = self._options

        # Retrieve integration k-grid
        bz = opt['bz']
        try:
            # If the BZ implements TRS (MonkhorstPack) then force it
            trs = bz._trs >= 0
        except:
            trs = opt['trs']

        if dtype is None:
            dtype = complex128

        # Now we are to calculate the real-space self-energy
        if E.imag == 0:
            E = E.real + 1j * opt['eta']

        # Used k-axes
        k_ax = self._k_axes

        k = _a.asarrayd(k)
        is_k = np.any(k != 0.)
        if is_k:
            axes = [self.semi.semi_inf] + k_ax.tolist()
            if np.any(k[axes] != 0.):
                raise ValueError(f'{self.__class__.__name__}.green requires k-point to be zero along the integrated axes.')
            if trs:
                raise ValueError(f'{self.__class__.__name__}.green requires a k-point sampled Green function to not use time reversal symmetry.')
            # Shift k-points to get the correct k-point in the larger one.
            bz._k += k.reshape(1, 3)

        # Self-energy function
        SE = self.semi.self_energy

        M0 = self.surface
        M0Pk = M0.Pk

        getrf = linalg_info('getrf', dtype)
        getri = linalg_info('getri', dtype)
        getri_lwork = linalg_info('getri_lwork', dtype)
        lwork = int(1.01 * _compute_lwork(getri_lwork, M0.shape[0]))
        def inv(A):
            lu, piv, info = getrf(A, overwrite_a=True)
            if info == 0:
                x, info = getri(lu, piv, lwork=lwork, overwrite_lu=True)
            if info != 0:
                raise ValueError(self.__class__.__name__ + '.green could not compute the inverse.')
            return x

        if M0.orthogonal:
            # Orthogonal *always* identity
            S0E = eye(len(M0), dtype=dtype) * E
            def _calc_green(k, dtype, surf_orbs, semi_bulk):
                invG = S0E - M0Pk(k, dtype=dtype, format='array', **kwargs)
                if semi_bulk:
                    invG[surf_orbs, surf_orbs.T] = SE(E, k, dtype=dtype, bulk=semi_bulk, **kwargs)
                else:
                    invG[surf_orbs, surf_orbs.T] -= SE(E, k, dtype=dtype, bulk=semi_bulk, **kwargs)
                return inv(invG)
        else:
            M0Sk = M0.Sk
            def _calc_green(k, dtype, surf_orbs, semi_bulk):
                invG = M0Sk(k, dtype=dtype, format='array') * E - M0Pk(k, dtype=dtype, format='array', **kwargs)
                if semi_bulk:
                    invG[surf_orbs, surf_orbs.T] = SE(E, k, dtype=dtype, bulk=semi_bulk, **kwargs)
                else:
                    invG[surf_orbs, surf_orbs.T] -= SE(E, k, dtype=dtype, bulk=semi_bulk, **kwargs)
                return inv(invG)

        # Create functions used to calculate the real-space Green function
        # For TRS we only-calculate +k and average by using G(k) = G(-k)^T
        # The extra arguments is because the internal decorator is actually pretty slow
        # to filter out unused arguments.

        # Define Bloch unfolding routine and number of tiles along the semi-inf direction
        bloch = Bloch(self._unfold)

        # If using Bloch's theorem we need to wrap the Green function calculation
        # as the method call.
        if len(bloch) > 1:
            def _func_bloch(k, dtype, surf_orbs, semi_bulk):
                return bloch(_calc_green, k, dtype=dtype, surf_orbs=surf_orbs, semi_bulk=semi_bulk)
        else:
            _func_bloch = _calc_green

        # calculate the Green function
        G = bz.asaverage().call(_func_bloch, dtype=dtype,
                                surf_orbs=self._surface_orbs,
                                semi_bulk=opt['semi_bulk'])

        if is_k:
            # Restore Brillouin zone k-points
            bz._k -= k.reshape(1, 3)

        if trs:
            # Faster to do it once, than per G
            return (G + G.T) * 0.5
        return G

    def clear(self):
        """ Clears the internal arrays created in `initialize` """
        del self._calc
