from __future__ import print_function, division

import numpy as np
from scipy.interpolate import CubicSpline

from sisl._help import _range as range
import sisl._array as _a
from .distribution import get_distribution
from .electron import EigenvalueElectron, EigenstateElectron, spin_squared
from .sparse import SparseOrbitalBZSpin

__all__ = ['Hamiltonian']


class Hamiltonian(SparseOrbitalBZSpin):
    """ Sparse Hamiltonian matrix object

    Assigning or changing Hamiltonian elements is as easy as with standard `numpy` assignments:

    >>> ham = Hamiltonian(...)
    >>> ham.H[1,2] = 0.1

    which assigns 0.1 as the coupling constant between orbital 2 and 3.
    (remember that Python is 0-based elements).

    Parameters
    ----------
    geometry : Geometry
      parent geometry to create a density matrix from. The density matrix will
      have size equivalent to the number of orbitals in the geometry
    dim : int or Spin, optional
      number of components per element, may be a `Spin` object
    dtype : np.dtype, optional
      data type contained in the density matrix. See details of `Spin` for default values.
    nnzpr : int, optional
      number of initially allocated memory per orbital in the density matrix.
      For increased performance this should be larger than the actual number of entries
      per orbital.
    spin : Spin, optional
      equivalent to `dim` argument. This keyword-only argument has precedence over `dim`.
    orthogonal : bool, optional
      whether the density matrix corresponds to a non-orthogonal basis. In this case
      the dimensionality of the density matrix is one more than `dim`.
      This is a keyword-only argument.
    """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        """ Initialize Hamiltonian """
        super(Hamiltonian, self).__init__(geometry, dim, dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        super(Hamiltonian, self)._reset()
        self.Hk = self.Pk
        self.dHk = self.dPk
        self.ddHk = self.ddPk

    def Hk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the Hamiltonian for a given k-point

        Creation and return of the Hamiltonian for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \mathbf H(k) = \mathbf H_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \mathbf H(k) = \mathbf H_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

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
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).
        spin : int, optional
           if the Hamiltonian is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the Hamiltonian is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        dHk : Hamiltonian derivative with respect to `k`
        ddHk : Hamiltonian double derivative with respect to `k`

        Returns
        -------
        object : the Hamiltonian matrix at :math:`k`. The returned object depends on `format`.
        """
        pass

    def dHk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the Hamiltonian derivative for a given k-point

        Creation and return of the Hamiltonian derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k \mathbf H_\alpha(k) = i R_\alpha \mathbf H_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.
        And :math:`\alpha` is one of the Cartesian directions.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \nabla_k \mathbf H_\alpha(k) = i r_\alpha \mathbf H_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

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
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).
        spin : int, optional
           if the Hamiltonian is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the Hamiltonian is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        Hk : Hamiltonian with respect to `k`
        ddHk : Hamiltonian double derivative with respect to `k`

        Returns
        -------
        tuple : for each of the Cartesian directions a :math:`\partial \mathbf H(k)/\partial k_\alpha` is returned.
        """
        pass

    def ddHk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the Hamiltonian double derivative for a given k-point

        Creation and return of the Hamiltonian double derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k^2 \mathbf H_{\alpha\beta}(k) = - R_\alpha R_\beta \mathbf H_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.
        And :math:`\alpha` and :math:`\beta` are one of the Cartesian directions.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \nabla_k^2 \mathbf H_{\alpha\beta}(k) = - r_\alpha r_\beta \mathbf H_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

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
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).
        spin : int, optional
           if the Hamiltonian is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the Hamiltonian is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        Hk : Hamiltonian with respect to `k`
        dHk : Hamiltonian derivative with respect to `k`

        Returns
        -------
        tuple of tuples : for each of the Cartesian directions
        """
        pass

    def _get_H(self):
        self._def_dim = self.UP
        return self

    def _set_H(self, key, value):
        if len(key) == 2:
            self._def_dim = self.UP
        self[key] = value

    H = property(_get_H, _set_H, doc="Access elements to the sparse Hamiltonian")

    def shift(self, E):
        r""" Shift the electronic structure by a constant energy

        This is equal to performing this operation:

        .. math::
           \mathbf H_\sigma = \mathbf H_\sigma + E \mathbf S

        where :math:`\mathbf H_\sigma` correspond to the spin diagonal components of the
        Hamiltonian.

        Parameters
        ----------
        E : float or (2,)
           the energy (in eV) to shift the electronic structure, if two values are passed
           the two first spin-components get shifted individually.
        """
        E = _a.asarrayd(E)
        if E.size == 1:
            E = np.tile(E, 2)

        if np.abs(E).sum() == 0.:
            # When the energy is zero, there is no shift
            return

        if self.orthogonal:
            for i in range(self.shape[0]):
                for j in range(min(self.spin.spins, 2)):
                    self[i, i, j] = self[i, i, j] + E[j]
        else:
            # For non-collinear and SO only the diagonal (real) components
            # should be shifted.
            for i in range(min(self.spin.spins, 2)):
                self._csr._D[:, i] += self._csr._D[:, self.S_idx] * E[i]

    def eigenvalue(self, k=(0, 0, 0), gauge='R', **kwargs):
        """ Calculate the eigenvalues at `k` and return an `EigenvalueElectron` object containing all eigenvalues for a given `k`

        Parameters
        ----------
        k : array_like*3, optional
            the k-point at which to evaluate the eigenvalues at
        gauge : str, optional
            the gauge used for calculating the eigenvalues
        sparse : bool, optional
            if ``True``, `eigsh` will be called, else `eigh` will be
            called (default).
        **kwargs : dict, optional
            passed arguments to the `eigh` routine

        See Also
        --------
        eigh : eigenvalue routine
        eigsh : eigenvalue routine

        Returns
        -------
        EigenvalueElectron
        """
        if kwargs.pop('sparse', False):
            e = self.eigsh(k, gauge=gauge, eigvals_only=True, **kwargs)
        else:
            e = self.eigh(k, gauge, eigvals_only=True, **kwargs)
        info = {'k': k,
                'gauge': gauge}
        if 'spin' in kwargs:
            info['spin'] = kwargs['spin']
        return EigenvalueElectron(e, self, **info)

    def eigenstate(self, k=(0, 0, 0), gauge='R', **kwargs):
        """ Calculate the eigenstates at `k` and return an `EigenstateElectron` object containing all eigenstates

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
        eigh : eigenvalue routine
        eigsh : eigenvalue routine

        Returns
        -------
        EigenstateElectron
        """
        if kwargs.pop('sparse', False):
            e, v = self.eigsh(k, gauge=gauge, eigvals_only=False, **kwargs)
        else:
            e, v = self.eigh(k, gauge, eigvals_only=False, **kwargs)
        info = {'k': k, 'gauge': gauge}
        if 'spin' in kwargs:
            info['spin'] = kwargs['spin']
        # Since eigh returns the eigenvectors [:, i] we have to transpose
        return EigenstateElectron(v.T, e, self, **info)

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads Hamiltonian from `Sile` using `read_hamiltonian`.

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
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

    def spin_squared(self, k=(0, 0, 0), n_up=None, n_down=None, **kwargs):
        r""" Calculate spin-squared expectation value, see `~sisl.physics.electron.spin_squared` for details

        Parameters
        ----------
        k : array_like, optional
            k-point at which the spin-squared expectation value is
        n_up : int, optional
            number of states for spin up configuration, default to all. All states up to and including
            `n_up`.
        n_down : int, optional
            same as `n_up` but for the spin-down configuration
        **kwargs : optional
            additional parameters passed to the `eigenstate` routine
        """
        if not self.spin.is_polarized:
            raise ValueError(self.__class__.__name__ + '.spin_squared requires as spin-polarized system')
        es_alpha = self.eigenstate(k, spin=0, **kwargs)
        if not n_up is None:
            es_alpha = es_alpha.sub(range(n_up))
        es_beta = self.eigenstate(k, spin=1, **kwargs)
        if not n_down is None:
            es_beta = es_beta.sub(range(n_down))
        # es_alpha.Sk should equal es_beta.Sk, so just pass one of them
        return spin_squared(es_alpha.state, es_beta.state, es_alpha.Sk())

    def velocity(self, k=(0, 0, 0), **kwargs):
        r""" Calculate the velocity for the eigenstates for a given `k` point

        Parameters
        ----------
        k : array_like, optional
            k-point at which the velocities are calculated
        **kwargs : optional
            additional parameters passed to the `eigenstate` routine

        See Also
        --------
        eigenstate : method used to calculate the eigenstates
        EigenvalueElectron.velocity : Underlying method used to calculate the velocity
        """
        return self.eigenstate(k, **kwargs).velocity()

    def spin_moment(self, k=(0, 0, 0), **kwargs):
        r""" Calculate the spin moment for the eigenstates for a given `k` point

        Parameters
        ----------
        k : array_like, optional
            k-point at which the spin moments are calculated
        **kwargs : optional
            additional parameters passed to the `eigenstate` routine

        See Also
        --------
        eigenstate : method used to calculate the eigenstates
        EigenvalueElectron.spin_moment : Underlying method used to calculate the spin moment
        """
        return self.eigenstate(k, **kwargs).spin_moment()

    def DOS(self, E, k=(0, 0, 0), distribution='gaussian', **kwargs):
        r""" Calculate the DOS at the given energies for a specific `k` point

        Parameters
        ----------
        E : array_like
            energies to calculate the DOS at
        k : array_like, optional
            k-point at which the DOS is calculated
        distribution : func or str, optional
            a function that accepts :math:`E-\epsilon` as argument and calculates the
            distribution function.
        **kwargs : optional
            additional parameters passed to the `eigenvalue` routine

        See Also
        --------
        sisl.physics.distribution : setup a distribution function, see details regarding the `distribution` argument
        eigenvalue : method used to calculate the eigenvalues
        PDOS : Calculate projected DOS
        EigenvalueElectron.DOS : Underlying method used to calculate the DOS
        """
        return self.eigenvalue(k, **kwargs).DOS(E, distribution)

    def PDOS(self, E, k=(0, 0, 0), distribution='gaussian', **kwargs):
        r""" Calculate the projected DOS at the given energies for a specific `k` point

        Parameters
        ----------
        E : array_like
            energies to calculate the projected DOS at
        k : array_like, optional
            k-point at which the projected DOS is calculated
        distribution : func or str, optional
            a function that accepts :math:`E-\epsilon` as argument and calculates the
            distribution function.
        **kwargs : optional
            additional parameters passed to the `eigenstate` routine

        See Also
        --------
        sisl.physics.distribution : setup a distribution function, see details regarding the `distribution` argument
        eigenstate : method used to calculate the eigenstates
        DOS : Calculate total DOS
        EigenstateElectron.PDOS : Underlying method used to calculate the projected DOS
        """
        return self.eigenstate(k, **kwargs).PDOS(E, distribution)

    def fermi_level(self, bz=None, q=None, distribution='fermi_dirac', q_tol=1e-12):
        """ Calculate the Fermi-level using a Brillouinzone sampling and a target charge

        The Fermi-level will be calculated using an iterative approach by first calculating all eigenvalues
        and subsequently fitting the Fermi level to the final charge (`q`).

        Parameters
        ----------
        bz : Brillouinzone, optional
            sampled k-points and weights, the ``bz.parent`` will be equal to this object upon return
            default to Gamma-point
        q : float, list of float, optional
            seeked charge, if not set will be equal to ``self.geometry.q0``. If a list of two is passed
            there will be calculated a Fermi-level per spin-channel. If the Hamiltonian is not spin-polarized
            the sum of the list will be used and only a single fermi-level will be returned.
        distribution : str, func, optional
            used distribution, must accept the keyword ``mu`` as parameter for the Fermi-level
        q_tol : float, optional
            tolerance of charge for finding the Fermi-level

        Returns
        -------
        fermi-level : the Fermi-level of the system (or two if two different charges are passed)
        """
        if bz is None:
            # Gamma-point only
            from .brillouinzone import BrillouinZone
            bz = BrillouinZone(self)
        else:
            # Overwrite the parent in bz
            bz.set_parent(self)
        # Ensure we are using asarray
        bz.asarray()

        if q is None:
            q = self.geometry.q0
        # Ensure we have an "array" in case of spin-polarized calculations
        q = np.asarray(q)

        if isinstance(distribution, str):
            distribution = get_distribution(distribution)

        # B-cast for easier weights
        w = bz.weight.reshape(-1, 1)

        # Internal class to calculate the Fermi-level
        def _Ef(q, eig):
            # We could reduce it depending on the temperature,
            # however the distribution does not have the kT
            # parameter available.
            Ef = np.average(eig[:, int(q)])

            l_Ef = []
            l_q = []
            def list_append(q, Ef):
                l_q.append(q)
                l_Ef.append(Ef)

            # Calculate guessed charge
            qt = (distribution(eig, mu=Ef) * w).sum()

            while abs(qt - q) > q_tol:
                # Add to cubic-spline
                list_append(qt, Ef)

                # Estimate new Fermi-level
                if len(l_q) > 1:
                    # We can do a spline interpolation
                    lq = np.array(l_q)
                    idx = np.argsort(lq)
                    lEf = np.array(l_Ef)
                    Ef = CubicSpline(lq[idx], lEf[idx], extrapolate=True)(q)
                else:
                    # Update limits
                    if qt > q:
                        Ef = Ef - 0.5
                    elif qt < q:
                        Ef = Ef + 0.5

                # Calculate new guessed charge
                qt = (distribution(eig, mu=Ef) * w).sum()

            return Ef

        if self.spin.is_polarized and q.size == 2:
            # We need to do Fermi-level separately since the user requests
            # separate fillings
            Ef = _a.emptyd(2)
            Ef[0] = _Ef(q[0], bz.eigh(spin=0))
            Ef[1] = _Ef(q[1], bz.eigh(spin=1))
        else:
            # Ensure a single charge
            q = q.sum()
            if self.spin.is_polarized:
                Ef = _Ef(q, np.concatenate([bz.eigh(spin=0),
                                            bz.eigh(spin=1)], axis=1))
            else:
                Ef = _Ef(q, bz.eigh())

        return Ef
