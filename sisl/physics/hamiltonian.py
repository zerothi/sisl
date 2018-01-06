from __future__ import print_function, division

import numpy as np

from sisl._help import _range as range, _str as str
from sisl.eigensystem import EigenSystem
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
            If ``None`` ``EigenState.distribution('gaussian')`` will be used.
        **kwargs: optional
            additional parameters passed to the `eigenstate` routine

        See Also
        --------
        eigenstate : method used to calculate the eigenstates
        PDOS : Calculate projected DOS
        EigenState.DOS : Underlying method used to calculate the DOS, see details regarding the `distribution` argument
        EigenState.PDOS : Underlying method used to calculate the DOS, see details regarding the `distribution` argument
        """
        return self.eigenstate(k, **kwargs).DOS(E, distribution)

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
            If ``None`` ``EigenState.distribution('gaussian')`` will be used.
        **kwargs: optional
            additional parameters passed to the `eigenstate` routine

        See Also
        --------
        eigenstate : method used to calculate the eigenstates
        DOS : Calculate total DOS
        EigenState.DOS : Underlying method used to calculate the DOS, see details regarding the `distribution` argument
        EigenState.PDOS : Underlying method used to calculate the DOS, see details regarding the `distribution` argument
        """
        return self.eigenstate(k, **kwargs).PDOS(E, distribution)


class EigenState(EigenSystem):
    """ Eigenstates associated by a Hamiltonian object

    This object can be generated from a Hamiltonian via `Hamiltonian.eigenstate`.
    Subsequent DOS calculations and/or wavefunction calculations (`Grid.psi`) may be
    performed using this object.
    """

    @classmethod
    def distribution(cls, method, smearing=0.1):
        r""" Create a distribution function for input in e.g. `DOS`. Gaussian, Lorentzian etc.

        In the following :math:`\epsilon` are the eigenvalues contained in this `EigenState`.

        The Gaussian distribution is calculated as:

        .. math::
            G(E) = \sum_i \frac{1}{\sqrt{2\pi\sigma^2}\exp\big[- (E - \epsilon_i)^2 / (2\sigma^2)\big]

        where :math:`\sigma` is the `smearing` parameter.

        The Lorentzian distribution is calculated as:

        .. math::
            L(E) = \sum_i \frac{1}{\pi}\frac{\gamma}{(E - \epsilon_i)^2 + \gamma^2}

        where :math:`\gamma` is the `smearing` parameter, note that here :math:`\gamma` is the
        half-width at half-maximum (:math:`2\gamma` would be the full-width at half-maximum).

        Parameters
        ----------
        method : {'gaussian', 'lorentzian'}
            the distribution function
        smearing : float, optional
            the smearing parameter for the method (:math:`\sigma` for Gaussian, etc.)

        Returns
        -------
        func : a function which accepts one argument
        """
        if method.lower() in ['gauss', 'gaussian']:
            exp = np.exp
            sigma2 = 2 * smearing ** 2
            pisigma = (np.pi * sigma2) ** .5
            def func(E):
                return exp(-E ** 2 / sigma2) / pisigma
        elif method.lower() in ['lorentz', 'lorentzian']:
            def func(E):
                return (smearing / np.pi) / (E ** 2 + smearing ** 2)
        else:
            raise ValueError(cls.__name__ + ".distribution currently only implements 'gaussian' or "
                             "'lorentzian' distribution functions")
        return func

    def DOS(self, E, distribution=None):
        r""" Calculate the DOS for the provided energies (`E`), using the supplied distribution function

        The Density Of States at a specific energy is calculated via the broadening function:

        .. math::
            \mathrm{DOS}(E) = \sum_i D(E-\epsilon_i) \approx\delta(E-\epsilon_i)

        where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
        used may be a user-defined function. Alternatively a distribution function may
        be aquired from `EigenState.distribution`.

        Parameters
        ----------
        E : array_like
            energies to calculate the DOS from
        distribution : func, optional
            a function that accepts :math:`E-\epsilon` as argument and calculates the
            distribution function.
            If ``None`` ``EigenState.distribution('gaussian')`` will be used.

        See Also
        --------
        distribution : a selected set of implemented distribution functions
        PDOS : the projected DOS

        Returns
        -------
        numpy.ndarray : DOS calculated at energies, has same length as `E`
        """
        if distribution is None:
            distribution = self.distribution('gaussian')
        elif isinstance(distribution, str):
            distribution = self.distribution(distribution)
        DOS = distribution(E - self.e[0])
        for i in range(1, len(self)):
            DOS += distribution(E - self.e[i])
        return DOS

    def PDOS(self, E, distribution=None):
        r""" Calculate the projected-DOS for the provided energies (`E`), using the supplied distribution function

        The projected DOS is calculated as:

        .. math::
             \mathrm{PDOS}_\nu(E) = \sum_i [\langle \psi_{i} | \mathbf S | \psi_{i}\rangle]_\nu D(E-\epsilon_i)

        where :math:`D(\Delta E)` is the distribution function used. Note that the distribution function
        used may be a user-defined function. Alternatively a distribution function may
        be aquired from `EigenState.distribution`.

        In case of an orthogonal basis set :math:`\mathbf S` is equal to the identity matrix.
        Note that `DOS` is the sum of the orbital projected DOS:

        .. math::
            \mathrm{DOS}(E) = \sum_\nu\mathrm{PDOS)_\nu(E)

        Parameters
        ----------
        E : array_like
            energies to calculate the projected-DOS from
        distribution : func, optional
            a function that accepts :math:`E-\epsilon` as argument and calculates the
            distribution function.
            If ``None`` ``EigenState.distribution('gaussian')`` will be used.

        See Also
        --------
        distribution : a selected set of implemented distribution functions
        DOS : the total DOS

        Returns
        -------
        numpy.ndarray : projected DOS calculated at energies, has dimension ``(self.size, len(E))``
        """
        if distribution is None:
            distribution = self.distribution('gaussian')
        elif isinstance(distribution, str):
            distribution = self.distribution(distribution)

        # Retrieve options for the Sk calculation
        k = self.info.get('k', (0, 0, 0))
        opt = {'k': self.info.get('k', (0, 0, 0))}
        if 'gauge' in self.info:
            opt['gauge'] = self.info['gauge']

        if isinstance(self.parent, Hamiltonian):
            # Calculate the overlap matrix
            Sk = self.parent.Sk(**opt)
            if self.parent.spin > Spin('p'):
                raise ValueError('Currently the PDOS for non-colinear and spin-orbit has not been checked')
        else:
            # Assume orthogonal basis set and Gamma-point
            # TODO raise warning, should we do this here?
            class _K(object):
                def dot(self, v):
                    return v
            Sk = _K()

        # Short-hands
        conj = np.conjugate
        add = np.add

        w = distribution(E - self.e[0]).reshape(1, -1)
        PDOS = (conj(self.v[0, :]) * Sk.dot(self.v[0, :])).real.reshape(-1, 1) * w
        for i in range(1, len(self)):
            w = distribution(E - self.e[i]).reshape(1, -1)
            add(PDOS, (conj(self.v[i, :]) * Sk.dot(self.v[i, :])).real.reshape(-1, 1) * w, out=PDOS)
        return PDOS

    def psi(self, grid):
        """ Calculate the wavefunction on `grid`

        Parameters
        ----------
        grid : Grid
            the grid to calculate the wavefunctions on (need not be commensurate with
            this parent objects geometry.
        """
        grid.psi(self, self.info.get('k', (0, 0, 0)))


# For backwards compatibility we also use TightBinding
# NOTE: that this is not sub-classed...
TightBinding = Hamiltonian
