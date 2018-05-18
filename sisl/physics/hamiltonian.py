from __future__ import print_function, division

import numpy as np
from numpy import dot
from scipy.sparse import csr_matrix

from sisl.selector import TimeSelector
from sisl._help import _range as range
import sisl._array as _a
from .electron import EigenvalueElectron, EigenstateElectron
from .sparse import SparseOrbitalBZSpin

__all__ = ['Hamiltonian']


class Hamiltonian(SparseOrbitalBZSpin):
    """ Sparse Hamiltonian matrix object

    Assigning or changing Hamiltonian elements is as easy as with standard `numpy` assignments:

    >>> ham = Hamiltonian(...) # doctest: +SKIP
    >>> ham.H[1,2] = 0.1 # doctest: +SKIP

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

        self.Hk = self.Pk
        self.dHk = TimeSelector([self._dHk_accummulate], True)
        self.dSk = TimeSelector([self._dSk_accummulate], True)

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

        where :math:`r` is the distance between the orbitals :math:`\nu` and :math:`\mu`.
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

        Returns
        -------
        * : the Hamiltonian with size ``(no, no)``. The returned object depends on `format`.
        """
        pass

    def _dHk_accummulate(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', spin=0):
        r""" Hamiltonian differentiated with respect to `k`

        .. math::
           \delta \mathbf H(k) = i \mathbf R e^{i\mathbf k\dot\mathbf R} \mathbf H

        Since :math:`\mathbf R` is a vector the :math:`\delta\mathbf H(k)` matrix returned
        has size ``(no * 3, no)`` where the first 3 rows correspond to the first orbital
        :math:`x`, :math:`y` and :math:`z` components.

        Parameters
        ----------
        k: array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge

        Returns
        -------
        csr_matrix : the dH matrix with size ``(no * 3, no)``.
        """
        if dtype is None:
            dtype = np.complex128

        if gauge != 'R':
            raise ValueError('Only the cell vector gauge has been implemented')

        k = np.asarray(k, np.float64)
        k.shape = (-1,)

        if not np.allclose(k, 0.):
            if np.dtype(dtype).kind != 'c':
                raise ValueError(self.__class__.__name__ + " setup at k different from Gamma requires a complex matrix")

        no = self.no

        # sparse matrix dimension (self.no)
        V = csr_matrix((len(self) * 3, len(self)), dtype=dtype)

        # Calculate all phases
        iRs = -1j * dot(self.sc.sc_off, self.cell)
        phases = np.exp(-1j * dot(dot(dot(self.rcell, k), self.cell), self.sc.sc_off.T))

        v = self.tocsr(spin)

        for si, (iR, phase) in enumerate(zip(iRs, phases)):
            vv = v[:, si*no:(si+1)*no] * phase
            V[0::3, :] += vv * iR[0]
            V[1::3, :] += vv * iR[1]
            V[2::3, :] += vv * iR[2]

        del v

        return V.asformat(format)

    def _dSk_accummulate(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Overlap matrix differentiated with respect to `k`

        .. math::
           \delta \mathbf S(k) = i \mathbf R e^{i\mathbf k\dot\mathbf R} \mathbf S

        Since :math:`\mathbf R` is a vector the :math:`\delta\mathbf S(k)` matrix returned
        has size ``(no * 3, no)`` where the first 3 rows correspond to the first orbital
        :math:`x`, :math:`y` and :math:`z` components.

        Parameters
        ----------
        k: array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge

        Returns
        -------
        csr_matrix : the dS matrix with size ``(no * 3, no)``.
        """
        if dtype is None:
            dtype = np.complex128

        if gauge != 'R':
            raise ValueError('Only the cell vector gauge has been implemented')

        k = np.asarray(k, np.float64)
        k.shape = (-1,)

        if not np.allclose(k, 0.):
            if np.dtype(dtype).kind != 'c':
                raise ValueError(self.__class__.__name__ + " setup at k different from Gamma requires a complex matrix")

        no = self.no

        # sparse matrix dimension (self.no)
        V = csr_matrix((len(self) * 3, len(self)), dtype=dtype)

        # Calculate all phases
        iRs = -1j * dot(self.sc.sc_off, self.cell)
        phases = np.exp(-1j * dot(dot(dot(self.rcell, k), self.cell), self.sc.sc_off.T))

        v = self.tocsr(self.S_idx)

        for si, (iR, phase) in enumerate(zip(iRs, phases)):
            vv = v[:, si*no:(si+1)*no] * phase
            V[0::3, :] += vv * iR[0]
            V[1::3, :] += vv * iR[1]
            V[2::3, :] += vv * iR[2]

        del v

        return V.asformat(format)

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
        E : float or (2,)
           the energy (in eV) to shift the electronic structure, if two values are passed
           the two first spin-components get shifted individually.
        """
        E = _a.asarrayd(E)
        if E.size == 1:
            E = np.tile(_a.asarrayd(E), 2)
        if not self.orthogonal:
            # For non-collinear and SO only the diagonal (real) components
            # should be shifted.
            for i in range(min(self.spin.spins, 2)):
                self._csr._D[:, i] += self._csr._D[:, self.S_idx] * E[i]
        else:
            for i in range(self.shape[0]):
                for j in range(min(self.spin.spins, 2)):
                    self[i, i, j] = self[i, i, j] + E[i]

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
        **kwargs: optional
            additional parameters passed to the `eigenstate` routine

        See Also
        --------
        sisl.physics.distribution : setup a distribution function, see details regarding the `distribution` argument
        eigenstate : method used to calculate the eigenstates
        PDOS : Calculate projected DOS
        EigenvalueElectron.DOS : Underlying method used to calculate the DOS
        EigenstateElectron.PDOS : Underlying method used to calculate the projected DOS
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
        **kwargs: optional
            additional parameters passed to the `eigenstate` routine

        See Also
        --------
        sisl.physics.distribution : setup a distribution function, see details regarding the `distribution` argument
        eigenstate : method used to calculate the eigenstates
        DOS : Calculate total DOS
        EigenvalueElectron.DOS : Underlying method used to calculate the DOS
        EigenstateElectron.PDOS : Underlying method used to calculate the projected DOS
        """
        return self.eigenstate(k, **kwargs).PDOS(E, distribution)
