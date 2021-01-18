import numpy as np

from sisl._internal import set_module
import sisl._array as _a
from sisl.messages import SislError
from .densitymatrix import _densitymatrix

__all__ = ['EnergyDensityMatrix']


@set_module("sisl.physics")
class EnergyDensityMatrix(_densitymatrix):
    """ Sparse energy density matrix object

    Assigning or changing elements is as easy as with standard `numpy` assignments:

    >>> EDM = EnergyDensityMatrix(...)
    >>> EDM.E[1,2] = 0.1

    which assigns 0.1 as the density element between orbital 2 and 3.
    (remember that Python is 0-based elements).

    Parameters
    ----------
    geometry : Geometry
      parent geometry to create a energy density matrix from. The energy density matrix will
      have size equivalent to the number of orbitals in the geometry
    dim : int or Spin, optional
      number of components per element, may be a `Spin` object
    dtype : np.dtype, optional
      data type contained in the energy density matrix. See details of `Spin` for default values.
    nnzpr : int, optional
      number of initially allocated memory per orbital in the energy density matrix.
      For increased performance this should be larger than the actual number of entries
      per orbital.
    spin : Spin, optional
      equivalent to `dim` argument. This keyword-only argument has precedence over `dim`.
    orthogonal : bool, optional
      whether the energy density matrix corresponds to a non-orthogonal basis. In this case
      the dimensionality of the energy density matrix is one more than `dim`.
      This is a keyword-only argument.
    """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        super().__init__(geometry, dim, dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        super()._reset()
        self.Ek = self.Pk
        self.dEk = self.dPk
        self.ddEk = self.ddPk

    @property
    def E(self):
        r""" Access the energy density matrix elements """
        self._def_dim = self.UP
        return self

    def Ek(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the energy density matrix for a given k-point

        Creation and return of the energy density matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \mathbf E(k) = \mathbf E_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \mathbf E(k) = \mathbf E_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like
           the k-point to setup the energy density matrix at
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
           Prefixing with 'sc:', or simply 'sc' returns the matrix in supercell format
           with phases.
        spin : int, optional
           if the energy density matrix is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the energy density matrix is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        dEk : Energy density matrix derivative with respect to `k`
        ddEk : Energy density matrix double derivative with respect to `k`

        Returns
        -------
        matrix : numpy.ndarray or scipy.sparse.*_matrix
            the energy density matrix at :math:`k`. The returned object depends on `format`.
        """
        pass

    def dEk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the energy density matrix derivative for a given k-point

        Creation and return of the energy density matrix derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k \mathbf E_\alpha(k) = i R_\alpha \mathbf E_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.
        And :math:`\alpha` is one of the Cartesian directions.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \nabla_k \mathbf E_\alpha(k) = i r_\alpha \mathbf E_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like
           the k-point to setup the energy density matrix at
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
           if the energy density matrix is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the energy density matrix is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        Ek : Energy density matrix with respect to `k`
        ddEk : Energy density matrix double derivative with respect to `k`

        Returns
        -------
        tuple
            for each of the Cartesian directions a :math:`\partial \mathbf E(k)/\partial k` is returned.
        """
        pass

    def ddEk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the energy density matrix double derivative for a given k-point

        Creation and return of the energy density matrix double derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k^2 \mathbf E_{\alpha\beta}(k) = - R_\alpha R_\beta \mathbf E_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.
        And :math:`\alpha` and :math:`\beta` are one of the Cartesian directions.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \nabla_k^2 \mathbf E_{\alpha\beta}(k) = - r_\alpha r_\beta \mathbf E_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like
           the k-point to setup the energy density matrix at
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
           if the energy density matrix is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the energy density matrix is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        Ek : Energy density matrix with respect to `k`
        dEk : Energy density matrix derivative with respect to `k`

        Returns
        -------
        list of matrices
            for each of the Cartesian directions (in Voigt representation); xx, yy, zz, zy, xz, xy
        """
        pass

    def shift(self, E, DM):
        r""" Shift the energy density matrix to a common energy by using a reference density matrix

        This is equal to performing this operation:

        .. math::
           \mathfrak E_\sigma = \mathfrak E_\sigma + E \boldsymbol \rho_\sigma

        where :math:`\mathfrak E_\sigma` correspond to the spin diagonal components of the
        energy density matrix and :math:`\boldsymbol \rho_\sigma` is the spin diagonal
        components of the corresponding density matrix.

        Parameters
        ----------
        E : float or (2,)
           the energy (in eV) to shift the energy density matrix, if two values are passed
           the two first spin-components get shifted individually.
        DM : DensityMatrix
           density matrix corresponding to the same geometry
        """
        if not self.spsame(DM):
            raise SislError(f"{self.__class__.__name__}.shift requires the input DM to have "
                            "the same sparsity as the shifted object.")

        E = _a.asarrayd(E)
        if E.size == 1:
            E = np.tile(E, 2)

        if np.abs(E).sum() == 0.:
            # When the energy is zero, there is no shift
            return

        for i in range(min(self.spin.spins, 2)):
            self._csr._D[:, i] += DM._csr._D[:, i] * E[i]

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads density matrix from `Sile` using `read_energy_density_matrix`.

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to read the density matrix
            and the overlap matrix (if any)
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_energy_density_matrix(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_energy_density_matrix(*args, **kwargs)
        else:
            with get_sile(sile) as fh:
                return fh.read_energy_density_matrix(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes a density matrix to the `Sile` as implemented in the :code:`Sile.write_energy_density_matrix` method """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_energy_density_matrix(self, *args, **kwargs)
        else:
            with get_sile(sile, 'w') as fh:
                fh.write_energy_density_matrix(self, *args, **kwargs)
