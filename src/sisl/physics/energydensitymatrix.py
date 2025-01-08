# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

import sisl._array as _a
from sisl._internal import set_module
from sisl.messages import SislError
from sisl.typing import GaugeType, KPoint

from .densitymatrix import _densitymatrix

__all__ = ["EnergyDensityMatrix"]


@set_module("sisl.physics")
class EnergyDensityMatrix(_densitymatrix):
    """Sparse energy density matrix object

    Assigning or changing elements is as easy as with standard `numpy` assignments:

    >>> EDM = EnergyDensityMatrix(...)
    >>> EDM.E[1,2] = 0.1

    which assigns 0.1 as the density element between orbital 2 and 3.
    (remember that Python is 0-based elements).

    For spin matrices the elements are defined with an extra dimension.

    For a polarized matrix:

    >>> M = EnergyDensityMatrix(..., spin="polarized")
    >>> M[0, 0, 0] = # onsite spin up
    >>> M[0, 0, 1] = # onsite spin down

    For non-colinear the indices are a bit more tricky:

    >>> M = EnergyDensityMatrix(..., spin="non-colinear")
    >>> M[0, 0, M.M11] = # Re(up-up)
    >>> M[0, 0, M.M22] = # Re(down-down)
    >>> M[0, 0, M.M12r] = # Re(up-down)
    >>> M[0, 0, M.M12i] = # Im(up-down)

    For spin-orbit it looks like this:

    >>> M = EnergyDensityMatrix(..., spin="spin-orbit")
    >>> M[0, 0, M.M11r] = # Re(up-up)
    >>> M[0, 0, M.M11i] = # Im(up-up)
    >>> M[0, 0, M.M22r] = # Re(down-down)
    >>> M[0, 0, M.M22i] = # Im(down-down)
    >>> M[0, 0, M.M12r] = # Re(up-down)
    >>> M[0, 0, M.M12i] = # Im(up-down)
    >>> M[0, 0, M.M21r] = # Re(down-up)
    >>> M[0, 0, M.M21i] = # Im(down-up)

    Thus the number of *orbitals* is unchanged but a sub-block exists for
    the spin-block.

    When transferring the matrix to a k-point the spin-box is local to each
    orbital, meaning that the spin-box for orbital i will be:

    >>> Ek = M.Ek()
    >>> Ek[i*2:(i+1)*2, i*2:(i+1)*2]

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
        r"""Access the energy density matrix elements"""
        self._def_dim = self.UP
        return self

    def Ek(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the energy density matrix for a given k-point

        Creation and return of the energy density matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the lattice vector gauge:

        .. math::
           \mathbf E(\mathbf k) = \mathbf E_{ij} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \mathbf E(\mathbf k) = \mathbf E_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k :
           the k-point to setup the energy density matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``lattice`` for cell vector gauge, and ``atomic`` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
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
            the energy density matrix at :math:`\mathbf k`. The returned object depends on `format`.
        """
        pass

    def dEk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the energy density matrix derivative for a given k-point

        Creation and return of the energy density matrix derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the lattice vector gauge:

        .. math::
           \nabla_{\mathbf k} \mathbf E_\alpha(\mathbf k) = i\mathbf R_\alpha \mathbf E_{ij} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.
        And :math:`\alpha` is one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_{\mathbf k} \mathbf E_\alpha(\mathbf k) = i\mathbf r_\alpha \mathbf E_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k :
           the k-point to setup the energy density matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``lattice`` for cell vector gauge, and ``atomic`` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
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
            for each of the Cartesian directions a :math:`\partial \mathbf E(\mathbf k)/\partial\mathbf k` is returned.
        """
        pass

    def ddEk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the energy density matrix double derivative for a given k-point

        Creation and return of the energy density matrix double derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the lattice vector gauge:

        .. math::
           \nabla_{\mathbf k^2} \mathbf E_{\alpha\beta}(\mathbf k) = -\mathbf R_\alpha\mathbf R_\beta \mathbf E_{ij} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.
        And :math:`\alpha` and :math:`\beta` are one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_{\mathbf k^2} \mathbf E_{\alpha\beta}(\mathbf k) = -\mathbf r_\alpha\mathbf r_\beta \mathbf E_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k :
           the k-point to setup the energy density matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``lattice`` for cell vector gauge, and ``atomic`` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
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
        r"""Shift the energy density matrix to a common energy by using a reference density matrix

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
            raise SislError(
                f"{self.__class__.__name__}.shift requires the input DM to have "
                "the same sparsity as the shifted object."
            )

        E = _a.asarrayd(E)
        if E.size == 1:
            E = np.tile(E, 2)

        if np.abs(E).sum() == 0.0:
            # When the energy is zero, there is no shift
            return

        for i in range(self.spin.spinor):
            self._csr._D[:, i].real += DM._csr._D[:, i].real * E[i]

    @staticmethod
    def read(sile, *args, **kwargs):
        """Reads density matrix from `Sile` using `read_energy_density_matrix`.

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
        from sisl.io import BaseSile, get_sile

        if isinstance(sile, BaseSile):
            return sile.read_energy_density_matrix(*args, **kwargs)
        else:
            with get_sile(sile, mode="r") as fh:
                return fh.read_energy_density_matrix(*args, **kwargs)
