# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

import sisl._array as _a
from sisl._internal import set_module
from sisl.typing import GaugeType, KPoint

from ._common import comply_gauge
from .distribution import get_distribution
from .electron import EigenstateElectron, EigenvalueElectron
from .sparse import SparseOrbitalBZSpin

__all__ = ["Hamiltonian"]


@set_module("sisl.physics")
class Hamiltonian(SparseOrbitalBZSpin):
    """Sparse Hamiltonian matrix object

    Assigning or changing Hamiltonian elements is as easy as with standard `numpy` assignments:

    >>> ham = Hamiltonian(...)
    >>> ham.H[1, 2] = 0.1

    which assigns 0.1 as the coupling constant between orbital 2 and 3.
    (remember that Python is 0-based elements).

    For spin matrices the elements are defined with an extra dimension.

    For a polarized matrix:

    >>> M = Hamiltonian(..., spin="polarized")
    >>> M[0, 0, 0] = # onsite spin up
    >>> M[0, 0, 1] = # onsite spin down

    For non-colinear the indices are a bit more tricky:

    >>> M = Hamiltonian(..., spin="non-colinear")
    >>> M[0, 0, M.M11] = # Re(up-up)
    >>> M[0, 0, M.M22] = # Re(down-down)
    >>> M[0, 0, M.M12r] = # Re(up-down)
    >>> M[0, 0, M.M12i] = # Im(up-down)

    For spin-orbit it looks like this:

    >>> M = Hamiltonian(..., spin="spin-orbit")
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

    >>> Hk = ham.Hk()
    >>> Hk[i*2:(i+1)*2, i*2:(i+1)*2]

    Parameters
    ----------
    geometry : Geometry
      parent geometry to create a Hamiltonian from. The Hamiltonian will
      have size equivalent to the number of orbitals in the geometry
    dim : int or Spin, optional
      number of components per element, may be a `Spin` object
    dtype : np.dtype, optional
      data type contained in the matrix. See details of `Spin` for default values.
    nnzpr : int, optional
      number of initially allocated memory per orbital in the matrix.
      For increased performance this should be larger than the actual number of entries
      per orbital.
    spin : Spin, optional
      equivalent to `dim` argument. This keyword-only argument has precedence over `dim`.
    orthogonal : bool, optional
      whether the matrix corresponds to a non-orthogonal basis. In this case
      the dimensionality of the matrix is one more than `dim`.
      This is a keyword-only argument.
    """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        """Initialize Hamiltonian"""
        super().__init__(geometry, dim, dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        super()._reset()
        self.Hk = self.Pk
        self.dHk = self.dPk
        self.ddHk = self.ddPk

    @property
    def H(self):
        r"""Access the Hamiltonian elements"""
        self._def_dim = self.UP
        return self

    def Hk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the Hamiltonian for a given k-point

        Creation and return of the Hamiltonian for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the lattice vector gauge:

        .. math::
           \mathbf H(\mathbf k) = \mathbf H_{ij} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \mathbf H(\mathbf k) = \mathbf H_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k :
           the k-point to setup the Hamiltonian at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``lattice`` for lattice vector gauge, and ``atomic`` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).
           Prefixing with 'sc:', or simply 'sc' returns the matrix in supercell format
           with phases. This is useful for e.g. bond-current calculations where individual
           hopping + phases are required.
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
        matrix : numpy.ndarray or scipy.sparse.*_matrix
            the Hamiltonian matrix at :math:`\mathbf k`. The returned object depends on `format`.
        """
        pass

    def dHk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the Hamiltonian derivative for a given k-point

        Creation and return of the Hamiltonian derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the lattice vector gauge:

        .. math::
           \nabla_{\mathbf k} \mathbf H_\alpha(\mathbf k) = i \mathbf R_\alpha \mathbf H_{ij} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.
        And :math:`\alpha` is one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_{\mathbf k} \mathbf H_\alpha(\mathbf k) = i \mathbf r_\alpha \mathbf H_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the atoms (for atom centered orbitals).

        Parameters
        ----------
        k :
           the k-point to setup the Hamiltonian at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``lattice`` for lattice vector gauge, and ``atomic`` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
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
        tuple
            for each of the Cartesian directions a :math:`\partial \mathbf H(\mathbf k)/\partial \mathbf k_\alpha` is returned.
        """
        pass

    def ddHk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the Hamiltonian double derivative for a given k-point

        Creation and return of the Hamiltonian double derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the lattice vector gauge:

        .. math::
           \nabla_{\mathbf k^2} \mathbf H_{\alpha\beta}(\mathbf k) = - \mathbf R_\alpha \mathbf R_\beta \mathbf H_{ij} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.
        And :math:`\alpha` and :math:`\beta` are one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_{\mathbf k^2} \mathbf H_{\alpha\beta}(\mathbf k) = -\mathbf r_\alpha\mathbf r_\beta \mathbf H_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the atoms.

        Parameters
        ----------
        k :
           the k-point to setup the Hamiltonian at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``lattice`` for lattice vector gauge, and ``atomic`` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
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
        list of matrices
            for each of the Cartesian directions (in Voigt representation); xx, yy, zz, zy, xz, xy
        """
        pass

    def shift(self, E):
        r"""Shift the electronic structure by a constant energy

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

        if np.abs(E).sum() == 0.0:
            # When the energy is zero, there is no shift
            return

        if self.spin.is_nambu:
            nspin = 2
        else:
            nspin = self.spin.spinor

        if self.orthogonal:
            for i in range(self.shape[0]):
                for j in range(nspin):
                    self[i, i, j] = self[i, i, j] + E[j]
        else:
            # For non-collinear and SO only the diagonal (real) components
            # should be shifted.
            for i in range(nspin):
                self._csr._D[:, i].real += self._csr._D[:, self.S_idx].real * E[i]

    def eigenvalue(self, k: KPoint = (0, 0, 0), gauge: GaugeType = "lattice", **kwargs):
        """Calculate the eigenvalues at `k` and return an `EigenvalueElectron` object containing all eigenvalues for a given `k`

        Parameters
        ----------
        k :
            the k-point at which to evaluate the eigenvalues at
        gauge :
            the gauge used for calculating the eigenvalues
        sparse : bool, optional
            if ``True``, `eigsh` will be called, else `eigh` will be
            called (default).
        format : str, optional
            see `eigh` for details, this will be passed to the EigenstateElectron
            instance to be used in subsequent calls, may speed up post-processing.
        **kwargs : dict, optional
            passed arguments to the eigenvalue calculator routine

        See Also
        --------
        eigh : dense eigenvalue routine
        eigsh : sparse eigenvalue routine

        Returns
        -------
        EigenvalueElectron
        """
        gauge = comply_gauge(gauge)
        format = kwargs.pop("format", None)
        if kwargs.pop("sparse", False):
            e = self.eigsh(k, gauge=gauge, eigvals_only=True, **kwargs)
        else:
            e = self.eigh(k, gauge, eigvals_only=True, **kwargs)
        info = {"k": k, "gauge": gauge}
        for name in ("spin",):
            if name in kwargs:
                info[name] = kwargs[name]
        if not format is None:
            info["format"] = format
        return EigenvalueElectron(e, self, **info)

    def eigenstate(self, k: KPoint = (0, 0, 0), gauge: GaugeType = "lattice", **kwargs):
        """Calculate the eigenstates at `k` and return an `EigenstateElectron` object containing all eigenstates

        Parameters
        ----------
        k :
            the k-point at which to evaluate the eigenstates at
        gauge :
            the gauge used for calculating the eigenstates
        sparse : bool, optional
            if ``True``, `eigsh` will be called, else `eigh` will be
            called (default).
        format : str, optional
            see `eigh` for details, this will be passed to the EigenstateElectron
            instance to be used in subsequent calls, may speed up post-processing.
        **kwargs : dict, optional
            passed arguments to the eigenvalue calculator routine

        See Also
        --------
        eigh : dense eigenvalue routine
        eigsh : sparse eigenvalue routine

        Returns
        -------
        EigenstateElectron
        """
        gauge = comply_gauge(gauge)
        format = kwargs.pop("format", None)
        if kwargs.pop("sparse", False):
            e, v = self.eigsh(k, gauge=gauge, eigvals_only=False, **kwargs)
        else:
            e, v = self.eigh(k, gauge, eigvals_only=False, **kwargs)
        info = {"k": k, "gauge": gauge}
        for name in ("spin",):
            if name in kwargs:
                info[name] = kwargs[name]
        if not format is None:
            info["format"] = format
        # Since eigh returns the eigenvectors [:, i] we have to transpose
        return EigenstateElectron(v.T, e, self, **info)

    @staticmethod
    def read(sile, *args, **kwargs):
        """Reads Hamiltonian from `Sile` using `read_hamiltonian`.

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
        from sisl.io import BaseSile, get_sile

        if isinstance(sile, BaseSile):
            return sile.read_hamiltonian(*args, **kwargs)
        else:
            with get_sile(sile, mode="r") as fh:
                return fh.read_hamiltonian(*args, **kwargs)

    def fermi_level(
        self,
        bz=None,
        q=None,
        distribution="fermi_dirac",
        q_tol: float = 1e-10,
        *,
        apply_kwargs=None,
    ):
        """Calculate the Fermi-level using a Brillouinzone sampling and a target charge

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
        apply_kwargs : dict, optional
           keyword arguments passed directly to ``bz.apply.renew(**apply_kwargs)``.

        Returns
        -------
        float or array_like
            the Fermi-level of the system (or two if two different charges are passed)
        """
        if bz is None:
            # Gamma-point only
            from .brillouinzone import BrillouinZone

            bz = BrillouinZone(self)
        else:
            # Overwrite the parent in bz
            bz.set_parent(self)

        if apply_kwargs is None:
            apply_kwargs = {}

        if q is None:
            if self.spin.is_unpolarized:
                q = self.geometry.q0 * 0.5
            else:
                q = self.geometry.q0

        # Ensure we have an "array" in case of spin-polarized calculations
        q = _a.asarrayd(q)
        if np.any(q <= 0.0):
            raise ValueError(
                f"{self.__class__.__name__}.fermi_level cannot calculate the Fermi level "
                "for 0 electrons."
            )

        if isinstance(distribution, str):
            distribution = get_distribution(distribution)

        # B-cast for easier weights
        w = bz.weight.reshape(-1, 1)

        # Internal class to calculate the Fermi-level
        def _Ef(q, eig):
            # We could reduce it depending on the temperature,
            # however the distribution does not have the kT
            # parameter available.
            nextafter = np.nextafter

            # calculate boundaries
            min_Ef, max_Ef = eig.min(), eig.max()
            Ef = (min_Ef + max_Ef) * 0.5
            while nextafter(min_Ef, max_Ef) < max_Ef:
                # Calculate guessed charge
                qt = (distribution(eig, mu=Ef) * w).sum()

                if abs(qt - q) < q_tol:
                    return Ef

                if qt >= q:
                    max_Ef = Ef
                elif qt <= q:
                    min_Ef = Ef
                Ef = (min_Ef + max_Ef) * 0.5

            return Ef

        # Retrieve dispatcher for averaging
        eigh = bz.apply.renew(**apply_kwargs).array.eigh

        if self.spin.is_polarized and q.size == 2:
            if np.any(q >= len(self)):
                raise ValueError(
                    f"{self.__class__.__name__}.fermi_level cannot calculate the Fermi level "
                    "for electrons ({q}) equal to or above number of orbitals ({len(self)})."
                )
            # We need to do Fermi-level separately since the user requests
            # separate fillings
            Ef = _a.emptyd(2)
            Ef[0] = _Ef(q[0], eigh(spin=0))
            Ef[1] = _Ef(q[1], eigh(spin=1))
        else:
            # Ensure a single charge
            q = q.sum()
            if q >= len(self):
                raise ValueError(
                    f"{self.__class__.__name__}.fermi_level cannot calculate the Fermi level "
                    "for electrons ({q}) equal to or above number of orbitals ({len(self)})."
                )
            if self.spin.is_polarized:
                Ef = _Ef(q, np.concatenate([eigh(spin=0), eigh(spin=1)], axis=1))
            else:
                Ef = _Ef(q, eigh())

        return Ef
