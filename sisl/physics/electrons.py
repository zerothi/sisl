from __future__ import print_function, division

import numpy as np
from numpy import add, conj

from sisl._help import dtype_complex_to_real, _range as range
from .distributions import distribution as dist_func
from .spin import Spin
from .state import Coefficient, State, StateC




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
        class S(object):
            @property
            @staticmethod
            def shape():
                n = eig_v.shape[1]
                return (n, n)
            @staticmethod
            def dot(v):
                return v

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

        # Initialize
        PDOS = np.empty([4, eig_v.shape[0], len(E)], dtype=dtype_complex_to_real(eig_v.dtype))

        d = distribution(E - eig[0]).reshape(1, -1)
        v = S.dot(eig_v[0].reshape(-1, 2))
        D = (conj(eig_v[0]) * v.ravel()).reshape(-1, 2) # diagonal PDOS
        PDOS[0, :, :] = D.sum(1).reshape(-1, 1) * d # total DOS
        PDOS[3, :, :] = (D[:, 0] - D[:, 1]).reshape(-1, 1) * d # z-dos
        D = (conj(eig_v[0, 1::2]) * 2 * v[:, 0]).reshape(-1, 1) # psi_down * psi_up * 2
        PDOS[1, :, :] = D.real * d # x-dos
        PDOS[2, :, :] = D.imag * d # y-dos
        for i in range(1, len(eig)):
            d = distribution(E - eig[i]).reshape(1, -1)
            v = S.dot(eig_v[i].reshape(-1, 2))
            D = (conj(eig_v[i]) * v.ravel()).reshape(-1, 2)
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
        class S(object):
            @property
            @staticmethod
            def shape():
                n = eig_v.shape[1] // 2
                return (n, n)
            @staticmethod
            def dot(v):
                return v

    if S.shape[1] == eig_v.shape[1]:
        S = S[::2, ::2]

    # Initialize
    s = np.empty([4, eig_v.shape[0]], dtype=dtype_complex_to_real(eig_v.dtype))

    # TODO consider doing this all in a few lines
    # TODO Since there are no energy dependencies here we can actually do all
    # TODO dot products in one go and then use b-casting rules. Should be much faster
    # TODO but also way more memory demanding!
    v = S.dot(eig_v[0].reshape(-1, 2))
    D = (conj(eig_v[0]) * v.ravel()).reshape(-1, 2) # diagonal elements
    s[0, 0] = D.sum() # total spin moment
    s[3, 0] = (D[:, 0] - D[:, 1]).sum() # S_z
    D = 2 * (conj(eig_v[0, 1::2]) * v[:, 0]).sum() # psi_down * psi_up * 2
    s[1, 0] = D.real # S_x
    s[2, 0] = D.imag # S_y
    for i in range(1, len(eig_v)):
        v = S.dot(eig_v[i].reshape(-1, 2))
        D = (conj(eig_v[i]) * v.ravel()).reshape(-1, 2)
        s[0, i] += D.sum()
        s[3, i] += (D[:, 0] - D[:, 1]).sum()
        D = 2 * (conj(eig_v[i, 1::2]) * v[:, 0]).sum()
        s[1, i] += D.real
        s[2, i] += D.imag

    return s


class _common_State(object):
    __slots__ = []

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
        from .hamiltonian import Hamiltonian

        if isinstance(self.parent, Hamiltonian):
            # Calculate the overlap matrix
            if not self.parent.orthogonal:
                opt = {'k': self.info.get('k', (0, 0, 0)),
                       'format': format}
                if 'gauge' in self.info:
                    opt['gauge'] = self.info['gauge']
                return self.parent.Sk(**opt)

        class __FakeSk(object):
            """ Replacement object which superseedes a matrix """
            __slots__ = []
            @staticmethod
            @property
            def shape():
                n = self.shape[1]
                return (n, n)
            @staticmethod
            def dot(v):
                return v

        if spin is None:
            return __FakeSk
        if spin.kind > Spin.POLARIZED:
            class __FakeSk(object):
                """ Replacement object which superseedes a matrix """
                __slots__ = []
                @staticmethod
                @property
                def shape():
                    n = self.shape[1] // 2
                    return (n, n)
                @staticmethod
                def dot(v):
                    return v
        return __FakeSk

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


class CoefficientElectron(Coefficient):
    pass


class StateElectron(State, _common_State):
    pass


class StateCElectron(StateC, _common_State):
    pass


class EigenvalueElectron(CoefficientElectron):
    """ Eigenvalues of electronic states, no eigenvectors retained

    This holds routines that enable the calculation of density of states.
    """
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


class EigenvectorsElectron(StateElectron):
    """ Eigenvectors of electronic states, no eigenvalues retained

    This holds routines that enable the calculation of spin moments.
    """
    pass


class EigenStateElectron(StateCElectron):
    """ Eigen states of electrons with eigenvectors and eigenvalues.

    This holds routines that enable the calculation of (projected) density of states,
    spin moments (spin texture).
    """
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
        return PDOS(E, self.eig, self.state, self.Sk(spin=spin), distribution, spin)
