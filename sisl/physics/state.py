# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from numbers import Real
import numpy as np
from numpy import einsum, exp
from numpy import ndarray, bool_

from sisl._help import dtype_real_to_complex
from sisl._internal import set_module, singledispatchmethod
from sisl.linalg import eigh_destroy
import sisl._array as _a
from sisl.messages import warn


__all__ = ['degenerate_decouple', 'Coefficient', 'State', 'StateC']

_abs = np.absolute
_phase = np.angle
_argmax = np.argmax
_append = np.append
_diff = np.diff
_dot = np.dot
_conj = np.conjugate
_outer_ = np.outer
_pi = np.pi
_pi2 = np.pi * 2


def _inner(v1, v2):
    return _dot(_conj(v1), v2)


@set_module("sisl.physics")
def degenerate_decouple(state, M):
    r""" Return `vec` decoupled via matrix `M`

    The decoupling algorithm is this recursive algorithm starting from :math:`i=0`:

    .. math::

       \mathbf p &= \mathbf V^\dagger \mathbf M_i \mathbf V
       \\
       \mathbf p \mathbf u &= \boldsymbol \lambda \mathbf u
       \\
       \mathbf V &= \mathbf u^T \mathbf V

    Parameters
    ----------
    state : numpy.ndarray or State
       states to be decoupled on matrices `M`
       The states must have C-ordering, i.e. ``[0, ...]`` is the first
       state.
    M : numpy.ndarray
       matrix to project to before disentangling the states
    """
    if isinstance(state, State):
        state.state = degenerate_decouple(state.state, M)
    else:
        # since M may be a sparse matrix, we cannot use __matmul__
        p = _conj(state) @ M.dot(state.T)
        state = eigh_destroy(p)[1].T @ state
    return state


class _FakeMatrix:
    """ Replacement object which superseedes a matrix """
    __slots__ = ('n', 'm')
    ndim = 2

    def __init__(self, n, m=None):
        self.n = n
        if m is None:
            m = n
        self.m = m

    @property
    def shape(self):
        return (self.n, self.m)

    @staticmethod
    def dot(v):
        return v

    def multiply(self, v):
        try:
            if v.shape == self.shape:
                diag = np.diagonal(v)
                out = np.zeros_like(v)
                np.fill_diagonal(out, diag)
                return out
        except:
            pass
        return v

    @property
    def T(self):
        return self


@set_module("sisl.physics")
class ParentContainer:
    """ A container for parent and information """
    __slots__ = ['parent', 'info']

    def __init__(self, parent, **info):
        self.parent = parent
        self.info = info

    @singledispatchmethod
    def _sanitize_index(self, idx):
        r""" Ensure indices are transferred to acceptable integers """
        if idx is None:
            # in case __len__ is not defined, this will fail...
            return np.arange(len(self))
        idx = _a.asarray(idx)
        if idx.size == 0:
            return _a.asarrayl([])
        elif idx.dtype == bool_:
            return idx.nonzero()[0]
        return idx

    @_sanitize_index.register(ndarray)
    def _(self, idx):
        if idx.dtype == bool_:
            return np.flatnonzero(idx)
        return idx


@set_module("sisl.physics")
class Coefficient(ParentContainer):
    """ An object holding coefficients for a parent with info

    Attributes
    ----------
    c : numpy.ndarray
        coefficients
    info : dict
        information regarding the creation of these coefficients
    parent : obj
        object from where the coefficients has been calculated, in one way or the other

    Parameters
    ----------
    c : array_like
       coefficients
    parent : obj, optional
       a parent object that defines the origin of the coefficient
    **info : dict, optional
       an info dictionary that turns into an attribute on the object.
       This `info` may contain anything that may be relevant for the coefficient.
    """
    __slots__ = ['c']

    def __init__(self, c, parent=None, **info):
        super().__init__(parent, **info)
        self.c = np.atleast_1d(c)

    def __str__(self):
        """ The string representation of this object """
        s = f"{self.__class__.__name__}{{coefficients: {len(self)}, kind: {self.dkind}"
        if self.parent is None:
            s += '}}'
        else:
            s += ',\n {}\n}}'.format(str(self.parent).replace('\n', '\n '))
        return s

    def __len__(self):
        """ Number of coefficients """
        return self.shape[0]

    @property
    def dtype(self):
        """ Data-type for the coefficients """
        return self.c.dtype

    @property
    def dkind(self):
        """ The data-type of the coefficient (in str) """
        return np.dtype(self.c.dtype).kind

    @property
    def shape(self):
        """ Returns the shape of the coefficients """
        return self.c.shape

    def copy(self):
        """ Return a copy (only the coefficients are copied). ``parent`` and ``info`` are passed by reference """
        copy = self.__class__(self.c.copy(), self.parent)
        copy.info = self.info
        return copy

    def degenerate(self, eps=1e-8):
        """ Find degenerate coefficients with a specified precision

        Parameters
        ----------
        eps : float, optional
           the precision above which coefficients are not considered degenerate

        Returns
        -------
        list of numpy.ndarray
            a list of indices
        """
        deg = list()
        sidx = np.argsort(self.c)
        dc = _diff(self.c[sidx])

        # Degenerate indices
        idx = (np.absolute(dc) <= eps).nonzero()[0]
        if len(idx) == 0:
            # There are no degenerate coefficients
            return deg

        # These are the points were we split the degeneracies
        seps = (_diff(idx) > 1).nonzero()[0]
        IDX = np.array_split(idx, seps + 1)
        for idx in IDX:
            deg.append(_append(sidx[idx], sidx[idx[-1] + 1]))
        return deg

    def sub(self, idx, inplace=False):
        """ Return a new coefficient with only the specified coefficients

        Parameters
        ----------
        idx : int or array_like
            indices that are retained in the returned object
        inplace : bool, optional
            whether the values will be retained inplace

        Returns
        -------
        Coefficient
            a new coefficient only containing the requested elements, only if `inplace` is false
        """
        idx = self._sanitize_index(idx)
        if inplace:
            self.c = self.c[idx]
        else:
            sub = self.__class__(self.c[idx], self.parent)
            sub.info = self.info
            return sub

    def remove(self, idx, inplace=False):
        """ Return a new coefficient without the specified coefficients

        Parameters
        ----------
        idx : int or array_like
            indices that are removed in the returned object
        inplace : bool, optional
            whether the values will be removed inplace

        Returns
        -------
        Coefficient
            a new coefficient without containing the requested elements
        """
        idx = np.delete(np.arange(len(self)), self._sanitize_index(idx))
        return self.sub(idx, inplace)

    def __getitem__(self, key):
        """ Return a new coefficient object with only one associated coefficient

        Parameters
        ----------
        key : int or array_like
           the indices for the returned coefficients

        Returns
        -------
        Coefficient
            a new coefficient *only* with the indexed values
        """
        return self.sub(key)

    def iter(self, asarray=False):
        """ An iterator looping over the coefficients in this system

        Parameters
        ----------
        asarray : bool, optional
           if true the yielded values are the coefficient vectors, i.e. a numpy array.
           Otherwise an equivalent object is yielded.

        Yields
        ------
        coeff : Coefficent
           the current coefficent as an object, only returned if `asarray` is false.
        coeff : numpy.ndarray
           the current the coefficient as an array, only returned if `asarray` is true.
        """
        if asarray:
            for i in range(len(self)):
                yield self.c[i]
        else:
            for i in range(len(self)):
                yield self.sub(i)

    __iter__ = iter


@set_module("sisl.physics")
class State(ParentContainer):
    """ An object handling a set of vectors describing a given *state*

    Attributes
    ----------
    state : numpy.ndarray
        state coefficients
    info : dict
        information regarding the creation of the states
    parent : obj
        object from where the states has been calculated, in one way or the other

    Parameters
    ----------
    state : array_like
       state vectors ``state[i, :]`` containing the i'th state vector
    parent : obj, optional
       a parent object that defines the origin of the state.
    **info : dict, optional
       an info dictionary that turns into an attribute on the object.
       This `info` may contain anything that may be relevant for the state.

    Notes
    -----
    This class should be subclassed!
    """
    __slots__ = ['state']

    def __init__(self, state, parent=None, **info):
        """ Define a state container with a given set of states """
        super().__init__(parent, **info)
        self.state = np.atleast_2d(state)

    def __str__(self):
        """ The string representation of this object """
        s = f"{self.__class__.__name__}{{states: {len(self)}, kind: {self.dkind}"
        if self.parent is None:
            s += '}'
        else:
            s += ',\n {}\n}}'.format(str(self.parent).replace('\n', '\n '))
        return s

    def __len__(self):
        """ Number of states """
        return self.shape[0]

    @property
    def dtype(self):
        """ Data-type for the state """
        return self.state.dtype

    @property
    def dkind(self):
        """ The data-type of the state (in str) """
        return np.dtype(self.state.dtype).kind

    @property
    def shape(self):
        """ Returns the shape of the state """
        return self.state.shape

    def copy(self):
        """ Return a copy (only the state is copied). ``parent`` and ``info`` are passed by reference """
        copy = self.__class__(self.state.copy(), self.parent)
        copy.info = self.info
        return copy

    def sub(self, idx, inplace=False):
        """ Return a new state with only the specified states

        Parameters
        ----------
        idx : int or array_like
            indices that are retained in the returned object
        inplace : bool, optional
            whether the values will be retained inplace

        Returns
        -------
        State
           a new state only containing the requested elements, only if `inplace` is false
        """
        idx = self._sanitize_index(idx)
        if inplace:
            self.state = self.state[idx]
        else:
            sub = self.__class__(self.state[idx], self.parent)
            sub.info = self.info
            return sub

    def remove(self, idx, inplace=False):
        """ Return a new state without the specified vectors

        Parameters
        ----------
        idx : int or array_like
            indices that are removed in the returned object
        inplace : bool, optional
            whether the values will be removed inplace

        Returns
        -------
        State
            a new state without containing the requested elements, only if `inplace` is false
        """
        idx = np.delete(np.arange(len(self)), self._sanitize_index(idx))
        return self.sub(idx, inplace)

    def translate(self, isc):
        r""" Translate the vectors to a new unit-cell position

        The method is thoroughly explained in `tile` while this one only
        selects the corresponding state vector

        Parameters
        ----------
        isc : (3,)
           number of offsets for the statevector

        See Also
        --------
        tile : equivalent method for generating more cells simultaneously
        """
        # the k-point gets reduced
        k = _a.asarrayd(self.info.get("k", [0]*3))
        assert len(isc) == 3

        s = self.copy()
        # translate the bloch coefficients with:
        #   exp(i k.T)
        # with T being
        #   i * a_0 + j * a_1 + k * a_2
        if not np.allclose(k, 0):
            # there will only be a phase if k != 0
            s.state *= exp(2j*_pi * k @ isc)
        return s

    def tile(self, reps, axis, normalize=False, offset=0):
        r"""Tile the state vectors for a new supercell

        Tiling a state vector makes use of the Bloch factors for a state by utilizing

        .. math::

           \psi_{\mathbf k}(\mathbf r + \mathbf T) \propto e^{i\mathbf k\cdot \mathbf T}

        where :math:`\mathbf T = i\mathbf a_0 + j\mathbf a_1 + l\mathbf a_2`. Note that `axis`
        selects which of the :math:`\mathbf a_i` vectors that are translated and `reps` corresponds
        to the :math:`i`, :math:`j` and :math:`l` variables. The `offset` moves the individual states
        by said amount, i.e. :math:`i\to i+\mathrm{offset}`.

        Parameters
        ----------
        reps : int
           number of repetitions along a specific lattice vector
        axis : int
           lattice vector to tile along
        normalize: bool, optional
           whether the states are normalized upon return, may be useful for
           eigenstates
        offset: float, optional
           the offset for the phase factors

        See Also
        --------
        Geometry.tile
        """
        # the parent gets tiled
        parent = self.parent.tile(reps, axis)
        # the k-point gets reduced
        k = _a.asarrayd(self.info.get("k", [0]*3))

        # now tile the state vectors
        state = np.tile(self.state, (1, reps)).astype(np.complex128, copy=False)
        # re-shape to apply phase-factors
        state.shape = (len(self), reps, -1)

        # Tiling stuff is trivial since we simply
        # translate the bloch coefficients with:
        #   exp(i k.T)
        # with T being
        #   i * a_0 + j * a_1 + k * a_2
        # We can leave out the lattice vectors entirely
        phase = exp(2j*_pi * k[axis] * (_a.aranged(reps) + offset))

        state *= phase.reshape(1, -1, 1)
        state.shape = (len(self), -1)

        # update new k; when we double the system, we halve the periodicity
        # and hence we need to account for this
        k[axis] = (k[axis] * reps % 1)
        while k[axis] > 0.5:
            k[axis] -= 1
        while k[axis] <= -0.5:
            k[axis] += 1

        # this allows us to make the same usable for StateC classes
        s = self.copy()
        s.parent = parent
        s.state = state
        # update the k-point
        s.info = dict(**self.info)
        s.info.update({'k': k})

        if normalize:
            return s.normalize()
        return s

    def __getitem__(self, key):
        """ Return a new state with only one associated state

        Parameters
        ----------
        key : int or array_like
           the indices for the returned states

        Returns
        -------
        State
            a new state *only* with the indexed values
        """
        return self.sub(key)

    def iter(self, asarray=False):
        """ An iterator looping over the states in this system

        Parameters
        ----------
        asarray : bool, optional
           if true the yielded values are the state vectors, i.e. a numpy array.
           Otherwise an equivalent object is yielded.

        Yields
        ------
        state : State
           a state *only* containing individual elements, if `asarray` is false
        state : numpy.ndarray
           a state *only* containing individual elements, if `asarray` is true
        """
        if asarray:
            for i in range(len(self)):
                yield self.state[i]
        else:
            for i in range(len(self)):
                yield self.sub(i)

    __iter__ = iter

    def norm(self):
        r""" Return a vector with the Euclidean norm of each state :math:`\sqrt{\langle\psi|\psi\rangle}`

        Returns
        -------
        numpy.ndarray
            the Euclidean norm for each state
        """
        return self.norm2() ** 0.5

    def norm2(self, sum=True):
        r""" Return a vector with the norm of each state :math:`\langle\psi|\psi\rangle`

        Parameters
        ----------
        sum : bool, optional
           for true only a single number per state will be returned, otherwise the norm
           per basis element will be returned.

        Returns
        -------
        numpy.ndarray
            the squared norm for each state
        """
        if sum:
            return self.inner()
        return _conj(self.state) * self.state

    def ipr(self, q=2):
        r""" Calculate the inverse participation ratio (IPR) for arbitrary `q` values

        The inverse participation ratio is defined as

        .. math::
            I_{q,i} = \frac{\sum_\nu |\psi_{i\nu}|^{2q}}{
               \big[\sum_\nu |\psi_{i\nu}|^2\big]^q}

        where :math:`i` is the band index and :math:`\nu` is the orbital.
        The order of the IPR is defaulted to :math:`q=2`, see :eq:`ipr2` for details.
        The IPR may be used to distinguish Anderson localization and extended
        states:

        .. math::
           :nowrap:
           :label: ipr2

           \begin{align}
           \lim_{L\to\infty} I_{2,i} = \left\{\begin{aligned}
                1/L^d & \text{extended state}
                \\
                const. & \text{localized state}
               \end{aligned}
           \end{align}

        For further details see [1]_. Note that for eigen states the IPR reduces to:

        .. math::
            I_{q,i} = \sum_\nu |\psi_{i\nu}|^{2q}

        since the denominator is :math:`1^{q} = 1`.

        Parameters
        ----------
        q : int, optional
          order parameter for the IPR

        References
        ----------
        .. [1] :doi:`N. C. Murphy *et.al.*, "Generalized inverse participation ratio as a possible measure of localization for interacting systems", PRB **83**, 184206 (2011) <10.1103/PhysRevB.83.184206>`
        """
        # This *has* to be a real value C * C^* == real
        state_abs2 = self.norm2(sum=False).real
        assert q >= 2, f"{self.__class__.__name__}.ipr requires q>=2"
        # abs2 is already having the exponent 2
        return (state_abs2 ** q).sum(-1) / state_abs2.sum(-1) ** q

    def normalize(self):
        r""" Return a normalized state where each state has :math:`|\psi|^2=1`

        This is roughly equivalent to:

        >>> state = State(np.arange(10))
        >>> n = state.norm()
        >>> norm_state = State(state.state / n.reshape(-1, 1))

        Notes
        -----
        This does *not* take into account a possible overlap matrix when non-orthogonal basis sets are used.

        Returns
        -------
        State
            a new state with all states normalized, otherwise equal to this
        """
        n = self.norm()
        s = self.__class__(self.state / n.reshape(-1, 1), parent=self.parent)
        s.info = self.info
        return s

    def outer(self, ket=None):
        r""" Return the outer product by :math:`\sum_i|\psi_i\rangle\langle\psi'_i|`

        Parameters
        ----------
        ket : State, optional
           the ket object to calculate the outer product of, if not passed it will do the outer
           product with itself. The object itself will always be the bra :math:`|\psi_i\rangle`

        Notes
        -----
        This does *not* take into account a possible overlap matrix when non-orthogonal basis sets are used.

        Returns
        -------
        numpy.ndarray
            a matrix with the sum of outer state products
        """
        if ket is None:
            ket = self.state
        elif isinstance(ket, State):
            ket = ket.state

        if not np.array_equal(self.shape, ket.shape):
            raise ValueError(f"{self.__class__.__name__}.outer requires the objects to have the same shape")
        return einsum('ki,kj->ij', self.state, _conj(ket))

    def inner(self, ket=None, matrix=None, diag=True):
        r""" Calculate the inner product as :math:`\mathbf A_{ij} = \langle\psi_i|\mathbf M|\psi'_j\rangle`

        Parameters
        ----------
        ket : State, optional
           the ket object to calculate the inner product with, if not passed it will do the inner
           product with itself. The object itself will always be the bra :math:`\langle\psi_i|`
        matrix : array_like, optional
           whether a matrix is sandwiched between the bra and ket, default to the identity matrix
        diag : bool, optional
           only return the diagonal matrix :math:`\mathbf A_{ii}`.

        Notes
        -----
        This does *not* take into account a possible overlap matrix when non-orthogonal basis sets are used.

        Raises
        ------
        ValueError
            if the number of state coefficients are different for the bra and ket

        Returns
        -------
        numpy.ndarray
            a matrix with the sum of inner state products
        """
        if matrix is None:
            M = _FakeMatrix(self.shape[-1])
        else:
            M = matrix
        ndim = M.ndim

        bra = self.state
        # decide on the ket
        if ket is None:
            ket = self.state
        elif isinstance(ket, State):
            # check whether this, and ket are both originating from
            # non-orthogonal basis. That would be non-ideal
            ket = ket.state
        if len(ket.shape) == 1:
            ket.shape = (1, -1)

        # They *must* have same number of basis points per state
        if self.shape[-1] != ket.shape[-1]:
            raise ValueError(f"{self.__class__.__name__}.inner requires the objects to have the same number of coefficients per vector {self.shape[-1]} != {ket.shape[-1]}")

        if diag:
            if len(bra) != len(ket):
                warn(f"{self.__class__.__name__}.inner matrix product is non-square, only the first {min(len(bra), len(ket))} diagonal elements will be returned.")
                if len(bra) < len(ket):
                    ket = ket[:len(bra)]
                else:
                    bra = bra[:len(ket)]
            if ndim == 2:
                Aij = einsum('ij,ji->i', _conj(bra), M.dot(ket.T))
            elif ndim == 1:
                Aij = einsum('ij,j,ij->i', _conj(bra), M, ket)
        elif ndim == 2:
            Aij = _conj(bra) @ M.dot(ket.T)
        elif ndim == 1:
            Aij = einsum('ij,j,kj->ik', _conj(bra), M, ket)
        return Aij

    def phase(self, method='max', ret_index=False):
        r""" Calculate the Euler angle (phase) for the elements of the state, in the range :math:`]-\pi;\pi]`

        Parameters
        ----------
        method : {'max', 'all'}
           for max, the phase for the element which has the largest absolute magnitude is returned,
           for all, all phases are calculated
        ret_index : bool, optional
           return indices for the elements used when ``method=='max'``
        """
        if method == 'max':
            idx = _argmax(_abs(self.state), 1)
            if ret_index:
                return _phase(self.state[:, idx]), idx
            return _phase(self.state[:, idx])
        elif method == 'all':
            return _phase(self.state)
        raise ValueError(f"{self.__class__.__name__}.phase only accepts method in [max, all]")

    def align_phase(self, other, ret_index=False, inplace=False):
        r""" Align `self` with the phases for `other`, a copy may be returned 

        States will be rotated by :math:`\pi` provided the phase difference between the states are above :math:`|\Delta\theta| > \pi/2`.

        Parameters
        ----------
        other : State
           the other state to align onto this state
        ret_index : bool, optional
           return which indices got swapped
        inplace : bool, optional
           rotate the states in-place

        See Also
        --------
        align_norm : re-order states such that site-norms have a smaller residual
        """
        other_phase, idx = other.phase(ret_index=True)
        phase = _phase(self.state[:, idx])

        # Calculate absolute phase difference
        abs_phase = _abs((phase - other_phase + _pi) % _pi2 - _pi)

        idx = (abs_phase > _pi / 2).nonzero()[0]

        ret = None
        if inplace:
            if ret_index:
                ret = ret_index
            self.state[idx] *= -1
            return ret

        out = self.copy()
        out.state[idx] *= -1
        if ret_index:
            return out, idx
        return out

    def align_norm(self, other, ret_index=False, inplace=False):
        r""" Align `self` with the site-norms of `other`, a copy may optionally be returned

        To determine the new ordering of `self` first calculate the residual norm of the site-norms.

        .. math::
           \delta N_{\alpha\beta} = \sum_i \big(\langle \psi^\alpha_i | \psi^\alpha_i\rangle - \langle \psi^\beta_i | \psi^\beta_i\rangle\big)^2

        where :math:`\alpha` and :math:`\beta` correspond to state indices in `self` and `other`, respectively.
        The new states (from `self`) returned is then ordered such that the index
        :math:`\alpha \equiv \beta'` where :math:`\delta N_{\alpha\beta}` is smallest.

        Parameters
        ----------
        other : State
           the other state to align against
        ret_index : bool, optional
           also return indices for the swapped indices
        inplace : bool, optional
           swap states in-place

        Returns
        -------
        self_swap : State
            A swapped instance of `self`, only if `inplace` is False
        index : array of int
            the indices that swaps `self` to be ``self_swap``, i.e. ``self_swap = self.sub(index)``
            Only if `inplace` is False and `ret_index` is True
        Notes
        -----
        The input state and output state have the same number of states, but their ordering is not necessarily the same.

        See Also
        --------
        align_phase : rotate states such that their phases align
        """
        snorm = self.norm2(False).real
        onorm = other.norm2(False).real

        # Now find new orderings
        show_warn = False
        sidx = _a.fulli(len(self), -1)
        oidx = _a.emptyi(len(self))
        for i in range(len(self)):
            R = snorm[i] - onorm
            R = einsum('ij,ij->i', R, R)

            # Figure out which band it should correspond to
            # find closest largest one
            for j in np.argsort(R):
                if j not in sidx[:i]:
                    sidx[i] = j
                    oidx[j] = i
                    break
                show_warn = True

        if show_warn:
            warn(f"{self.__class__.__name__}.align_norm found multiple possible candidates with minimal residue, swapping not unique")

        if inplace:
            self.sub(oidx, inplace=True)
            if ret_index:
                return oidx
        elif ret_index:
            return self.sub(oidx), oidx
        else:
            return self.sub(oidx)

    def rotate(self, phi=0., individual=False):
        r""" Rotate all states (in-place) to rotate the largest component to be along the angle `phi`

        The states will be rotated according to:

        .. math::

            S' = S / S^\dagger_{\phi-\mathrm{max}} \exp (i \phi),

        where :math:`S^\dagger_{\phi-\mathrm{max}}` is the phase of the component with the largest amplitude
        and :math:`\phi` is the angle to align on.

        Parameters
        ----------
        phi : float, optional
           angle to align the state at (in radians), 0 is the positive real axis
        individual : bool, optional
           whether the rotation is per state, or a single maximum component is chosen.
        """
        # Convert angle to complex phase
        phi = exp(1j * phi)
        s = self.state.view()
        if individual:
            for i in range(len(self)):
                # Find the maximum amplitude index
                idx = _argmax(_abs(s[i, :]))
                s[i, :] *= phi * _conj(s[i, idx] / _abs(s[i, idx]))
        else:
            # Find the maximum amplitude index among all elements
            idx = np.unravel_index(_argmax(_abs(s)), s.shape)
            s *= phi * _conj(s[idx] / _abs(s[idx]))

    def change_gauge(self, gauge, offset=(0, 0, 0)):
        r""" In-place change of the gauge of the state coefficients

        The two gauges are related through:

        .. math::

            \tilde C_j = e^{i\mathbf k\mathbf r_j} C_j

        where :math:`C_j` and :math:`\tilde C_j` belongs to the ``r`` and ``R`` gauge, respectively.

        Parameters
        ----------
        gauge : {'R', 'r'}
            specify the new gauge for the mode coefficients
        offset : array_like, optional
            whether the coordinates should be offset by another phase-factor
        """
        # These calls will fail if the gauge is not specified.
        # In that case it will not do anything
        if self.info.get('gauge', gauge) == gauge:
            # Quick return
            return

        # Update gauge value
        self.info['gauge'] = gauge

        # Check that we can do a gauge transformation
        k = _a.asarrayd(self.info.get('k', [0., 0., 0.]))
        if k.dot(k) <= 0.000000001:
            return

        g = self.parent.geometry
        xyz = g.xyz + offset
        phase = xyz[g.o2a(_a.arangei(g.no)), :] @ (k @ g.rcell)

        try:
            if not self.parent.spin.is_diagonal:
                # for NC/SOC we have a 2x2 spin-box per orbital
                phase = np.repeat(phase, 2)
        except:
            pass

        if gauge == 'r':
            # R -> r gauge tranformation.
            self.state *= exp(-1j * phase).reshape(1, -1)
        elif gauge == 'R':
            # r -> R gauge tranformation.
            self.state *= exp(1j * phase).reshape(1, -1)

    # def toStateC(self, norm=1.):
    #     r""" Transforms the states into normalized values equal to `norm` and specifies the coefficients in `StateC` as the norm

    #     This is an easy method to renormalize the state vectors to a common (or state dependent) normalization constant.

    #     .. math::
    #         c_i &= \sqrt{\langle \psi_i | \psi_i\rangle} / \mathrm{norm} \\
    #           |\psi_i\rangle &= | \psi_i\rangle / c_i

    #     Parameters
    #     ----------
    #     norm : value, array_like
    #         the resulting norm of all (or individual states)

    #     Returns
    #     -------
    #     StateC
    #        a new coefficient state object with associated coefficients
    #     """
    #     n = len(self)
    #     norm = _a.asarray(norm).ravel()
    #     if norm.size == 1 and n > 1:
    #         norm = np.tile(norm, n)
    #     elif norm.size != n:
    #         raise ValueError(self.__class__.__name__ + '.toStateC requires the input norm to be a single float or having equal length as this state!')

    #     # Correct data-type
    #     if norm.dtype in [np.complex64, np.complex128]:
    #         dtype = norm.dtype
    #     else:
    #         dtype = dtype_complex_to_real(self.dtype)

    #     # TODO check datatype if norm is complex but state is real
    #     c = np.empty(n, dtype=dtype)
    #     state = np.empty(self.shape, dtype=self.dtype)

    #     for i in range(n):
    #         c[i] = (_inner1(self.state[i].ravel()).astype(dtype) / norm[i]) ** 0.5
    #         state[i, ...] = self.state[i, ...] / c[i]

    #     cs = StateC(state, c, parent=self.parent)
    #     cs.info = self.info
    #     return cs


# Although the StateC could inherit from both Coefficient and State
# there are problems with __slots__ and multiple inheritance schemes.
# I.e. we are forced to do *one* inheritance, which we choose to be State.
@set_module("sisl.physics")
class StateC(State):
    """ An object handling a set of vectors describing a given *state* with associated coefficients `c`

    Attributes
    ----------
    c : numpy.ndarray
        coefficients assigned to each state
    state : numpy.ndarray
        state coefficients
    info : dict
        information regarding the creation of the states
    parent : obj
        object from where the states has been calculated, in one way or the other

    Parameters
    ----------
    state : array_like
       state vectors ``state[i, :]`` containing the i'th state vector
    c : array_like
       coefficients for the states ``c[i]`` containing the i'th coefficient
    parent : obj, optional
       a parent object that defines the origin of the state.
    **info : dict, optional
       an info dictionary that turns into an attribute on the object.
       This `info` may contain anything that may be relevant for the state.

    Notes
    -----
    This class should be subclassed!
    """
    __slots__ = ['c']

    def __init__(self, state, c, parent=None, **info):
        """ Define a state container with a given set of states and coefficients for the states """
        super().__init__(state, parent, **info)
        self.c = np.atleast_1d(c)
        if len(self.c) != len(self.state):
            raise ValueError(f"{self.__class__.__name__} could not be created with coefficients and states "
                             "having unequal length.")

    def copy(self):
        """ Return a copy (only the coefficients and states are copied), ``parent`` and ``info`` are passed by reference """
        copy = self.__class__(self.state.copy(), self.c.copy(), self.parent)
        copy.info = self.info
        return copy

    def normalize(self):
        r""" Return a normalized state where each state has :math:`|\psi|^2=1`

        This is roughly equivalent to:

        >>> state = StateC(np.arange(10), 1)
        >>> n = state.norm()
        >>> norm_state = StateC(state.state / n.reshape(-1, 1), state.c.copy())
        >>> norm_state.c[0] == 1

        Returns
        -------
        State
            a new state with all states normalized, otherwise equal to this
        """
        n = self.norm()
        s = self.__class__(self.state / n.reshape(-1, 1), self.c.copy(), parent=self.parent)
        s.info = self.info
        return s

    def outer(self, idx=None):
        r""" Return the outer product for the indices `idx` (or all if ``None``) by :math:`\sum_i|\psi_i\rangle c_i\langle\psi_i|`

        Parameters
        ----------
        idx : int or array_like, optional
           only perform an outer product of the specified indices, otherwise all states are used

        Returns
        -------
        numpy.ndarray
            a matrix with the sum of outer state products
        """
        if idx is None:
            return einsum('k,ki,kj->ij', self.c, self.state, _conj(self.state))
        idx = self._sanitize_index(idx).ravel()
        return einsum('k,ki,kj->ij', self.c[idx], self.state[idx], _conj(self.state[idx]))

    def sort(self, ascending=True):
        """ Sort and return a new `StateC` by sorting the coefficients (default to ascending)

        Parameters
        ----------
        ascending : bool, optional
            sort the contained elements ascending, else they will be sorted descending
        """
        if ascending:
            idx = np.argsort(self.c)
        else:
            idx = np.argsort(-self.c)
        return self.sub(idx)

    def derivative(self, order=1, degenerate=1e-5, degenerate_dir=(1, 1, 1), matrix=False):
        r""" Calculate the derivative with respect to :math:`\mathbf k` for a set of states up to a given order

        These are calculated using the analytic expression (:math:`\alpha` corresponding to the Cartesian directions),
        here only shown for the 1st order derivative:

        .. math::

           \mathbf{d}_{ij\alpha} = \langle \psi_j |
                    \frac{\partial}{\partial\mathbf k_\alpha} \mathbf H(\mathbf k) | \psi_i \rangle

        In case of non-orthogonal basis the equations substitutes :math:`\mathbf H(\mathbf k)` by
        :math:`\mathbf H(\mathbf k) - \epsilon_i\mathbf S(\mathbf k)`.

        The 2nd order derivatives are calculated with the Berry curvature correction:

        .. math::

           \mathbf d^2_{ij\alpha\beta} = \langle\psi_j|
               \frac{\partial^2}{\partial\mathbf k_\alpha\partial\mathbf k_\beta} \mathbf H(\mathbf k) | \psi_i\rangle
               - \frac12\frac{\mathbf{d}_{ij\alpha}\mathbf{d}_{ij\beta}}
                     {\epsilon_j - \epsilon_i}

        Notes
        -----

        When requesting 2nd derivatives it will not be advisable to use a `sub` before
        calculating the derivatives since the 1st order perturbation uses the energy
        differences (Berry contribution) and the 1st derivative matrix for correcting the curvature.

        For states at the :math:`\Gamma` point you may get warnings about casting complex numbers
        to reals. In these cases you should force the state at the :math:`\Gamma` point to be calculated
        in complex numbers to enable the correct decoupling.

        Parameters
        ----------
        order : {1, 2}
           an integer specifying which order of the derivative is being calculated.
        degenerate : float or list of array_like, optional
           If a float is passed it is regarded as the degeneracy tolerance used to calculate the degeneracy
           levels. Defaults to 1e-5 eV.
           If a list, it contains the indices of degenerate states. In that case a prior diagonalization
           is required to decouple them. See `degenerate_dir` for the sum of directions.
        degenerate_dir : (3,), optional
           a direction used for degenerate decoupling. The decoupling based on the velocity along this direction
        matrix : bool, optional
           whether the full matrix or only the diagonal components are returned

        See Also
        --------
        SparseOrbitalBZ.dPk : function for generating the matrix derivatives
        SparseOrbitalBZ.dSk : function for generating the matrix derivatives in non-orthogonal basis

        Returns
        -------
        dv, ddv
            if `matrix` is false, they are per state with shape ``(state.shape[0], *)``, ddv is only
            returned if ``order>=2``
        dv, ddv
            if `matrix` is true, they are per state with shape ``(state.shape[0], state.shape[0], *)``, ddv is only
            returned if ``order>=2``
        """

        # Figure out arguments
        opt = {
            "k": self.info.get("k", (0, 0, 0)),
            "dtype": dtype_real_to_complex(self.dtype),
        }

        if degenerate is None:
            pass
        elif isinstance(degenerate, Real):
            degenerate = self.degenerate(degenerate)

        if order not in (1, 2):
            raise NotImplementedError(f"{self.__class__.__name__}.derivative required order to be in this list: [1, 2], higher order derivatives are not implemented")

        def add_keys(opt, *keys):
            for key in keys:
                # Store gauge, if present
                val = self.info.get(key)
                if val is not None:
                    # this must mean the methods accepts this as an argument
                    opt[key] = val

        add_keys(opt, "gauge", "format")

        # Initialize variables
        ddPk = dSk = ddSk = None

        # Figure out if we are dealing with a non-orthogonal basis set
        is_orthogonal = True
        try:
            if not self.parent.orthogonal:
                is_orthogonal = False
                dSk = self.parent.dSk(**opt)
                if order > 1:
                    ddSk = self.parent.ddSk(**opt)
        except: pass

        # Now figure out if spin is a thing
        add_keys(opt, "spin")
        dPk = self.parent.dPk(**opt)
        if order > 1:
            ddPk = self.parent.ddPk(**opt)

        # short-hand, everything will now be done *in-place* of this object
        state = self.state
        # in case the state is not an Eigenstate*
        energy = self.c

        # Now parse the degeneracy handling
        if degenerate is not None:
            # normalize direction
            degenerate_dir = _a.asarrayd(degenerate_dir)
            degenerate_dir /= (degenerate_dir ** 2).sum() ** 0.5

            # de-coupling is only done for the 1st derivative

            # create the degeneracy decoupling projector
            deg_dPk = sum(d*dh for d, dh in zip(degenerate_dir, dPk))

            if is_orthogonal:
                for deg in degenerate:
                    # Set the average energy
                    energy[deg] = np.average(energy[deg])
                    # Now diagonalize to find the contributions from individual states
                    # then re-construct the seperated degenerate states
                    # Since we do this for all directions we should decouple them all
                    state[deg] = degenerate_decouple(state[deg], deg_dPk)
            else:
                for deg in degenerate:
                    e = np.average(energy[deg])
                    energy[deg] = e
                    deg_dSk = sum((d*e)*ds for d, ds in zip(degenerate_dir, dSk))
                    state[deg] = degenerate_decouple(state[deg], deg_dPk - deg_dSk)

        # States have been decoupled and we can calculate things now
        # number of states
        nstate, lstate = state.shape
        # we calculate the first derivative matrix
        cstate = _conj(state)

        # We split everything up into orthogonal and non-orthogonal
        # This reduces if-checks
        if is_orthogonal:

            if matrix or order > 1:
                # calculate the full matrix
                v = np.empty((nstate, nstate, 3), dtype=opt["dtype"])
                v[:, :, 0] = (cstate @ dPk[0].dot(state.T)).T
                v[:, :, 1] = (cstate @ dPk[1].dot(state.T)).T
                v[:, :, 2] = (cstate @ dPk[2].dot(state.T)).T

                if matrix:
                    ret = (v,)
                else:
                    # diagonal of nd removes axis0 and axis1 and appends a new
                    # axis to the end, this is not what we want
                    ret = (np.diagonal(v).T,)

            else:
                # calculate projections on states
                v = np.empty((nstate, 3), dtype=opt["dtype"])
                v[:, 0] = (cstate * dPk[0].dot(state.T).T).sum(1)
                v[:, 1] = (cstate * dPk[1].dot(state.T).T).sum(1)
                v[:, 2] = (cstate * dPk[2].dot(state.T).T).sum(1)

                ret = (v,)

            if order > 1:
                # Now calculate the 2nd order corrections
                # loop energies

                if matrix:
                    vv = np.empty((nstate, nstate, 6), dtype=opt["dtype"])
                    for s, e in enumerate(energy):
                        de = energy - e
                        # add factor 2 here
                        np.divide(2, de, where=(de != 0), out=de)

                        # we will use this multiple times
                        absv = np.absolute(v[s]).T

                        # calculate 2nd derivative
                        # xx
                        vv[s, :, 0] = cstate @ ddPk[0].dot(state[s]) - de * absv[0] ** 2
                        # yy
                        vv[s, :, 1] = cstate @ ddPk[1].dot(state[s]) - de * absv[1] ** 2
                        # zz
                        vv[s, :, 2] = cstate @ ddPk[2].dot(state[s]) - de * absv[2] ** 2
                        # yz
                        vv[s, :, 3] = cstate @ ddPk[3].dot(state[s]) - de * absv[1] * absv[2]
                        # xz
                        vv[s, :, 4] = cstate @ ddPk[4].dot(state[s]) - de * absv[0] * absv[2]
                        # xy
                        vv[s, :, 5] = cstate @ ddPk[5].dot(state[s]) - de * absv[0] * absv[1]

                else:
                    vv = np.empty((nstate, 6), dtype=opt["dtype"])
                    for s, e in enumerate(energy):
                        de = energy - e
                        # add factor 2 here
                        np.divide(2, de, where=(de != 0), out=de)

                        # we will use this multiple times, .T just for easier access
                        absv = np.absolute(v[s]).T

                        # calculate 2nd derivative
                        # xx
                        vv[s, 0] = cstate[s] @ ddPk[0].dot(state[s]) - de @ absv[0] ** 2
                        # yy
                        vv[s, 1] = cstate[s] @ ddPk[1].dot(state[s]) - de @ absv[1] ** 2
                        # zz
                        vv[s, 2] = cstate[s] @ ddPk[2].dot(state[s]) - de @ absv[2] ** 2
                        # yz
                        vv[s, 3] = cstate[s] @ ddPk[3].dot(state[s]) - de @ (absv[1] * absv[2])
                        # xz
                        vv[s, 4] = cstate[s] @ ddPk[4].dot(state[s]) - de @ (absv[0] * absv[2])
                        # xy
                        vv[s, 5] = cstate[s] @ ddPk[5].dot(state[s]) - de @ (absv[0] * absv[1])

                ret += (vv,)
        else:
            # non-orthogonal basis set

            if matrix or order > 1:
                # calculate the full matrix
                v = np.empty((nstate, nstate, 3), dtype=opt["dtype"])
                for s, e in enumerate(energy):
                    v[s, :, 0] = cstate @ (dPk[0] - e * dSk[0]).dot(state[s])
                    v[s, :, 1] = cstate @ (dPk[1] - e * dSk[1]).dot(state[s])
                    v[s, :, 2] = cstate @ (dPk[2] - e * dSk[2]).dot(state[s])

                if matrix:
                    ret = (v,)
                else:
                    # diagonal of nd removes axis0 and axis1 and appends a new
                    # axis to the end, this is not what we want
                    ret = (np.diagonal(v).T,)

            else:
                # calculate diagonal components on states
                v = np.empty((nstate, 3), dtype=opt["dtype"])
                for s, e in enumerate(energy):
                    v[s, 0] = cstate[s] @ (dPk[0] - e * dSk[0]).dot(state[s])
                    v[s, 1] = cstate[s] @ (dPk[1] - e * dSk[1]).dot(state[s])
                    v[s, 2] = cstate[s] @ (dPk[2] - e * dSk[2]).dot(state[s])

                ret = (v,)

            if order > 1:
                # Now calculate the 2nd order corrections
                # loop energies

                if matrix:
                    vv = np.empty((nstate, nstate, 6), dtype=opt["dtype"])
                    for s, e in enumerate(energy):
                        de = energy - e
                        # add factor 2 here
                        np.divide(2, de, where=(de != 0), out=de)

                        # we will use this multiple times:
                        absv = np.absolute(v[s]).T

                        # calculate 2nd derivative
                        # xx
                        vv[s, :, 0] = cstate @ (ddPk[0] - e * ddSk[0]).dot(state[s]) - de * absv[0] ** 2
                        # yy
                        vv[s, :, 1] = cstate @ (ddPk[1] - e * ddSk[1]).dot(state[s]) - de * absv[1] ** 2
                        # zz
                        vv[s, :, 2] = cstate @ (ddPk[2] - e * ddSk[2]).dot(state[s]) - de * absv[2] ** 2
                        # yz
                        vv[s, :, 3] = cstate @ (ddPk[3] - e * ddSk[3]).dot(state[s]) - de * absv[1] * absv[2]
                        # xz
                        vv[s, :, 4] = cstate @ (ddPk[4] - e * ddSk[4]).dot(state[s]) - de * absv[0] * absv[2]
                        # xy
                        vv[s, :, 5] = cstate @ (ddPk[5] - e * ddSk[5]).dot(state[s]) - de * absv[0] * absv[1]

                else:
                    vv = np.empty((nstate, 6), dtype=opt["dtype"])
                    for s, e in enumerate(energy):
                        de = energy - e
                        # add factor 2 here
                        np.divide(2, de, where=(de != 0), out=de)

                        # we will use this multiple times:
                        absv = np.absolute(v[s]).T

                        # calculate 2nd derivative
                        # xx
                        vv[s, 0] = cstate[s] @ (ddPk[0] - e * ddSk[0]).dot(state[s]) - de @ absv[0] ** 2
                        # yy
                        vv[s, 1] = cstate[s] @ (ddPk[1] - e * ddSk[1]).dot(state[s]) - de @ absv[1] ** 2
                        # zz
                        vv[s, 2] = cstate[s] @ (ddPk[2] - e * ddSk[2]).dot(state[s]) - de @ absv[2] ** 2
                        # yz
                        vv[s, 3] = cstate[s] @ (ddPk[3] - e * ddSk[3]).dot(state[s]) - de @ (absv[1] * absv[2])
                        # xz
                        vv[s, 4] = cstate[s] @ (ddPk[4] - e * ddSk[4]).dot(state[s]) - de @ (absv[0] * absv[2])
                        # xy
                        vv[s, 5] = cstate[s] @ (ddPk[5] - e * ddSk[5]).dot(state[s]) - de @ (absv[0] * absv[1])

                ret += (vv,)

        if len(ret) == 1:
            return ret[0]
        return ret

    def degenerate(self, eps):
        """ Find degenerate coefficients with a specified precision

        Parameters
        ----------
        eps : float
           the precision above which coefficients are not considered degenerate

        Returns
        -------
        list of numpy.ndarray
            a list of indices
        """
        deg = list()
        sidx = np.argsort(self.c)
        dc = _diff(self.c[sidx])

        # Degenerate indices
        idx = (dc < eps).nonzero()[0]
        if len(idx) == 0:
            # There are no degenerate coefficients
            return deg

        # These are the points were we split the degeneracies
        seps = (_diff(idx) > 1).nonzero()[0]
        IDX = np.array_split(idx, seps + 1)
        for idx in IDX:
            deg.append(_append(sidx[idx], sidx[idx[-1] + 1]))
        return deg

    def sub(self, idx, inplace=False):
        """ Return a new state with only the specified states

        Parameters
        ----------
        idx : int or array_like
            indices that are retained in the returned object
        inplace : bool, optional
            whether the values will be retained inplace

        Returns
        -------
        StateC
            a new object with a subset of the states, only if `inplace` is false
        """
        idx = self._sanitize_index(idx).ravel()
        if inplace:
            self.state = self.state[idx]
            self.c = self.c[idx]
        else:
            sub = self.__class__(self.state[idx, ...], self.c[idx], self.parent)
            sub.info = self.info
            return sub

    def remove(self, idx, inplace=False):
        """ Return a new state without the specified indices

        Parameters
        ----------
        idx : int or array_like
            indices that are removed in the returned object
        inplace : bool, optional
            whether the values will be removed inplace

        Returns
        -------
        StateC
            a new state without containing the requested elements, only if `inplace` is false
        """
        idx = np.delete(np.arange(len(self)), self._sanitize_index(idx))
        return self.sub(idx, inplace)

    def asState(self):
        s = State(self.state.copy(), self.parent)
        s.info = self.info
        return s

    def asCoefficient(self):
        c = Coefficient(self.c.copy(), self.parent)
        c.info = self.info
        return c
