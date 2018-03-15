from __future__ import print_function, division

import numpy as np

import sisl._array as _a
from sisl._help import dtype_complex_to_real
from sisl._help import _zip as zip, _range as range


__all__ = ['State', 'CoeffState']

_dot = np.dot
_conj = np.conjugate
_outer_ = np.outer


def _outer(v):
    return _outer_(v, _conj(v))


def _idot(v):
    return _dot(_conj(v), v)


def _couter(c, v):
    return _outer_(v * c, _conj(v))


class State(object):
    """ An object handling a set of vectors describing a given *state*

    Notes
    -----
    This class should be subclassed!

    See Also
    --------
    StateAtom : a state defined on each atomic site
    StateOrbital : a state defined on each orbital
    """
    __slots__ = ['state', 'parent', 'info']

    def __init__(self, state, parent=None, **info):
        """ Define a state container with a given set of states

        Parameters
        ----------
        state : array_like
           state vectors ``state[i, :]`` containing the i'th state vector
        parent : obj, optional
           a parent object that defines the origin of the state.
        **info : dict, optional
           an info dictionary that turns into an attribute on the object.
           This `info` may contain anything that may be relevant for the state.
        """
        self.state = state
        self.parent = parent
        self.info = info
        self.__state_finalize__()

    def __state_finalize__(self):
        """ Finalizer for the state object. """
        if not self.state is None:
            self.state = _a.asarray(self.state)
            # Correct for vector
            if self.state.ndim == 1:
                self.state.shape = (1, -1)

    def __repr__(self):
        """ The string representation of this object """
        s = self.__class__.__name__ + '{{dim: {0}'.format(len(self))
        if self.parent is None:
            s += '}}'
        else:
            s += '\n {}}}'.format(repr(self.parent).replace('\n', '\n '))
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

    def __getitem__(self, key):
        """ Return a new state with only one associated state

        Parameters
        ----------
        key : int or array_like
           the indices for the returned states

        Returns
        -------
        state : State
            a new state *only* with the indexed values
        """
        return self.sub(key)

    def iter(self):
        """ Return an iterator looping over the states in this system

        Yields
        ------
        state : State
           a state *only* containing individual elements
        """
        for i in range(len(self)):
            yield self.sub(i)

    __iter__ = iter

    def copy(self):
        """ Return a copy (only the state is copied). ``parent`` and ``info`` are passed by reference """
        copy = self.__class__(self.state.copy(), self.parent)
        copy.info = self.info
        return copy

    def norm(self):
        r""" Return a vector with the norm of each state :math:`\sqrt{\langle\psi|\psi\rangle}`

        Returns
        -------
        np.ndarray : the normalization for each state
        """
        return np.sqrt(self.norm2())

    def norm2(self):
        r""" Return a vector with the norm of each state :math:`\langle\psi|\psi\rangle`

        Returns
        -------
        np.ndarray : the normalization for each state
        """
        dtype = dtype_complex_to_real(self.dtype)
        shape = self.shape[:-1]
        n = np.empty(shape, dtype=dtype)

        # A state should only be up to
        if len(shape) == 1:
            for i in range(shape[0]):
                n[i] = _idot(self.state[i, :]).astype(n.dtype)
        elif len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    n[i, j] = _idot(self.state[i, j, :]).astype(n.dtype)
        else:
            raise SislError(self.__class__.__name__ + '.norm2 does not implement more than 2 dimensions on states.')
        return n

    def normalize(self):
        r""" Return a normalized state where each state has :math:`|\psi|^2=1`

        This is roughly equivalent to:

        >>> state = State(np.arange(10))
        >>> n = state.norm()
        >>> norm_state = State(state.state / n.reshape(-1, 1))

        Returns
        -------
        state : a new state with all states normalized, otherwise equal to this
        """
        n = self.norm()
        s = self.__class__(self.state / n.reshape(n.shape + (1,)), parent=self.parent)
        s.info = self.info
        return s

    def outer(self, idx=None):
        r""" Return the outer product for the indices `idx` (or all if ``None``) by :math:`\sum_i|\psi_i\rangle\langle\psi_i|`

        Parameters
        ----------
        idx : int or array_like, optional
           only perform an outer product of the specified indices, otherwise all states are used

        Returns
        -------
        np.ndarray : a matrix with the sum of outer state products
        """
        if idx is None:
            m = _outer(self.state[0, :])
            for i in range(1, len(self)):
                m += _outer(self.state[i, :])
            return m
        idx = _a.asarrayi(idx).ravel()
        m = _outer(self.state[idx[0], :])
        for i in idx[1:]:
            m += _outer(self.state[i, :])
        return m

    def sub(self, idx):
        """ Return a new state with only the specified states

        Parameters
        ----------
        idx : int or array_like
            indices that are retained in the returned object

        Returns
        -------
        state : a new state only containing the requested elements
        """
        idx = _a.asarrayi(idx).ravel() # this ensures that the first dimension is preserved
        sub = self.__class__(self.state[idx].copy(), self.parent)
        sub.info = self.info
        return sub

    def toCoeffState(self, norm=1.):
        r""" Transforms the states into normalized values equal to `norm` and specifies the coefficients in `CoeffState` as the norm

        This is an easy method to renormalize the state vectors to a common (or state dependent) normalization constant.

        .. math::
            c_i &= \sqrt{\langle \psi_i | \psi_i\rangle} / \mathrm{norm} \\
              |\psi_i\rangle &= | \psi_i\rangle / c_i

        Parameters
        ----------
        norm : value, array_like
            the resulting norm of all (or individual states)

        Returns
        -------
        CoeffState : a new coefficient state object with associated coefficients
        """
        n = len(self)
        norm = _a.asarray(norm).ravel()
        if norm.size == 1 and n > 1:
            norm = np.tile(norm, n)
        elif norm.size != n:
            raise ValueError(self.__class__.__name__ + '.toCoeffState requires the input norm to be a single float or having equal length to the state!')

        # Correct data-type
        if norm.dtype in [np.complex64, np.complex128]:
            cdtype = norm.dtype
        else:
            cdtype = dtype_complex_to_real(self.dtype)

        # TODO check datatype if norm is complex but state is real
        c = np.empty(n, dtype=cdtype)
        state = np.empty(self.shape, dtype=self.dtype)

        for i in range(n):
            c[i] = (_idot(self.state[i].ravel()).astype(c.dtype) / norm[i]) ** 0.5
            state[i, ...] = self.state[i, ...] / c[i]

        cs = CoeffState(c, state, parent=self.parent)
        cs.info = self.info
        return cs


class CoeffState(State):
    """ An object handling a set of vectors describing a given *state* with associated coefficients `c`

    Notes
    -----
    This class should be subclassed!

    See Also
    --------
    State
    StateAtom : a state defined on each atomic site
    StateOrbital : a state defined on each orbital
    """
    __slots__ = ['c']

    def __init__(self, c, state, parent=None, **info):
        """ Define a state container with a given set of states and coefficients for the states

        Parameters
        ----------
        c : array_like
           coefficients for the states ``c[i]`` containing the i'th coefficient
        state : array_like
           state vectors ``state[i, :]`` containing the i'th state vector
        parent : obj, optional
           a parent object that defines the origin of the state.
        **info : dict, optional
           an info dictionary that turns into an attribute on the object.
           This `info` may contain anything that may be relevant for the state.
        """
        self.c = np.asarray(c).ravel()
        super(CoeffState, self).__init__(state, parent)
        self.info = info

    def __len__(self):
        """ Length of `CoeffState` """
        return len(self.c)

    def copy(self):
        """ Return a copy (only the coefficients and states are copied). Parent and info are passed by reference """
        if self.state is None:
            copy = self.__class__(self.c.copy(), None, self.parent)
        else:
            copy = self.__class__(self.c.copy(), self.state.copy(), self.parent)
        copy.info = self.info
        return copy

    def normalize(self):
        r""" Return a normalized state where each state has :math:`|\psi|^2=1`

        This is roughly equivalent to:

        >>> state = CoeffState(1, np.arange(10))
        >>> n = state.norm()
        >>> norm_state = CoeffState(state.c.copy(), state.state / n.reshape(-1, 1))

        Returns
        -------
        state : a new state with all states normalized, otherwise equal to this
        """
        n = self.norm()
        s = self.__class__(self.c.copy(), self.state / n.reshape(n.shape + (1,)), parent=self.parent)
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
        np.ndarray : a matrix with the sum of outer state products
        """
        if idx is None:
            m = _couter(self.c[0], self.state[0].ravel())
            for i in range(1, len(self)):
                m += _couter(self.c[i], self.state[i].ravel())
            return m
        idx = _a.asarrayi(idx).ravel()
        m = _couter(self.c[idx[0]], self.state[idx[0]].ravel())
        for i in idx[1:]:
            m += _couter(self.c[i], self.state[i].ravel())
        return m

    def sort(self, ascending=True):
        """ Sort and return a new `CoeffState` by sorting the coefficients (default to ascending)

        Parameters
        ----------
        ascending : bool, optional
            sort the contained elements ascending, else they will be sorced descending
        """
        if ascending:
            idx = np.argsort(self.c)
        else:
            idx = np.argsort(-self.c)
        return self.sub(idx)

    def sub(self, idx):
        """ Return a new state with only the specified states

        Parameters
        ----------
        idx : int or array_like
            indices that are retained in the returned object

        Returns
        -------
        CoeffState : a new object with a subset of the states
        """
        idx = _a.asarrayi(idx).ravel()
        sub = self.__class__(self.c[idx], self.state[idx, ...], self.parent)
        sub.info = self.info
        return sub

    toCoeffState = None
