from __future__ import print_function, division

import numpy as np

import sisl._array as _a
from sisl._help import dtype_complex_to_real
from sisl._help import _range as range


__all__ = ['Coefficient', 'State', 'StateC']

_append = np.append
_diff = np.diff
_dot = np.dot
_conj = np.conjugate
_outer_ = np.outer


def _outer(v):
    return _outer_(v, _conj(v))


def _idot(v):
    return _dot(_conj(v), v)


def _couter(c, v):
    return _outer_(v * c, _conj(v))


class ParentContainer(object):
    """ A container for parent and information """
    __slots__ = ['parent', 'info']

    def __init__(self, parent, **info):
        self.parent = parent
        self.info = info


class Coefficient(ParentContainer):
    """ An object holding coefficients for a parent with info

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
        super(Coefficient, self).__init__(parent, **info)
        self.c = np.atleast_1d(c)

    def __str__(self):
        """ The string representation of this object """
        s = self.__class__.__name__ + '{{coefficients: {0}'.format(len(self))
        if self.parent is None:
            s += '}}'
        else:
            s += '\n {}}}'.format(str(self.parent).replace('\n', '\n '))
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

    def degenerate(self, eps):
        """ Find degenerate coefficients with a specified precision

        Parameters
        ----------
        eps : float
           the precision above which coefficients are not considered degenerate

        Returns
        -------
        list of numpy.ndarray: a list of indices
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

    def sub(self, idx):
        """ Return a new coefficient with only the specified coefficients

        Parameters
        ----------
        idx : int or array_like
            indices that are retained in the returned object

        Returns
        -------
        Coefficient
            a new coefficient only containing the requested elements
        """
        idx = _a.asarrayi(idx).ravel() # this ensures that the first dimension is preserved
        sub = self.__class__(self.c[idx].copy(), self.parent)
        sub.info = self.info
        return sub

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
        asarray: bool, optional
           if true the yielded values are the coefficient vectors, i.e. a numpy array.
           Otherwise an equivalent object is yielded.

        Yields
        ------
        coeff: Coefficent
           the current coefficent as an object, only returned if `asarray` is false.
        coeff: numpy.ndarray
           the current the coefficient as an array, only returned if `asarray` is true.
        """
        if asarray:
            for i in range(len(self)):
                yield self.c[i]
        else:
            for i in range(len(self)):
                yield self.sub(i)

    __iter__ = iter


class State(ParentContainer):
    """ An object handling a set of vectors describing a given *state*

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
        super(State, self).__init__(parent, **info)
        self.state = np.atleast_2d(state)

    def __str__(self):
        """ The string representation of this object """
        s = self.__class__.__name__ + '{{states: {0}'.format(len(self))
        if self.parent is None:
            s += '}}'
        else:
            s += '\n {}}}'.format(str(self.parent).replace('\n', '\n '))
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

    def sub(self, idx):
        """ Return a new state with only the specified states

        Parameters
        ----------
        idx : int or array_like
            indices that are retained in the returned object

        Returns
        -------
        State
           a new state only containing the requested elements
        """
        idx = _a.asarrayi(idx).ravel() # this ensures that the first dimension is preserved
        sub = self.__class__(self.state[idx].copy(), self.parent)
        sub.info = self.info
        return sub

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
        asarray: bool, optional
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
        r""" Return a vector with the norm of each state :math:`\sqrt{\langle\psi|\psi\rangle}`

        Returns
        -------
        numpy.ndarray
            the normalization for each state
        """
        return np.sqrt(self.norm2())

    def norm2(self, sum=True):
        r""" Return a vector with the norm of each state :math:`\langle\psi|\psi\rangle`

        Parameters
        ----------
        sum : bool, optional
           if true the summed site square is returned (a vector). For false a matrix
           with normalization squared per site is returned.

        Returns
        -------
        numpy.ndarray
            the normalization for each state
        """
        if not sum:
            return (conj(self.state) * self.state.T).real

        dtype = dtype_complex_to_real(self.dtype)

        N = len(self)
        n = np.empty(N, dtype=dtype)

        for i in range(N):
            n[i] = _idot(self.state[i, :]).real

        return n

    def normalize(self):
        r""" Return a normalized state where each state has :math:`|\psi|^2=1`

        This is roughly equivalent to:

        >>> state = State(np.arange(10))
        >>> n = state.norm()
        >>> norm_state = State(state.state / n.reshape(-1, 1))

        Returns
        -------
        State
            a new state with all states normalized, otherwise equal to this
        """
        n = self.norm()
        s = self.__class__(self.state / n.reshape(-1, 1), parent=self.parent)
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
        numpy.ndarray
            a matrix with the sum of outer state products
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
    #         c[i] = (_idot(self.state[i].ravel()).astype(dtype) / norm[i]) ** 0.5
    #         state[i, ...] = self.state[i, ...] / c[i]

    #     cs = StateC(state, c, parent=self.parent)
    #     cs.info = self.info
    #     return cs


# Although the StateC could inherit from both Coefficient and State
# there are problems with __slots__ and multiple inheritance schemes.
# I.e. we are forced to do *one* inheritance, which we choose to be State.
class StateC(State):
    """ An object handling a set of vectors describing a given *state* with associated coefficients `c`

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
        super(StateC, self).__init__(state, parent, **info)
        self.c = np.atleast_1d(c)
        if len(self.c) != len(self.state):
            raise ValueError(self.__class__.__name__ + ' could not be created with coefficients and states '
                             'having unequal length.')

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
        state : a new state with all states normalized, otherwise equal to this
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
        numpy.ndarray : a matrix with the sum of outer state products
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

    def degenerate(self, eps):
        """ Find degenerate coefficients with a specified precision

        Parameters
        ----------
        eps : float
           the precision above which coefficients are not considered degenerate

        Returns
        -------
        list of numpy.ndarray: a list of indices
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

    def sub(self, idx):
        """ Return a new state with only the specified states

        Parameters
        ----------
        idx : int or array_like
            indices that are retained in the returned object

        Returns
        -------
        StateC : a new object with a subset of the states
        """
        idx = _a.asarrayi(idx).ravel()
        sub = self.__class__(self.state[idx, ...], self.c[idx], self.parent)
        sub.info = self.info
        return sub

    def asState(self):
        s = State(self.state.copy(), self.parent)
        s.info = self.info
        return s

    def asCoefficient(self):
        c = Coefficient(self.c.copy(), self.parent)
        c.info = self.info
        return c
