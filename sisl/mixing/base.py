# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections import deque
from numbers import Integral
import operator as op
from abc import abstractmethod

from sisl._internal import set_module


__all__ = [
    "BaseMixer",
    "CompositeMixer",
    "BaseWeightMixer",
    "BaseHistoryWeightMixer",
    "StepMixer",
    "History",
]


@set_module("sisl.mixing")
class BaseMixer:
    r""" Base class mixer """
    __slots__ = ()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Mix quantities based on arguments """

    def __add__(self, other):
        return CompositeMixer(op.add, self, other)

    def __radd__(self, other):
        return CompositeMixer(op.add, other, self)

    def __sub__(self, other):
        return CompositeMixer(op.sub, self, other)

    def __rsub__(self, other):
        return CompositeMixer(op.sub, other, self)

    def __mul__(self, factor):
        return CompositeMixer(op.mul, self, factor)

    def __rmul__(self, factor):
        return CompositeMixer(op.mul, self, factor)

    def __truediv__(self, divisor):
        return CompositeMixer(op.truediv, self, divisor)

    def __rtruediv__(self, divisor):
        return CompositeMixer(op.truediv, divisor, self)

    def __neg__(self):
        return CompositeMixer(op.mul, -1, self)

    def __pow__(self, other):
        return CompositeMixer(op.pow, self, other)

    def __rpow__(self, other):
        return CompositeMixer(op.pow, other, self)


@set_module("sisl.mixing")
class CompositeMixer(BaseMixer):
    """ Placeholder for two metrics """

    __slots__ = ("_op", "A", "B")

    def __init__(self, op, A, B):
        self._op = op
        self.A = A
        self.B = B

    def __call__(self, *args, **kwargs):
        if isinstance(self.A, BaseMixer):
            A = self.A(*args, **kwargs)
        else:
            A = self.A
        if isinstance(self.B, BaseMixer):
            B = self.B(*args, **kwargs)
        else:
            B = self.B
        return self._op(A, B)

    def __str__(self):
        if isinstance(self.A, BaseMixer):
            A = "({})".format(repr(self.A).replace('\n', '\n '))
        else:
            A = f"{self.A}"
        if isinstance(self.B, BaseMixer):
            B = "({})".format(repr(self.B).replace('\n', '\n '))
        else:
            B = f"{self.B}"
        return f"{self.__class__.__name__}{{{self._op.__name__}({A}, {B})}}"


@set_module("sisl.mixing")
class BaseWeightMixer(BaseMixer):
    r""" Base class mixer """
    __slots__ = ("_weight",)

    def __init__(self, weight=0.2):
        self.set_weight(weight)

    @property
    def weight(self):
        """ This mixers mixing weight, the weight is the fractional contribution of the derivative """
        return self._weight

    def set_weight(self, weight):
        """ Set a new weight for this mixer

        Parameters
        ----------
        weight : float
           the new weight for this mixer, it must be bigger than 0
        """
        assert weight > 0, "Weight must be larger than 0"
        self._weight = weight


@set_module("sisl.mixing")
class BaseHistoryWeightMixer(BaseWeightMixer):
    r""" Base class mixer with history """
    __slots__ = ("_history",)

    def __init__(self, weight=0.2, history=0):
        super().__init__(weight)
        self.set_history(history)

    def __call__(self, *args, append=True):
        """ Append data to thMix quantities based on arguments """
        if not append:
            # do nothing
            return

        # remove none from the args
        args = list(filter(lambda arg: arg is not None, args))

        # append *args
        self.history.append(*args)

    @property
    def history(self):
        """ History object tracked by this mixer """
        return self._history

    def set_history(self, history):
        """ Replace the current history in the mixer with a new one

        Parameters
        ----------
        history : int or History
           if an int a new History object will be created with that number of history elements
           Otherwise the object will be directly attached to the mixer.
        """
        if isinstance(history, Integral):
            history = History(history)
        self._history = history


@set_module("sisl.mixing")
class StepMixer(BaseMixer):
    """ Step between different mixers in a user-defined fashion

    This is handy for creating variable mixing schemes that alternates (or differently)
    between multiple mixers.


    Examples
    --------

    Alternate between two mixers:

    >>> mixer = StepMixer(
    ...        StepMixer.yield_repeat(mix1, 1),
    ...        StepMixer.yield_repeat(mix2, 1))

    One may also create custom based generators
    which can interact with the mixers in between
    different mixers:

    >>> def gen():
    ...     yield mix1
    ...     mix1.history.clear()
    ...     yield mix1
    ...     yield mix1
    ...     yield mix2

    A restart mixer for history mixers:

    >>> def gen():
    ...     for _ in range(50):
    ...         yield mix
    ...     mix.history.clear()
    """

    __slots__ = ("_yield_func", "_yield_mixer", "_mixer")

    def __init__(self, *yield_funcs):
        self._yield_func = self.yield_chain(*yield_funcs)

        self._yield_mixer = self._yield_func()

        # We force a mixer to be in the queue.
        # This is necessary so that attributes may be accessed
        self._mixer = next(self._yield_mixer)

    def next(self):
        """ Return the current mixer, and step the internal mixer """
        mixer = self._mixer
        try:
            self._mixer = next(self._yield_mixer)
        except StopIteration:
            self._yield_mixer = self._yield_func()
            self._mixer = next(self._yield_mixer)
        return mixer

    @property
    def mixer(self):
        """ Return the current mixer """
        return self._mixer

    def __call__(self, *args, **kwargs):
        """ Apply the mixing routine """
        return self.next()(*args, **kwargs)

    def __getattr__(self, attr):
        """ Divert all unknown attributes to the current mixer

        Note that available attributes may be different for different
        mixers.
        """
        return getattr(self.mixer, attr)

    @classmethod
    def yield_repeat(cls, mixer, n):
        """ Returns a function which repeats `mixer` `n` times """
        if n == 1:
            def yield_repeat():
                f""" Yield the mixer {mixer} 1 time """
                yield mixer
        else:
            def yield_repeat():
                f""" Yield the mixer {mixer} {n} times """
                for _ in range(n):
                    yield mixer
        return yield_repeat

    @classmethod
    def yield_chain(cls, *yield_funcs):
        """ Returns a function which yields from each of the function arguments in turn

        Basically equivalent to a function which does this:

        >>> for yield_func in yield_funcs:
        ...     yield from yield_func()

        Parameters
        *yield_funcs : list of callable
           every function will be ``yield from``
        """
        if len(yield_funcs) == 1:
            return yield_funcs[0]
        def yield_chain():
            f""" Yield from the different yield generators """
            for yield_func in yield_funcs:
                yield from yield_func()
        return yield_chain


@set_module("sisl.mixing")
class History:
    r""" A history class for retaining a set of history elements

    A history class may contain several different variables in a `collections.deque`
    list allowing easy managing of the length of the history.

    Attributes
    ----------
    history_max : int or tuple of int
       maximum number of history elements

    Parameters
    ----------
    history : int, optional
       number of maximum history elements stored
    """

    def __init__(self, history=2):
        # Create a list of queues
        self._hist = deque(maxlen=history)

    def __str__(self):
        """ str of the object """
        return f"{self.__class__.__name__}{{history: {self.elements}/{self.max_elements}}}"

    @property
    def max_elements(self):
        r""" Maximum number of elements stored in the history for each variable """
        return self._hist.maxlen

    @property
    def elements(self):
        r""" Number of elements in the history """
        return len(self._hist)

    def __len__(self):
        return self.elements

    def __getitem__(self, key):
        return self._hist[key]

    def __setitem__(self, key, value):
        self._hist[key] = value

    def __delitem__(self, key):
        self.clear(key)

    def append(self, *args):
        r""" Add variables to the history

        Parameters
        ----------
        *args : tuple of object
            each variable will be added to the history of the mixer
        """
        self._hist.append(args)

    def clear(self, index=None):
        r""" Clear variables to the history

        Parameters
        ----------
        index : int or array_like of int
            which indices of the history we should clear
        """
        if index is None:
            self._hist.clear()
            return

        if isinstance(index, Integral):
            index = [index]
        else:
            index = list(index)
        # We need to ensure we delete in the correct order
        index.sort(reverse=True)
        for i in index:
            del self._hist[i]
