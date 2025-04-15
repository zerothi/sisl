# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import operator as op
from abc import abstractmethod
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from numbers import Integral
from typing import Any, Optional, TypeVar, Union

from sisl._internal import set_module

__all__ = [
    "BaseMixer",
    "CompositeMixer",
    "BaseWeightMixer",
    "BaseHistoryWeightMixer",
    "StepMixer",
    "History",
]

T = TypeVar("T")
TypeBaseMixer = "BaseMixer"
TypeCompositeMixer = "CompositeMixer"
TypeStepMixer = "StepMixer"
TypeWeight = Union[float, int]
TypeHistory = "History"
TypeArgHistory = Union[int, TypeHistory]
# we don't use the Generator as we don't use the SendType/ReturnType
TypeStepCallable = Callable[[], Iterator[TypeBaseMixer]]
TypeMetric = Callable[[Any, Any], Any]


@set_module("sisl.mixing")
class BaseMixer:
    r"""Base class mixer"""

    __slots__ = ()

    @abstractmethod
    def __call__(self, f: T, df: T, *args: Any, **kwargs: Any) -> T:
        """Mix quantities based on arguments"""

    def __add__(self, other: Union[float, int, TypeBaseMixer]) -> TypeCompositeMixer:
        return CompositeMixer(op.add, self, other)

    def __radd__(self, other: Union[float, int, TypeBaseMixer]) -> TypeCompositeMixer:
        return CompositeMixer(op.add, other, self)

    def __sub__(self, other: Union[float, int, TypeBaseMixer]) -> TypeCompositeMixer:
        return CompositeMixer(op.sub, self, other)

    def __rsub__(self, other: Union[float, int, TypeBaseMixer]) -> TypeCompositeMixer:
        return CompositeMixer(op.sub, other, self)

    def __mul__(self, factor: Union[float, int, TypeBaseMixer]) -> TypeCompositeMixer:
        return CompositeMixer(op.mul, self, factor)

    def __rmul__(self, factor: Union[float, int, TypeBaseMixer]) -> TypeCompositeMixer:
        return CompositeMixer(op.mul, self, factor)

    def __truediv__(
        self, divisor: Union[float, int, TypeBaseMixer]
    ) -> TypeCompositeMixer:
        return CompositeMixer(op.truediv, self, divisor)

    def __rtruediv__(
        self, divisor: Union[float, int, TypeBaseMixer]
    ) -> TypeCompositeMixer:
        return CompositeMixer(op.truediv, divisor, self)

    def __neg__(self) -> TypeCompositeMixer:
        return CompositeMixer(op.mul, -1, self)

    def __pow__(self, other: Union[float, int, TypeBaseMixer]) -> TypeCompositeMixer:
        return CompositeMixer(op.pow, self, other)

    def __rpow__(self, other: Union[float, int, TypeBaseMixer]) -> TypeCompositeMixer:
        return CompositeMixer(op.pow, other, self)


@set_module("sisl.mixing")
class CompositeMixer(BaseMixer):
    """Placeholder for two metrics"""

    __slots__ = ("_op", "A", "B")

    def __init__(self, op: Callable[[Any, Any], Any], A: Any, B: Any):
        self._op = op
        self.A = A
        self.B = B

    def __call__(self, f: T, df: T, *args: Any, **kwargs: Any) -> T:
        if isinstance(self.A, BaseMixer):
            A = self.A(f, df, *args, **kwargs)
        else:
            A = self.A
        if isinstance(self.B, BaseMixer):
            B = self.B(f, df, *args, **kwargs)
        else:
            B = self.B
        return self._op(A, B)

    def __str__(self) -> str:
        if isinstance(self.A, BaseMixer):
            A = "({})".format(repr(self.A).replace("\n", "\n "))
        else:
            A = f"{self.A}"
        if isinstance(self.B, BaseMixer):
            B = "({})".format(repr(self.B).replace("\n", "\n "))
        else:
            B = f"{self.B}"
        return f"{self.__class__.__name__}{{{self._op.__name__}({A}, {B})}}"


@set_module("sisl.mixing")
class BaseWeightMixer(BaseMixer):
    r"""Base class mixer"""

    __slots__ = ("_weight",)

    def __init__(self, weight: TypeWeight = 0.2):
        self.set_weight(weight)

    @property
    def weight(self) -> TypeWeight:
        """This mixers mixing weight, the weight is the fractional contribution of the derivative"""
        return self._weight

    def set_weight(self, weight: TypeWeight):
        """Set a new weight for this mixer

        Parameters
        ----------
        weight :
           the new weight for this mixer, it must be bigger than 0
        """
        assert weight > 0, "Weight must be larger than 0"
        self._weight = weight


@set_module("sisl.mixing")
class BaseHistoryWeightMixer(BaseWeightMixer):
    r"""Base class mixer with history"""

    __slots__ = ("_history",)

    def __init__(self, weight: TypeWeight = 0.2, history: TypeArgHistory = 0):
        super().__init__(weight)
        self.set_history(history)

    def __str__(self) -> str:
        r"""String representation"""
        hist = str(self.history).replace("\n", "\n  ")
        return f"{self.__class__.__name__}{{weight: {self.weight:.4f},\n  {hist}\n}}"

    def __repr__(self) -> str:
        r"""String representation"""
        hist = len(self.history)
        max_hist = self.history.max_elements
        return f"{self.__class__.__name__}{{weight: {self.weight:.4f}, history={hist}|{max_hist}}}"

    def __call__(self, f: T, df: T, *args: Any, append: bool = True) -> None:
        """Append data to the history (omitting None values)!"""
        if not append:
            # do nothing
            return

        # remove none from the args
        args = list(filter(lambda arg: arg is not None, args))

        # append *args
        self.history.append(f, df, *args)

    @property
    def history(self) -> TypeHistory:
        """History object tracked by this mixer"""
        return self._history

    def set_history(self, history: TypeArgHistory) -> None:
        """Replace the current history in the mixer with a new one

        Parameters
        ----------
        history :
           if an int a new History object will be created with that number of history elements
           Otherwise the object will be directly attached to the mixer.
        """
        if isinstance(history, Integral):
            history = History(history)
        self._history = history


@set_module("sisl.mixing")
class StepMixer(BaseMixer):
    """Step between different mixers in a user-defined fashion

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

    def __init__(self, *yield_funcs: TypeStepCallable):
        self._yield_func = self.yield_chain(*yield_funcs)

        self._yield_mixer = self._yield_func()

        # We force a mixer to be in the queue.
        # This is necessary so that attributes may be accessed
        self._mixer = next(self._yield_mixer)

    def next(self) -> TypeBaseMixer:
        """Return the current mixer, and step the internal mixer"""
        mixer = self._mixer
        try:
            self._mixer = next(self._yield_mixer)
        except StopIteration:
            # reset the generator
            self._yield_mixer = self._yield_func()
            self._mixer = next(self._yield_mixer)
        return mixer

    @property
    def mixer(self) -> TypeBaseMixer:
        """Return the current mixer"""
        return self._mixer

    def __call__(self, f: T, df: T, *args: Any, **kwargs: Any) -> T:
        """Apply the mixing routine"""
        return self.next()(f, df, *args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        """Divert all unknown attributes to the current mixer

        Note that available attributes may be different for different
        mixers.
        """
        return getattr(self.mixer, attr)

    @classmethod
    def yield_repeat(
        cls: TypeStepMixer, mixer: TypeBaseMixer, n: int
    ) -> TypeStepCallable:
        """Returns a function which repeats `mixer` `n` times"""
        if n == 1:

            def yield_repeat() -> Iterator[TypeBaseMixer]:
                f"""Yield the mixer {mixer} 1 time"""
                yield mixer

        else:

            def yield_repeat() -> Iterator[TypeBaseMixer]:
                f"""Yield the mixer {mixer} {n} times"""
                for _ in range(n):
                    yield mixer

        return yield_repeat

    @classmethod
    def yield_chain(
        cls: TypeStepMixer, *yield_funcs: TypeStepCallable
    ) -> TypeStepCallable:
        """Returns a function which yields from each of the function arguments in turn

        Basically equivalent to a function which does this:

        >>> for yield_func in yield_funcs:
        ...     yield from yield_func()

        Parameters
        ----------
        *yield_funcs :
             every function will be ``yield from``
        """
        if len(yield_funcs) == 1:
            return yield_funcs[0]

        def yield_chain() -> Iterator[TypeBaseMixer]:
            f"""Yield from the different yield generators"""
            for yield_func in yield_funcs:
                yield from yield_func()

        return yield_chain


@set_module("sisl.mixing")
class History:
    r"""A history class for retaining a set of history elements

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

    def __init__(self, history: int = 2):
        # Create a list of queues
        self._hist = deque(maxlen=history)

    def __str__(self) -> str:
        """str of the object"""
        return (
            f"{self.__class__.__name__}{{history: {self.elements}/{self.max_elements}}}"
        )

    @property
    def max_elements(self) -> int:
        r"""Maximum number of elements stored in the history for each variable"""
        return self._hist.maxlen

    @property
    def elements(self) -> int:
        r"""Number of elements in the history"""
        return len(self._hist)

    def __len__(self) -> int:
        return self.elements

    def __getitem__(self, key: int) -> Any:
        return self._hist[key]

    def __setitem__(self, key: int, value: Any) -> None:
        self._hist[key] = value

    def __delitem__(self, key: Union[int, Sequence[int]]) -> None:
        self.clear(key)

    def append(self, *variables: Any) -> None:
        r"""Add variables to the history

        Internally, the list of variables will be added to the queue, it is up
        to the implementation to use the appended values.

        Parameters
        ----------
        *variables :
            each variable will be added to the history of the mixer
        """
        self._hist.append(variables)

    def clear(self, index: Optional[Union[int, Sequence[int]]] = None) -> None:
        r"""Clear variables to the history

        Parameters
        ----------
        index :
            which indices of the history we should clear
        """
        if index is None:
            self._hist.clear()
            return

        if isinstance(index, Integral):
            index = [index]

        # Reverse sort so we can delete without breaking the
        # order of the elements
        index = sorted(index, reverse=True)

        for i in index:
            del self._hist[i]
