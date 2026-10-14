# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from collections import ChainMap, defaultdict
from collections.abc import Iterable, Sequence
from difflib import get_close_matches
from functools import lru_cache, singledispatchmethod, wraps
from typing import Optional

import numpy as np

from ._internal import set_module

__all__ = ["Category", "CompositeCategory", "NullCategory"]
__all__ += ["AndCategory", "OrCategory", "XOrCategory"]
__all__ += ["InstanceCache"]

_log = logging.getLogger(__name__)


class InstanceCache:
    """Wraps an instance to cache *all* results based on `functools.lru_cache`

    Parameters
    ----------
    obj : object
        the object to get cached results
    lru_size : int or dict, optional
        initial size of the lru cache. For integers this
        is the default size of the cache, for a dictionary
        it should return the ``maxsize`` argument for `functools.lru_cache`
    no_cache : searchable (list or dict)
        a list-like (or dictionary) for searching for
        methods that don't require caches (e.g. small methods)
    """

    def __init__(self, obj, lru_size=1, no_cache=None):
        self.__obj = obj

        # Handle user input for lru_size
        if isinstance(lru_size, defaultdict):
            # fine, user did everything good
            self.__lru_size = lru_size
        elif isinstance(lru_size, dict):
            default = lru_size.pop("default", 1)
            self.__lru_size = ChainMap(lru_size, defaultdict(lambda: default))
        else:
            self.__lru_size = defaultdict(lambda: lru_size)

        if no_cache is None:
            self.__no_cache = []
        else:
            self.__no_cache = no_cache

    def __getattr__(self, name):
        attr = getattr(self.__obj, name)
        # Check if the attribute has the cached functionality
        try:
            attr.cache_info()
        except AttributeError:
            # Fix it and set it to this one
            if name in self.__no_cache:
                # We have to make it cacheable
                maxsize = self.__lru_size[name]
                if maxsize != 0:
                    attr = wraps(attr)(lru_cache(maxsize)(attr))

        # offload the attribute to this class (to minimize overhead)
        object.__setattr__(self, name, attr)
        return attr


class CategoryMeta(ABCMeta):
    """
    Metaclass that defines how category classes should behave.
    """

    def __call__(cls, *args, **kwargs):
        """Instantiate the category, or build it from keywords

        A family class (e.g. `AtomCategory`) called with only keyword
        arguments returns ``cls.kw(**kwargs)``.

        Examples
        -----------

        >>> AtomZ(6) # returns an AtomZ category with the Z parameter set to 6
        >>> AtomCategory(Z=6) # returns exactly the same since it uses `AtomCategory.kw`
        """
        if not args and not cls._keys:
            return cls.kw(**kwargs)
        return super().__call__(*args, **kwargs)


@set_module("sisl.category")
class Category(metaclass=CategoryMeta):
    r"""A category

    Sub-classes register the keywords that `kw` matches through
    class arguments.

    Parameters
    ----------
    key : str, optional
       class argument, the keyword that selects this category in `kw`.
       Defaults to the class name without the family prefix
       (``AtomNeighbors`` gets ``Neighbors``).
    aliases : tuple of str, optional
       class argument, other keywords that select this category.
    family : str, optional
       class argument, declares a family class (e.g. `AtomCategory`).
       The value is the prefix removed from the names of its sub-classes.
       A family class has no keywords, and calling it with keywords
       builds categories through `kw`.

    Examples
    --------
    >>> class AtomCoordination(AtomCategory, key="coordination", aliases=("cn",)):
    ...     ...
    >>> AtomCategory(cn=3) # an AtomCoordination category
    """

    __slots__ = ("_name", "_wrapper")
    _key_prefix: str = ""
    _keys: tuple[str, ...] = ()

    def __init_subclass__(
        cls,
        /,
        key: Optional[str] = None,
        aliases: Sequence[str] = (),
        family: Optional[str] = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        if family is not None:
            cls._key_prefix = family
            cls._keys = ()
            return
        if key is None:
            key = cls.__name__.removeprefix(cls._key_prefix)
        cls._keys = (key, *aliases)

    def __init__(self, name=None):
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

    @property
    def name(self):
        r"""Name of category"""
        return self._name

    @name.setter
    def name(self, name):
        r"""Override the name of the categorization"""
        self._name = name

    @classmethod
    def is_class(cls, name: str, case: bool = True) -> bool:
        r"""Query whether `name` is one of the keywords of this class

        The keywords are the ``key`` and ``aliases`` class arguments.
        Sub-classes may override this method for other matching rules.

        Parameters
        ----------
        name :
           the keyword to match
        case :
           whether the match is case-sensitive
        """
        if case:
            return name in cls._keys
        name = name.lower()
        return any(name == key.lower() for key in cls._keys)

    @classmethod
    def kw(cls, **kwargs):
        """Create categories based on keywords

        This will search through the inherited classes and
        return and & category object for all keywords.

        Since this is a class method one should use this
        on the base category class in the given section
        of the code.

        Each keyword matches a sub-class through `is_class`, first
        case-sensitive, then case-insensitive.
        The value is passed to the matched class:

        - ``True`` creates it without arguments (``odd=True``),
        - a `dict` is passed as keyword arguments (``odd={}``),
        - anything else is passed as the single argument (``Z="C"``).

        Raises
        ------
        TypeError
           if no keywords are given
        ValueError
           if a keyword matches no, or more than one, sub-class
        """
        if not kwargs:
            raise TypeError(
                f"{cls.__name__} requires at least one keyword argument, "
                f"e.g. {cls.__name__}(Z=6)."
            )

        subcls = set()
        work = [cls]
        while work:
            parent = work.pop()
            for child in parent.__subclasses__():
                if child not in subcls:
                    subcls.add(child)
                    work.append(child)

        del work

        def get_cat(cl, args):
            if args is True:
                return cl()
            if isinstance(args, dict):
                return cl(**args)
            return cl(args)

        # Now search keywords and create category
        cat = None
        for key, args in kwargs.items():
            found = None
            # First search case-sensitive
            for cl in subcls:
                if cl.is_class(key):
                    if found:
                        raise ValueError(
                            f"{cls.__name__}.kw got a non-unique argument for category name:\n"
                            f"    Searching for {key} and found matches {found.__name__} and {cl.__name__}."
                        )
                    found = cl

            if found is None:
                for cl in subcls:
                    if cl.is_class(key, case=False):
                        if found:
                            raise ValueError(
                                f"{cls.__name__}.kw got a non-unique argument for category name:\n"
                                f"    Searching for {key} and found matches {found.__name__.lower()} and {cl.__name__.lower()}."
                            )
                        found = cl

            if found is None:
                keys = sorted({k for cl in subcls for k in cl._keys})
                close = get_close_matches(key, keys, n=3)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise ValueError(
                    f"{cls.__name__}.kw got an unknown keyword {key!r}.{hint}\n"
                    f"    Valid keywords are: {', '.join(keys)}."
                )

            if cat is None:
                cat = get_cat(found, args)
            else:
                cat = cat & get_cat(found, args)

        return cat

    @abstractmethod
    def categorize(self, *args, **kwargs):
        r"""Do categorization"""
        pass

    def mask(self, *args, **kwargs) -> np.ndarray:
        r"""Boolean array which is true for the objects in this category

        The arguments are the same as for `categorize`.
        The parts of a composite category share their calculations,
        e.g. the neighbor search of `AtomNeighbors`.

        Returns
        -------
        numpy.ndarray
           1D boolean array, one value per categorized object.
        """
        return self._mask(*args, cache={}, **kwargs)

    def _mask(self, *args, cache: dict, **kwargs) -> np.ndarray:
        cat = self.categorize(*args, **kwargs)
        if not isinstance(cat, Iterable):
            cat = [cat]
        return np.fromiter((not isinstance(c, NullCategory) for c in cat), dtype=bool)

    def __str__(self):
        r"""String representation of the class (non-distinguishable between equivalent classifiers)"""
        return self.name

    def __repr__(self):
        r"""String representation of the class (non-distinguishable between equivalent classifiers)"""
        return self.name

    @singledispatchmethod
    def __eq__(self, other):
        """Comparison of two categories, they are compared by class-type"""
        # This is not totally safe since composites *could* be generated
        # in different sequences and result in the same boolean expression.
        # This we do not check and thus are not fool proof...
        # The exact action also depends on whether we are dealing with
        # an And/Or/XOr operation....
        # I.e. something like
        # (A & B & C) != (A & C & B)
        # (A ^ B ^ C) != (C ^ A ^ B)
        if isinstance(self, CompositeCategory):
            if isinstance(other, CompositeCategory):
                return self.__class__ is other.__class__ and (
                    self.A == other.A
                    and self.B == other.B
                    or self.A == other.B
                    and self.B == other.A
                )
            # if neither is a compositecategory, then they cannot
            # be the same category
            return False
        if self.__class__ != other.__class__:
            return False
        return self == other

    @__eq__.register(Iterable)
    def _(self, other):
        return [self.__eq__(o) for o in other]

    def __ne__(self, other):
        eq = self == other
        if isinstance(eq, Iterable):
            return [not e for e in eq]
        return not eq

    # Implement logical operators to enable composition of sets
    def __and__(self, other):
        return AndCategory(self, other)

    def __or__(self, other):
        return OrCategory(self, other)

    def __xor__(self, other):
        return XOrCategory(self, other)

    def __invert__(self):
        if isinstance(self, NotCategory):
            return self._cat
        return NotCategory(self)


@set_module("sisl.category")
class GenericCategory(Category):
    """Used to indicate that the category does not act on specific objects

    It serves to identify categories such as `NullCategory`, `NotCategory`
    and `CompositeCategory` and distinguish them from categories that have
    a specific object in which they act.
    """

    __slots__ = ()

    @classmethod
    def is_class(cls, name, case=True):
        # never allow one to match a generic class
        # I.e. you can't instantiate a Null/Not/And/Or/XOr category by name
        return False


@set_module("sisl.category")
class NullCategory(GenericCategory):
    r"""Special Null class which always represents a classification not being *anything*"""

    __slots__ = ()

    def categorize(self, *args, **kwargs):
        return self

    @singledispatchmethod
    def __eq__(self, other):
        if other is None:
            return True
        return self.__class__ == other.__class__

    @__eq__.register(Iterable)
    def _(self, other):
        return super().__eq__(other)

    @property
    def name(self):
        return "∅"

    @name.setter
    def name(self, name):
        raise ValueError(
            f"One cannot overwrite the name of a {self.__class__.__name__}"
        )


@set_module("sisl.category")
class NotCategory(GenericCategory):
    """A class returning the *opposite* of this class (NullCategory) if it is categorized as such"""

    __slots__ = ("_cat",)

    def __init__(self, cat):
        super().__init__()
        if isinstance(cat, CompositeCategory):
            self.name = f"~({cat})"
        else:
            self.name = f"~{cat}"
        self._cat = cat

    def _mask(self, *args, cache: dict, **kwargs) -> np.ndarray:
        return ~self._cat._mask(*args, cache=cache, **kwargs)

    def categorize(self, *args, **kwargs):
        r"""Base method for queriyng whether an object is a certain category"""
        cat = self._cat.categorize(*args, **kwargs)

        _null = NullCategory()

        def check(cat):
            if isinstance(cat, NullCategory):
                return self
            return _null

        if isinstance(cat, Iterable):
            return [check(c) for c in cat]
        return check(cat)

    @singledispatchmethod
    def __eq__(self, other):
        if isinstance(other, NotCategory):
            return self._cat == other._cat
        return False

    @__eq__.register(Iterable)
    def _(self, other):
        # this will call the list approach
        return super().__eq__(other)


def _composite_name(sep):
    def getter(self):
        if self._name is not None:
            return self._name

        # Name is unset, we simply return the other parts
        if isinstance(self.A, CompositeCategory):
            nameA = f"({self.A.name})"
        else:
            nameA = self.A.name
        if isinstance(self.B, CompositeCategory):
            nameB = f"({self.B.name})"
        else:
            nameB = self.B.name

        return f"{nameA} {sep} {nameB}"

    def setter(self, name):
        self._name = name

    return property(getter, setter)


@set_module("sisl.category")
class CompositeCategory(GenericCategory):
    """A composite class consisting of two categories, an abstract class to always be inherited

    This should take 2 categories as arguments

    Parameters
    ----------
    A : Category
       the left hand side of the set operation
    B : Category
       the right hand side of the set operation
    """

    __slots__ = ("A", "B")

    def __init__(self, A, B):
        # To ensure we always get composite name
        super().__init__()
        self._name = None
        self.A = A
        self.B = B

    def __init_subclass__(cls, /, composite_name: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = _composite_name(composite_name)


@set_module("sisl.category")
class OrCategory(CompositeCategory, composite_name="|"):
    """A  class consisting of two categories

    This should take 2 categories as arguments and a binary operator to define
    how the categories are related.

    Parameters
    ----------
    A : Category
       the left hand side of the set operation
    B : Category
       the right hand side of the set operation
    """

    __slots__ = ()

    def _mask(self, *args, cache: dict, **kwargs) -> np.ndarray:
        return self.A._mask(*args, cache=cache, **kwargs) | self.B._mask(
            *args, cache=cache, **kwargs
        )

    def categorize(self, *args, **kwargs):
        r"""Base method for queriyng whether an object is a certain category"""
        catA = self.A.categorize(*args, **kwargs)

        if isinstance(catA, Iterable):
            if all(map(lambda a: not isinstance(a, NullCategory), catA)):
                return catA
        elif not isinstance(catA, NullCategory):
            return catA

        catB = self.B.categorize(*args, **kwargs)

        def cmp(a, b):
            if isinstance(a, NullCategory):
                return b
            return a

        if isinstance(catA, Iterable):
            return [cmp(a, b) for a, b in zip(catA, catB)]
        return cmp(catA, catB)


@set_module("sisl.category")
class AndCategory(CompositeCategory, composite_name="&"):
    """A  class consisting of two categories

    This should take 2 categories as arguments and a binary operator to define
    how the categories are related.

    Parameters
    ----------
    A : Category
       the left hand side of the set operation
    B : Category
       the right hand side of the set operation
    """

    __slots__ = ()

    def _mask(self, *args, cache: dict, **kwargs) -> np.ndarray:
        return self.A._mask(*args, cache=cache, **kwargs) & self.B._mask(
            *args, cache=cache, **kwargs
        )

    def categorize(self, *args, **kwargs):
        r"""Base method for queriyng whether an object is a certain category"""
        catA = self.A.categorize(*args, **kwargs)

        if isinstance(catA, Iterable):
            if all(map(lambda a: isinstance(a, NullCategory), catA)):
                return catA
        elif isinstance(catA, NullCategory):
            return catA

        # We can now get B and categorize
        catB = self.B.categorize(*args, **kwargs)

        def cmp(a, b):
            if isinstance(a, NullCategory):
                return a
            if isinstance(b, NullCategory):
                return b
            return self

        if isinstance(catA, Iterable):
            return [cmp(a, b) for a, b in zip(catA, catB)]
        return cmp(catA, catB)


@set_module("sisl.category")
class XOrCategory(CompositeCategory, composite_name="⊕"):
    """A  class consisting of two categories

    This should take 2 categories as arguments and a binary operator to define
    how the categories are related.

    Parameters
    ----------
    A : Category
       the left hand side of the set operation
    B : Category
       the right hand side of the set operation
    """

    __slots__ = ()

    def _mask(self, *args, cache: dict, **kwargs) -> np.ndarray:
        return self.A._mask(*args, cache=cache, **kwargs) ^ self.B._mask(
            *args, cache=cache, **kwargs
        )

    def categorize(self, *args, **kwargs):
        r"""Base method for queriyng whether an object is a certain category"""
        catA = self.A.categorize(*args, **kwargs)
        catB = self.B.categorize(*args, **kwargs)

        def cmp(a, b):
            if isinstance(a, NullCategory):
                return b
            if isinstance(b, NullCategory):
                return a
            # both are not NullCategory, in which case nothing
            # is exclusive, so we return the NullCategory
            return NullCategory()

        if isinstance(catA, Iterable):
            return [cmp(a, b) for a, b in zip(catA, catB)]
        return cmp(catA, catB)
