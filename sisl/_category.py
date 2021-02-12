from collections import defaultdict, ChainMap
from collections.abc import Iterable
from abc import ABCMeta, abstractmethod
from functools import lru_cache
import numpy as np

from ._internal import set_module, singledispatchmethod

__all__ = ["Category", "CompositeCategory", "NullCategory"]
__all__ += ["AndCategory", "OrCategory", "XOrCategory"]
__all__ += ["InstanceCache"]


class InstanceCache:
    """ Wraps an instance to cache *all* results based on `functools.lru_cache`

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
        """
        If a category class is called, we will attempt to instantiate it.

        However, it may be that this is a parent class (e.g. `AtomCategory`)
        that does not make sense to instantiate. Since this classes are abstract,
        they will raise an error that we will use to build the categories that the user
        requested.

        Examples
        -----------

        >>> AtomZ(6) # returns an AtomZ category with the Z parameter set to 6
        >>> AtomCategory(Z=6) # returns exactly the same since it uses `AtomCategory.kw`
        """
        try:
            return super().__call__(*args, **kwargs)
        except TypeError as e:
            if len(args) == 0:
                return cls.kw(**kwargs)
            # If args were provided, the user probably didn't want to use the category builder,
            # so we are going to let the exception be raised
            raise e


@set_module("sisl.category")
class Category(metaclass=CategoryMeta):
    r""" A category """
    __slots__ = ("_name", "_wrapper")

    def __init__(self, name=None):
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

    @property
    def name(self):
        r""" Name of category """
        return self._name

    def set_name(self, name):
        r""" Override the name of the categorization """
        self._name = name

    @classmethod
    @abstractmethod
    def is_class(cls, name, case=True):
        r""" Query whether `name` matches the class name by removing a prefix `kw`

        This is important to ensure that users match the full class name
        by omitting the prefix returned from this method.

        This is an abstract method to ensure sub-classes of `Category`
        implements it.

        For instance:

        .. code::

            class MyCategory(Category):

                @classmethod
                def is_class(cls, name):
                     # strip "My" and do comparison
                     return cl.__name__.lower()[2:] == name.lower()

        would enable one to compare against the *base* category scheme.

        This has the option to search case-sensitivity or not.
        """
        pass

    @classmethod
    def kw(cls, **kwargs):
        """ Create categories based on keywords

        This will search through the inherited classes and
        return and & category object for all keywords.

        Since this is a class method one should use this
        on the base category class in the given section
        of the code.
        """

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
                        raise ValueError(f"{cls.__name__}.kw got a non-unique argument for category name:\n"
                                         f"    Searching for {key} and found matches {found.__name__} and {cl.__name__}.")
                    found = cl

            if found is None:
                for cl in subcls:
                    if cl.is_class(key, case=False):
                        if found:
                            raise ValueError(f"{cls.__name__}.kw got a non-unique argument for category name:\n"
                                             f"    Searching for {key} and found matches {found.__name__.lower()} and {cl.__name__.lower()}.")
                        found = cl

            if found is None:
                raise ValueError(f"{cls.__name__}.kw got an argument for category name:\n"
                                 f"    Searching for {key} but found no matches.")

            if cat is None:
                cat = get_cat(found, args)
            else:
                cat = cat & get_cat(found, args)

        return cat

    @abstractmethod
    def categorize(self, *args, **kwargs):
        r""" Do categorization """
        pass

    def __str__(self):
        r""" String representation of the class (non-distinguishable between equivalent classifiers) """
        return self.name

    def __repr__(self):
        r""" String representation of the class (non-distinguishable between equivalent classifiers) """
        return self.name

    @singledispatchmethod
    def __eq__(self, other):
        """ Comparison of two categories, they are compared by class-type """
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
                return (self.__class__ is other.__class__ and
                        (self.A == other.A and self.B == other.B or
                         self.A == other.B and self.B == other.A))
            # if neither is a compositecategory, then they cannot
            # be the same category
            return False
        elif self.__class__ != other.__class__:
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
    @classmethod
    def is_class(cls, name):
        # never allow one to match a generic class
        # I.e. you can't instantiate a Null/Not/And/Or/XOr category by name
        return False


@set_module("sisl.category")
class NullCategory(GenericCategory):
    r""" Special Null class which always represents a classification not being *anything* """
    __slots__ = tuple()

    def __init__(self):
        pass

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


@set_module("sisl.category")
class NotCategory(GenericCategory):
    """ A class returning the *opposite* of this class (NullCategory) if it is categorized as such """
    __slots__ = ("_cat",)

    def __init__(self, cat):
        super().__init__()
        if isinstance(cat, CompositeCategory):
            self.set_name(f"~({cat})")
        else:
            self.set_name(f"~{cat}")
        self._cat = cat

    def categorize(self, *args, **kwargs):
        r""" Base method for queriyng whether an object is a certain category """
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


@set_module("sisl.category")
class CompositeCategory(GenericCategory):
    """ A composite class consisting of two categories, an abstract class to always be inherited

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

    def categorize(self, *args, **kwargs):
        r""" Base method for queriyng whether an object is a certain category """
        catA = self.A.categorize(*args, **kwargs)
        catB = self.B.categorize(*args, **kwargs)
        return catA, catB


def _composite_name(sep):
    def name(self):
        if not self._name is None:
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

    return property(name)


@set_module("sisl.category")
class OrCategory(CompositeCategory):
    """ A  class consisting of two categories

    This should take 2 categories as arguments and a binary operator to define
    how the categories are related.

    Parameters
    ----------
    A : Category
       the left hand side of the set operation
    B : Category
       the right hand side of the set operation
    """
    __slots__ = tuple()

    def categorize(self, *args, **kwargs):
        r""" Base method for queriyng whether an object is a certain category """
        catA, catB = super().categorize(*args, **kwargs)

        def cmp(a, b):
            if isinstance(a, NullCategory):
                return b
            return a

        if isinstance(catA, Iterable):
            return [cmp(a, b) for a, b in zip(catA, catB)]
        return cmp(catA, catB)

    name = _composite_name("|")


@set_module("sisl.category")
class AndCategory(CompositeCategory):
    """ A  class consisting of two categories

    This should take 2 categories as arguments and a binary operator to define
    how the categories are related.

    Parameters
    ----------
    A : Category
       the left hand side of the set operation
    B : Category
       the right hand side of the set operation
    """
    __slots__ = tuple()

    def categorize(self, *args, **kwargs):
        r""" Base method for queriyng whether an object is a certain category """
        catA, catB = super().categorize(*args, **kwargs)

        def cmp(a, b):
            if isinstance(a, NullCategory):
                return a
            elif isinstance(b, NullCategory):
                return b
            return self

        if isinstance(catA, Iterable):
            return [cmp(a, b) for a, b in zip(catA, catB)]
        return cmp(catA, catB)

    name = _composite_name("&")


@set_module("sisl.category")
class XOrCategory(CompositeCategory):
    """ A  class consisting of two categories

    This should take 2 categories as arguments and a binary operator to define
    how the categories are related.

    Parameters
    ----------
    A : Category
       the left hand side of the set operation
    B : Category
       the right hand side of the set operation
    """
    __slots__ = tuple()

    def categorize(self, *args, **kwargs):
        r""" Base method for queriyng whether an object is a certain category """
        catA, catB = super().categorize(*args, **kwargs)

        def cmp(a, b):
            if isinstance(a, NullCategory):
                return b
            elif isinstance(b, NullCategory):
                return a
            # both are not NullCategory, in which case nothing
            # is exclusive, so we return the NullCategory
            return NullCategory()

        if isinstance(catA, Iterable):
            return [cmp(a, b) for a, b in zip(catA, catB)]
        return cmp(catA, catB)

    name = _composite_name("⊕")
