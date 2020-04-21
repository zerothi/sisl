from abc import ABCMeta, abstractmethod
from functools import wraps

__all__ = ["AbstractDispatch", "ObjectDispatcher", "ClassDispatcher"]


class AbstractDispatch(metaclass=ABCMeta):
    r""" Dispatcher class used for dispatching function calls """

    def __init__(self, obj):
        self._obj = obj

    @abstractmethod
    def dispatch(self, method):
        """ Create dispatched method with correctly wrapped documentation

        This should return a function that mimics method but wraps it
        in some way.
        """
        @wraps(method)
        def func(at_least_same_args_as_method):
            return method(at_least_same_args_as_method)
        return func

    def __getattr__(self, key):
        # Retrieve method from object
        method = getattr(self._obj, key)

        return self.dispatch(method)


class ObjectDispatcher:
    # We need to hide the methods and objects
    # since we are going to retrieve dispatchs from the object it-self
    __slots__ = ("_dispatchs", "_obj")

    def __init__(self, dispatchs, obj):
        self._dispatchs = dispatchs
        self._obj = obj

    def __getitem__(self, key):
        """ Retrieve dispatched dispatchs by hash (allows functions to be dispatched) """
        return self._dispatchs[key](self._obj)

    def __getattr__(self, key):
        """ Retrieve dispatched method by name """
        return self._dispatchs[key](self._obj)


class ClassDispatcher:
    __slots__ = ["_dispatchs"]

    def __init__(self):
        self._dispatchs = dict()

    def __get__(self, instance, owner):
        """ Class dispatcher retrieval

        When directly retrieved from the class we return it-self to
        allow interaction with the dispatcher.

        When retrieved from an object it returns an `ObjectDispatcher`
        which contains the current dispatchs allowed to be dispatched through.
        """
        if instance is None:
            return self
        return ObjectDispatcher(self._dispatchs, instance)

    def register(self, dispatch, key=None):
        """ Register a dispatch class

        Parameter
        ---------
        dispatch : AbstractDispatch
            dispatch class to be registered
        key : *hashable*, optional
            hashable key used in the dictionary look-up
            Will default to ``dispatch.__name__.lower()``
        """
        if key is None:
            key = dispatch.__name__.lower()
        self._dispatchs[key] = dispatch
