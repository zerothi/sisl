from abc import ABCMeta, abstractmethod
from functools import wraps

__all__ = ["AbstractDispatch", "ObjectDispatcher", "ClassDispatcher"]


class AbstractDispatch(metaclass=ABCMeta):
    r""" Dispatcher class used for dispatching function calls """

    def __init__(self, obj):
        self._obj = obj

    def __str__(self):
        return f"{self.__class__.__name__}"

    ####
    # Only the following methods are necessary for the dispatch method to work
    ####

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

    def __call__(self, method, *args, **kwargs):
        return self.dispatch(method)(*args, **kwargs)

    def __getattr__(self, key):
        method = getattr(self._obj, key)
        return self.dispatch(method)


class ObjectDispatcher:
    # We need to hide the methods and objects
    # since we are going to retrieve dispatchs from the object it-self
    __slots__ = ("_obj", "_dispatchs")

    def __init__(self, obj, dispatchs=None):
        self._obj = obj
        if dispatchs is None:
            dispatchs = dict()
        self._dispatchs = dispatchs

    def __len__(self):
        return len(self._dispatchs)

    def __str__(self):
        obj = str(self._obj).replace("\n", "\n ")
        dispatchs = ",\n ".join(
            map(lambda kv: f"{kv[0]} = " + str(kv[1](object())).replace("\n", "\n "),
                self._dispatchs.items()
            )
        )
        return f"{self.__class__.__name__}{{dispatchs: {len(self)},\n {obj},\n {dispatchs}\n}}"

    ####
    # Only the following methods are necessary for the dispatch method to work
    ####

    def register(self, key, dispatch):
        """ Register a dispatch class to this object and to the object class instance (if existing)

        Parameter
        ---------
        key : *any hashable*
            key used in the dictionary look-up for the dispatch class
        dispatch : AbstractDispatch
            dispatch class to be registered
        """
        cls_dispatch = getattr(self._obj.__class__, "dispatch", None)
        if cls_dispatch:
            cls_dispatch.register(key, dispatch)
        # Since this instance is already created, we have to add it here.
        # This has the side-effect that already stored dispatch (of ObjectDispatcher)
        # will not get these.
        self._dispatchs[key] = dispatch

    def __getitem__(self, key):
        r""" Retrieve dispatched dispatchs by hash (allows functions to be dispatched) """
        return self._dispatchs[key](self._obj)

    def __getattr__(self, key):
        """ Retrieve dispatched method by name """
        return self._dispatchs[key](self._obj)


class ClassDispatcher:
    __slots__ = ["_dispatchs"]

    def __init__(self):
        self._dispatchs = dict()

    def __len__(self):
        return len(self._dispatchs)

    def __str__(self):
        # We know how to create an object, passing 1 argument (an object)
        # We will fake this to get a str representation.
        dispatchs = ",\n ".join(
            map(lambda kv: f"{kv[0]} = " + str(kv[1](object())).replace("\n", "\n "),
                self._dispatchs.items()
            )
        )
        return f"{self.__class__.__name__}{{dispatchs: {len(self)},\n {dispatchs}\n}}"

    ####
    # Only the following methods are necessary for the dispatch method to work
    ####

    def __get__(self, instance, owner):
        """ Class dispatcher retrieval

        When directly retrieved from the class we return it-self to
        allow interaction with the dispatcher.

        When retrieved from an object it returns an `ObjectDispatcher`
        which contains the current dispatchs allowed to be dispatched through.
        """
        if instance is None:
            return self
        return ObjectDispatcher(instance, self._dispatchs)

    def register(self, key, dispatch):
        """ Register a dispatch class

        Parameter
        ---------
        key : *any hashable*
            key used in the dictionary look-up for the dispatch class
        dispatch : AbstractDispatch
            dispatch class to be registered
        """
        self._dispatchs[key] = dispatch
