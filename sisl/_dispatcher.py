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
        def func(*args, **kwargs):
            return method(*args, **kwargs)
        return func

    def __call__(self, method):
        return self.dispatch(method)

    def __getattr__(self, key):
        method = getattr(self._obj, key)
        return self.dispatch(method)


class ObjectDispatcher:
    # We need to hide the methods and objects
    # since we are going to retrieve dispatchs from the object it-self
    __slots__ = ("_obj", "_dispatchs", "_default")

    def __init__(self, obj, dispatchs=None, default=None):
        self._obj = obj
        if dispatchs is None:
            dispatchs = dict()
        self._dispatchs = dispatchs
        self._default = default

    def __len__(self):
        return len(self._dispatchs)

    def __str__(self):
        obj = str(self._obj).replace("\n", "\n ")
        def toline(kv):
            k, v = kv
            if k == self._default:
                return f"*{k} = " + str(v(object())).replace("\n", "\n ")
            return f" {k} = " + str(v(object())).replace("\n", "\n ")
        dispatchs = ",\n ".join(map(toline, self._dispatchs.items()))
        return f"{self.__class__.__name__}{{dispatchs: {len(self)},\n {obj},\n {dispatchs}\n}}"

    ####
    # Only the following methods are necessary for the dispatch method to work
    ####

    def register(self, key, dispatch, default=False, to_class=True):
        """ Register a dispatch class to this object and to the object class instance (if existing)

        Parameter
        ---------
        key : *any hashable*
            key used in the dictionary look-up for the dispatch class
        dispatch : AbstractDispatch
            dispatch class to be registered
        default : bool, optional
            this dispatch class will be the default on this object _only_.
            To register a class as the default class-wide, do this on the class
            variable.
        to_class : bool, optional
            whether the dispatch class will also be registered with the
            contained object's class instance
        """
        if to_class:
            cls_dispatch = getattr(self._obj.__class__, "dispatch", None)
            if cls_dispatch:
                cls_dispatch.register(key, dispatch)
        # Since this instance is already created, we have to add it here.
        # This has the side-effect that already stored dispatch (of ObjectDispatcher)
        # will not get these.
        self._dispatchs[key] = dispatch
        if default:
            self._default = key

    def __getitem__(self, key):
        r""" Retrieve dispatched dispatchs by hash (allows functions to be dispatched) """
        return self._dispatchs[key](self._obj)

    def __getattr__(self, key):
        """ Retrieve dispatched method by name """
        if key in self._dispatchs:
            return self._dispatchs[key](self._obj)
        return getattr(self._dispatchs[self._default](self._obj), key)


class ClassDispatcher:
    __slots__ = ("_dispatchs", "_default")

    def __init__(self):
        self._dispatchs = dict()
        self._default = None

    def __len__(self):
        return len(self._dispatchs)

    def __str__(self):
        # We know how to create an object, passing 1 argument (an object)
        # We will fake this to get a str representation.
        def toline(kv):
            k, v = kv
            if k == self._default:
                return f"*{k} = " + str(v(object())).replace("\n", "\n ")
            return f" {k} = " + str(v(object())).replace("\n", "\n ")
        dispatchs = ",\n ".join(map(toline, self._dispatchs.items()))
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
        return ObjectDispatcher(instance, self._dispatchs, default=self._default)

    def register(self, key, dispatch, default=False):
        """ Register a dispatch class

        Parameter
        ---------
        key : *any hashable*
            key used in the dictionary look-up for the dispatch class
        dispatch : AbstractDispatch
            dispatch class to be registered
        default : bool, optional
            if true, this `dispatch` will be the default in case the
            `ObjectDispatcher` cannot find the requested object.
        """
        self._dispatchs[key] = dispatch
        if default:
            self._default = key
