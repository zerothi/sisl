from abc import ABCMeta, abstractmethod
from functools import wraps

__all__ = ["AbstractDispatcher", "ObjectDispatcher", "ClassDispatcher"]

class AbstractDispatcher(metaclass=ABCMeta):
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
    # since we are going to retrieve methods from the object it-self
    __slots__ = ("_methods", "_obj")

    def __init__(self, methods, obj):
        self._methods = methods
        self._obj = obj

    def __getitem__(self, key):
        """ Retrieve dispatched methods by hash (allows functions to be dispatched) """
        return self._methods[key](self._obj)
                
    def __getattr__(self, key):
        """ Retrieve dispatched method by name """
        return self._methods[key](self._obj)


class ClassDispatcher:
    __slots__ = ["_methods"]
    
    def __init__(self):
        self._methods = dict()

    def __get__(self, instance, owner):
        """ Class dispatcher retrieval

        When directly retrieved from the class we return it-self to
        allow interaction with the dispatcher.
       
        When retrieved from an object it returns an `ObjectDispatcher`
        which contains the current methods allowed to be dispatched through.
        """
        if instance is None:
            return self
        return ObjectDispatcher(self._methods, instance)

    def register(self, *args, **kwargs):
        """ Register a dispatched method or class """
        for arg in args:
            if isinstance(arg, tuple):
                self._methods[arg[0]] = arg[1]
            else:
                try:
                    self._methods[arg.__name__] = arg
                except:
                    self._methods[arg] = arg
        for key, arg in kwargs.items():
            self._methods[key] = arg
