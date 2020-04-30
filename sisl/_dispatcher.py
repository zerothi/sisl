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
        if callable(method):
            return self.dispatch(method)
        return method


class Dispatcher:
    __slots__ = ("_dispatchs", "_default", "__name__")

    def __init__(self, dispatchs=None, default=None):
        if dispatchs is None:
            dispatchs = dict()
        self._dispatchs = dispatchs
        self._default = default
        self.__name__ = self.__class__.__name__

    def __len__(self):
        return len(self._dispatchs)

    def __str__(self):
        def toline(kv):
            k, v = kv
            if k == self._default:
                return f"*{k} = " + str(v(object())).replace("\n", "\n ")
            return f" {k} = " + str(v(object())).replace("\n", "\n ")
        dispatchs = ",\n ".join(map(toline, self._dispatchs.items()))
        return f"{self.__name__}{{dispatchs: {len(self)},\n {dispatchs}\n}}"

    def register(self, key, dispatch, default=False):
        """ Register a dispatch class to this container

        Parameter
        ---------
        key : *any hashable*
            key used in the dictionary look-up for the dispatch class
        dispatch : AbstractDispatch
            dispatch class to be registered
        default : bool, optional
            if true, `dispatch` will be the default when requesting it
        """
        self._dispatchs[key] = dispatch
        if default:
            self._default = key


class MethodDispatcher(Dispatcher):
    __slots__ = ("_method", "_obj")

    def __init__(self, method, dispatchs=None, default=None, obj=None):
        super().__init__(dispatchs, default)
        # This will probably fail for PYTHONOPTIMIZE=2
        self._method = method

        # In python3 a method *always* have the __self__ key
        # In case the method is bound on a class.
        if obj is None:
            self._obj = getattr(method, "__self__", None)
        else:
            self._obj = obj

        # Make function documentation local to __call__
        self.__call__.__func__.__doc__ = method.__doc__
        # Storing the name is required for help on functions
        self.__name__ = method.__name__

    def __call__(self, *args, **kwargs):
        if self._default is None:
            return self._method(*args, **kwargs)
        return self._dispatchs[self._default](self._obj).dispatch(self._method)(*args, **kwargs)

    def __getitem__(self, key):
        r""" Get method using dispatch according to `key` """
        return self._dispatchs[key](self._obj).dispatch(self._method)

    __getattr__ = __getitem__


class ObjectDispatcher(Dispatcher):
    # We need to hide the methods and objects
    # since we are going to retrieve dispatchs from the object it-self
    __slots__ = ("_obj", "_obj_getattr", "_cls_attr_name")

    def __init__(self, obj, dispatchs=None, default=None, cls_attr_name=None, obj_getattr=None):
        super().__init__(dispatchs, default)
        self._obj = obj
        if obj_getattr is None:
            def obj_getattr(obj, key):
                return getattr(obj, key)
        self._obj_getattr = obj_getattr
        self._cls_attr_name = cls_attr_name

    def __str__(self):
        obj = str(self._obj).replace("\n", "\n ")
        return super().__str__().replace(",\n", f",\n {obj},\n", 1)

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
        super().register(key, dispatch, default)
        if to_class:
            cls_dispatch = getattr(self._obj.__class__, self._cls_attr_name, None)
            if isinstance(cls_dispatch, ClassDispatcher):
                cls_dispatch.register(key, dispatch)

    def __getitem__(self, key):
        r""" Retrieve dispatched dispatchs by hash (allows functions to be dispatched) """
        return self._dispatchs[key](self._obj)

    def __getattr__(self, key):
        """ Retrieve dispatched method by name, or if the name does not exist return a MethodDispatcher """
        if key in self._dispatchs:
            return self._dispatchs[key](self._obj)

        attr = self._obj_getattr(self._obj, key)
        if callable(attr):
            # This will also ensure that if the user calls immediately after it will use the default
            return MethodDispatcher(attr, dispatchs=self._dispatchs,
                                    default=self._default, obj=self._obj)
        return attr


class ClassDispatcher(Dispatcher):
    __slots__ = ("_obj_getattr", "_attr_name")

    def __init__(self, name, dispatchs=None, default=None, obj_getattr=None):
        # obj_getattr is necessary for the ObjectDispatcher to create the correct
        # MethodDispatcher
        super().__init__(dispatchs, default)
        if obj_getattr is None:
            def obj_getattr(obj, key):
                return getattr(obj, key)
        self._obj_getattr = obj_getattr
        # the name of the ClassDispatcher attribute in the class
        self._attr_name = name

    def __get__(self, instance, owner):
        """ Class dispatcher retrieval

        When directly retrieved from the class we return it-self to
        allow interaction with the dispatcher.

        When retrieved from an object it returns an `ObjectDispatcher`
        which contains the current dispatchs allowed to be dispatched through.
        """
        if instance is None:
            return self
        return ObjectDispatcher(instance, self._dispatchs,
                                default=self._default,
                                cls_attr_name=self._attr_name,
                                obj_getattr=self._obj_getattr)
