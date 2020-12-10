""" Dispatcher classes for handling methods dispatched for wrapped function calls.

This method allows classes to dispatch methods through other classes.

Here is a small snippet showing how to utilize this module.

"""

from abc import ABCMeta, abstractmethod
from functools import wraps


__all__ = ["AbstractDispatch", "ObjectDispatcher", "ClassDispatcher"]


def _dict_to_str(name, d, parser=None):
    """ Convert a dict to __str__ representation """
    if parser is None:
        def parser(kv):
            return f" {kv[0]}: {kv[1]}"
    d_str = ",\n ".join(map(parser, d.items()))
    if len(d_str) > 0:
        return f"{name} ({len(d)}): [\n {d_str}\n ]"
    return ""


class AbstractDispatch(metaclass=ABCMeta):
    r""" Dispatcher class used for dispatching function calls """
    __slots__ = ("_obj", "_attrs")

    def __init__(self, obj, **attrs):
        self._obj = obj
        # Local dictionary with attributes.
        # This could in principle contain anything.
        self._attrs = attrs

    def __call__(self, method):
        return self.dispatch(method)

    def __str__(self):
        obj = str(self._obj).replace("\n", "\n ")
        attrs = _dict_to_str("attrs", self._attrs)
        if len(attrs) == 0:
            return f"{self.__class__.__name__}{{{obj}}}"
        return f"{self.__class__.__name__}{{{obj}, {attrs}\n}}"

    @abstractmethod
    def dispatch(self, method):
        """ Create dispatched method with correctly wrapped documentation

        This should return a function that mimics method but wraps it
        in some way.

        A basic interception would be

        .. code:: python
            @wraps(method)
            def func(*args, **kwargs):
                return method(*args, **kwargs)

        """
        pass

    def __getattr__(self, key):
        attr = getattr(self._obj, key)
        if callable(attr):
            return self.dispatch(attr)
        return attr


class Dispatcher:
    __slots__ = ("_dispatchs", "_default", "__name__", "_attrs")

    def __init__(self, dispatchs=None, default=None, **attrs):
        if dispatchs is None:
            dispatchs = dict()
        self._dispatchs = dispatchs
        self._default = default
        self.__name__ = self.__class__.__name__
        # Attributes associated with the dispatcher
        self._attrs = attrs

    def __len__(self):
        return len(self._dispatchs)

    def __str__(self):
        def toline(kv):
            k, v = kv
            v = str(v(object())).replace("\n", "\n ")
            if k == self._default:
                return f"*{k} = {v}"
            return f" {k} = {v}"
        dispatchs = _dict_to_str("dispatchs", self._dispatchs, parser=toline)
        attrs = _dict_to_str("attrs", self._attrs)
        if len(attrs) == 0:
            return f"{self.__name__}{{{dispatchs}\n}}"
        return f"{self.__name__}{{{dispatchs},\n {attrs}\n}}"

    def register(self, key, dispatch, default=False, overwrite=False):
        """ Register a dispatch class to this container

        Parameter
        ---------
        key : *any hashable*
            key used in the dictionary look-up for the dispatch class
        dispatch : AbstractDispatch
            dispatch class to be registered
        default : bool, optional
            if true, `dispatch` will be the default when requesting it
        overwrite : bool, optional
            if true and `key` already exists in the list of dispatchs, then
            it will be overwritten, otherwise a `LookupError` is raised.
        """
        if not overwrite and key in self._dispatchs:
            raise LookupError(f"{self.__class__.__name__} already has {key} registered (and overwrite is false)")
        self._dispatchs[key] = dispatch
        if default:
            self._default = key


class MethodDispatcher(Dispatcher):
    __slots__ = ("_method", "_obj")

    def __init__(self, method, dispatchs=None, default=None, obj=None, **attrs):
        super().__init__(dispatchs, default, **attrs)
        self._method = method

        # In python3 a method *always* have the __self__ key
        # In case the method is bound on a class.
        if obj is None:
            self._obj = getattr(method, "__self__", None)
        else:
            self._obj = obj

        # This will probably fail for PYTHONOPTIMIZE=2
        self.__call__.__func__.__doc__ = method.__doc__
        # Storing the name is required for help on functions
        self.__name__ = method.__name__

    def __call__(self, *args, **kwargs):
        if self._default is None:
            return self._method(*args, **kwargs)
        return self._dispatchs[self._default](self._obj, **self._attrs).dispatch(self._method)(*args, **kwargs)

    def __getitem__(self, key):
        r""" Get method using dispatch according to `key` """
        return self._dispatchs[key](self._obj, **self._attrs).dispatch(self._method)

    __getattr__ = __getitem__


class ObjectDispatcher(Dispatcher):
    # We need to hide the methods and objects
    # since we are going to retrieve dispatchs from the object it-self
    __slots__ = ("_obj", "_obj_getattr", "_cls_attr_name")

    def __init__(self, obj, dispatchs=None, default=None, cls_attr_name=None, obj_getattr=None, **attrs):
        super().__init__(dispatchs, default, **attrs)
        self._obj = obj
        if obj_getattr is None:
            def obj_getattr(obj, key):
                return getattr(obj, key)
        self._obj_getattr = obj_getattr
        self._cls_attr_name = cls_attr_name

    def __str__(self):
        obj = str(self._obj).replace("\n", "\n ")
        return super().__str__().replace("{", f"{{\n {obj},\n ", 1)

    def register(self, key, dispatch, default=False, overwrite=False, to_class=True):
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
        overwrite : bool, optional
            if true and `key` already exists in the list of dispatchs, then
            it will be overwritten, otherwise a `LookupError` is raised.
        to_class : bool, optional
            whether the dispatch class will also be registered with the
            contained object's class instance
        """
        super().register(key, dispatch, default, overwrite)
        if to_class:
            cls_dispatch = getattr(self._obj.__class__, self._cls_attr_name, None)
            if isinstance(cls_dispatch, ClassDispatcher):
                cls_dispatch.register(key, dispatch, overwrite=overwrite)

    def __call__(self, **attrs):
        # Return a new instance of this object (with correct attributes)
        overlap = self._attrs.keys() & attrs.keys()
        # Create new attributes without overlaps
        new_attrs = {key: self._attrs[key]
                     for key in self._attrs if key not in overlap}
        new_attrs.update(attrs)
        return self.__class__(self._obj, self._dispatchs, self._default,
                              self._cls_attr_name, self._obj_getattr, **new_attrs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __getitem__(self, key):
        r""" Retrieve dispatched dispatchs by hash (allows functions to be dispatched) """
        return self._dispatchs[key](self._obj, **self._attrs)

    def __getattr__(self, key):
        """ Retrieve dispatched method by name, or if the name does not exist return a MethodDispatcher """
        if key in self._dispatchs:
            return self._dispatchs[key](self._obj, **self._attrs)

        attr = self._obj_getattr(self._obj, key)
        if callable(attr):
            # This will also ensure that if the user calls immediately after it will use the default
            return MethodDispatcher(attr, dispatchs=self._dispatchs,
                                    default=self._default, obj=self._obj, **self._attrs)
        return attr


class ClassDispatcher(Dispatcher):
    __slots__ = ("_obj_getattr", "_attr_name")

    def __init__(self, name, dispatchs=None, default=None, obj_getattr=None, **attrs):
        # obj_getattr is necessary for the ObjectDispatcher to create the correct
        # MethodDispatcher
        super().__init__(dispatchs, default, **attrs)
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
                                obj_getattr=self._obj_getattr,
                                **self._attrs)


class ClassTypeDispatcher(ClassDispatcher):
    __slots__ = tuple()

    def __get__(self, instance, owner):
        return self

    def __call__(self, *args, **kwargs):
        # now check the first argument
        if len(args) > 0:
            typ = type(args[0])
        elif self._default is None:
            raise ValueError(f"{self.__class__.__name__} could not find any type from the input arguments and no default are defined.")
        else:
            return self._dispatchs[self._default](*args, **kwargs)
        return self._dispatchs[typ](*args, **kwargs)

    def __getitem__(self, key):
        r""" Get method using dispatch according to `key` """
        return self._dispatchs[key]

    __getattr__ = __getitem__


'''
For use when doing cached dispatechers
class CachedClassDispatcher(ClassDispatcher):
    __slots__ = ("_obj_getattr", "_attr_name")

    def __init__(self, name, dispatchs=None, default=None, obj_getattr=None, **attrs):
        # obj_getattr is necessary for the ObjectDispatcher to create the correct
        # MethodDispatcher
        super().__init__(dispatchs, default, **attrs)
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
        dispatcher = ObjectDispatcher(instance, self._dispatchs,
                                default=self._default,
                                cls_attr_name=self._attr_name,
                                obj_getattr=self._obj_getattr,
                                **self._attrs)
        object.__setattr__(instance, self._attr_name, dispatcher)
        return dispatcher
'''
