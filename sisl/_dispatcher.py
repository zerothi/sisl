""" Dispatcher classes for handling methods dispatched for wrapped function calls.

This method allows classes to dispatch methods through other classes.

Here is a small snippet showing how to utilize this module.

"""

from abc import ABCMeta, abstractmethod
from functools import wraps
from collections import namedtuple


__all__ = ["AbstractDispatch", "ObjectDispatcher", "MethodDispatcher",
           "ErrorDispatcher", "ClassDispatcher", "TypeDispatcher"]


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

    def renew(self, **attrs):
        """ Create a new class with updated attributes """
        return self.__class__(self._obj, **{**self._attrs, **attrs})

    def __call__(self, *args, **kwargs):
        return self.dispatch(*args, **kwargs)

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


class AbstractDispatcher:
    """ A container for dispatchers

    This is an abstract class holding the dispatch classes (`AbstractDispatch`)
    and the attributes that are associated with the dispatchers.
    """
    __slots__ = ("_dispatchs", "_default", "__name__", "_attrs")

    def __init__(self, dispatchs=None, default=None, **attrs):
        if dispatchs is None:
            dispatchs = dict()
        self._dispatchs = dispatchs
        self._default = default
        self.__name__ = self.__class__.__name__
        # Attributes associated with the dispatcher
        self._attrs = attrs

    def renew(self, **attrs):
        """ Create a new class with updated attributes """
        return self.__class__(self._dispatchs, self._default, **{**self._attrs, **attrs})

    def __len__(self):
        return len(self._dispatchs)

    def __str__(self):
        def toline(kv):
            k, v = kv
            v = str(v("<self>")).replace("\n", "\n ")
            if k == self._default:
                return f"*{k} = {v}"
            return f" {k} = {v}"
        dispatchs = _dict_to_str("dispatchs", self._dispatchs, parser=toline)
        attrs = _dict_to_str("attrs", self._attrs)
        if len(attrs) == 0:
            return f"{self.__name__}{{{dispatchs}\n}}"
        return f"{self.__name__}{{{dispatchs},\n {attrs}\n}}"

    def __setitem__(self, key, dispatch):
        """ Registers a dispatch method (using `register` with default values)

        Parameters
        ----------
        key : *any hashable*
            key used in the dictionary look-up for the dispatch class
        dispatch : AbstractDispatch
            dispatch class to be registered
        """
        self.register(key, dispatch)

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
        if key in self._dispatchs and not overwrite:
            raise LookupError(f"{self.__class__.__name__} already has {key} registered (and overwrite is false)")
        self._dispatchs[key] = dispatch
        if default:
            self._default = key


class ErrorDispatcher(AbstractDispatcher):
    """ Faulty handler to ensure that certain operations are not allowed

    This may for instance be used with ``ClassDispatcher(instance_dispatcher=ErrorDispatcher)``
    to ensure that a certain dispatch attribute will never be called on an instance.
    It won't work on type_dispatcher due to not being able to call `register`.
    """
    __slots__ = ()

    def __init__(self, obj, *args, **kwargs):
        raise ValueError(f"Dispatcher on {obj} must not be called in this way, see documentation.")


class MethodDispatcher(AbstractDispatcher):
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

    def renew(self, **attrs):
        """ Create a new class with updated attributes """
        return self.__class__(self._method, self._dispatchs, self._default,
                              self._obj, **{**self._attrs, **attrs})

    def __call__(self, *args, **kwargs):
        if self._default is None:
            return self._method(*args, **kwargs)
        return self._dispatchs[self._default](self._obj, **self._attrs).dispatch(self._method)(*args, **kwargs)

    def __getitem__(self, key):
        r""" Get method using dispatch according to `key` """
        return self._dispatchs[key](self._obj, **self._attrs).dispatch(self._method)

    __getattr__ = __getitem__


class ObjectDispatcher(AbstractDispatcher):
    """ A dispatcher relying on object lookups

    This dispatcher wraps a method call with lookup tables and possible defaults.

    Examples
    --------
    >>> a = ObjectDispatcher(lambda x: print(x))
    >>> class DoubleCall(AbstractDispatch):
    ...    def dispatch(self, method):
    ...        def func(x):
    ...            method(x)
    ...            method(x)
    ...        return func
    >>> a.register("double", DoubleCall)
    >>> a.double("hello world")
    hello world
    hello world
    """
    __slots__ = ("_obj", "_obj_getattr", "_cls_attr_name")

    def __init__(self, obj, dispatchs=None, default=None, cls_attr_name=None, obj_getattr=None, **attrs):
        super().__init__(dispatchs, default, **attrs)
        self._obj = obj
        if obj_getattr is None:
            def obj_getattr(obj, key):
                return getattr(obj, key)
        self._obj_getattr = obj_getattr
        self._cls_attr_name = cls_attr_name

    def renew(self, **attrs):
        """ Create a new class with updated attributes """
        return self.__class__(self._obj, self._dispatchs, self._default,
                              self._cls_attr_name, self._obj_getattr,
                              **{**self._attrs, **attrs})

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


class TypeDispatcher(ObjectDispatcher):
    """ A dispatcher relying on type lookups

    This dispatcher may be called directly and will query the dispatch method
    through the type of the first argument.

    Examples
    --------
    >>> a = TypeDispatcher("a")
    >>> class MyDispatch(AbstractDispatch):
    ...    def dispatch(self, arg):
    ...        print(arg)
    >>> a.register(str, MyDispatch)
    >>> a("hello world")
    hello world
    """
    __slots__ = ()

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
        super().register(key, dispatch, default, overwrite, to_class=False)
        if to_class:
            cls_dispatch = getattr(self._obj, self._cls_attr_name, None)
            if isinstance(cls_dispatch, ClassDispatcher):
                cls_dispatch.register(key, dispatch, overwrite=overwrite)

    def __call__(self, obj, *args, **kwargs):
        # A call on a TypeDispatcher forces at least a single argument
        # where the type is being dispatched.

        # Figure out if obj is a class or not
        if isinstance(obj, type):
            typ = obj
        else:
            # If not, then get the type (basically same as obj.__class__)
            typ = type(obj)

        # if you want obj to be a type, then the dispatcher should control that
        return self._dispatchs[typ](self._obj)(obj, *args, **kwargs)

    def __getitem__(self, key):
        r""" Retrieve dispatched dispatchs by hash (allows functions to be dispatched) """
        return self._dispatchs[key](self._obj, **self._attrs)


class ClassDispatcher(AbstractDispatcher):
    """ A dispatcher for classes, using `__get__` it converts into `ObjectDispatcher` upon invocation from an object, or a `TypeDispatcher` when invoked from a class

    This is a class-placeholder allowing a dispatecher to be a class attribute and converted into an
    `ObjectDispatcher` when invoked from an object.

    If it is called on the class, it will return a `TypeDispatcher`.

    This class should be an attribute of a class. It heavily relies on the `__get__` special
    method.

    Parameters
    ----------
    name : str
       name of the attribute in the class
    dispatchs : dict, optional
       dictionary of dispatch methods
    obj_getattr : callable, optional
       method with 2 arguments, an ``obj`` and the ``attr`` which may be used
       to control how the attribute is called.
    instance_dispatcher : AbstractDispatcher, optional
       control how instance dispatchers are handled through `__get__` method.
       This controls the dispatcher used if called from an instance.
    type_dispatcher : AbstractDispatcher, optional
       control how class dispatchers are handled through `__get__` method.
       This controls the dispatcher used if called from a class.

    Examples
    --------
    >>> class A:
    ...   new = ClassDispatcher("new", obj_getattr=lambda obj, attr: getattr(obj.sub, attr))

    The above defers any attributes to the contained `A.sub` attribute.
    """
    __slots__ = ("_obj_getattr", "_attr_name", "_get")

    def __init__(self, attr_name, dispatchs=None, default=None,
                 obj_getattr=None,
                 instance_dispatcher=ObjectDispatcher,
                 type_dispatcher=None,
                 **attrs):
        # obj_getattr is necessary for the ObjectDispatcher to create the correct
        # MethodDispatcher
        super().__init__(dispatchs, default, **attrs)
        # the name of the ClassDispatcher attribute in the class
        self._attr_name = attr_name
        p = namedtuple("get_class", ["instance", "type"])
        self._get = p(instance_dispatcher, type_dispatcher)

        # Default the obj_getattr
        if obj_getattr is None:
            def obj_getattr(obj, key):
                return getattr(obj, key)
        self._obj_getattr = obj_getattr

    def renew(self, **attrs):
        """ Create a new class with updated attributes """
        return self.__class__(self._attr_name, self._dispatchs, self._default,
                              self._obj_getattr,
                              self._get.instance, self._get.type,
                              **{**self._attrs, **attrs})

    def __get__(self, instance, owner):
        """ Class dispatcher retrieval

        The returned class depends on the setup of the `ClassDispatcher`.

        If called on an instance, it will return a class ``self._get.instance``
        class object.

        If called on a class (type), it will return a class ``self._get.type``.

        If the returned class is None, it will return ``self``.
        """
        if instance is None:
            inst = owner
            cls = self._get.type
        else:
            cls = self._get.instance
            if issubclass(cls, TypeDispatcher):
                inst = owner
            else:
                inst = instance
        if cls is None:
            return self
        return cls(inst, self._dispatchs, default=self._default,
                   cls_attr_name=self._attr_name,
                   obj_getattr=self._obj_getattr,
                   **self._attrs)


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
