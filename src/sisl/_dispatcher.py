# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" Dispatcher classes for handling methods dispatched for wrapped function calls.

This method allows classes to dispatch methods through other classes.

Here is a small snippet showing how to utilize this module.

"""
import inspect
import logging
from abc import ABCMeta, abstractmethod
from collections import ChainMap, namedtuple
from functools import update_wrapper
from typing import Any, Callable, Union

from sisl.utils._search_mro import find_implementation

__all__ = [
    "AbstractDispatch",
    "ObjectDispatcher",
    "MethodDispatcher",
    "ErrorDispatcher",
    "ClassDispatcher",
    "TypeDispatcher",
]

_log = logging.getLogger(__name__)


def _dict_to_str(name, d, parser=None):
    """Convert a dict to __str__ representation"""
    if parser is None:

        def parser(kv):
            return f" {kv[0]}: {kv[1]}"

    d_str = ",\n ".join(map(parser, d.items()))
    if len(d_str) > 0:
        return f"{name} ({len(d)}): [\n {d_str}\n ]"
    return ""


class AbstractDispatch(metaclass=ABCMeta):
    r"""Dispatcher class used for dispatching function calls"""

    def __init__(self, obj, **attrs):
        self._obj = obj
        # Local dictionary with attributes.
        # This could in principle contain anything.
        self._attrs = attrs
        _log.debug(f"__init__ {self.__class__.__name__}", extra={"obj": self})

    @classmethod
    def __from_function__(
        cls, func: Optional[Callable] = None, *, name: Optional[str] = None, **kwargs
    ) -> Self:
        """Wrap function to return a new class (in the named function) for easier
        handling

        Parameters
        ----------
        func :
            the function to be wrapped to a new type.
            If not specified the remaining arguments can be used to return
            a decorator.
        name :
            Returned class name of the decorated function.
        **kwargs :
            Methods added to the ``type(name, (cls,), kwargs)``.

        Examples
        --------

        >>> @DispatchClass.__from_function__
        ... def func(msg):
        ...     print(msg)
        >>> print(func)
        <class '__main__.func'>

        >>> @DispatchClass.__from_function__(name="Hello")
        ... def func(msg):
        ...     print(msg)
        >>> print(func)
        <class '__main__.Hello'>
        """
        if func is None:

            def decorator(func) -> Self:
                return cls.__from_function__(func, name=name, **kwargs)

            return decorator

        if name is None:
            name = func.__name__

        things = dict(__signature__=inspect.signature(func), dispatch=func)
        for attr in (
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
            "__type_params__",
        ):
            try:
                value = getattr(func, attr)
            except AttributeError:
                pass
            else:
                things[attr] = value

        # Add user-defined details
        things.update(**kwargs)

        return type(name, (cls,), things)

    def copy(self):
        """Create a copy of this object (will not copy `obj`)"""
        _log.debug(f"copy {self.__class__.__name__}", extra={"obj": self})
        return self.__class__(self._obj, **self._attrs)

    def renew(self, **attrs):
        """Create a new class with updated attributes"""
        _log.debug(f"renew {self.__class__.__name__}", extra={"obj": self})
        return self.__class__(self._obj, **{**self._attrs, **attrs})

    def __call__(self, *args, **kwargs):
        _log.debug(f"call {self.__class__.__name__}{args}", extra={"obj": self})
        return self.dispatch(*args, **kwargs)

    def __str__(self):
        obj = str(self._obj).replace("\n", "\n ")
        attrs = _dict_to_str("attrs", self._attrs)
        if len(attrs) == 0:
            return f"{self.__class__.__name__}{{{obj}}}"
        return f"{self.__class__.__name__}{{{obj}, {attrs}\n}}"

    def __repr__(self):
        nattrs = len(self._attrs)
        return f"<{self.__class__.__name__}{{{self._obj!r}, nattrs={nattrs}}}>"

    def _get_object(self):
        """Retrieves the object (self._obj) but also checks that the object is in fact an object (`type`)

        This will fail if the dispatch method has been called on the class (not an instance).
        """
        obj = self._obj
        if isinstance(obj, type):
            raise ValueError(f"Dispatcher on {obj} must not be called on the class.")
        return obj

    def _get_class(self, allow_instance=False):
        """Retrieves the object (self._obj) but also checks that the object is a class (not an instance, `type`)

        This will fail if the dispatch method has been called on an instance (not a class).
        """
        obj = self._obj
        if isinstance(obj, type):
            return obj
        if allow_instance:
            return obj.__class__
        raise ValueError(f"Dispatcher on {obj} must not be called on the instance.")

    @abstractmethod
    def dispatch(self, method):
        """Create dispatched method with correctly wrapped documentation

        This should return a function that mimics method but wraps it
        in some way.

        A basic interception would be

        .. code:: python

            @wraps(method)
            def func(*args, **kwargs):
                return method(*args, **kwargs)

        """

    def __getattr__(self, key):
        attr = getattr(self._obj, key)
        if callable(attr):
            return self.dispatch(attr)
        return attr


def _get_dispatch(dispatcher: AbstractDispatcher, key: Any):
    """Return the dispatch contained in `obj._dispatchs`"""
    dispatchs = dispatcher._dispatchs
    if isinstance(key, type) and key not in dispatchs:
        dispatch = find_implementation(key, dispatchs)
        # we will register for a faster look-up next time.
        dispatcher.register(key, dispatch)
    else:
        dispatch = dispatchs.get(key)
    if dispatch is None:
        raise KeyError(f"{dispatcher.__class__.__name__} has no dispatch for {key}.")
    return dispatch


class AbstractDispatcher(metaclass=ABCMeta):
    """A container for dispatchers

    This is an abstract class holding the dispatch classes (`AbstractDispatch`)
    and the attributes that are associated with the dispatchers.
    """

    def __init__(self, dispatchs=None, default=None, **attrs):
        if dispatchs is None:
            dispatchs = {}
        if not isinstance(dispatchs, ChainMap):
            dispatchs = ChainMap(dispatchs)
        # we will always use a chainmap to store the dispatches
        # We must not *copy*
        # It should be the same memory location in case we are
        # passing around the dispatch sequences
        self._dispatchs = dispatchs
        self._default = default
        self.__name__ = self.__class__.__name__
        # Attributes associated with the dispatcher
        self._attrs = attrs
        _log.debug(f"__init__ {self.__class__.__name__}", extra={"obj": self})

    def copy(self):
        """Create a copy of this object (making a new child for the dispatch lookup)"""
        _log.debug(f"copy {self.__class__.__name__}", extra={"obj": self})
        return self.__class__(self._dispatchs.new_child(), self._default, **self._attrs)

    def renew(self, **attrs):
        """Create a new class with updated attributes"""
        _log.debug(
            f"renew {self.__class__.__name__}{tuple(attrs.keys())}", extra={"obj": self}
        )
        return self.__class__(
            self._dispatchs, self._default, **{**self._attrs, **attrs}
        )

    def __len__(self):
        return len(self._dispatchs)

    def __repr__(self):
        ndispatchs = len(self._dispatchs)
        nattrs = len(self._attrs)
        return f"<{self.__name__}{{ndispatchs={ndispatchs}, nattrs={nattrs}}}>"

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
            if len(self._dispatchs) == 0:
                return f"{self.__name__}{{<empty>}}"
            return f"{self.__name__}{{{dispatchs}\n}}"
        return f"{self.__name__}{{{dispatchs},\n {attrs}\n}}"

    def __setitem__(self, key, dispatch):
        """Registers a dispatch method (using `register` with default values)

        Parameters
        ----------
        key : *any hashable*
            key used in the dictionary look-up for the dispatch class
        dispatch : AbstractDispatch
            dispatch class to be registered
        """
        self.register(key, dispatch)

    def __dir__(self):
        """Return instances belonging to this object"""
        return list(self._dispatchs.keys()) + ["renew", "register"]

    def register(self, key, dispatch, default=False, overwrite=True):
        """Register a dispatch class to this container

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
        _log.debug(
            f"register {self.__class__.__name__}(key: {key})", extra={"obj": self}
        )
        if key in self._dispatchs.maps[0] and not overwrite:
            raise LookupError(
                f"{self.__class__.__name__} already has {key} registered (and overwrite is false)"
            )
        self._dispatchs[key] = dispatch
        if default:
            self._default = key


class ErrorDispatcher(AbstractDispatcher):
    """Faulty handler to ensure that certain operations are not allowed

    This may for instance be used with ``ClassDispatcher(instance_dispatcher=ErrorDispatcher)``
    to ensure that a certain dispatch attribute will never be called on an instance.
    It won't work on type_dispatcher due to not being able to call `register`.
    """

    def __init__(self, obj, *args, **kwargs):  # pylint: disable=W0231
        raise ValueError(
            f"Dispatcher on {obj} must not be called in this way, see documentation."
        )


class MethodDispatcher(AbstractDispatcher):
    def __init__(self, method, dispatchs=None, default=None, obj=None, **attrs):
        super().__init__(dispatchs, default, **attrs)
        update_wrapper(self, method)

        # In python3 a method *always* have the __self__ key
        # In case the method is bound on a class.
        if obj is None:
            self._obj = getattr(method, "__self__", None)
        else:
            self._obj = obj

        _log.debug(f"__init__ {self.__class__.__name__}", extra={"obj": self})

    def copy(self):
        """Create a copy of this object (making a new child for the dispatch lookup)"""
        _log.debug(f"copy {self.__class__.__name__}", extra={"obj": self})
        return self.__class__(
            self.__wrapped__,
            self._dispatchs.new_child(),
            self._default,
            self._obj,
            **self._attrs,
        )

    def renew(self, **attrs):
        """Create a new class with updated attributes"""
        _log.debug(f"renew {self.__class__.__name__}", extra={"obj": self})
        return self.__class__(
            self.__wrapped__,
            self._dispatchs,
            self._default,
            self._obj,
            **{**self._attrs, **attrs},
        )

    def __call__(self, *args, **kwargs):
        _log.debug(f"call {self.__class__.__name__}{args}", extra={"obj": self})
        if self._default is None:
            return self.__wrapped__(*args, **kwargs)
        return self._dispatchs[self._default](self._obj, **self._attrs).dispatch(
            self.__wrapped__
        )(*args, **kwargs)

    def __getitem__(self, key):
        r"""Get method using dispatch according to `key`"""
        _log.debug(
            f"__getitem__ {self.__class__.__name__},key={key}", extra={"obj": self}
        )
        dispatch = _get_dispatch(self, key)
        return dispatch(self._obj, **self._attrs).dispatch(self.__wrapped__)

    __getattr__ = __getitem__


def _parse_obj_getattr(func):
    """Parse `func` for all methods"""
    if func is None:
        # return common handler
        return getattr
    elif isinstance(func, str):
        # One can make getattr fail regardless of what to fetch from
        # the object
        if func == "error":

            def func(obj, key):
                raise AttributeError(
                    f"{obj} does not implement the '{key}' dispatcher, "
                    "are you using it incorrectly?"
                )

            return func
        raise NotImplementedError(
            f"Defaulting the obj_getattr argument only accepts [error], got {func}."
        )
    return func


class ObjectDispatcher(AbstractDispatcher):
    """A dispatcher relying on object lookups

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

    def __init__(
        self,
        obj,
        dispatchs=None,
        default=None,
        attr_name=None,
        obj_getattr=None,
        **attrs,
    ):
        super().__init__(dispatchs, default, **attrs)
        self._obj = obj
        self._obj_getattr = _parse_obj_getattr(obj_getattr)
        self._attr_name = attr_name
        _log.debug(f"__init__ {self.__class__.__name__}", extra={"obj": self})

    def copy(self):
        """Create a copy of this object (making a new child for the dispatch lookup)"""
        _log.debug(f"copy {self.__class__.__name__}", extra={"obj": self})
        return self.__class__(
            self._obj,
            dispatchs=self._dispatchs.new_child(),
            default=self._default,
            attr_name=self._attr_name,
            obj_getattr=self._obj_getattr,
            **self._attrs,
        )

    def renew(self, **attrs):
        """Create a new class with updated attributes"""
        _log.debug(f"renew {self.__class__.__name__}", extra={"obj": self})
        return self.__class__(
            self._obj,
            dispatchs=self._dispatchs,
            default=self._default,
            attr_name=self._attr_name,
            obj_getattr=self._obj_getattr,
            **{**self._attrs, **attrs},
        )

    def __call__(self, obj, *args, **kwargs):
        _log.debug(
            f"call {self.__class__.__name__}{{obj={obj!s}}}", extra={"obj": self}
        )
        return self[obj](*args, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__}{{obj={self._obj!r}}}>"

    def __str__(self):
        obj = str(self._obj).replace("\n", "\n ")
        # super() returns the super class, not the super-instance.
        # Hence we need to call the explicit function
        return super().__str__().replace("{", f"{{\n {obj},\n ", 1)

    def register(self, key, dispatch, default=False, overwrite=True, to_class=True):
        """Register a dispatch class to this object and to the object class instance (if existing)

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
        _log.debug(
            f"register {self.__class__.__name__}(key: {key})", extra={"obj": self}
        )
        super().register(key, dispatch, default, overwrite)
        if to_class:
            cls_dispatch = getattr(self._obj.__class__, self._attr_name, None)
            if isinstance(cls_dispatch, ClassDispatcher):
                cls_dispatch.register(key, dispatch, overwrite=overwrite)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __getitem__(self, key):
        r"""Retrieve dispatched dispatchs by hash (allows functions to be dispatched)"""
        _log.debug(
            f"__getitem__ {self.__class__.__name__},key={key}", extra={"obj": self}
        )
        dispatch = _get_dispatch(self, key)
        return dispatch(self._obj, **self._attrs)

    def __getattr__(self, key):
        """Retrieve dispatched method by name, or if the name does not exist return a MethodDispatcher"""
        # Attribute retrieval will never be a class, so this will be directly
        # inferable in the dictionary.
        if key in self._dispatchs:
            _log.debug(
                f"__getattr__ {self.__class__.__name__},dispatch={key}",
                extra={"obj": self},
            )
            return self._dispatchs[key](self._obj, **self._attrs)

        attr = self._obj_getattr(self._obj, key)
        if callable(attr):
            _log.debug(
                f"__getattr__ {self.__class__.__name__},method-dispatch={key}",
                extra={"obj": self},
            )
            # This will also ensure that if the user calls immediately after it will use the default
            return MethodDispatcher(
                attr, self._dispatchs, self._default, self._obj, **self._attrs
            )
        _log.debug(
            f"__getattr__ {self.__class__.__name__},method={key}", extra={"obj": self}
        )
        return attr


class TypeDispatcher(ObjectDispatcher):
    """A dispatcher relying on type lookups

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

    def register(self, key, dispatch, default=False, overwrite=True, to_class=True):
        """Register a dispatch class to this object and to the object class instance (if existing)

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
        _log.debug(
            f"register {self.__class__.__name__}(key: {key})", extra={"obj": self}
        )
        super().register(key, dispatch, default, overwrite, to_class=False)
        if to_class:
            cls_dispatch = getattr(self._obj, self._attr_name, None)
            if isinstance(cls_dispatch, ClassDispatcher):
                cls_dispatch.register(key, dispatch, overwrite=overwrite)

    def __call__(self, obj: Any, *args, **kwargs) -> Any:
        # A call on a TypeDispatcher forces at least a single argument
        # where the type is being dispatched.

        # Figure out if obj is a class or not
        if isinstance(obj, type):
            typ = obj
        else:
            # If not, then get the type (basically same as obj.__class__)
            typ = type(obj)

        # if you want obj to be a type, then the dispatcher should control that
        _log.debug(f"call {self.__class__.__name__}{args}", extra={"obj": self})
        dispatch = _get_dispatch(self, typ)
        return dispatch(self._obj)(obj, *args, **kwargs)


class ClassDispatcher(AbstractDispatcher):
    """A dispatcher for classes, using `__get__` it converts into `ObjectDispatcher` upon invocation from an object, or a `TypeDispatcher` when invoked from a class

    This is a class-placeholder allowing a dispatcher to be a class attribute and converted into an
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

    def __init__(
        self,
        attr_name,
        dispatchs=None,
        default=None,
        obj_getattr=None,
        instance_dispatcher=ObjectDispatcher,
        type_dispatcher=TypeDispatcher,
        **attrs,
    ):
        # obj_getattr is necessary for the ObjectDispatcher to create the correct
        # MethodDispatcher
        super().__init__(dispatchs, default, **attrs)
        # the name of the ClassDispatcher attribute in the class
        self._attr_name = attr_name
        p = namedtuple("get_class", ["instance", "type"])
        self._get = p(instance_dispatcher, type_dispatcher)

        self._obj_getattr = _parse_obj_getattr(obj_getattr)
        _log.debug(f"__init__ {self.__class__.__name__}", extra={"obj": self})

    def copy(self):
        """Create a copy of this object (making a new child for the dispatch lookup)"""
        _log.debug(f"copy {self.__class__.__name__}", extra={"obj": self})
        return self.__class__(
            self._attr_name,
            self._dispatchs.new_child(),
            self._default,
            self._obj_getattr,
            self._get.instance,
            self._get.type,
            **self._attrs,
        )

    def renew(self, **attrs):
        """Create a new class with updated attributes"""
        _log.debug(f"renew {self.__class__.__name__}", extra={"obj": self})
        return self.__class__(
            self._attr_name,
            self._dispatchs,
            self._default,
            self._obj_getattr,
            self._get.instance,
            self._get.type,
            **{**self._attrs, **attrs},
        )

    def __get__(self, instance, owner):
        """Class dispatcher retrieval

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
        _log.debug(
            f"__get__ {self.__class__.__name__},instance={instance!r},inst={inst!r},owner={owner!r},cls={cls!r}",
            extra={"obj": self},
        )
        if cls is None:
            return self
        return cls(
            inst,
            self._dispatchs,
            default=self._default,
            attr_name=self._attr_name,
            obj_getattr=self._obj_getattr,
            **self._attrs,
        )


'''
For use when doing cached dispatchers
class CachedClassDispatcher(ClassDispatcher):

    def __init__(self, name, dispatchs=None, default=None, obj_getattr=None, **attrs):
        # obj_getattr is necessary for the ObjectDispatcher to create the correct
        # MethodDispatcher
        super().__init__(dispatchs, default, **attrs)
        self._obj_getattr = _parse_obj_getattr(obj_getattr)
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
                                attr_name=self._attr_name,
                                obj_getattr=self._obj_getattr,
                                **self._attrs)
        object.__setattr__(instance, self._attr_name, dispatcher)
        return dispatcher
'''
