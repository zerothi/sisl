# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import inspect
import types
from functools import singledispatch
from textwrap import dedent
from typing import Any, Optional

from sisl._lib._docscrape import FunctionDoc
from sisl.messages import SislError, warn
from sisl.typing import FuncType

__all__ = ["register_sisl_function"]


# Internal dictionary holding names of functions that
# are the wrapper functions that can actually be used for
# registrering.
# This ensures that the method
_registry = {}


def register_sisl_function(name: str, cls: type, module: Optional[str] = None):
    """Decorate a function and hook it into the `cls` for calling it directly"""

    def decorator(func: FuncType):
        nonlocal name, cls, module

        if module is None:
            module = cls.__module__
        func.__module__ = module

        if callable(name):
            name = name(func.__name__)

        old_method = getattr(cls, name, None)
        if old_method is not None:
            # Check if it is abstract, in which case we can overwrite it
            if getattr(old_method, "__isabstractmethod__", False):
                # If it is abstract, allow to overwrite it!
                old_method = None

        if old_method is not None:
            # Check that the attribute is actually created on the class it-self
            # This will prohibit warn against functions for derived classes
            if old_method.__qualname__.startswith(cls.__name__):
                warn(
                    f"registration of function {repr(func)} under name "
                    f"{repr(name)} for type {repr(cls)} is overriding a "
                    "pre-existing "
                    "attribute with the same name."
                )

        # Patch the parameter list by removing the first
        # entry. That is the class instance
        fdoc = FunctionDoc(func, role="meth")
        sig = inspect.signature(func)

        # Since sometimes we don't have the object documented
        # we will have to check whether the signature is the same as the
        # parameters list.
        sig_params = [p for p in sig.parameters.values()]

        # Check if the documentation has an entry:
        if len(fdoc["Parameters"]) > 0:
            if sig_params[0].name == fdoc["Parameters"][0].name:
                fdoc["Parameters"] = fdoc["Parameters"][1:]

        # Remove the first signature
        sig_params[0] = sig_params[0].replace(
            name="self", annotation=inspect.Parameter.empty
        )
        sig = sig.replace(parameters=sig_params)

        # Replace signature and documentation
        # We can't replace the doc since it isn't compatible with the
        # documentation style.
        # I.e. the numpydoc converts it to include rst stuff which isn't parseable.
        # This becomes problematic as it produces completely unreadable documentation
        # func.__doc__ = str(fdoc).strip()
        # This will also change the signature specification
        # in the overload method. That is not desired... :(
        # func.__signature__ = sig

        # Assign the function
        setattr(cls, name, func)
        if not hasattr(cls, "_funcs"):
            cls._funcs = set()
        cls._funcs.add(name)

        return func

    return decorator


def _append_doc_dispatch(
    method: FuncType, cls: type, module: Optional[str] = None
) -> None:
    """Append to the doc-string of the dispatch method retrieved by `method` that the `cls` class can be used"""
    global _registry

    # get method name
    name = method.__name__

    if module is None:
        module = cls.__module__

    # retrieve dispatch method
    method_registry = _registry[name]

    # Append to doc string
    doc = f"\n{module}.{cls.__name__}.{name} : equivalent to ``{name}({cls.__name__.lower()}, ...)``."
    method_registry.__doc__ += doc


def register_sisl_dispatch(
    cls: Optional[type] = None,
    *,
    cls_name: Optional[str] = None,
    module: Optional[str] = None,
):
    """Create a new dispatch method from a method

    If the method has not been registered before, it will be created.
    """
    global _registry

    def deco(method: FuncType):
        nonlocal cls, cls_name, module

        name = method.__name__
        if cls_name is None:
            cls_name = name

        if module is None and cls is not None:
            # default the module
            module = cls.__module__

        if name not in _registry:
            # create a new method that will be stored
            # as a place-holder for the dispatch methods.

            # The default method needs access to it-self.
            # In this way we can figure out if there are
            # some mechanism by which we can recover
            # a meaningful action.
            # I.e. this small hack will allow one to do this:
            #  @register_sisl_dispatch(Geometry)
            #  def func(geometry: Geometry, ...)
            #
            #  from ase import Atoms
            #  func(Atoms(...), ...)
            #
            # The reason is that func won't find any registered
            # classes under Atoms, and so it will run through
            # the registered classes, trying out `cls.new` for each
            # of them. And then re-call the function it self.

            def method_registry(obj: Any, *args, **kwargs):
                nonlocal name, method_registry

                # Obviously, the method has been called without finding the
                # correct dispatch method.
                def skip_builtins(items):
                    for cls, cls_func in items:
                        if not isinstance(cls, types.BuiltinFunctionType):
                            yield cls, cls_func

                # Whether to return a sisl object
                ret_sisl = kwargs.pop("ret_sisl", False)

                if ret_sisl:
                    # No matter what, we simply return the object
                    def parse_obj(cls, input_obj, output_obj):
                        return output_obj

                else:
                    # It depends, if a single value returned, or any value
                    # in a tuple return is a sisl_obj corresponding to `cls`
                    # then we'll convert that to the corresponding input object
                    def parse_obj(cls, input_obj, output_obj):
                        if isinstance(output_obj, tuple):
                            return tuple(
                                parse_obj(cls, input_obj, obj) for obj in output_obj
                            )
                        if isinstance(output_obj, cls):
                            # return object back-converted to the input_obj type
                            return output_obj.to(type(input_obj))
                        return output_obj

                for cls, cls_func in skip_builtins(method_registry.registry.items()):

                    # Try and get the conversion method
                    # Currently we'll only use `new` for consistency
                    # I don't know if this will work for other
                    # keys as well...
                    new = getattr(cls, "new", None)

                    if new is not None:
                        try:
                            # Try and convert to a sisl-compatible object
                            sisl_obj = cls.new(obj)
                            sisl_obj = cls_func(sisl_obj, *args, **kwargs)
                            return parse_obj(cls, obj, sisl_obj)

                        except KeyError:
                            pass

                raise SislError(
                    f"Calling '{name}' with a non-registered type, {type(obj)} has not "
                    "been registered, and cannot be converted to a sisl type."
                )

            doc = dedent(
                f"""\
                    Dispatcher for `{name}`

                    See Also
                    --------
                    """
            )
            method_registry.__doc__ = doc
            method_registry.__name__ = name
            method_registry.__module__ = module
            _registry[name] = singledispatch(method_registry)

        # Retrieve the dispatched method
        method_registry = _registry[name]
        # Retrieve old methods, so we can extract what to dispatch methods
        # too
        # We need to copy the keys list, otherwise it will be a weakref to the
        # items in the registry of the function
        keys = method_registry.registry.keys()
        old_registry = set(keys)
        method_registry.register(cls, method)
        new_registry = set(keys)

        for cls in new_registry - old_registry:
            _append_doc_dispatch(method, cls)
            register_sisl_function(cls_name, cls)(method)

        return method

    return deco


def expose_registered_methods(module: str):
    """Expose methods registered to the module `module`

    Parameters
    ----------
    module :
        Only register methods that belongs to `module`.
    """
    global _registry
    import importlib

    # import the module
    mod = importlib.import_module(module)
    for name, method in _registry.items():
        if method.__module__ == module:
            setattr(mod, name, method)
