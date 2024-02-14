# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import singledispatch
from textwrap import dedent
from typing import Callable, Optional

from sisl.messages import SislError, warn
from sisl.typing import FuncType

__all__ = ["register_sisl_function"]


# Internal dictionary holding names of functions that
# are the wrapper functions that can actually be used for
# registrering.
# This ensures that the method
_registry = {}


def register_sisl_function(name: str, cls: type, module: Optional[str] = None):
    """Decorate a function and hook it into the {cls} for calling it directly"""

    def decorator(func: FuncType):
        nonlocal name, cls, module

        if module is None:
            module = cls.__module__
        func.__module__ = module

        if callable(name):
            name = name(func.__name__)

        old_method = getattr(cls, name, None)
        if old_method is not None:
            # Check that the attribute is actually created on the class it-self
            # This will prohibit warn against functions for derived classes
            if old_method.__qualname__.startswith(cls.__name__):
                warn(
                    f"registration of function {repr(func)} under name "
                    f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                    f"attribute with the same name."
                )

        setattr(cls, name, func)
        if not hasattr(cls, "_funcs"):
            cls._funcs = set()
        cls._funcs.add(name)
        return func

    return decorator


def _append_doc_dispatch(method: FuncType, cls: type):
    """Append to the doc-string of the dispatch method retrieved by `method` that the `cls` class can be used"""
    global _registry

    # get method name
    name = method.__name__

    # retrieve dispatch method
    method_registry = _registry[name]

    # Append to doc string
    doc = f"\n{cls.__name__}.{name}: equivalent to ``{name}({cls.__name__.lower()}, ...)``."
    method_registry.__doc__ += doc


def register_sisl_dispatch(
    cls: Optional[type] = None,
    *,
    cls_name: Optional[str] = None,
    module: Optional[str] = None,
):
    """Create a new dispatch method from a method

    If the method has not been registrered before, it will be created.
    """
    global _registry

    def deco(method: FuncType):
        nonlocal cls, cls_name, module

        name = method.__name__
        if cls_name is None:
            cls_name = name
        if name not in _registry:
            # create a new method that will be stored
            # as a place-holder for the dispatch methods.

            def method_registry(obj, *args, **kwargs):
                raise SislError(
                    f"Calling '{name}' with a non-registered type, {type(obj)} has not been registered."
                )

            doc = dedent(
                f"""\
                    Dispatcher for '{name}'

                    See also
                    --------
                    """
            )
            method_registry.__doc__ = doc
            method_registry.__name__ = name
            method_registry.__module__ = "sisl"
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
    module: str
        only register methods that belongs to `module`
    """
    global _registry
    import importlib

    # import the module
    mod = importlib.import_module(module)
    for name, method in _registry.items():
        if method.__module__ == module:
            setattr(mod, name, method)
