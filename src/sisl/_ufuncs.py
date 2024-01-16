# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from functools import singledispatch, wraps
from textwrap import dedent
from typing import Callable, Optional

from sisl.messages import SislError, warn

__all__ = ["register_sisl_function"]


# Internal dictionary holding names of functions that
# are the wrapper functions that can actually be used for
# registrering.
# This ensures that the method
_registry = {}


def register_sisl_function(name, cls, module=None):
    """Decorate a function and hook it into the {cls} for calling it directly"""

    def decorator(func):
        nonlocal name, cls, module

        if module is None:
            module = cls.__module__
        func.__module__ = module

        if callable(name):
            name = name(func.__name__)

        if hasattr(cls, name):
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


def _append_doc_dispatch(method: Callable, cls: type):
    """Append to the doc-string of the dispatch method retrieved by `method` that the `cls` class can be used"""
    global _registry

    # get method name
    name = method.__name__

    # retrieve dispatch method
    method_registry = _registry[name]

    # Append to doc string
    doc = f"\n{cls.__name__}.{name}: when first argument is of type '{cls.__name__}'"
    method_registry.__doc__ += doc


def register_sisl_dispatch(
    module: Optional[str] = None, method: Optional[Callable] = None
):
    """Create a new dispatch method from a method

    If the method has not been registrered before, it will be created.
    """
    global _registry
    if method is None:
        if module is None:
            return lambda f: register_sisl_dispatch(method=f)
        if isinstance(module, str):
            return lambda f: register_sisl_dispatch(module, method=f)
        return register_sisl_dispatch(method=module)

    name = method.__name__
    if name not in _registry:

        def method_registry(*args, **kwargs):
            """Dispatch method"""
            raise SislError

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
    method_registry.register(method)
    new_registry = set(keys)

    for cls in new_registry - old_registry:
        _append_doc_dispatch(method, cls)
        register_sisl_function(name, cls)(method)

    return method


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
