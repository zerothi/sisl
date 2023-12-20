# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from sisl.messages import warn

__all__ = ["register_sisl_function"]


def register_sisl_function(name, cls, module=None):
    """Decorate a function and hook it into the {cls} for calling it directly"""

    def decorator(func):
        nonlocal name, cls, module

        # TODO add check that the first argument of `func` is proper class (for self)
        if module is not None:
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
        cls._funcs.add(name)
        return func

    return decorator
