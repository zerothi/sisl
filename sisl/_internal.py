r""" Internal sisl-only methods that should not be used outside """
import sys

# override module level, inspired by numpy

__all__ = ["set_module", "singledispatchmethod"]


def set_module(module):
    r"""Decorator for overriding __module__ on a function or class"""
    def deco(f_or_c):
        if module is not None:
            f_or_c.__module__ = module
        return f_or_c
    return deco


# Wrap functools singledispatchmethod
# see here:
#   https://github.com/ikalnytskyi/singledispatchmethod/blob/master/src/singledispatchmethod.py
if sys.version_info[1] >= 8:
    from functools import singledispatchmethod
else:
    from functools import singledispatch, update_wrapper

    class singledispatchmethod(object):
        """Single-dispatch generic method descriptor."""

        def __init__(self, func):
            if not callable(func) and not hasattr(func, "__get__"):
                raise TypeError("{!r} is not callable or a descriptor".format(func))

            self.dispatcher = singledispatch(func)
            self.func = func

        def register(self, cls, method=None):
            return self.dispatcher.register(cls, func=method)

        def __get__(self, obj, cls):
            def _method(*args, **kwargs):
                method = self.dispatcher.dispatch(args[0].__class__)
                return method.__get__(obj, cls)(*args, **kwargs)

            _method.__isabstractmethod__ = self.__isabstractmethod__
            _method.register = self.register
            update_wrapper(_method, self.func)
            return _method

        @property
        def __isabstractmethod__(self):
            return getattr(self.func, "__isabstractmethod__", False)
