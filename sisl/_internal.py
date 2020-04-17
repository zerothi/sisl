r""" Internal sisl-only methods that should not be used outside """

# override module level, inspired by numpy
def set_module(module):
    r"""Decorator for overriding __module__ on a function or class"""
    def deco(f_or_c):
        if module is not None:
            f_or_c.__module__ = module
        return f_or_c
    return deco
