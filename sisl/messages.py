""" Module to expose messages to users

These routines implement a layer of interaction with the user.
The warning routines and error handling should be passed through these
routines.

More specifically we do not like the *very* verbose warnings issued through

>>> import warnings
>>> warnings.warn('Help!')
__main__:1: UserWarning: Help!
>>> warnings.showwarning(SislWarning('Help!'), SislWarning, 'W', 'sisl', line='')
:sisl: UserWarning: Help!
>>> warnings.showwarning(UserWarning('Help!'), Warning, 'W', 'sisl', line='')
:sisl: UserWarning: Help!

We prefer the later which is particularly useful when the installation path is
complex.
"""
import warnings


__all__ = ['SislDeprecation', 'SislInfo', 'SislWarning', 'SislException', 'SislError']
__all__ += ['warn', 'info']

# The local registry for warnings issued
_sisl_warn_registry = {}


class SislException(Exception):
    """ Sisl exception """
    pass


class SislError(SislException):
    """ Sisl error """
    pass


class SislWarning(SislException, UserWarning):
    """ Sisl warnings """
    pass


class SislDeprecation(SislWarning):
    """ Sisl deprecation informations """
    pass


class SislInfo(SislWarning):
    """ Sisl informations """
    pass


def deprecate(message):
    """ Issue sisl deprecation warnings

    Parameters
    ----------
    message: str
    """
    warnings.warn_explicit(message, SislDeprecation, 'dep', 0, registry=_sisl_warn_registry)


def warn(message, category=None):
    """ Show warnings in short context form with sisl

    Parameters
    ----------
    message: str, Warning
       the warning to issue, default to issue a `SislWarning`
    category: Warning, optional
       the category of the warning to issue. Default to `SislWarning', unless `message` is
       a subclass of `Warning`
    """
    if isinstance(message, Warning):
        category = message.__class__
    elif category is None:
        category = SislWarning
    warnings.warn_explicit(message, category, 'warn', 0, registry=_sisl_warn_registry)


def info(message, category=None):
    """ Show info in short context form with sisl

    Parameters
    ----------
    message: str, Warning
       the information to issue, default to issue a `SislInfo`
    category: Warning, optional
       the category of the warning to issue. Default to `SislInfo', unless `message` is
       a subclass of `Warning`
    """
    if isinstance(message, Warning):
        category = message.__class__
    elif category is None:
        category = SislInfo
    warnings.warn_explicit(message, category, 'info', 0, registry=_sisl_warn_registry)
