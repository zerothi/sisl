# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

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
from functools import wraps

from ._environ import get_environ_variable
from ._help import has_module
from ._internal import set_module

__all__ = ["SislDeprecation", "SislInfo", "SislWarning", "SislException", "SislError"]
__all__ += ["warn", "info", "deprecate", "deprecation", "deprecate_argument"]
__all__ += ["progressbar", "tqdm_eta"]

# The local registry for warnings issued
_sisl_warn_registry = {}


@set_module("sisl")
class SislException(Exception):
    """Sisl exception"""


@set_module("sisl")
class SislError(SislException):
    """Sisl error"""


@set_module("sisl")
class SislWarning(SislException, UserWarning):
    """Sisl warnings"""


@set_module("sisl")
class SislDeprecation(SislWarning, FutureWarning):
    """Sisl deprecation warnings for end-users"""


@set_module("sisl")
class SislInfo(SislWarning):
    """Sisl informations"""


@set_module("sisl")
def deprecate(message, from_version=None, remove_version=None):
    """Issue sisl deprecation warnings

    Parameters
    ----------
    message : str
       the displayed message
    from_version : optional
       which version this deprecation was introduced
    remove_version : optional
       which version of sisl this will be removed
    """
    if from_version is not None:
        message = f"{message} [>={from_version}]"
    if remove_version is not None:
        message = f"{message} [removed in {remove_version}]"
    warnings.warn_explicit(
        message, SislDeprecation, "dep", 0, registry=_sisl_warn_registry
    )


@set_module("sisl")
def deprecate_argument(old, new, message, from_version=None, remove_version=None):
    """Decorator for deprecating `old` argument, and replacing it with `new`

    The old keyword argument is still retained.

    If `new` is none, it will be deleted.
    """

    def deco(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if old in kwargs:
                deprecate(
                    f"{func.__name__} {message}",
                    from_version=from_version,
                    remove_version=remove_version,
                )
                if new is not None:
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapped

    return deco


@set_module("sisl")
def deprecation(message, from_version=None, remove_version=None):
    """Decorator for deprecating a method or a class

    Parameters
    ----------
    message : str
       message displayed
    from_version : optional
       which version this deprecation was introduced
    remove_version : optional
       which version of sisl this will be removed
    """

    def install_deprecate(cls_or_func):
        if isinstance(cls_or_func, type):
            # we have a class
            class wrapped(cls_or_func):
                @deprecation(message, from_version, remove_version)
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

        else:

            @wraps(cls_or_func)
            def wrapped(*args, **kwargs):
                deprecate(message, from_version, remove_version)
                return cls_or_func(*args, **kwargs)

        return wrapped

    return install_deprecate


@set_module("sisl")
def warn(message, category=None, register=False):
    """Show warnings in short context form with sisl

    Parameters
    ----------
    message : str, Warning
       the warning to issue, default to issue a `SislWarning`
    category : Warning, optional
       the category of the warning to issue. Default to `SislWarning', unless `message` is
       a subclass of `Warning`
    register : bool, optional
       whether the warning is registered to limit the number of times this is output
    """
    if isinstance(message, Warning):
        category = message.__class__
    elif category is None:
        category = SislWarning
    if register:
        warnings.warn_explicit(
            message, category, "warn", 0, registry=_sisl_warn_registry
        )
    else:
        warnings.warn_explicit(message, category, "warn", 0)


@set_module("sisl")
def info(message, category=None, register=False):
    """Show info in short context form with sisl

    Parameters
    ----------
    message : str, Warning
       the information to issue, default to issue a `SislInfo`
    category : Warning, optional
       the category of the warning to issue. Default to `SislInfo', unless `message` is
       a subclass of `Warning`
    register : bool, optional
       whether the information is registered to limit the number of times this is output
    """
    if isinstance(message, Warning):
        category = message.__class__
    elif category is None:
        category = SislInfo
    if register:
        warnings.warn_explicit(
            message, category, "info", 0, registry=_sisl_warn_registry
        )
    else:
        warnings.warn_explicit(message, category, "info", 0)


# https://stackoverflow.com/a/39662359/827281
def is_jupyter_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


# Figure out if we can import tqdm.
# If so, simply use the progressbar class there.
# Otherwise, create a fake one.
if has_module("tqdm"):
    if is_jupyter_notebook():
        from tqdm.notebook import tqdm as _tqdm
    else:
        from tqdm import tqdm as _tqdm
else:
    # Necessary methods used
    from sys import stdout as _stdout
    from time import time as _time

    class _tqdm:
        """Fake tqdm progress-bar. I should update this to also work in regular instances"""

        __slots__ = ["total", "desc", "t0", "n", "l"]

        def __init__(self, total, desc, unit):
            # Only inform them when we hit this
            info(
                "Please install tqdm (pip install tqdm) for better looking progress bars",
                register=True,
            )
            self.total = total
            self.desc = desc
            _stdout.write(f"{self.desc}  ETA = ?????h ??m ????s\r")
            _stdout.flush()
            self.t0 = _time()
            self.n = 0
            self.l = total

        def update(self, n=1):
            self.n += n
            self.l -= n
            m, s = divmod((_time() - self.t0) / self.n * self.l, 60)
            h, m = divmod(m, 60)
            _stdout.write(f"{self.desc}  ETA = {int(h):5d}h {int(m):2d}m {s:4.1f}s\r")
            _stdout.flush()

        def close(self):
            m, s = divmod(_time() - self.t0, 60)
            h, m = divmod(m, 60)
            _stdout.write(
                f"{self.desc} finished after {int(h):d}h {int(m):d}m {s:.1f}s\r"
            )
            _stdout.flush()


_default_eta = get_environ_variable("SISL_SHOW_PROGRESS")


@set_module("sisl")
def progressbar(total, desc, unit, eta, **kwargs):
    """Create a progress bar in when it is requested. Otherwise returns a fake object

    Parameters
    ----------
    total : int
       number of total iterations
    desc : str
       description on the stdout when running the progressbar
    unit : str
       unit shown in the progressbar
    eta : bool or str or None
       if True a progressbar is returned (default from ``tqdm``). Else a fake instance is returned.
       If a str, that will be used as the description.
       If None, use SISL_SHOW_PROGRESS environment variable as the value

    Returns
    -------
    object
       progress bar if `eta` is true, otherwise an object which does nothing
    """
    if eta is None:
        eta = _default_eta
    if eta:
        if isinstance(eta, str):
            desc = eta
        bar = _tqdm(total=total, desc=desc, unit=unit, **kwargs)
    else:
        # Since the eta bar is not needed we simply create a fake object which
        # has the required 2 methods, update and close.
        class Fake:
            __slots__ = []

            def update(self, n=1):
                pass

            def close(self):
                pass

        bar = Fake()
    return bar
