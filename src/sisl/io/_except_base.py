# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps
from typing import Optional, Union

from sisl._internal import set_module
from sisl.messages import SislError, SislException, SislInfo, SislWarning, info, warn

__all__ = [
    "SileError",
    "SileWarning",
    "SileInfo",
    "MissingInputSileException",
    "MissingInputSileError",
    "MissingInputSileInfo",
    "MissingInputSileWarning",
    "missing_input",
]


@set_module("sisl.io")
class SileError(SislError, IOError):
    """Define an error object related to the Sile objects"""

    def __init__(self, value, obj=None):
        self.value = value
        self.obj = obj

    def __str__(self):
        if self.obj:
            return f"{self.value!s} in {self.obj!s}"
        return self.value


@set_module("sisl.io")
class SileWarning(SislWarning):
    """Warnings that informs users of things to be carefull about when using their retrieved data

    These warnings should be issued whenever a read/write routine is unable to retrieve all information
    but are non-influential in the sense that sisl is still able to perform the action.
    """


@set_module("sisl.io")
class SileInfo(SislInfo):
    """Information for the user, this is hidden in a warning, but is not as severe so as to issue a warning."""


InputsType = Optional[Union[list[tuple[str, str]], list[str], str]]


class MissingInputSileException(SislException):
    """Container for constructing error/warnings when a fdf flag is missing from the input file.

    This error message should preferably be raised through:

    >>> raise InheritedClass(method, ["Diag.ParallelOverk"]) from exc

    to retain a proper context.
    """

    def __init__(
        self, executable: str, inputs: InputsType, method: Callable, msg: str = ""
    ):
        # Formulate the error message
        try:
            name = f"{method.__self__.__class__.__name__}.{method.__name__}"
        except AttributeError:
            name = f"{method.__name__}"

        if isinstance(inputs, str):
            inputs = [inputs]

        def parse(v):
            if isinstance(v, str):
                return f"  * {v}"
            return f"  * {' '.join(v)}"

        str_except = ""
        if inputs is not None:
            str_except = "\n".join(map(parse, inputs))

        super().__init__(
            f"{msg}\nData from method '{name}' failed due to missing output values.\n\n"
            f"This is because of missing options in the input file for executable {executable}.\n"
            f"Please read up on the following flags in the manual of '{executable}' to figure out "
            f"how to retrieve the expected quantities:\n{str_except}"
        )


class MissingInputSileError(MissingInputSileException, SileError):
    """Issued when specific flags in the input file can be used to extract data"""


class MissingInputSileWarning(MissingInputSileException, SileWarning):
    """Issued when specific flags in the input file can be used to extract data"""


class MissingInputSileInfo(MissingInputSileException, SileInfo):
    """Issued when specific flags in the input file can be used to extract data"""


def missing_input(
    executable: str,
    inputs: InputsType,
    what: MissingInputSileException,
    when_exception: Exception = Exception,
):
    """Issues warnings, or errors based on `when` and `what`"""

    def decorator(func):
        what_inst = what(executable, inputs, func)

        if issubclass(when_exception, Warning):
            # we should do a warning catch thing
            @wraps(func)
            def deco(*args, **kwargs):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    ret = func(*args, **kwargs)

                if len(w) > 0:
                    if isinstance(what_inst, MissingInputSileWarning):
                        warn(what_inst)
                    elif isinstance(what_inst, MissingInputSileInfo):
                        info(what_inst)
                    elif isinstance(what_inst, MissingInputSileError):
                        raise what_inst

                return ret

        else:
            # it must be ane error to be raised
            @wraps(func)
            def deco(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except what as ke:
                    raise what_inst from ke.__cause__
                except when_exception as ke:
                    raise what_inst from ke

        return deco

    return decorator
