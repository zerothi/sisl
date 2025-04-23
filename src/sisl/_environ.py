# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
from typing import Any, Union

__all__ = ["register_environ_variable", "get_environ_variable", "sisl_environ"]


# Local variable for retaining the variables, may be used for
# extroversion
SISL_ENVIRON = {}


@contextmanager
def sisl_environ(**environ):
    r"""Create a new context for temporary overwriting the sisl environment variables

    Parameters
    ----------
    environ : dict, optional
        the temporary environment variables that should be used in this context
    """
    global SISL_ENVIRON
    old = {}
    for key, value in environ.items():
        old[key] = SISL_ENVIRON[key]["value"]
        SISL_ENVIRON[key]["value"] = value
    yield  # nothing to yield
    for key in environ:
        SISL_ENVIRON[key]["value"] = old[key]


def register_environ_variable(
    name: str,
    default: Any,
    description: str = None,
    process: Callable[[Any], Any] = None,
):
    """Register a new global sisl environment variable.

    Parameters
    -----------
    name: str or list-like of str
        the name of the environment variable. Needs to
        be correctly prefixed with "SISL_".
    default: any, optional
        the default value for this environment variable
    description: str, optional
        a description of what this variable does.
    process : callable, optional
        a callable which will be used to post-process the value when retrieving
        it.

    Raises
    ------
    ValueError
       if `name` does not start with "SISL_"
    """
    if not name.startswith("SISL_"):
        raise ValueError("register_environ_variable: name should start with 'SISL_'")

    if process is None:

        def process(arg):
            return arg

    global SISL_ENVIRON

    if name in SISL_ENVIRON:
        raise NameError(f"register_environ_variable: name {name} already registered")

    SISL_ENVIRON[name] = {
        "default": default,
        "description": description,
        "process": process,
        "value": os.environ.get(name, default),
    }


def get_environ_variable(name: str):
    """Gets the value of a registered environment variable.

    Parameters
    -----------
    name: str
        the name of the environment variable.
    """
    variable = SISL_ENVIRON[name]
    return variable["process"](variable["value"])


def _abs_path(path: str):
    path = Path(path)
    return path.resolve()


def _float_or_int(value: Union[str, float, int]):
    value = float(value)
    # See if it is an integer
    if value == int(value):
        value = int(value)
    return value


register_environ_variable(
    "SISL_LOG_FILE",
    "",
    "Log file to write into. If empty, do not log.",
    process=_abs_path,
)

register_environ_variable(
    "SISL_LOG_LEVEL",
    "INFO",
    "Define the log level used when writing to the file. Should be importable from logging module.",
    lambda x: x.upper(),
)


register_environ_variable(
    "SISL_NUM_PROCS",
    1,
    "Maximum number of CPU's used for parallel computing (len(os.sched_getaffinity(0)) is a good guess)",
    process=int,
)


register_environ_variable(
    "SISL_PAR_CHUNKSIZE",
    0.1,
    "Default chunksize for parallel processing, can severely impact performance.",
    process=_float_or_int,
)

register_environ_variable(
    "SISL_TMP",
    ".sisl_tmp",
    "Path where temporary files should be stored",
    process=_abs_path,
)

register_environ_variable(
    "SISL_CONFIGDIR",
    Path.home() / ".config" / "sisl",
    "Directory where configuration files for sisl should be stored",
    process=_abs_path,
)

register_environ_variable(
    "SISL_FILES_TESTS",
    "_THIS_DIRECTORY_DOES_NOT_EXIST_",
    dedent(
        """\
                          Full path of the sisl/files folder.
                          Generally this is only used for tests and for documentations.
                          """
    ),
    process=_abs_path,
)

register_environ_variable(
    "SISL_SHOW_PROGRESS",
    "false",
    "Whether routines which can enable progress bars should show them by default or not.",
    process=lambda val: val and val.lower().strip() in ("1", "t", "true"),
)

register_environ_variable(
    "SISL_IO_DEFAULT",
    "",
    "The default DFT code for processing files, Siles will be compared with endswidth(<>).",
    process=lambda val: val.lower(),
)

register_environ_variable(
    "SISL_CODATA",
    "2022",
    "The CODATA year of units and constants used by sisl.",
    process=lambda val: val.lower(),
)
