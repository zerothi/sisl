# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import gzip
import inspect
import logging
import re
import tempfile
import zipfile
from collections.abc import Callable
from functools import wraps
from io import IOBase
from itertools import product
from operator import contains
from os.path import basename, splitext
from pathlib import Path
from textwrap import dedent, indent
from types import MethodType
from typing import Any, Literal, Optional, Union

from sisl._environ import get_environ_variable
from sisl._help import has_module
from sisl._internal import set_module
from sisl.messages import deprecate, info, warn
from sisl.utils.misc import str_spec

from . import _except_base, _except_objects
from ._except_base import *
from ._except_objects import *
from ._help import *
from ._zipfile import ZipPath

# Public used objects
__all__ = ["add_sile", "get_sile_class", "get_sile", "get_siles", "get_sile_rules"]

__all__ += [
    "BaseSile",
    "BaseBufferSile",
    "BufferSile",
    "BufferSileCDF",
    "BufferSileBin",
    "Sile",
    "SileCDF",
    "SileBin",
]
__all__.extend(_except_base.__all__)
__all__.extend(_except_objects.__all__)

# Decorators or sile-specific functions
__all__ += ["sile_fh_open", "sile_raise_write", "sile_raise_read"]

# Global container of all Sile rules
# This list of tuples is formed as
#  [('fdf', fdfSileSiesta, fdfSileSiesta),
#   ('fdf', ncSileSiesta, fdfSileSiesta)]
# [0] is the file suffix
# [1] is the base class that may be queried
# [2] is the actual class the file represents
# This enables one to add several files with the
# same extension and query it based on a sub-class
__sile_rules = []
__siles = []


class _sile_rule:
    """Internal data-structure to check whether a file is the same as this sile"""

    COMPARISONS = {
        "contains": contains,
        "in": contains,
        "endswith": str.endswith,
        "startswith": str.startswith,
    }

    __slots__ = ("cls", "case", "suffix", "gzip", "bases", "base_names")

    def __init__(self, cls, suffix, case=True, gzip=False):
        self.cls = cls
        self.case = case
        if case:
            self.suffix = suffix
        else:
            self.suffix = suffix.lower()
        self.gzip = gzip
        self.bases = self.build_bases()
        self.base_names = [c.__name__.lower() for c in self.bases]

    def __str__(self):
        s = ",\n ".join(map(lambda x: x.__name, self.bases))
        return (
            f"{self.cls.__name__}{{case={self.case}, "
            f"suffix={self.suffix}, gzip={self.gzip},\n {s}\n}}"
        )

    def __repr__(self):
        return (
            f"<{self.cls.__name__}, case={self.case}, "
            f"suffix={self.suffix}, gzip={self.gzip}>"
        )

    def build_bases(self):
        """Return a list of all classes that this file is inheriting from (except Sile, SileBin or SileCDF)"""
        children = list(self.cls.__bases__) + [self.cls]
        nl = -1
        while len(children) != nl:
            nl = len(children)
            # Remove baseclasses everybody have
            for obj in (object, BaseSile, Sile, SileBin, SileCDF):
                try:
                    i = children.index(obj)
                    children.pop(i)
                except Exception:
                    pass
            for child in list(children):  # ensure we have a copy for infinite loops
                for c in child.__bases__:
                    if c not in children:
                        children.append(c)

        return children

    def in_bases(self, base, method="contains"):
        """Whether any of the inherited bases compares with `base` in their class-name (lower-case sensitive)"""
        if base is None:
            return True
        elif isinstance(base, object):
            base = base.__name__.lower()
        comparison = self.COMPARISONS[method]
        for b in self.base_names:
            if comparison(b, base):
                return True
        return False

    def get_base(self, base, method="contains"):
        """Whether any of the inherited bases compares with `base` in their class-name (lower-case sensitive)"""
        if base is None:
            return None
        comparison = self.COMPARISONS[method]
        for bn, b in zip(self.base_names, self.bases):
            if comparison(bn, base):
                return b
        return None

    def in_class(self, base, method="contains"):
        """Whether any of the inherited bases compares with `base` in their class-name (lower-case sensitive)"""
        if base is None:
            return False
        comparison = self.COMPARISONS[method]
        return comparison(self.cls.__name__.lower(), base)

    def is_suffix(self, suffix):
        if not self.case:
            suffix = suffix.lower()
        # Now check names and (possibly gzip)
        my_suffix = self.suffix
        if suffix == my_suffix:
            return True
        if not self.gzip:
            return False
        return suffix == f"{my_suffix}.gz"

    def is_class(self, cls):
        if cls is None:
            return False
        return self.cls == cls

    def is_subclass(self, cls):
        if cls is None:
            return False
        return issubclass(self.cls, cls)


@set_module("sisl.io")
def add_sile(suffix, cls, case=True, gzip=False):
    """Add files to the global lookup table

    Public for attaching lookup tables for allowing
    users to attach files externally.

    Parameters
    ----------
    suffix : str
         The file-name suffix, it can be several file endings (.TBT.nc)
    cls : child of BaseSile
         An object that is associated with the respective file.
         It must be inherited from `BaseSile`.
    case : bool, optional
         Whether case sensitivity is applicable for determining file.
    gzip : bool, optional
         Whether files with ``.gz`` endings can be read.
         This option should only be given to files with ASCII text
         output.
    """
    global __sile_rules, __siles

    if not issubclass(cls, BaseSile):
        raise ValueError(f"Class {cls.__name__} must be a subclass of BaseSile!")

    # Only add pure suffixes...
    if suffix.startswith("."):
        suffix = suffix[1:]

    # If it isn't already in the list of
    # Siles, add it.
    if cls not in __siles:
        __siles.append(cls)

    # Add the rule of the sile to the list of rules.
    __sile_rules.append(_sile_rule(cls, suffix, case=case, gzip=gzip))


@set_module("sisl.io")
def get_sile_class(filename, *args, **kwargs):
    """Retrieve a class from the global lookup table via filename and the extension

    Parameters
    ----------
    filename : str
       the file to be quried for a correct file object.
       This file name may contain {<class-name>} which sets
       `cls` in case `cls` is not set.
       For instance:

          water.xyz

       will return an `xyzSile`.
    cls : class, optional
       In case there are several files with similar file-suffixes
       you may query the exact base-class that should be chosen.
       If there are several files with similar file-endings this
       function returns a random one.
    """
    global __sile_rules, __siles

    # This ensures that the first argument need not be cls
    cls = kwargs.pop("cls", None)

    # Split filename into proper file name and
    # the Specification of the type
    tmp_file, specification = str_spec(str(filename))

    # Now check whether we have a specific checker
    if specification is None:
        # ensure to grab nothing
        specification = get_environ_variable("SISL_IO_DEFAULT").strip()
    elif specification.strip() == "":
        specification = ""

    # extract which comparsion method
    if "=" in specification:
        method, cls_search = specification.split("=", 1)
        if "=" in cls_search:
            raise ValueError(
                f"Comparison specification currently only supports one level of comparison(single =); got {specification}"
            )
    else:
        method, cls_search = "contains", specification

    # searchable rules
    eligible_rules = []
    if cls is None and not cls_search is None:
        # cls has not been set, and fcls is found
        # Figure out if fcls is a valid sile, if not
        # do nothing (it may be part of the file name)
        # Which is REALLY obscure... but....)
        cls_searchl = cls_search.lower()
        for sr in __sile_rules:
            if sr.in_class(cls_searchl, method=method):
                eligible_rules.append(sr)

        if eligible_rules:
            # we have at least one eligible rule
            filename = tmp_file
        else:
            warn(
                f"Specification requirement of the file did not result in any found files: {specification}"
            )

    else:
        # search everything
        eligible_rules = __sile_rules

    if eligible_rules:
        if len(eligible_rules) == 1:
            return eligible_rules[0].cls
    else:
        # nothing has been found, this may meen that we need to search *everything*
        eligible_rules = __sile_rules

    try:
        # Create list of endings on this file
        f = basename(filename)
        end_list = []
        end = ""

        def try_methods(eligibles, prefixes=("read_",)):
            """return only those who can actually perform the read actions"""

            def has(keys):
                nonlocal prefixes
                has_keys = []
                for key in keys:
                    for prefix in prefixes:
                        if key.startswith(prefix):
                            has_keys.append(key)
                return has_keys

            outs = []
            for e in eligibles:
                attrs = has(e.cls.__dict__.keys())
                try:
                    sile = e.cls(filename)
                except Exception:
                    outs.append(e)
                    continue
                for attr in attrs:
                    try:
                        getattr(sile, attr)()
                        # if one succeeds, we will assume it is working
                        outs.append(e)
                        break
                    except Exception:
                        pass
            return outs

        # Check for files without ending, or that they are directly zipped
        lext = splitext(f)
        while len(lext[1]) > 0:
            end = f"{lext[1]}{end}"
            if end[0] == ".":
                end_list.append(end[1:])
            else:
                end_list.append(end)
            lext = splitext(lext[0])

        # We also check the entire file name
        end_list.append(f)
        # Reverse to start by the longest extension
        # (allows grid.nc extensions, etc.)
        end_list = list(reversed(end_list))

        def get_eligibles(end, rules):
            nonlocal cls
            eligibles = []
            for sr in rules:
                if sr.is_class(cls):
                    # class-specification has precedence
                    # This should only occur when the
                    # class-specification is exact (i.e. xyzSile)
                    return [sr]
                elif sr.is_suffix(end):
                    if sr.is_subclass(cls):
                        return [sr]
                    eligibles.append(sr)
            return eligibles

        # First we check for class AND file ending
        for end, rules in product(end_list, (eligible_rules, __sile_rules)):
            eligibles = get_eligibles(end, rules)
            # Determine whether we have found a compatible sile
            if len(eligibles) == 1:
                return eligibles[0].cls
            elif len(eligibles) > 1:
                workable_eligibles = try_methods(eligibles)
                if len(workable_eligibles) == 1:
                    return workable_eligibles[0].cls
                raise ValueError(
                    f"Cannot determine the exact Sile requested, multiple hits: {tuple(e.cls.__name__ for e in eligibles)}"
                )

        # Print-out error on which extensions it tried (and full filename)
        if len(end_list) == 1:
            ext_list = end_list
        else:
            ext_list = end_list[1:]
        raise NotImplementedError(
            f"Sile for file '{filename}' ({ext_list}) could not be found, "
            "possibly the file has not been implemented."
        )

    except Exception as e:
        raise e


@set_module("sisl.io")
def get_sile(file, *args, **kwargs):
    """Retrieve an object from the global lookup table via filename and the extension

    Internally this is roughly equivalent to ``get_sile_class(...)()``.

    When the file suffix is not recognized and you know which file-type it is
    it is recommended to get the file class from the known suffixes and use that
    class to construct the file object, see examples.

    Parameters
    ----------
    file : str or pathlib.Path
       the file to be quried for a correct `Sile` object.
       This file name may contain {<class-name>} which sets
       `cls` in case `cls` is not set.
       For instance ``get_sile("water.dat{xyzSile}")``
       will read the file ``water.dat`` using the `xyzSile` class.
    cls : class
       In case there are several files with similar file-suffixes
       you may query the exact base-less that should be chosen.
       If there are several files with similar file-endings this
       function returns a random one.

    Examples
    --------
    A file named ``water.dat`` contains the xyz-coordinates as the `xyzSile`.
    One can forcefully get the sile by:

    >>> obj = get_sile("water.dat{xyzSile}")

    Alternatively one can query the xyz file and use that class reference
    in future instantiations. This ensures a future proof method without
    explicitly typing the Sile object.

    >>> cls = get_sile_class("anyfile.xyz")
    >>> obj = cls("water.dat")
    >>> another_xyz = cls("water2.dat")

    To narrow the search one can clarify whether it should start or
    end with a string:

    >>> cls = get_sile_class("water.dat{startswith=xyz}")
    """
    cls = kwargs.pop("cls", None)
    sile = get_sile_class(file, *args, cls=cls, **kwargs)

    # Get the file path with the potential {specification}
    # removed from the end. However, if this is a zipfile
    # path we need to preserve the original zipfile.
    if isinstance(file, zipfile.Path):
        internal_path = Path(str(file)).relative_to(file.root.filename)
        clean_filename = str_spec(str(internal_path))[0]
        file = ZipPath(file.root, clean_filename)
    else:
        clean_filename = str_spec(str(file))[0]
        file = Path(clean_filename)

    return sile(file, *args, **kwargs)


@set_module("sisl.io")
def get_siles(attrs=None):
    """Retrieve all files with specific attributes or methods

    Parameters
    ----------
    attrs : list of attribute names
       limits the returned objects to those that have
       the given attributes ``hasattr(sile, attrs)``, default ``[None]``
    """
    global __siles

    if attrs is None:
        attrs = [None]

    if len(attrs) == 1 and attrs[0] is None:
        return list(__siles)

    siles = []
    for sile in __siles:
        for attr in attrs:
            try:
                if hasattr(sile, attr):
                    siles.append(sile)
                    break
            except TypeError as e:
                if str(e).startswith("[SileBinder]"):
                    # a non-bounded deferred class created using the SileBinder
                    # TODO work around this check
                    siles.append(sile)

    return siles


@set_module("sisl.io")
def get_sile_rules(attrs=None, cls=None):
    """
    Retrieve all sile rules of siles with specific attributes or methods

    Parameters
    ----------
    attrs : list of attribute names
       limits the returned objects to those that have
       the given attributes ``hasattr(sile, attrs)``, default ``[None]``
    """
    global __sile_rules

    if attrs is None and cls is None:
        return list(__sile_rules)

    sile_rules = []
    for rule in __sile_rules:
        sile = rule.cls
        if sile is cls:
            sile_rules.append(rule)
        elif attrs:
            for attr in attrs:
                if hasattr(sile, attr):
                    sile_rules.append(rule)
                    break

    return sile_rules


@set_module("sisl.io")
class BaseSile:
    """Base class for all sisl files"""

    def __new__(cls, filename, *args, **kwargs):
        # check whether filename is an actual str, or StringIO or some buffer
        if not isinstance(filename, IOBase):
            # this is just a regular sile opening
            return super().__new__(cls)

        return super().__new__(cls._buffer_cls)

    def __init_subclass__(cls, buffer_cls=None):
        if issubclass(cls, BaseBufferSile):
            # return since it already inherits BufferSile
            return

        buffer_extension_cls = getattr(cls, "_buffer_extension_cls", None)

        if buffer_extension_cls is None and buffer_cls is None:
            cls._buffer_cls = None
        else:
            if buffer_cls is None:
                buffer_cls = type(
                    f"{cls.__name__}Buffer",
                    (buffer_extension_cls, cls),
                    # Ensure the module is the same
                    {"__module__": cls.__module__},
                )
            elif not issubclass(buffer_cls, BaseBufferSile):
                raise TypeError(
                    f"The passed buffer_cls should inherit from sisl.io.BufferSile to "
                    "ensure correct behaviour."
                )

            cls._buffer_cls = buffer_cls

    def __init__(self, *args, **kwargs):
        """Just to pass away the args and kwargs"""

    def _sanitize_filename(self, filename) -> Union[Path, ZipPath]:
        """Sanitize the filename to be a ``Path`` or ``ZipPath`` object.

        If the filename is a ``zipfile.Path``, a ``ZipPath`` or a path with
        a .zip file in the middle of it, it will be converted to a ``ZipPath``.

        Otherwise, it will be converted to a ``Path`` object.

        Parameters
        ----------
        filename :
            The filename to be sanitized.
        """
        if isinstance(filename, zipfile.Path):
            filename = ZipPath.from_zipfile_path(filename)
        elif not isinstance(filename, ZipPath):
            filename = Path(filename)

            # Try to convert to a ZipPath, which will only succeed if there
            # is a .zip file in the middle of the path
            try:
                filename = ZipPath.from_path(filename, self._mode)
            except FileNotFoundError:
                pass

        return filename

    @property
    def file(self):
        """File of the current `Sile`"""
        return self._file

    @property
    def base_file(self):
        """File of the current `Sile`"""
        return basename(self._file)

    def base_directory(self, relative_to="."):
        """Retrieve the base directory of the file, relative to the path `relative_to`"""
        if isinstance(relative_to, str):
            relative_to = Path(relative_to)
        try:
            d = self._directory.relative_to(relative_to.resolve())
        except ValueError:  # in case they are not relative
            d = self._directory
        return d

    def dir_file(self, filename=None, filename_base=""):
        """File of the current `Sile`"""
        if filename is None:
            filename = Path(self._file).name
        return self._directory / filename_base / filename

    def read(self, *args, **kwargs):
        """Generic read method which should be overloaded in child-classes

        Parameters
        ----------
        kwargs :
          keyword arguments will try and search for the attribute ``read_<>``
          and call it with the remaining ``**kwargs`` as arguments.
        """
        for key in kwargs.keys():
            # Loop all keys and try to read the quantities
            if hasattr(self, f"read_{key}"):
                func = getattr(self, f"read_{key}")
                # Call read
                return func(kwargs[key], **kwargs)

    # Options for writing
    # The default routine for writing
    _write_default = None
    # Whether only the default should be used
    # when issuing a write
    _write_default_only = False

    def write(self, *args, **kwargs):
        """Generic write method which should be overloaded in child-classes

        Parameters
        ----------
        **kwargs :
          keyword arguments will try and search for the attribute `write_<>`
          and call it with the remaining ``**kwargs`` as arguments.
        """
        if self._write_default is not None:
            self._write_default(*args, **kwargs)
            if self._write_default_only:
                return

        for key in kwargs.keys():
            # Loop all keys and try to write the quantities
            if hasattr(self, f"write_{key}"):
                func = getattr(self, f"write_{key}")
                # Call write
                func(kwargs[key], **kwargs)

    def _setup(self, *args, **kwargs):
        """Setup the `Sile` after initialization

        Inherited method for setting up the sile.

        This method can be overwritten.
        """

    def _base_setup(self, *args, **kwargs):
        """Setup the `Sile` after initialization

        Inherited method for setting up the sile.

        This method **must** be overwritten *and*
        end with ``self._setup()``.
        """
        base = kwargs.get("base", None)
        if base is None:
            # Extract from filename
            self._directory = self._file.parent
        else:
            self._directory = base
        if not str(self._directory):
            self._directory = "."

        if isinstance(self._directory, (str, Path)):
            self._directory = self._sanitize_filename(Path(self._directory).resolve())

        self._setup(*args, **kwargs)

    def _base_file(self, f):
        """Make `f` refer to the file with the appropriate base directory"""
        return self._directory / f

    def __getattr__(self, name):
        """Override to check the handle"""
        if name == "fh":
            raise AttributeError(
                f"The filehandle for {self.file} has not been opened yet..."
            )
        if name == "read_supercell" and hasattr(self, "read_lattice"):
            deprecate(
                f"{self.__class__.__name__}.read_supercell is deprecated in favor of read_lattice",
                "0.15",
                "0.17",
            )
            return getattr(self, "read_lattice")
        if name == "write_supercell" and hasattr(self, "write_lattice"):
            deprecate(
                f"{self.__class__.__name__}.write_supercell is deprecated in favor of write_lattice",
                "0.15",
                "0.17",
            )
            return getattr(self, "write_lattice")
        return getattr(self.fh, name)

    @classmethod
    def _ArgumentParser_args_single(cls):
        """Default arguments for the Sile"""
        return {}

    # Define the custom ArgumentParser
    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that may be available for this Sile

        Parameters
        ----------
        p : ArgumentParser
           the argument parser to add the arguments to.
        """
        raise NotImplementedError(
            f"The ArgumentParser of '{self.__class__.__name__}' has not been implemented yet."
        )

    def ArgumentParser_out(self, p=None, *args, **kwargs):
        """Appends additional arguments based on the output of the file

        Parameters
        ----------
        p : ArgumentParser
           the argument parser to add the arguments to.
        """

    def _log(self, msg, *args, level=logging.INFO, **kwargs):
        """Provide a log message to the logging mechanism"""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(
                f"{self.__module__}.{self.__class__.__name__}|{self.base_file}"
            )
        logger = self._logger
        logger.log(level, msg, *args, **kwargs)

    def __str__(self):
        """Return a representation of the `Sile`"""
        # Check if the directory is relative to the current path
        # If so, only print the relative path, otherwise print the full path
        d = self.base_directory()
        return f"{self.__class__.__name__}({self.base_file!s}, base={d!s})"


def sile_fh_open(from_closed: bool = False, reset: Callable[[BaseSile], None] = None):
    """Method decorator for objects to directly implement opening of the
    file-handle upon entry (if it isn't already).

    Parameters
    ----------
    from_closed :
       ensure the wrapped function *must* open the file, otherwise it will seek to 0.
    reset :
       in case the file gets opened a new, then the `reset` method will be called as
       ``reset(self)``
    """
    if reset is None:

        def reset(self):
            pass

    if from_closed:

        def _wrapper(func):
            nonlocal reset

            @wraps(func)
            def pre_open(self, *args, **kwargs):
                with self:
                    # This happens when the file seeks to 0,
                    # so basically the same as re-opening the file
                    reset(self)
                    return func(self, *args, **kwargs)

            return pre_open

    else:

        def _wrapper(func):
            nonlocal reset

            @wraps(func)
            def pre_open(self, *args, **kwargs):
                if hasattr(self, "fh"):
                    return func(self, *args, **kwargs)
                with self:
                    reset(self)
                    return func(self, *args, **kwargs)

            return pre_open

    return _wrapper


class BaseBufferSile:

    def __getattribute__(self, name):

        attr = super().__getattribute__(name)

        if name.startswith("read_"):
            attr = self.wrap_read(attr, name)
        elif name.startswith("write_"):
            attr = self.wrap_write(attr, name)

        return attr

    def wrap_read(self, func: Callable, name: str):
        """Wrap the read function to ensure it is called with the correct file handle"""
        return func

    def wrap_write(self, func: Callable, name: str):
        """Wrap the write function to ensure it is called with the correct file handle"""
        return func


@set_module("sisl.io")
class BufferSile(BaseBufferSile):
    """Sile for handling `StringIO` and `TextIOBase` objects

    These are basically meant for users passing down the above objects
    """

    def __init__(self, filename, *args, **kwargs):
        # here, filename is actually a file-handle.
        # However, to accommodate keyword arguments we *must* have the same name
        filehandle = filename

        try:
            filename = Path(filehandle.name)
        except AttributeError:
            # this is not optimal, it will be the current directory, but one should not be able
            # to write to it
            filename = Path()

        try:
            mode = filehandle.mode
        except AttributeError:
            # a StringIO will always be able to read *and* write
            # to its buffer
            mode = "rw"

        self.fh = filehandle
        self._fh_init_tell = filehandle.tell()

        # pass mode to the file to let it know what happened
        # we can't use super here due to the coupling to the Sile class
        super().__init__(filename, mode, *args, **kwargs)

    def _open(self):
        self.fh.seek(self._fh_init_tell)
        self._line = 0

    def __exit__(self, type, value, traceback):
        # we will not close a file-handle
        self._line = 0
        return False

    def close(self):
        """Will not close the file since this is passed by the user"""


@set_module("sisl.io")
class BufferSileCDF(BaseBufferSile):

    def __init__(self, filename, mode="r", *args, **kwargs):
        # here, filename is actually a file-handle.
        # However, to accommodate keyword arguments we *must* have the same name
        filehandle = filename

        try:
            filename = Path(filehandle.name)
        except AttributeError:
            filename = Path("dummy")

        if isinstance(filehandle, zipfile.ZipExtFile) and mode == "r":
            # For some convoluted reason that I don't manage to understand, if
            # the buffer is a zipfile in reading mode and we have the filename
            # set to the real name, the read data will be completely wrong
            # (tests will fail if this line is commented out)
            filename = "dummy"

        # Remove the b from the mode, as netCDF4 only accepts "r" or "w"
        mode = mode.replace("b", "")

        if hasattr(filehandle, "mode") and filehandle.mode.replace("b", "") != mode:
            raise ValueError(
                f"The filehandle's mode ({filehandle.mode}) does not match the sile's mode ({mode})"
            )

        self._buffer = filehandle
        self._wrapped = False

        # pass mode to the file to let it know what happened
        # we can't use super here due to the coupling to the Sile class
        super().__init__(filename, mode, *args, **kwargs)

    def _open(self):

        if "fh" not in self.__dict__:
            self.__dict__["fh"] = netCDF4.Dataset(
                str(self.file),
                self._mode,
                format="NETCDF4",
                memory=self._buffer.read() if self._mode == "r" else 4,
            )

        return self

    def close(self):
        if self._mode.startswith("w"):
            self._buffer.write(self.fh.close().tobytes())

    def wrap_write(self, func: Callable, name: str):
        if self._wrapped:
            return func

        @wraps(func)
        def _wrapped(*args, **kwargs):
            res = func(*args, **kwargs)
            self.close()
            self._wrapped = False
            return res

        self._wrapped = True

        return _wrapped


@set_module("sisl.io")
class BufferSileBin(BaseBufferSile):

    def __init__(self, filename, *args, mode="r", close: bool = False, **kwargs):
        # here, filename is actually a file-handle.
        # However, to accommodate keyword arguments we *must* have the same name
        filehandle = filename

        try:
            filename = Path(filehandle.name)
        except AttributeError:
            # this is not optimal, it will be the current directory, but one should not be able
            # to write to it
            filename = Path("dummy")

        mode = mode.replace("b", "")

        if hasattr(filehandle, "mode") and filehandle.mode.replace("b", "") != mode:
            raise ValueError(
                f"The filehandle mode ({filehandle.mode}) does not match the sile's mode ({mode})"
            )

        self._buffer = filehandle
        self._close = close

        self._temp_file = None
        # pass mode to the file to let it know what happened
        # we can't use super here due to the coupling to the Sile class
        super().__init__(filename, mode, *args, **kwargs)

    def close(self):
        """"""
        if self._mode.startswith("w"):
            with open(self._file, "rb") as fp:
                data = fp.read()

            self._buffer.write(data)

        Path(self._temp_file.name).unlink()
        self._temp_file = None
        if self._close:
            self._buffer.close()

    def _wrap_read_write(self, func, copy_buffer_content: bool = False):

        if self._temp_file is not None:
            return func

        self._temp_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        temp_path = Path(self._temp_file.name)

        # Set state to account for the fact that we have written the temp file.
        self._file = temp_path

        if copy_buffer_content:
            # Read the data from the buffer
            data = self._buffer.read()

            # Write it to the temporary file
            with self._temp_file as fp:
                fp.write(data)

        @wraps(func)
        def _wrapped(*args, **kwargs):
            # If the file is inside a zip fortran can't read directly from it.
            # We therefore store the contents in a temporary file and set this
            # as the file to be read from.
            res = func(*args, **kwargs)
            self.close()
            return res

        return _wrapped

    def wrap_read(self, func: Callable, name: str):
        return self._wrap_read_write(func, copy_buffer_content=True)

    def wrap_write(self, func: Callable, name: str):
        return self._wrap_read_write(func, copy_buffer_content=False)


@set_module("sisl.io")
class Info:
    """An info class that creates .info with inherent properties

    These properties can be added at will.
    """

    # default to be empty
    _info_attributes_: List[InfoAttr] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = _Info(self)

    class InfoAttr:
        """Holder for parsing lines and extracting information from text files

        This consists of:

        name:
            the name of the attribute
            This will be the `sile.info.<name>` access point.
        searcher:
            the regular expression used to match a line.
            If a `str`, it will be compiled *as is* to a regex pattern.
            `regex.match(line)` will be used to check if the value should be updated.
            It can also be a direct method called
        parser:
            if `regex.match(line)` returns a match that is true, then this parser will
            be executed.
            The parser *must* be a function accepting two arguments:

                def parser(attr, instance, match)

            where `attr` is this object, and `match` is the match done on the line.
            (Note that `match.string` will return the full line used to match against).
        updatable:
            control whether a new match on the line will update using `parser`.
            If false, only the first match will update the value
        default:
            the default value of the attribute
        found:
            whether the value has been found in the file.
        not_found: callable or str or Exception
            what to do when the attribute is not found, defaults to raise a SislInfo.
            It should accept 2 arguments, the object calling it, and the attribute.

                def not_found(obj, attr): # do something
        """

        __slots__ = (
            "name",
            "searcher",
            "parser",
            "updatable",
            "default",
            "value",
            "found",
            "doc",
            "not_found",
        )

        def __init__(
            self,
            name: str,
            searcher: Union[Callable[[InfoAttr, BaseSile, str], str], str, re.Pattern],
            parser: Callable[
                [InfoAttr, BaseSile, Union[str, re.Match]], Any
            ] = lambda attr, inst, line: line,
            doc: str = "",
            updatable: bool = False,
            default: Optional[Any] = None,
            found: bool = False,
            not_found: Union[None, str, Callable[[Any, InfoAttr], None]] = None,
            instance: Any = None,
        ):
            self.name = name

            if isinstance(searcher, str):
                searcher = re.compile(searcher)

            elif searcher is None:
                searcher = lambda info, instance, line: line

            if isinstance(searcher, re.Pattern):

                def used_searcher(info, instance, line):
                    nonlocal searcher

                    match = searcher.match(line)
                    if match:
                        info.value = info.parser(info, instance, match)
                        # print(f"found {info.name}={info.value} with {line}")
                        info.found = True
                        return True

                    return False

                used_searcher.pattern = searcher.pattern
            else:

                used_searcher = searcher
                used_searcher.pattern = "<custom>"

            self.searcher = used_searcher
            self.parser = parser
            self.updatable = updatable

            # Figure out if `self` is in the arguments of `default`
            # If so, instance bind it, use MethodType
            if callable(default) and instance is not None:
                if "self" in inspect.signature(default).parameters:
                    # Do a type-binding
                    default = MethodType(default, instance)

            self.default = default
            # Also set the actual value to the default one
            self.value = default
            self.found = found
            self.doc = doc

            def not_found_factory(method):
                # first check for a class, then check whether it is a specific class
                # otherwise a TypeError would be raised...
                if isinstance(method, type) and issubclass(method, BaseException):

                    def not_found(obj, attr):
                        raise method(
                            f"Attribute {attr.name} could not be found in {obj}."
                        )

                else:

                    def not_found(obj, attr):
                        method(f"Attribute {attr.name} could not be found in {obj}.")

                return not_found

            if not_found is None:
                not_found = not_found_factory(info)

            elif isinstance(not_found, str):
                if not_found == "info":
                    not_found = not_found_factory(info)
                elif not_found == "warn":
                    not_found = not_found_factory(warn)
                elif not_found == "error":
                    not_found = not_found_factory(KeyError)
                elif not_found == "ignore":
                    not_found = lambda obj, attr: None
                else:
                    raise ValueError(
                        f"{self.__class__.__name__} instantiated with unrecognized value in 'not_found' argument, got {not_found}."
                    )

            elif isinstance(not_found, type) and issubclass(not_found, BaseException):
                not_found = not_found_factory(not_found)

            self.not_found = not_found

        def process(self, instance, line):
            if self.found and not self.updatable:
                return False

            return self.searcher(self, instance, line)

        def reset(self):
            """Reset the property to the default value"""
            self.value = self.default

        def copy(self, instance: Any = None):
            obj = self.__class__(
                name=self.name,
                searcher=self.searcher,
                parser=self.parser,
                doc=self.doc,
                updatable=self.updatable,
                default=self.value,
                found=self.found,
                not_found=self.not_found,
                instance=instance,
            )
            return obj

        def documentation(self):
            """Returns a documentation string for this object"""
            if self.doc:
                doc = "\n" + indent(dedent(self.doc), " " * 4)
            else:
                doc = ""
            return f"{self.name}[{self.value}]: r'{self.searcher.pattern}'{doc}"

    class _Info:
        """The actual .info object that will attached to the instance.

        As of now this is problematic to document.
        We should figure out a way to do that.
        """

        def __init__(self, instance):
            # attach this info instance to the instance
            self._instance = instance
            self._attrs = []
            self._properties = []
            self._searching = False

            # add the properties
            for prop in instance._info_attributes_:
                if isinstance(prop, dict):
                    prop = instance.InfoAttr(instance=instance, **prop)
                elif isinstance(prop, (tuple, list)):
                    prop = instance.InfoAttr(*prop, instance=instance)
                else:
                    prop = prop.copy(instance=instance)
                self.add_property(prop)

            # Patch the readline of the instance
            def patch(info):
                # grab the function to be patched
                properties = info._properties
                func = info._instance.readline

                @wraps(func)
                def readline(*args, **kwargs):
                    line = func(*args, **kwargs)
                    for prop in properties:
                        prop.process(info._instance, line)
                    return line

                return readline

            if len(self) > 0:
                self._instance.readline = patch(self)

        def add_property(self, prop: InfoAttr) -> None:
            """Add a new property to be reachable from the .info"""
            self._attrs.append(prop.name)
            self._properties.append(prop)

        def get_property(self, prop: str) -> None:
            """Add a new property to be reachable from the .info"""
            if prop not in self._attrs:
                inst = self._instance
                raise AttributeError(
                    f"{inst.__class__.__name__}.info.{prop} does not exist, did you mistype?"
                )

            idx = self._attrs.index(prop)
            return self._properties[idx]

        def __str__(self):
            """Return a string of the contained attributes, with the values they currently contain"""
            return "\n".join([p.documentation() for p in self._properties])

        def __len__(self) -> int:
            return len(self._properties)

        def __getattr__(self, attr):
            """Overwrite the attribute retrieval to be able to fetch the actual values from the information"""
            inst = self._instance
            prop = self.get_property(attr)

            if prop.found or self._searching:
                # only when hitting the new line will this change...
                return prop.value

            # we need to parse the rest of the file
            # This is not ideal, but...
            self._searching = True
            loc = None
            try:
                loc = inst.fh.tell()
            except AttributeError:
                pass
            with inst:
                line = inst.readline()
                while not (prop.found or line == ""):
                    line = inst.readline()
            if loc is not None:
                inst.fh.seek(loc)

            if not prop.found:
                prop.not_found(inst, prop)

            self._searching = False
            return prop.value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info = self._Info(self)


@set_module("sisl.io")
class Sile(Info, BaseSile):
    """Base class for ASCII files

    All ASCII files that needs to be added to the global lookup table can
    with benefit inherit this class.

    By subclassing a `Sile` one can manually specify the buffer class used
    when passing a `buffer_cls` keyword argument. This enables one to overwrite
    buffer classes for custom siles.

    >>> class mySile(otherSislSile, buffer_cls=myBufferClass): ...
    """

    _buffer_extension_cls = BufferSile

    def __init__(self, filename, mode="r", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = mode
        self._file = self._sanitize_filename(filename)

        comment = kwargs.pop("comment", None)
        if isinstance(comment, (list, tuple)):
            self._comment = list(comment)
        elif not comment is None:
            self._comment = [comment]
        else:
            self._comment = []

        self._fh_opens = 0
        self._line = 0

        # Initialize
        self._base_setup(*args, **kwargs)

    def _open(self):
        # track how many times this has been called
        if hasattr(self, "fh"):
            self.fh.seek(0)
        else:
            self._fh_opens = 0

            if self.file.suffix == ".gz":
                if self._mode == "r":
                    # assume the file is a text file and open in text-mode
                    self.fh = gzip.open(str(self.file), mode="rt")
                else:
                    # assume this is opening in binary or write mode
                    self.fh = gzip.open(str(self.file), mode=self._mode)
            else:
                self.fh = self.file.open(self._mode)

        # the file should restart the file-read (as per instructed)
        self._line = 0

        self._fh_opens += 1

    def __enter__(self):
        """Opens the output file and returns it self"""
        self._open()
        return self

    def __exit__(self, type, value, traceback):
        # clean-up so that it does not exist
        self.close()
        return False

    def close(self):
        # decrement calls
        self._fh_opens -= 1
        if self._fh_opens <= 0:
            self._line = 0
            self.fh.close()
            delattr(self, "fh")
            self._fh_opens = 0

    @staticmethod
    def is_keys(keys):
        """Returns true if ``not isinstance(keys, str)``"""
        return not isinstance(keys, str)

    @staticmethod
    def key2case(key, case):
        """Converts str/list of keywords to proper case"""
        if case:
            return key
        return key.lower()

    @staticmethod
    def keys2case(keys, case):
        """Converts str/list of keywords to proper case"""
        if case:
            return keys
        return [k.lower() for k in keys]

    @staticmethod
    def line_has_key(line, key, case=True):
        if case:
            return line.find(key) >= 0
        return line.lower().find(key.lower()) >= 0

    @staticmethod
    def line_has_keys(line, keys, case=True):
        found = False
        if case:
            for key in keys:
                found |= line.find(key) >= 0
        else:
            l = line.lower()
            for key in keys:
                found |= l.find(key.lower()) >= 0
        return found

    def __iter__(self):
        """Reading the entire content, without regarding comments"""
        l = self.readline(comment=True)
        while l:
            yield l
            l = self.readline(comment=True)

    def readline(self, comment: bool = False) -> str:
        r"""Reads the next line of the file"""
        l = self.fh.readline()
        self._line += 1
        if comment:
            return l
        while starts_with_list(l, self._comment):
            l = self.fh.readline()
            self._line += 1
        return l

    def step_to(
        self, keywords, case=True, allow_reread=True, ret_index=False, reopen=False
    ):
        r"""Steps the file-handle until the keyword(s) is found in the input

        Parameters
        ----------
        keywords : str or list
            keyword(s) to find in the sile
        case : bool, optional
            whether to search case sensitive
        allow_reread : bool, optional
            whether the search from current position is allowed to
            loop back to the beginning
        ret_index : bool, optional
            returns the index in the keyword list that was matched in the search
        reopen : bool, optional
            if True, the search is forced to start from the beginning
            of the sile (search after sile close and reopen)

        Returns
        -------
        found : bool
            whether the search was successful or not
        l : str
            the found line that matches the keyword(s)
        idx : int, optional
            index of the keyword from the list that was matched
        """

        if reopen:
            # ensure file is read from the beginning
            self.close()
            self._open()

        # If keyword is a list, it just matches one of the inputs
        found = False
        # The previously read line...
        line = self._line
        if isinstance(keywords, str):
            # convert to list
            keywords = [keywords]
        keys = self.keys2case(keywords, case)

        while not found:
            l = self.readline()
            if l == "":
                break
            found = self.line_has_keys(l, keys, case)

        if not found and (l == "" and line > 0) and allow_reread:
            # We may be in the case where the user request
            # reading the same twice...
            # So we need to re-read the file...
            self.close()
            # Re-open the file...
            self._open()

            # Try and read again
            while not found and self._line <= line:
                l = self.readline()
                if l == "":
                    break
                found = self.line_has_keys(l, keys, case)

        if ret_index:
            idx = -1
            if found:
                idx = 0

            # force return an index
            for i, key in enumerate(keys):
                if self.line_has_key(l, key, case):
                    return found, l, i
            return found, l, idx

        # sometimes the line contains information, as a
        # default we return the line found
        return found, l

    def _write(self, *args, **kwargs):
        """Wrapper to default the write statements"""
        self.fh.write(*args, **kwargs)


# Instead of importing netCDF4 on each invocation
# of the __enter__ functioon (below), we make
# a pass around it
if has_module("netCDF4"):
    import netCDF4
else:

    class _mock_netCDF4:
        def __getattr__(self, attr):
            import sys

            exe = Path(sys.executable).name
            msg = f"Could not import netCDF4. Please install it using '{exe} -m pip install netCDF4'"
            raise SileError(msg)

    netCDF4 = _mock_netCDF4()


@set_module("sisl.io")
class SileCDF(BaseSile):
    """Creates/Opens a SileCDF

    Opens a SileCDF with `mode` and compression level `lvl`.
    If `mode` is in read-mode (r) the compression level
    is ignored.

    The final `access` parameter sets how the file should be
    open and subsequently accessed.

    0) means direct file access for every variable read
    1) means stores certain variables in the object.
    """

    _buffer_extension_cls = BufferSileCDF

    _is_inside_zip: bool = False
    _buffer_instance: Optional[BufferSileCDF] = None

    def __init__(self, filename, mode="r", lvl=0, access=1, *args, **kwargs):
        # Open mode
        self._mode = mode
        # Get file
        self._file = self._sanitize_filename(filename)
        self._is_inside_zip = isinstance(self._file, ZipPath)
        self._buffer_instance = None
        # Save compression internally
        self._lvl = lvl
        # Initialize the _data dictionary for access == 1
        self._data = dict()
        if self.file.is_file():
            self._access = access
        else:
            # If it has not been created we should not try
            # and read anything, no matter what the user says
            self._access = 0

            # The CDF file can easily open the file
        if kwargs.pop("_open", True):
            self._open()

        # Must call setup-methods
        self._base_setup(*args, **kwargs)

    @property
    def _cmp_args(self):
        """Returns the compression arguments for the NetCDF file

        >>> nc.createVariable(..., **self._cmp_args)
        """
        return {"zlib": self._lvl > 0, "complevel": self._lvl}

    def __enter__(self):
        """Opens the output file and returns it self"""
        self._open()
        return self

    def _open(self):
        """Opens the output file and returns it self"""
        # We do the import here
        if "fh" not in self.__dict__:

            if self._is_inside_zip:
                if self._buffer_instance is None:
                    self._buffer_instance = self._buffer_cls(
                        self._file.open(self._mode + "b"),
                        mode=self._mode,
                    )

                    self.fh = self._buffer_instance.fh

            else:
                self.__dict__["fh"] = netCDF4.Dataset(
                    str(self.file), self._mode, format="NETCDF4"
                )

        return self

    def __getattribute__(self, name):

        is_read = name.startswith("read_")
        is_write = name.startswith("write_")

        if (
            (is_read or is_write)
            and self._is_inside_zip
            and self._buffer_instance is not None
        ):
            return getattr(self._buffer_instance, name)

        return super().__getattribute__(name)

    def __exit__(self, type, value, traceback):
        self.close()
        return False

    def close(self):
        if self._buffer_instance is not None:
            self._buffer_instance._buffer.close()
            self._buffer_instance = None
            self.fh = None
        else:
            self.fh.close()

    def _dimension(self, name, tree=None):
        """Local method for obtaing the dimension in a certain tree"""
        return self._dimensions(self, name, tree)

    @staticmethod
    def _dimensions(n, name, tree=None):
        """Retrieve  method to get the NetCDF variable"""
        if tree is None:
            return n.dimensions[name]

        g = n
        if isinstance(tree, list):
            for t in tree:
                g = g.groups[t]
        else:
            g = g.groups[tree]

        return g.dimensions[name]

    def _variable(self, name, tree=None):
        """Local method for obtaining the data from the SileCDF.

        This method returns the variable as-is.
        """
        if self._access > 0 and tree is None:
            if name in self._data:
                return self._data[name]
        return self._variables(self, name, tree=tree)

    def _value(self, name, tree=None):
        """Local method for obtaining the data from the SileCDF.

        This method returns the value of the variable.
        """
        return self._variable(name, tree)[:]

    @staticmethod
    def _variables(n, name, tree=None):
        """Retrieve  method to get the NetCDF variable"""
        if tree is None:
            return n.variables[name]

        g = n
        if isinstance(tree, list):
            for t in tree:
                g = g.groups[t]
        else:
            g = g.groups[tree]

        return g.variables[name]

    @staticmethod
    def _crt_grp(n, name):
        if "/" in name:  # this is NetCDF, so / is fixed as seperator!
            groups = name.split("/")
            grp = n
            for group in groups:
                if len(group) > 0:
                    grp = SileCDF._crt_grp(grp, group)
            return grp

        if name in n.groups:
            return n.groups[name]
        return n.createGroup(name)

    @staticmethod
    def _crt_dim(n, name, l):
        if name in n.dimensions:
            return n.dimensions[name]
        return n.createDimension(name, l)

    @staticmethod
    def _crt_var(n, name, *args, **kwargs):
        if name in n.variables:
            return n.variables[name]

        if "attrs" in kwargs:
            attrs = kwargs.pop("attrs")
        else:
            attrs = None
        var = n.createVariable(name, *args, **kwargs)
        if attrs is not None:
            for name, value in attrs.items():
                setattr(var, name, value)
        return var

    @classmethod
    def isDimension(cls, obj):
        """Return true if ``obj`` is an instance of the NetCDF4 ``Dimension`` type

        This is just a wrapper for ``isinstance(obj, netCDF4.Dimension)``.
        """
        return isinstance(obj, netCDF4.Dimension)

    @classmethod
    def isVariable(cls, obj):
        """Return true if ``obj`` is an instance of the NetCDF4 ``Variable`` type

        This is just a wrapper for ``isinstance(obj, netCDF4.Variable)``.
        """
        return isinstance(obj, netCDF4.Variable)

    @classmethod
    def isGroup(cls, obj):
        """Return true if ``obj`` is an instance of the NetCDF4 ``Group`` type

        This is just a wrapper for ``isinstance(obj, netCDF4.Group)``.
        """
        return isinstance(obj, netCDF4.Group)

    @classmethod
    def isDataset(cls, obj):
        """Return true if ``obj`` is an instance of the NetCDF4 ``Dataset`` type

        This is just a wrapper for ``isinstance(obj, netCDF4.Dataset)``.
        """
        return isinstance(obj, netCDF4.Dataset)

    isRoot = isDataset

    def iter(self, group=True, dimension=True, variable=True, levels=-1, root=None):
        """Iterator on all groups, variables and dimensions.

        This iterator iterates through all groups, variables and dimensions in the ``Dataset``

        The generator sequence will _always_ be:

          1. Group
          2. Dimensions in group
          3. Variables in group

        As the dimensions are generated before the variables it is possible to copy
        groups, dimensions, and then variables such that one always ensures correct
        dependencies in the generation of a new ``SileCDF``.

        Parameters
        ----------
        group : ``bool`` (`True`)
           whether the iterator yields `Group` instances
        dimension : ``bool`` (`True`)
           whether the iterator yields `Dimension` instances
        variable : ``bool`` (`True`)
           whether the iterator yields `Variable` instances
        levels : ``int`` (`-1`)
           number of levels to traverse, with respect to ``root`` variable, i.e. number of
           sub-groups this iterator will return.
        root : ``str`` (`None`)
           the base root to start iterating from.

        Examples
        --------

        Script for looping and checking each instance.

        >>> for gv in self.iter():
        ...     if self.isGroup(gv):
        ...         # is group
        ...     elif self.isDimension(gv):
        ...         # is dimension
        ...     elif self.isVariable(gv):
        ...         # is variable

        """
        if root is None:
            head = self.fh
        else:
            head = self.fh[root]

        # Start by returning the root group
        if group:
            yield head

        if dimension:
            yield from head.dimensions.values()
        if variable:
            yield from head.variables.values()

        if levels == 0:
            # Stop the iterator
            return

        for grp in head.groups.values():
            yield from self.iter(
                group, dimension, variable, levels=levels - 1, root=grp.path
            )

    __iter__ = iter


@set_module("sisl.io")
class SileBin(BaseSile):
    """Creates/Opens a SileBin

    Opens a SileBin with `mode` (b).
    If `mode` is in read-mode (r).
    """

    _buffer_extension_cls = BufferSileBin

    _is_inside_zip: bool = False
    _buffer_instance: Optional[BufferSileBin] = None

    def __init__(self, filename, mode="r", *args, **kwargs):
        # Open mode
        self._mode = mode.replace("b", "") + "b"
        # Get file
        self._file = self._sanitize_filename(filename)

        self._is_inside_zip = isinstance(self._file, ZipPath)
        self._buffer_instance = None

        # Must call setup-methods
        self._base_setup(*args, **kwargs)

    def __getattribute__(self, name):
        # If the file is inside a zip fortran can't read directly from it.
        # We open the buffer from the zipfile and then pass it to the BufferSileBin
        # class, which will handle things as with any other buffer
        is_read_write = name.startswith("read_") or name.startswith("write_")
        if is_read_write and self._is_inside_zip:
            if self._buffer_instance is None:
                self._buffer_instance = self._buffer_cls(
                    self._file.open(self._mode), mode=self._mode, close=True
                )

            return getattr(self._buffer_instance, name)

        return super().__getattribute__(name)

    def __enter__(self):
        """Opens the output file and returns it self"""
        return self

    def __exit__(self, type, value, traceback):
        return False


def sile_raise_write(self, ok=("w", "a")):
    is_ok = False
    for O in ok:
        is_ok = is_ok or (O in self._mode)
    if not is_ok:
        raise SileError(
            (
                "Writing to file not possible; allowed "
                f"modes={ok}, used mode={self._mode}"
            ),
            self,
        )


def sile_raise_read(self, ok=("r", "a")):
    is_ok = False
    for O in ok:
        is_ok = is_ok or (O in self._mode)
    if not is_ok:
        raise SileError(
            f"Reading file not possible; allowed "
            f"modes={ok}, used mode={self._mode}",
            self,
        )
