# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from functools import wraps
from os.path import splitext, basename
import gzip
from io import TextIOBase
from pathlib import Path

from sisl._internal import set_module
from sisl.messages import SislWarning, SislInfo
from sisl.utils.misc import str_spec
from ._help import *


# Public used objects
__all__ = [
    'add_sile',
    'get_sile_class',
    'get_sile',
    'get_siles',
    'get_sile_rules'
]

__all__ += [
    'BaseSile',
    'BufferSile',
    'Sile',
    'SileCDF',
    'SileBin',
    'SileError',
    'SileWarning',
    'SileInfo',
]

# Decorators or sile-specific functions
__all__ += [
    'sile_fh_open',
    'sile_raise_write',
    'sile_raise_read'
]

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
    """ Internal data-structure to check whether a file is the same as this sile """

    __slots__ = ('cls', 'case', 'suffix', 'gzip', 'bases', 'base_names')

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
        s = '{cls}{{case={case}, suffix={suffix}, gzip={gzip},\n '.format(cls=self.cls.__name__, case=self.case,
                                                                          suffix=self.suffix, gzip=self.gzip)
        for b in self.bases:
            s += f' {b.__name__},\n '
        return s[:-3] + '\n}'

    def build_bases(self):
        """ Return a list of all classes that this file is inheriting from (except Sile, SileBin or SileCDF) """
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
            for child in list(children): # ensure we have a copy for infinite loops
                for c in child.__bases__:
                    if c not in children:
                        children.append(c)

        return children

    def in_bases(self, base):
        """ Whether any of the inherited bases starts with `base` in their class-name (non-case sensitive) """
        if base is None:
            return True
        elif isinstance(base, object):
            base = base.__name__.lower()
        for b in self.base_names:
            if b.startswith(base) or base in b:
                return True
        return False

    def get_base(self, base):
        """ Whether any of the inherited bases starts with `base` in their class-name (non-case sensitive) """
        if base is None:
            return None
        for bn, b in zip(self.base_names, self.bases):
            if bn.startswith(base) or base in bn:
                return b
        return None

    def in_class(self, base):
        """ Whether any of the inherited bases starts with `base` in their class-name (non-case sensitive) """
        if base is None:
            return False
        n = self.cls.__name__.lower()
        return (n.startswith(base) or base in n)

    def is_suffix(self, suffix):
        if not self.case:
            suffix = suffix.lower()
        # Now check names and (possibly gzip)
        my_suffix = self.suffix
        if suffix == my_suffix:
            return True
        if not self.gzip:
            return False
        return suffix == (my_suffix + ".gz")

    def is_class(self, cls):
        return self.cls == cls

    def is_subclass(self, cls):
        return issubclass(self.cls, cls)


@set_module("sisl.io")
def add_sile(suffix, cls, case=True, gzip=False):
    """ Add files to the global lookup table

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

    # Only add pure suffixes...
    if suffix.startswith('.'):
        suffix = suffix[1:]

    # If it isn't already in the list of
    # Siles, add it.
    if cls not in __siles:
        __siles.append(cls)

    # Add the rule of the sile to the list of rules.
    __sile_rules.append(_sile_rule(cls, suffix, case=case, gzip=gzip))


@set_module("sisl.io")
def get_sile_class(filename, *args, **kwargs):
    """ Retrieve a class from the global lookup table via filename and the extension

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
    cls = kwargs.pop('cls', None)

    # Split filename into proper file name and
    # the Specification of the type
    tmp_file, fcls = str_spec(str(filename))

    if cls is None and not fcls is None:
        # cls has not been set, and fcls is found
        # Figure out if fcls is a valid sile, if not
        # do nothing (it may be part of the file name)
        # Which is REALLY obscure... but....)
        fclsl = fcls.lower()
        for sr in __sile_rules:
            if sr.in_class(fclsl):
                cls = sr.cls
            else:
                cls = sr.get_base(fclsl)
            if cls is not None:
                filename = tmp_file
                break

    try:
        # Create list of endings on this file
        f = basename(filename)
        end_list = []
        end = ''

        # Check for files without ending, or that they are directly zipped
        lext = splitext(f)
        while len(lext[1]) > 0:
            end = lext[1] + end
            if end[0] == '.':
                end_list.append(end[1:])
            else:
                end_list.append(end)
            lext = splitext(lext[0])

        # We also check the entire file name
        #  (mainly for VASP)
        end_list.append(f)
        # Reverse to start by the longest extension
        # (allows grid.nc extensions, etc.)
        end_list = list(reversed(end_list))

        # First we check for class AND file ending
        clss = None
        for end in end_list:
            for sr in __sile_rules:
                if sr.is_class(cls):
                    # class-specification has precedence
                    # This should only occur when the
                    # class-specification is exact (i.e. xyzSile)
                    return sr.cls
                elif sr.is_suffix(end):
                    if cls is None:
                        return sr.cls
                    elif sr.is_subclass(cls):
                        return sr.cls
                    clss = sr.cls
            if clss is not None:
                return clss

        if clss is None:
            raise NotImplementedError("Sile for file '{}' could not be found, "
                                      "possibly the file has not been implemented.".format(filename))
        return clss

    except Exception as e:
        raise e


@set_module("sisl.io")
def get_sile(file, *args, **kwargs):
    """ Retrieve an object from the global lookup table via filename and the extension

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
       you may query the exact base-class that should be chosen.
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
    """
    cls = kwargs.pop('cls', None)
    sile = get_sile_class(file, *args, cls=cls, **kwargs)
    return sile(Path(str_spec(str(file))[0]), *args, **kwargs)


@set_module("sisl.io")
def get_siles(attrs=None):
    """ Retrieve all files with specific attributes or methods

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
            if hasattr(sile, attr):
                siles.append(sile)
                break

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
    """ Base class for all sisl files """

    @property
    def file(self):
        """ File of the current `Sile` """
        return self._file

    @property
    def base_file(self):
        """ File of the current `Sile` """
        return basename(self._file)

    def dir_file(self, filename=None, filename_base=""):
        """ File of the current `Sile` """
        if filename is None:
            filename = Path(self._file).name
        return self._directory / filename_base / filename

    def read(self, *args, **kwargs):
        """ Generic read method which should be overloaded in child-classes

        Parameters
        ----------
        kwargs :
          keyword arguments will try and search for the attribute ``read_<>``
          and call it with the remaining ``**kwargs`` as arguments.
        """
        for key in kwargs.keys():
            # Loop all keys and try to read the quantities
            if hasattr(self, "read_" + key):
                func = getattr(self, "read_" + key)
                # Call read
                return func(kwargs[key], **kwargs)

    # Options for writing
    # The default routine for writing
    _write_default = None
    # Whether only the default should be used
    # when issuing a write
    _write_default_only = False

    def write(self, *args, **kwargs):
        """ Generic write method which should be overloaded in child-classes

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
            if hasattr(self, "write_" + key):
                func = getattr(self, "write_" + key)
                # Call write
                func(kwargs[key], **kwargs)

    def _setup(self, *args, **kwargs):
        """ Setup the `Sile` after initialization

        Inherited method for setting up the sile.

        This method can be overwritten.
        """
        pass

    def _base_setup(self, *args, **kwargs):
        """ Setup the `Sile` after initialization

        Inherited method for setting up the sile.

        This method **must** be overwritten *and*
        end with ``self._setup()``.
        """
        base = kwargs.get('base', None)
        if base is None:
            # Extract from filename
            self._directory = Path(self._file).parent
        else:
            self._directory = base
        if not str(self._directory):
            self._directory = '.'
        self._directory = Path(self._directory).resolve()

        self._setup(*args, **kwargs)

    def _base_file(self, f):
        """ Make `f` refer to the file with the appropriate base directory """
        return self._directory / f

    def __getattr__(self, name):
        """ Override to check the handle """
        if name == 'fh':
            raise AttributeError(f"The filehandle for {self.file} has not been opened yet...")
        return getattr(self.fh, name)

    @classmethod
    def _ArgumentParser_args_single(cls):
        """ Default arguments for the Sile """
        return {}

    # Define the custom ArgumentParser
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that may be available for this Sile

        Parameters
        ----------
        p : ArgumentParser
           the argument parser to add the arguments to.
        """
        raise NotImplementedError(f"The ArgumentParser of '{self.__class__.__name__}' has not been implemented yet.")

    def ArgumentParser_out(self, p=None, *args, **kwargs):
        """ Appends additional arguments based on the output of the file

        Parameters
        ----------
        p : ArgumentParser
           the argument parser to add the arguments to.
        """
        pass

    def __str__(self):
        """ Return a representation of the `Sile` """
        # Check if the directory is relative to the current path
        # If so, only print the relative path, otherwise print the full path
        d = self._directory
        try:
            # bypass d.is_relative_to, added in 3.9
            d = d.relative_to(Path('.').resolve())
        except Exception: pass
        return f"{self.__class__.__name__}({self.base_file!s}, base={d!s})"


def sile_fh_open(from_closed=False):
    """ Method decorator for objects to directly implement opening of the
    file-handle upon entry (if it isn't already).

    Parameters
    ----------
    from_closed : bool, optional
       ensure the wrapped function *must* open the file, otherwise it will seek to 0.
    """
    if from_closed:
        def _wrapper(func):
            @wraps(func)
            def pre_open(self, *args, **kwargs):
                if hasattr(self, "fh"):
                    self.fh.seek(0)
                    return func(self, *args, **kwargs)
                with self:
                    return func(self, *args, **kwargs)
            return pre_open
    else:
        def _wrapper(func):
            @wraps(func)
            def pre_open(self, *args, **kwargs):
                if hasattr(self, "fh"):
                    return func(self, *args, **kwargs)
                with self:
                    return func(self, *args, **kwargs)
            return pre_open
    return _wrapper


@set_module("sisl.io")
class BufferSile:
    """ Sile for handling `StringIO` and `TextIOBase` objects

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
        """ Will not close the file since this is passed by the user """
        pass


@set_module("sisl.io")
class Sile(BaseSile):
    """ Base class for ASCII files

    All ASCII files that needs to be added to the global lookup table can
    with benefit inherit this class.

    By subclassing a `Sile` one can manually specify the buffer class used
    when passing a `buffer_cls` keyword argument. This enables one to overwrite
    buffer classes for custom siles.

    >>> class mySile(otherSislSile, buffer_cls=myBufferClass): ...
    """

    def __new__(cls, filename, *args, **kwargs):

        # check whether filename is an actual str, or StringIO or some buffer
        if not isinstance(filename, TextIOBase):
            # this is just a regular sile opening
            return super().__new__(cls)

        return super().__new__(cls._buffer_cls)

    def __init_subclass__(cls, buffer_cls=None):
        if issubclass(cls, BufferSile):
            # return since it already inherits BufferSile
            return

        if buffer_cls is None:
            buffer_cls = type(f"{cls.__name__}Buffer", (BufferSile, cls),
                              # Ensure the module is the same
                              {"__module__": cls.__module__})
        elif not issubclass(buffer_cls, BufferSile):
            raise TypeError(f"The passed buffer_cls should inherit from sisl.io.BufferSile to "
                            "ensure correct behaviour.")

        cls._buffer_cls = buffer_cls

    def __init__(self, filename, mode='r', *args, **kwargs):
        self._file = Path(filename)
        self._mode = mode

        comment = kwargs.pop("comment", None)
        if isinstance(comment, (list, tuple)):
            self._comment = list(comment)
        elif not comment is None:
            self._comment = [comment]
        else:
            self._comment = []

        self._line = 0

        # Initialize
        self._base_setup(*args, **kwargs)

    def _open(self):
        if self.file.suffix == ".gz":
            if self._mode == 'r':
                # assume the file is a text file and open in text-mode
                self.fh = gzip.open(str(self.file), mode='rt')
            else:
                # assume this is opening in binary or write mode
                self.fh = gzip.open(str(self.file), mode=self._mode)
        else:
            self.fh = self.file.open(self._mode)
        self._line = 0

    def __enter__(self):
        """ Opens the output file and returns it self """
        self._open()
        return self

    def __exit__(self, type, value, traceback):
        # clean-up so that it does not exist
        self.close()
        self._line = 0
        return False

    def close(self):
        self.fh.close()
        delattr(self, 'fh')

    @staticmethod
    def is_keys(keys):
        """ Returns true if ``not isinstance(keys, str)`` """
        return not isinstance(keys, str)

    @staticmethod
    def key2case(key, case):
        """ Converts str/list of keywords to proper case """
        if case:
            return key
        return key.lower()

    @staticmethod
    def keys2case(keys, case):
        """ Converts str/list of keywords to proper case """
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
        """ Reading the entire content, without regarding comments """
        l = self.readline(comment=True)
        while l:
            yield l
            l = self.readline(comment=True)

    def readline(self, comment=False):
        r""" Reads the next line of the file """
        l = self.fh.readline()
        self._line += 1
        if comment:
            return l
        while starts_with_list(l, self._comment):
            l = self.fh.readline()
            self._line += 1
        return l

    def step_to(self, keywords, case=True, allow_reread=True, ret_index=False, reopen=False):
        r""" Steps the file-handle until the keyword(s) is found in the input

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
            if l == '':
                break
            found = self.line_has_keys(l, keys, case)

        if not found and (l == '' and line > 0) and allow_reread:
            # We may be in the case where the user request
            # reading the same twice...
            # So we need to re-read the file...
            self.close()
            # Re-open the file...
            self._open()

            # Try and read again
            while not found and self._line <= line:
                l = self.readline()
                if l == '':
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
        """ Wrapper to default the write statements """
        self.fh.write(*args, **kwargs)


# Instead of importing netCDF4 on each invocation
# of the __enter__ functioon (below), we make
# a pass around it
_netCDF4 = None


def _import_netCDF4():
    global _netCDF4
    if _netCDF4 is None:
        import netCDF4 as _netCDF4


@set_module("sisl.io")
class SileCDF(BaseSile):
    """ Creates/Opens a SileCDF

    Opens a SileCDF with `mode` and compression level `lvl`.
    If `mode` is in read-mode (r) the compression level
    is ignored.

    The final `access` parameter sets how the file should be
    open and subsequently accessed.

    0) means direct file access for every variable read
    1) means stores certain variables in the object.
    """

    def __init__(self, filename, mode='r', lvl=0, access=1, *args, **kwargs):
        self._file = Path(filename)
        # Open mode
        self._mode = mode
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
        if kwargs.pop('_open', True):
            _import_netCDF4()
            self.__dict__['fh'] = _netCDF4.Dataset(str(self.file), self._mode,
                                                   format='NETCDF4')

        # Must call setup-methods
        self._base_setup(*args, **kwargs)

    @property
    def _cmp_args(self):
        """ Returns the compression arguments for the NetCDF file

        >>> nc.createVariable(..., **self._cmp_args)
        """
        return {'zlib': self._lvl > 0, 'complevel': self._lvl}

    def __enter__(self):
        """ Opens the output file and returns it self """
        # We do the import here
        if 'fh' not in self.__dict__:
            _import_netCDF4()
            self.__dict__['fh'] = _netCDF4.Dataset(str(self.file), self._mode, format='NETCDF4')
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        return False

    def close(self):
        self.fh.close()

    def _dimension(self, name, tree=None):
        """ Local method for obtaing the dimension in a certain tree """
        return self._dimensions(self, name, tree)

    @staticmethod
    def _dimensions(n, name, tree=None):
        """ Retrieve  method to get the NetCDF variable """
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
        """ Local method for obtaining the data from the SileCDF.

        This method returns the variable as-is.
        """
        if self._access > 0 and tree is None:
            if name in self._data:
                return self._data[name]
        return self._variables(self, name, tree=tree)

    def _value(self, name, tree=None):
        """ Local method for obtaining the data from the SileCDF.

        This method returns the value of the variable.
        """
        return self._variable(name, tree)[:]

    @staticmethod
    def _variables(n, name, tree=None):
        """ Retrieve  method to get the NetCDF variable """
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
        if '/' in name: # this is NetCDF, so / is fixed as seperator!
            groups = name.split('/')
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

        if 'attrs' in kwargs:
            attrs = kwargs.pop('attrs')
        else:
            attrs = None
        var = n.createVariable(name, *args, **kwargs)
        if attrs is not None:
            for name, value in attrs.items():
                setattr(var, name, value)
        return var

    @classmethod
    def isDimension(cls, obj):
        """ Return true if ``obj`` is an instance of the NetCDF4 ``Dimension`` type

        This is just a wrapper for ``isinstance(obj, netCDF4.Dimension)``.
        """
        return isinstance(obj, _netCDF4.Dimension)

    @classmethod
    def isVariable(cls, obj):
        """ Return true if ``obj`` is an instance of the NetCDF4 ``Variable`` type

        This is just a wrapper for ``isinstance(obj, netCDF4.Variable)``.
        """
        return isinstance(obj, _netCDF4.Variable)

    @classmethod
    def isGroup(cls, obj):
        """ Return true if ``obj`` is an instance of the NetCDF4 ``Group`` type

        This is just a wrapper for ``isinstance(obj, netCDF4.Group)``.
        """
        return isinstance(obj, _netCDF4.Group)

    @classmethod
    def isDataset(cls, obj):
        """ Return true if ``obj`` is an instance of the NetCDF4 ``Dataset`` type

        This is just a wrapper for ``isinstance(obj, netCDF4.Dataset)``.
        """
        return isinstance(obj, _netCDF4.Dataset)
    isRoot = isDataset

    def iter(self, group=True, dimension=True, variable=True, levels=-1, root=None):
        """ Iterator on all groups, variables and dimensions.

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
            yield from self.iter(group, dimension, variable,
                                 levels=levels-1, root=grp.path)

    __iter__ = iter


@set_module("sisl.io")
class SileBin(BaseSile):
    """ Creates/Opens a SileBin

    Opens a SileBin with `mode` (b).
    If `mode` is in read-mode (r).
    """

    def __init__(self, filename, mode='r', *args, **kwargs):
        self._file = Path(filename)
        # Open mode
        self._mode = mode.replace('b', '') + 'b'

        # Must call setup-methods
        self._base_setup(*args, **kwargs)

    def __enter__(self):
        """ Opens the output file and returns it self """
        return self

    def __exit__(self, type, value, traceback):
        return False


def sile_raise_write(self, ok=('w', 'a')):
    is_ok = False
    for O in ok:
        is_ok = is_ok or (O in self._mode)
    if not is_ok:
        raise SileError(('Writing to file not possible; allowed '
                         'modes={}, used mode={}'.format(ok, self._mode)), self)


def sile_raise_read(self, ok=('r', 'a')):
    is_ok = False
    for O in ok:
        is_ok = is_ok or (O in self._mode)
    if not is_ok:
        raise SileError(f"Reading file not possible; allowed "
                        f"modes={ok}, used mode={self._mode}", self)


@set_module("sisl.io")
class SileError(IOError):
    """ Define an error object related to the Sile objects """

    def __init__(self, value, obj=None):
        self.value = value
        self.obj = obj

    def __str__(self):
        if self.obj:
            return f"{self.value!s} in {self.obj!s}"
        else:
            return self.value


@set_module("sisl.io")
class SileWarning(SislWarning):
    """ Warnings that informs users of things to be carefull about when using their retrieved data

    These warnings should be issued whenever a read/write routine is unable to retrieve all information
    but are non-influential in the sense that sisl is still able to perform the action.
    """
    pass


@set_module("sisl.io")
class SileInfo(SislInfo):
    """ Information for the user, this is hidden in a warning, but is not as severe so as to issue a warning. """
    pass
