from __future__ import print_function, division

from os.path import splitext, isfile
import gzip

import numpy as np

from sisl import Geometry
from sisl.utils.misc import str_spec
from ._help import *


# Public used objects
__all__ = [
    'add_sile',
    'get_sile_class',
    'get_sile',
    'get_siles']

__all__ += [
    'BaseSile',
    'Sile',
    'SileCDF',
    'SileBin',
    'SileError',
    ]

# Decorators or sile-specific functions
__all__ += [
    'Sile_fh_open',
    'sile_raise_write',
    'sile_raise_read']

# Global container of all Sile rules
# This list of tuples is formed as
#  [('fdf', fdfSileSiesta, fdfSileSiesta),
#   ('fdf', ncSileSiesta, fdfSileSiesta)]
# [0] is the file-endding
# [1] is the base class that may be queried
# [2] is the actual class the file represents
# This enables one to add several files with the
# same extension and query it based on a sub-class
__sile_rules = []
__siles = []


def add_sile(ending, cls, case=True, gzip=False, _parent_cls=None):
    """
    Public for attaching lookup tables for allowing
    users to attach files for the IOSile function call

    Parameters
    ----------
    ending : str
         The file-name ending, it can be several file endings (.TBT.nc)
    cls : `BaseSile` child
         An object that is associated with the respective file.
         It must be inherited from a `BaseSile`.
    case : bool, (True)
         Whether case sensitivity is applicable for determining
         file.
    gzip : bool, (False)
         Whether files with `.gz` endings can be read.
         This option should only be given to files with ASCII text
         output.
         It will automatically call:

          >>> add_sile(ending+'.gz',...,gzip=False)

         to add the gzipped file to the list of possible files.
    """
    global __sile_rules, __siles

    # Only add pure suffixes...
    if ending[0] == '.':
        add_sile(ending[1:], cls, case=case, gzip=gzip, _parent_cls=_parent_cls)
        return

    # The parent_obj is the actual class used to construct
    # the output file
    if _parent_cls is None:
        _parent_cls = cls

    # If it isn't already in the list of
    # Siles, add it.
    if cls not in __siles:
        __siles.append(cls)

    # These classes will never be added to the
    # children. It makes no sense to
    # have bases which are common for all files.
    # Or does it?
    # What if a file-extension may both represent a
    # formatted, or a binary file?
    # In that case should:
    #  def_cls = [object, BaseSile]
    def_cls = [object, BaseSile, Sile, SileBin, SileCDF]

    # First we find the base-class
    # This base class must not be
    #  BaseSile
    #  Sile
    #  SileBin
    #  SileCDF
    def get_children(cls):
        # List all childern
        children = list(cls.__bases__)
        for child in children:
            cchildren = get_children(child)
            for cchild in cchildren:
                if cchild not in children:
                    children.append(cchild)
        remove_cls(children, def_cls)
        return children

    def remove_cls(l, clss):
        for ccls in clss:
            if ccls in l:
                l.remove(ccls)

    # Finally we append the child objects
    inherited = get_children(cls)

    # Now we should remove all objects which are descendants from
    # another object in the list
    # We also default this base-class to be removed
    rem = [object, cls]
    for ccls in inherited:
        inh = get_children(ccls)
        for cccls in inh:
            if cccls in inherited:
                rem.append(cccls)
    remove_cls(inherited, rem)

    # Now, we finally have a list of classes which
    # are a single sub-class of the actual class.
    for ccls in inherited:
        add_sile(ending, ccls, case=case, gzip=gzip, _parent_cls=_parent_cls)

    # If the gzip is none, we decide whether we can
    # read gzipped files
    # In particular, if the cls is a `Sile`, we allow
    # such reading
    if not case:
        add_sile(ending.lower(), cls, gzip=gzip, _parent_cls=_parent_cls)
        add_sile(ending.upper(), cls, gzip=gzip, _parent_cls=_parent_cls)

    else:
        # Add the rule of the sile to the list of rules.
        __sile_rules.append((ending, cls, _parent_cls))
        if gzip:
            add_sile(ending + '.gz', cls, case=case, _parent_cls=_parent_cls)


def get_sile_class(file, *args, **kwargs):
    """ Guess the ``Sile`` class corresponding to the input file and return the class

    Parameters
    ----------
    file : str
       the file to be quried for a correct `Sile` object.
       This file name may contain {<class-name>} which sets
       `cls` in case `cls` is not set.
       For instance:
          water.xyz
       will return an ``XYZSile``. 
    cls : class
       In case there are several files with similar file-suffixes
       you may query the exact base-class that should be chosen.
       If there are several ``Sile``s with similar file-endings this
       function returns a random one.
    """
    global __sile_rules, __siles

    # This ensures that the first argument need not be cls
    cls = kwargs.pop('cls', None)

    # Split filename into proper file name and
    # the specification of the type
    tmp_file, fcls = str_spec(file)

    if cls is None and not fcls is None:
        # cls has not been set, and fcls is found
        # Figure out if fcls is a valid sile, if not
        # do nothing (it may be part of the file name)
        # Which is REALLY obscure... but....)
        for sile in __siles:
            if sile.__name__.lower().startswith(fcls.lower()):
                cls = sile
                # Make sure that {class-name} is
                # removed from the file name
                file = tmp_file
                break

    try:
        # Create list of endings on this file
        f = file
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
        for end in end_list:
            for suf, base, fobj in __sile_rules:
                if end != suf:
                    continue
                if cls is None:
                    return fobj
                elif cls == base:
                    return fobj

        # Now we skip the limitation of the suffix,
        # now only the base-class is necessary.
        for end in end_list:

            # Check for object
            for suf, base, fobj in __sile_rules:
                if cls == base:
                    return fobj

        del end_list

        raise NotImplementedError('sile not implemented: {}'.format(file))
    except NotImplementedError as e:
        pass
    except Exception as e:
        import traceback as t
        t.print_exc()
        raise e
    raise NotImplementedError("Sile for file '"+ file + "' could not be found, possibly the file has not been implemented.")


def get_sile(file, *args, **kwargs):
    """ Guess the ``Sile`` corresponding to the input file and return an open object of the corresponding ``Sile``

    Parameters
    ----------
    file : str
       the file to be quried for a correct `Sile` object.
       This file name may contain {<class-name>} which sets
       `cls` in case `cls` is not set.
       For instance:
          water.dat{XYZSile}
       will read the file water.dat as an `XYZSile`.
    cls : class
       In case there are several files with similar file-suffixes
       you may query the exact base-class that should be chosen.
       If there are several `Sile`s with similar file-endings this
       function returns a random one.
    """
    cls = kwargs.pop('cls', None)
    sile = get_sile_class(file, *args, cls=cls, **kwargs)
    return sile(str_spec(file)[0], *args, **kwargs)


def get_siles(attrs=[None]):
    """ Returns all siles with a specific attribute (or all)

    Parameters
    ----------
    attrs : list of attribute names
       limits the returned sile-objects to those that have
       the given attributes `hasattr(sile, attrs)`
    """
    global __siles

    if len(attrs) == 1 and attrs[0] is None:
        return list(__siles)

    siles = []
    for sile in __siles:
        for attr in attrs:
            if hasattr(sile, attr):
                siles.append(sile)
                break

    return siles


class BaseSile(object):
    """ Base class for the Siles """

    def read(self, *args, **kwargs):
        """ Generic read method which should be overloaded in child-classes 

        Parameters
        ----------
        **kwargs :
          keyword arguments will try and search for the attribute `read_<>`
          and call it with the remaining ``**kwargs`` as arguments.
        """
        for key in kwargs.keys():
            # Loop all keys and try to read the quantities
            if hasattr(self, "read_" + key):
                func = getattr(self, "read_" + key)
                # Call read
                return func(kwargs[key], **kwargs)

    def read_sc(self, *args, **kwargs):
        """ Deprecated function which is superseeded by `read_supercell` """
        if getattr(self, 'read_supercell'):
            return self.read_supercell(*args, **kwargs)
        raise ValueError('read_sc is deprecated, please use read_supercell')

    def read_geom(self, *args, **kwargs):
        """ Deprecated function which is superseeded by `read_geometry` """
        if getattr(self, 'read_geometry'):
            return self.read_geometry(*args, **kwargs)
        raise ValueError('read_geom is deprecated, please use read_geometry')

    def read_es(self, *args, **kwargs):
        """ Deprecated function which is superseeded by `read_hamiltonian` """
        if getattr(self, 'read_hamiltonian'):
            return self.read_hamiltonian(*args, **kwargs)
        raise ValueError('read_es is deprecated, please use read_hamiltonian')

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

    def write_sc(self, *args, **kwargs):
        """ Deprecated function which is superseeded by `write_supercell` """
        if getattr(self, 'write_supercell'):
            return self.write_supercell(*args, **kwargs)
        raise ValueError('write_sc is deprecated, please use write_supercell')

    def write_geom(self, *args, **kwargs):
        """ Deprecated function which is superseeded by `write_geometry` """
        if getattr(self, 'write_geometry'):
            return self.write_geometry(*args, **kwargs)
        raise ValueError('write_geom is deprecated, please use write_geometry')

    def write_es(self, *args, **kwargs):
        """ Deprecated function which is superseeded by `write_hamiltonian` """
        if getattr(self, 'write_hamiltonian'):
            return self.write_hamiltonian(*args, **kwargs)
        raise ValueError('write_es is deprecated, please use write_hamiltonian')

    def _setup(self, *args, **kwargs):
        """ Setup the `Sile` after initialization

        Inherited method for setting up the sile.

        This method can be overwritten.
        """
        pass

    def __setup(self, *args, **kwargs):
        """ Setup the `Sile` after initialization

        Inherited method for setting up the sile.

        This method **must** be overwritten *and*
        end with ``self._setup()``.
        """
        self._setup(*args, **kwargs)

    def __getattr__(self, name):
        """ Override to check the handle """
        if name == 'fh':
            raise AttributeError("The filehandle has not been opened yet...")
        return getattr(self.fh, name)

    @classmethod
    def _ArgumentParser_args_single(cls):
        """ Default arguments for the Sile """
        return {}

    # Define the custom ArgumentParser
    def ArgumentParser(self, parser=None, *args, **kwargs):
        """ Returns the arguments that may be available for this Sile

        Parameters
        ----------
        parser: ArgumentParser
           the argument parser to add the arguments to.
        """
        raise NotImplementedError("The ArgumentParser of '"+self.__class__.__name__+"' has not been implemented yet.")

    def ArgumentParser_out(self, parser=None, *args, **kwargs):
        """ Appends additional arguments based on the output of the file

        Parameters
        ----------
        parser: ArgumentParser
           the argument parser to add the arguments to.
        """
        pass

    def __repr__(self):
        """ Return a representation of the `Sile` """
        return ''.join([self.__class__.__name__, '(', self.file, ')'])


def Sile_fh_open(func):
    """ Method decorator for objects to directly implement opening of the
    file-handle upon entry (if it isn't already).
    """

    def pre_open(self, *args, **kwargs):
        if hasattr(self, "fh"):
            return func(self, *args, **kwargs)
        else:
            with self:
                return func(self, *args, **kwargs)
    return pre_open


class Sile(BaseSile):
    """ Class to contain a file with easy access """

    def __init__(self, filename, mode='r', comment='#'):

        self._file = filename
        self._mode = mode
        if isinstance(comment, (list, tuple)):
            self._comment = list(comment)
        else:
            self._comment = [comment]
        self._line = 0

        # Initialize
        self.__setup()

    @property
    def file(self):
        """ File of the current `Sile` """
        return self._file

    def __setup(self):
        """ Setup the `Sile` after initialization

        Inherited method for setting up the sile.

        This method must **never** be overwritten.
        """
        self._setup()

    def _open(self):
        if self.file.endswith('gz'):
            self.fh = gzip.open(self.file)
        else:
            self.fh = open(self.file, self._mode)
        self._line = 0

    def __enter__(self):
        """ Opens the output file and returns it self """
        self._open()
        return self

    def __exit__(self, type, value, traceback):
        self.fh.close()
        # clean-up so that it does not exist
        delattr(self, 'fh')
        self._line = 0
        return False

    @staticmethod
    def is_keys(keys):
        """ Returns true if ``isinstance(keys,(list,np.ndarray))`` """
        return isinstance(keys, (list, np.ndarray))

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
        return line.lower().find(key) >= 0

    @staticmethod
    def line_has_keys(line, keys, case=True):
        found = False
        if case:
            for key in keys:
                found |= line.find(key) >= 0
        else:
            l = line.lower()
            for key in keys:
                found |= l.find(key) >= 0
        return found

    def readline(self, comment=False):
        """ Reads the next line of the file """
        l = self.fh.readline()
        self._line += 1
        if comment:
            return l
        while starts_with_list(l, self._comment):
            l = self.fh.readline()
            self._line += 1
        return l

    def step_to(self, keywords, case=True, reread=True):
        """ Steps the file-handle until the keyword is found in the input """
        # If keyword is a list, it just matches one of the inputs
        found = False
        # The previously read line...
        line = self._line

        # Do checking outside line checks
        if self.is_keys(keywords):
            line_has = self.line_has_keys
            keys = self.keys2case(keywords, case)
        else:
            line_has = self.line_has_key
            keys = self.key2case(keywords, case)

        while not found:
            l = self.readline()
            if l == '':
                break
            found = line_has(l, keys, case=case)

        if not found and (l == '' and line > 0) and reread:
            # We may be in the case where the user request
            # reading the same twice...
            # So we need to re-read the file...
            self.fh.close()
            # Re-open the file...
            self._open()

            # Try and read again
            while not found and self._line <= line:
                l = self.readline()
                if l == '':
                    break
                found = line_has(l, keys, case=case)

        # sometimes the line contains information, as a
        # default we return the line found
        return found, l

    def step_either(self, keywords, case=True):
        """ Steps the file-handle until the keyword is found in the input """
        # If keyword is a list, it just matches one of the inputs

        # Do checking outside line checks
        if self.is_keys(keywords):
            line_has = self.line_has_key
            keys = self.keys2case(keywords, case)
        else:
            found, l = self.step_to(keywords, case)
            return found, 0, l

        # initialize
        found = False
        j = -1

        while not found:
            l = self.readline()
            if l == '':
                break

            for i, key in enumerate(keys):
                found = line_has(l, key, case=case)
                if found:
                    j = i
                    break

        # sometimes the line contains information, as a
        # default we return the line found
        return found, j, l

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


class SileCDF(BaseSile):
    """ Class to contain a file with easy access
    The file format for this file is the NetCDF file format """

    def __init__(self, filename, mode='r', lvl=0, access=1, _open=True):
        """ Creates/Opens a SileCDF

        Opens a SileCDF with `mode` and compression level `lvl`.
        If `mode` is in read-mode (r) the compression level
        is ignored.

        The final `access` parameter sets how the file should be
        open and subsequently accessed.

        0) means direct file access for every variable read
        1) means stores certain variables in the object.
        """

        self._file = filename
        # Open mode
        self._mode = mode
        # Save compression internally
        self._lvl = lvl
        # Initialize the _data dictionary for access == 1
        self._data = dict()
        if isfile(self.file):
            self._access = access
        else:
            # If it has not been created we should not try
            # and read anything, no matter what the user says
            self._access = 0

        # The CDF file can easily open the file
        if _open:
            global _import_netCDF4
            _import_netCDF4()
            self.__dict__['fh'] = _netCDF4.Dataset(self.file, self._mode,
                                                   format='NETCDF4')

        # Must call setup-methods
        self.__setup()

    @property
    def file(self):
        """ Filename of the current `Sile` """
        return self._file

    def _setup(self, *args, **kwargs):
        """ Simple setup that needs to be overloaded """
        pass

    def __setup(self):
        """ Setup `SileCDF` after initialization """
        self._setup()

    @property
    def _cmp_args(self):
        """ Returns the compression arguments for the NetCDF file

        Do
          >>> nc.createVariable(..., **self._cmp_args)
        """
        return {'zlib': self._lvl > 0, 'complevel': self._lvl}

    def __enter__(self):
        """ Opens the output file and returns it self """
        # We do the import here
        return self

    def __exit__(self, type, value, traceback):
        return False

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
        if self._access > 0:
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
        if '/' in name:
            groups = name.split('/')
            grp = n
            for group in groups:
                if len(group) > 0:
                    grp = _crt_grp(grp, group)
            return grp

        if name in n.groups:
            return n.groups[name]
        return n.createGroup(name)

    @staticmethod
    def _crt_dim(n, name, l):
        if name in n.dimensions:
            return
        n.createDimension(name, l)

    @staticmethod
    def _crt_var(n, name, *args, **kwargs):
        if name in n.variables:
            return n.variables[name]

        attr = None
        if 'attr' in kwargs:
            attr = kwargs.pop('attr')
        v = n.createVariable(name, *args, **kwargs)
        if attr is not None:
            for name in attr:
                setattr(v, name, attr[name])
        return v

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
        >>>     if self.isGroup(gv):
        >>>         # is group
        >>>     elif self.isDimension(gv):
        >>>         # is dimension
        >>>     elif self.isVariable(gv):
        >>>         # is variable

        """
        if root is None:
            head = self.fh
        else:
            head = self.fh[root]

        # Start by returning the root group
        if group:
            yield head

        if dimension:
            for dim in head.dimensions.values():
                yield dim
        if variable:
            for var in head.variables.values():
                yield var

        if levels == 0:
            # Stop the iterator
            return

        for grp in head.groups.values():
            for dvg in self.iter(group, dimension, variable,
                                 levels=levels-1, root=grp.path):
                yield dvg

    __iter__ = iter


class SileBin(BaseSile):
    """ Class to contain a file with easy access
    The file format for this file is a binary format.
    """

    def __init__(self, filename, mode='r'):
        """ Creates/Opens a SileBin

        Opens a SileBin with `mode` (b).
        If `mode` is in read-mode (r).
        """

        self._file = filename
        # Open mode
        self._mode = mode.replace('b', '') + 'b'

        # Must call setup-methods
        self.__setup()

    @property
    def file(self):
        """ File of the current `Sile` """
        return self._file

    def _setup(self, *args, **kwargs):
        """ Simple setup that needs to be overwritten """
        pass

    def __setup(self):
        """ Setup `SileBin` after initialization """
        self._setup()

    def __enter__(self):
        """ Opens the output file and returns it self """
        return self

    def __exit__(self, type, value, traceback):
        return False


class SileError(IOError):
    """ Define an error object related to the Sile objects """

    def __init__(self, value, obj=None):
        self.value = value
        self.obj = obj

    def __str__(self):
        s = ''
        if self.obj:
            s = self.obj.__name__ + '(' + self.obj.file + ')'

        return self.value + ' in ' + s


def sile_raise_write(self):
    if not ('w' in self._mode or 'a' in self._mode):
        raise SileError('Writing to a read-only file not possible', self)


def sile_raise_read(self):
    if not ('r' in self._mode or 'a' in self._mode):
        raise SileError('Reading a write-only file not possible', self)
