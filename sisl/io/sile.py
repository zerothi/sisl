from __future__ import print_function, division

from os.path import splitext, isfile
import gzip

import numpy as np

from sisl import Geometry
from sisl.utils.misc import name_spec
from ._help import *


# Public used objects
__all__ = [
    'add_sile',
    'get_sile',
    'get_siles']

__all__ += [
    'BaseSile',
    'Sile',
    'SileCDF',
    'SileBin',
    'SileError',
    ]

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
        __sile_rules.append( (ending, cls, _parent_cls) )
        if gzip:
            add_sile(ending + '.gz', cls, case=case, _parent_cls=_parent_cls)



def get_sile(file, *args, **kwargs):
    """
    Guess the file handle for the input file and return
    and object with the file handle.
    
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
    global __sile_rules, __siles

    # This ensures that the first argument
    # need not be cls
    cls = kwargs.pop('cls', None)
    
    # Split filename into proper file name and
    # the specification of the type
    tmp_file, fcls = name_spec(file)

    if cls is None and not fcls is None:
        # cls has not been set, and fcls is found
        # Figure out if fcls is a valid sile, if not
        # do nothing (it may be part of the file name)
        # Which is REALLY obscure... but....)
        for sile in __siles:
            if sile.__name__.lower() == fcls.lower():
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
                    return fobj(file, *args, **kwargs)
                elif cls == base:
                    return fobj(file, *args, **kwargs)

        # Now we skip the limitation of the suffix,
        # now only the base-class is necessary.
        for end in end_list:

            # Check for object
            for suf, base, fobj in __sile_rules:
                if cls == base:
                    return fobj(file, *args, **kwargs)

        del end_list
        
        raise Exception('sile not implemented')
    except Exception as e:
        print(e)
        raise NotImplementedError("File '"+ file + "' requested could not be found, possibly the file has not been implemented.")
    raise NotImplementedError("File '"+ file + "' requested could not be found, possibly the file has not been implemented.")


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

        self.file = filename
        self._mode = mode
        self._comment = [comment]
        self._line = 0

        # Initialize
        self.__setup()


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


    def step_to(self, keywords, case=True):
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

        if not found and (l == '' and line > 0):
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

    def write(self, *args, **kwargs):
        """
        Wrapper for the file-handle write statement
        """
        for arg in args:
            if isinstance(arg, Geometry):
                self.write_geom(arg)
        if 'geom' in kwargs:
            self.write_geom(kwargs['geom'])

    def _write(self, *args, **kwargs):
        """ Wrapper to default the write statements """
        self.fh.write(*args, **kwargs)


# Instead of importing netCDF4 on each invocation
# of the __enter__ function (below), we make
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

        self.file = filename
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


    def _setup(self,*args, **kwargs):
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
        return self._variables(self, name, tree=tree)[:]
    
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


class SileBin(BaseSile):
    """ Class to contain a file with easy access
    The file format for this file is a binary format.
    """

    def __init__(self, filename, mode='r'):
        """ Creates/Opens a SileBin

        Opens a SileBin with `mode` (b).
        If `mode` is in read-mode (r).
        """

        self.file = filename
        # Open mode
        self._mode = mode.replace('b','') + 'b'

        # Must call setup-methods
        self.__setup()


    def _setup(self,*args, **kwargs):
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
