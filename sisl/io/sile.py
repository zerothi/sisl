from __future__ import print_function, division

from os.path import splitext, isfile
import gzip

import numpy as np

from sisl import Geometry
from ._help import *


# Public used objects
__all__ = [
    'sile_objects', 
    'add_sile',
    'get_sile']

__all__ += [
    'BaseSile',
    'Sile',
    'NCSile',
    'SileError',
    ]

__all__ += [
    'Sile_fh_open',
    'sile_raise_write',
    'sile_raise_read']

# Global container of all Sile rules
sile_objects = {}

def add_sile(ending, obj, case=True, gzip=False):
    """
    Public for attaching lookup tables for allowing
    users to attach files for the IOSile function call

    Parameters
    ----------
    ending : str
         The file-name ending, it can be several file endings (.TBT.nc)
    obj : `BaseSile` child
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
    global sile_objects
    # If the gzip is none, we decide whether we can
    # read gzipped files
    # In particular, if the obj is a `Sile`, we allow
    # such reading
    if gzip:
        add_sile(ending + '.gz', obj, case=case)
    if not case:
        add_sile(ending.lower(), obj, gzip=gzip)
        add_sile(ending.upper(), obj, gzip=gzip)
        return
    sile_objects[ending] = obj
    if ending[0] == '.':
        sile_objects[ending[1:]] = obj
    else:
        sile_objects['.' + ending] = obj


def get_sile(file, *args, **kwargs):
    """
    Guess the file handle for the input file and return
    and object with the file handle.
    """
    try:
        # Create list of endings on this file
        f = file
        end_list = []
        end = ''

        # Check for files without ending, or that they are directly zipped
        lext = splitext(f)
        while len(lext[1]) > 0:
            end = lext[1] + end
            end_list.append(end)
            lext = splitext(lext[0])

        # We also check the entire file name
        #  (mainly for VASP)
        end_list.append(f)

        while end_list:
            end = end_list.pop()

            # Check for ending and possibly
            # return object
            if end in sile_objects:
                return sile_objects[end](file, *args, **kwargs)

        raise Exception('print fail')
    except Exception as e:
        print(e)
    raise NotImplementedError("File requested could not be found, possibly the file has not been implemented.")


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



class NCSile(BaseSile):
    """ Class to contain a file with easy access
    The file format for this file is the NetCDF file format """

    def __init__(self, filename, mode='r', lvl=0, access=1):
        """ Creates/Opens a NCSile

        Opens a NCSile with `mode` and compression level `lvl`.
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
        if isfile(self.file):
            self._access = access
        else:
            # If it has not been created we should not try
            # and read anything, no matter what the user says
            self._access = 0

        # Must call setup-methods
        self.__setup()


    def _setup(self,*args, **kwargs):
        """ Simple setup that needs to be overwritten """
        pass


    def __setup(self):
        """ Setup `NCSile` after initialization """
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
        global _import_netCDF4
        _import_netCDF4()
        self.__dict__['fh'] = _netCDF4.Dataset(self.file, self._mode, 
                                               format='NETCDF4')
        return self


    def __exit__(self, type, value, traceback):
        if 'fh' in self.__dict__:
            self.__dict__['fh'].close()
            # clean-up so that it does not exist
            del self.__dict__['fh']
        return False


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
