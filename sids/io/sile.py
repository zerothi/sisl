from __future__ import print_function, division

from sids import Geometry
from sids.io._help import *

import numpy as np

__all__ = ['Sile','NCSile','SileError','sile_raise_write','sile_raise_read']


class Sile(object):
    """ Class to contain a file with easy access """
    _mode = 'r'
    _comment = ['#']

    def __init__(self,filename,mode=None):
        self.file = filename
        if mode:
            self._mode = mode
        
    def __enter__(self):
        """ Opens the output file and returns it self """
        self.fh = open(self.file,self._mode)
        return self

    def __exit__(self, type, value, traceback):
        self.fh.close()
        # clean-up so that it does not exist
        del self.fh
        return False

    @staticmethod
    def line_has_key(line,keyword,case=True):
        found = False
        if isinstance(keyword,(list,np.ndarray)):
            if not case: keyword = [k.lower() for k in keyword]
            if not case: 
                l = line.lower()
            else:
                l = line
            for key in keyword:
                found |= l.find(key) >= 0
        else:
            if not case: keyword = keyword.lower()
            if not case: 
                l = line.lower()
            else:
                l = line
            found = l.find(keyword) >= 0
        return found    

    def readline(self,comment=False):
        """ Reads the next line of the file """
        l = self.fh.readline()
        if comment: return l
        while starts_with_list(l,self._comment):
            l = self.fh.readline()
        return l

    def step_to(self,keyword,case=True):
        """ Steps the file-handle until the keyword is found in the input """
        # If keyword is a list, it just matches one of the inputs
        found = False

        while not found:
            l = self.readline()
            if l == '': return found,''
            found = self.line_has_key(l,keyword,case=case)
            
        # sometimes the line contains information, as a
        # default we return the line found
        return found, l

    def write(self,*args,**kwargs):
        """
        Wrapper for the file-handle write statement
        """
        for arg in args:
            if isinstance(arg,Geometry):
                self.write_geom(arg)
        if 'geom' in kwargs: 
            self.write_geom(kwargs['geom'])
        

    def _write(self,*args,**kwargs):
        """ Wrapper to default the write statements """
        self.fh.write(*args,**kwargs)

class NCSile(object):
    """ Class to contain a file with easy access
    The file format for this file is the NetCDF file format """
    _mode = 'r'
    _lvl = 0

    def __init__(self,filename,mode=None,lvl=0):
        self.file = filename
        if mode:
            self._mode = mode
        # Save compression internally
        self._lvl = lvl

    def _cmp_args(self):
        """ Returns the compression arguments for the NetCDF file

        Do
          >>> nc.createVariable(..., **self._cmp_args)
        """
        return {'zlib':self._lvl>0,'complevel':self._lvl}
        
    def __enter__(self):
        """ Opens the output file and returns it self """
        # We do the import here
        try:
            import netCDF4
        except:
            raise ImportError("Could not import required module netCDF4, "+
                              "please add netCDF4 to your path and re-run.")
        self.fh = netCDF4.Dataset(self.file,self._mode,format='NETCDF4')
        return self

    def __getattr__(self,attr):
        """ Bypass attributes to directly interact with the NetCDF model """
        return getattr(self.fh,attr,None)

    def __exit__(self, type, value, traceback):
        self.fh.close()
        # clean-up so that it does not exist
        del self.fh
        return False

    @staticmethod
    def _crt_grp(n,name):
        if name in n.groups: return n.groups[name]
        return n.createGroup(name)

    @staticmethod
    def _crt_dim(n,name,l):
        if name in n.dimensions: return
        n.createDimension(name,l)

    @staticmethod
    def _crt_var(n,name,*args,**kwargs):
        if name in n.variables: return n.variables[name]
        return n.createVariable(name,*args,**kwargs)


class SileError(IOError):
    """ Define an error object related to the Sile objects """

    def __init__(self, value,obj=None):
        self.value = value
        self.obj = obj

    def __str__(self):
        s = ''
        if self.obj:
            s = self.obj.__name__ + '(' + self.obj.file + ')'
            
        return self.value + ' in ' + s

def sile_raise_write(self):
    if not ('w' in self._mode or 'a' in self._mode):
        raise SileError('Writing to a read-only file not possible',self)
def sile_raise_read(self):
    if not ('r' in self._mode or 'a' in self._mode):
        raise SileError('Reading a write-only file not possible',self)

if __name__ == "__main__":
    i1 = Sile('f.dat','a')
    i2 = Sile('f.dat')

    print(i1._mode)
    print(i2._mode)

