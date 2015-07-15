"""
IO imports
"""
from __future__ import print_function, division
from os.path import splitext
import sys

from sids.io.sile import *

# Import the different Sile objects
# enabling the actual print-out
from sids.io.fdf import *
from sids.io.gulp import *
from sids.io.siesta import *
from sids.io.tb import *
from sids.io.tbtrans import *
from sids.io.xyz import *
from sids.io.xv import *

# Default functions in this top module
__all__ = ['add_sile','get_sile']

# Extend all by the sub-modules
def extendall(mod):
    global __all__
    __all__.extend(sys.modules[mod].__dict__['__all__'])

extendall('sids.io.sile')
extendall('sids.io.fdf')
extendall('sids.io.gulp')
extendall('sids.io.siesta')
extendall('sids.io.tb')
extendall('sids.io.tbtrans')
extendall('sids.io.xyz')
extendall('sids.io.xv')


# This is a file chooser which from the file-ending tries to 
# determine which kind of file we are dealing with.

_objs = {}

def add_sile(ending,obj,case=True):
    """
    Public for attaching lookup tables for allowing
    users to attach files for the IOSile function call
    """
    global _objs
    if not case:
        add_sile(ending.lower(),obj)
        add_sile(ending.upper(),obj)
        return
    _objs[ending] = obj
    if ending[0] == '.':
        _objs[ending[1:]] = obj
    else:
        _objs['.'+ending] = obj


add_sile('xyz',XYZSile,case=False)
add_sile('fdf',FDFSile,case=False)
add_sile('nc',SIESTASile,case=False)
add_sile('tb',TBSile,case=False)
add_sile('TBT.nc',TBtransSile)
add_sile('got',GULPSile)
add_sile('XV',XVSile)

def get_sile(file,*args,**kwargs):
    """ 
    Guess the file handle for the input file and return
    and object with the file handle.
    """
    try:
        # Create list of endings on this file
        end_list = []
        f = file
        end = ''
        while True:
            lext = splitext(f)
            end = lext[1] + end
            end_list.append(end)
            if len(lext[1]) == 0: break
            f = lext[0]
        while end_list:
            end = end_list.pop()
            try:
                return _objs[end](file,*args,**kwargs)
            except:
                pass
        raise Exception('print fail')
    except Exception as e:
        raise NotImplementedError("File requested could not be found, possibly the file has "+
                                  "not been implemented.")


if __name__ == "__main__":
    
    assert isinstance(get_sile('test.xyz'),XYZSile),"Returning incorrect object"
    assert isinstance(get_sile('test.XYZ'),XYZSile),"Returning incorrect object"
    try:
        io = get_sile('test.xz')
        assert False,"Returning something which should not return"
    except: pass
    assert isinstance(get_sile('test.fdf'),FDFSile),"Returning incorrect object"
    assert isinstance(get_sile('test.FDF'),FDFSile),"Returning incorrect object"
    assert isinstance(get_sile('test.nc'),SIESTASile),"Returning incorrect object"
    assert isinstance(get_sile('test.NC'),SIESTASile),"Returning incorrect object"
    assert isinstance(get_sile('test.tb'),TBSile),"Returning incorrect object"
    assert isinstance(get_sile('test.TB'),TBSile),"Returning incorrect object"
    assert isinstance(get_sile('test.got'),GULPSile),"Returning incorrect object"
    assert isinstance(get_sile('test.XV'),XVSile),"Returning incorrect object"
    assert isinstance(get_sile('test.TBT.nc'),TBtransSile),"Returning incorrect object"
    assert isinstance(get_sile('test.ueontsh.nc.XV'),XVSile),"Returning incorrect object"
    print('Finished tests successfully')


