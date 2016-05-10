"""
IO imports
"""
from __future__ import print_function, division
from os.path import splitext
import sys

from .sile import *

# Import the different Sile objects
# enabling the actual print-out
from .bigdft_ascii import *
from .cube import *
from .fdf import *
from .gulp import *
from .siesta import *
from .siesta_grid import *
from .tb import *
from .tbtrans import *
from .vasp import *
from .xyz import *
from .xv import *

# Default functions in this top module
__all__ = ['add_sile', 'get_sile']

# Extend all by the sub-modules


def extendall(mod):
    global __all__
    __all__.extend(sys.modules[mod].__dict__['__all__'])

extendall('sisl.io.sile')
extendall('sisl.io.bigdft_ascii')
extendall('sisl.io.cube')
extendall('sisl.io.fdf')
extendall('sisl.io.gulp')
extendall('sisl.io.siesta')
extendall('sisl.io.siesta_grid')
extendall('sisl.io.tb')
extendall('sisl.io.tbtrans')
extendall('sisl.io.vasp')
extendall('sisl.io.xyz')
extendall('sisl.io.xv')


# This is a file chooser which from the file-ending tries to
# determine which kind of file we are dealing with.

_objs = {}


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
    global _objs
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
    _objs[ending] = obj
    if ending[0] == '.':
        _objs[ending[1:]] = obj
    else:
        _objs['.' + ending] = obj


# Sile's
add_sile('ascii', BigDFTASCIISile, case=False, gzip=True)
add_sile('cube', CUBESile, case=False, gzip=True)
add_sile('fdf', FDFSile, case=False, gzip=True)
add_sile('gout', GULPSile, gzip=True)
add_sile('tb', TBSile, case=False, gzip=True)
add_sile('xyz', XYZSile, case=False, gzip=True)
add_sile('XV', XVSile, gzip=True)
add_sile('CONTCAR', POSCARSile, gzip=True)
add_sile('POSCAR', POSCARSile, gzip=True)

# NCSile's
add_sile('nc', SIESTASile, case=False)
add_sile('grid.nc', SIESTAGridSile, case=False)
add_sile('TBT.nc', TBtransSile)
add_sile('PHT.nc', PHtransSile)


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
            if end in _objs:
                return _objs[end](file, *args, **kwargs)

        raise Exception('print fail')
    except Exception as e:
        print(e)
        raise NotImplementedError(
            ("File requested could not be found, possibly the file has ",
             "not been implemented."))


if __name__ == "__main__":

    assert isinstance(get_sile('test.cube'),
                      CUBESile), "Returning incorrect object"
    assert isinstance(get_sile('test.CUBE'),
                      CUBESile), "Returning incorrect object"

    assert isinstance(get_sile('test.xyz'),
                      XYZSile), "Returning incorrect object"
    assert isinstance(get_sile('test.XYZ'),
                      XYZSile), "Returning incorrect object"
    try:
        io = get_sile('test.xz')
        assert False, "Returning something which should not return"
    except:
        pass
    assert isinstance(get_sile('test.fdf'),
                      FDFSile), "Returning incorrect object"
    assert isinstance(get_sile('test.FDF'),
                      FDFSile), "Returning incorrect object"
    assert isinstance(get_sile('test.nc'),
                      SIESTASile), "Returning incorrect object"
    assert isinstance(get_sile('test.NC'),
                      SIESTASile), "Returning incorrect object"
    assert isinstance(get_sile('test.tb'),
                      TBSile), "Returning incorrect object"
    assert isinstance(get_sile('test.TB'),
                      TBSile), "Returning incorrect object"
    assert isinstance(get_sile('test.got'),
                      GULPSile), "Returning incorrect object"
    assert isinstance(get_sile('test.XV'),
                      XVSile), "Returning incorrect object"
    assert isinstance(get_sile('test.TBT.nc'),
                      TBtransSile), "Returning incorrect object"
    assert isinstance(get_sile('test.ueontsh.nc.XV'),
                      XVSile), "Returning incorrect object"
    print('Finished tests successfully')
