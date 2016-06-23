"""
IO imports
"""
from __future__ import print_function, division
import sys

from ._help import extendall
from .sile import *

# Import the different Sile objects
# enabling the actual print-out
from .bigdft import *
from .cube import *
from .gulp import *
from .siesta import *
from .tb import *
from .vasp import *
from .xyz import *

# Default functions in this top module
__all__ = ['add_sile', 'get_sile']

extendall(__all__, 'sisl.io.sile')

extendall(__all__, 'sisl.io.bigdft')
extendall(__all__, 'sisl.io.cube')
extendall(__all__, 'sisl.io.gulp')
extendall(__all__, 'sisl.io.tb')
extendall(__all__, 'sisl.io.siesta')
extendall(__all__, 'sisl.io.vasp')
extendall(__all__, 'sisl.io.xyz')


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
