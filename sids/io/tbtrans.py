"""
Sile object for reading TBtrans binary files
"""
from __future__ import print_function

# Import sile objects
from sids.io.sile import *

# Import the geometry object
from sids.geom import Geometry, Atom

import numpy as np

__all__ = ['TBtransSile']


class TBtransSile(NCSile):
    """ TBtrans file object """

    def read_geom(self):
        """ Returns Geometry object from a *.TBT.nc file 
        """
        if not hasattr(self,'fh'):
            with self:
                return self.read_geom()

        cell = np.array(self.variables['cell'][:],np.float)
        cell.shape = (3,3)
        xyz = np.array(self.variables['xa'][:],np.float)
        xyz.shape = (-1,3)

        # Create list with correct number of orbitals
        lasto = np.array(self.variables['lasto'][:],np.int)
        orbs = np.append([lasto[0]],np.diff(lasto))
        orbs = np.array(orbs,np.int)

        atms = [Atom(Z='H',orbs=o) for o in orbs]
            
        # Create and return geometry object
        geom = Geometry(cell,xyz,atoms=atms)
        geom.cell *= geom.Length
        geom.xyz *= geom.Length
        return geom

    
    def write_geom(self):
        """ This does not work """
        raise ValueError(self.__class__.__name__+" can not write a geometry")

