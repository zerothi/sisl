"""
Sile object for reading TBtrans binary files
"""
from __future__ import print_function

# Import sile objects
from sids.io.sile import *

# Import the geometry object
from sids import Geometry, Atom, SuperCell

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

        cell = np.array(self.variables['cell'][:],np.float64)
        cell.shape = (3,3)
        xyz = np.array(self.variables['xa'][:],np.float64)
        xyz.shape = (-1,3)

        # Create list with correct number of orbitals
        lasto = np.array(self.variables['lasto'][:],np.int32)
        orbs = np.append([lasto[0]],np.diff(lasto))
        orbs = np.array(orbs,np.int32)

        atms = [Atom(Z='H',orbs=o) for o in orbs]
            
        # Create and return geometry object
        geom = Geometry(xyz,atoms=atms,sc=SuperCell(cell))
        geom.sc.cell *= geom.Length
        geom.set_supercell(geom.sc.copy())
        geom.xyz *= geom.Length
        return geom

    
    def write_geom(self):
        """ This does not work """
        raise ValueError(self.__class__.__name__+" can not write a geometry")

