"""
Sile object for reading/writing XV files
"""

from __future__ import print_function

# Import sile objects
from sids.io.sile import *

# Import the geometry object
from sids.geom import Geometry, Atom

import numpy as np

__all__ = ['XVSile']


class XVSile(Sile):
    """ XV file object """
    # These are the comments
    _comment = []

    def write_geom(self,geom):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self:
                return self.write_geom(geom)

        # Write unit-cell
        tmp = np.zeros(6,np.float)
        fmt = ('   ' + '{:18.9f}'*3)*2 + '\n'
        for i in range(3):
            tmp[0:3] = geom.cell[i,:] / geom.Length
            self._write(fmt.format(*tmp))
        self._write('{:12d}\n'.format(geom.na))
        fmt  = '{:3d}{:6d}'
        fmt += '{:18.9f}'*3 + '   ' + '{:18.9f}'*3
        fmt += '\n'
        for ia, a, ips in geom.iter_species():
            tmp[0:3] = geom.xyz[ia,:] / geom.Length
            self._write(fmt.format(ips+1,a.Z,*tmp))

    def read_geom(self):
        """ Returns Geometry object from the XV file 
        """
        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_geom()

        cell = np.empty([3,3],np.float)
        for i in range(3):
            cell[i,:] = np.fromstring(self.readline(), dtype=float, sep = ' ')[0:3]
        cell *= Geometry.Length
        # Read number of atoms
        na = int(self.readline())
        atms = [None] * na
        xyz = np.empty([na,3],np.float)
        line = np.empty(8,np.float)
        for ia in xrange(na):
            line[:] = np.fromstring(self.readline(),dtype=float,sep = ' ')[0:8]
            atms[ia] = Atom[int(line[1])]
            xyz[ia,:] = line[2:5]
        xyz *= Geometry.Length

        return Geometry(cell=cell,xyz=xyz,atoms=atms)


if __name__ == "__main__":
    # Create geometry
    alat = 3.57
    dist = alat * 3. **.5 / 4
    C = Atom(Z=6,R=dist * 1.01,orbs=2)
    geom = Geometry(cell=np.array([[0,1,1],
                                   [1,0,1],
                                   [1,1,0]],np.float) * alat/2,
                    xyz = np.array([[0,0,0],[1,1,1]],np.float)*alat/4,
                    atoms = C )
    # Write stuff
    print(geom)
    geom.write(XVSile('diamond.XV','w'))
    geomr = XVSile('diamond.XV','r').read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)
