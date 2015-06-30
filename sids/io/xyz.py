"""
Sile object for reading/writing FDF files
"""

from __future__ import print_function

# Import sile objects
from sids.io.sile import *

# Import the geometry object
from sids.geom import Geometry, Atom

import numpy as np

class XYZSile(Sile):
    """ XYZ file object """
    # These are the comments
    _comment = []

    def write_geom(self,geom,fmt='.5f'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        with self as fh:

            # Write out the cell
            fh._write('  '+str(len(geom))+'\n')
            # We write the cell coordinates as the cell coordinates
            fmt_str = '{{:{0}}} '.format(fmt)*9 + '\n'
            fh._write(fmt_str.format(*geom.cell.flatten()))

            fmt_str = '{{0:2s}}  {{1:{0}}}  {{2:{0}}}  {{3:{0}}}\n'.format(fmt)
            for ia,a,isp in geom.iter_species():
                fh._write(fmt_str.format(a.symbol,*geom.xyz[ia,:]))


    def read_geom(self):
        """ Returns Geometry object from the XYZ file 

        NOTE: Unit-cell is the Eucledian 3D space.
        """
        cell = np.asarray(np.diagflat([1]*3),np.float64)
        with self as fh:
            l = fh.readline()
            na = int(l)
            l = fh.readline()
            l = l.split()
            if len(l) == 9:
                # we possibly have the cell as a comment
                cell.shape = (9,)
                for i,il in enumerate(l): 
                    cell[i] = float(il)
                cell.shape = (3,3)
            sp = [None] * na
            xyz = np.empty([na,3],np.float64)
            for ia in xrange(na):
                l = fh.readline().split()
                sp[ia] = l.pop(0)
                xyz[ia,:] = [float(k) for k in l[:3]]
        return Geometry(cell=cell,xyz=xyz,atoms=sp)

if __name__ == "__main__":
    # Create geometry
    alat = 3.57
    dist = alat * 3. **.5 / 4
    C = Atom(Z=6,R=dist * 1.01,orbs=2)
    geom = Geometry(cell=np.array([[0,1,1],
                                   [1,0,1],
                                   [1,1,0]],np.float64) * alat/2,
                    xyz = np.array([[0,0,0],[1,1,1]],np.float64)*alat/4,
                    atoms = C )
    # Write stuff
    print(geom)
    geom.write(XYZSile('diamond.xyz','w'))
    io = XYZSile('diamond.xyz','r')
    geomr = io.read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)
