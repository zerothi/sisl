"""
Sile object for reading/writing CONTCAR/POSCAR files
"""

from __future__ import print_function

# Import sile objects
from sids.io.sile import *

# Import the geometry object
from sids import Geometry, Atom, SuperCell

import numpy as np

__all__ = ['POSCARSile']


class POSCARSile(Sile):
    """ CAR file object 
    This file-object handles both POSCAR and CONTCAR files
    """
    # These are the comments
    _comment = []
    _scale = 1.

    def write_geom(self,geom):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self:
                return self.write_geom(geom)

        # LABEL
        self._write('sids output\n')

        # Scale
        self._write('  1.\n')

        # Write unit-cell
        fmt = ('   ' + '{:18.9f}'*3)*2 + '\n'
        for i in range(3):
            tmp[0:3] = geom.cell[i,:]
            self._write(fmt.format(*geom.cell[i,:]))

        # Figure out how many species
        d = []
        for ia,a,idx_specie in geom.iter_species():
            if idx_specie > len(d):
                d.append(0)
            d[idx_specie] += + 1
        fmt = ' ' + '{:d}' * len(d) + '\n'
        self._write(fmt.format(*d))
        self._write('Cartesian\n')

        fmt = '{:18.9f}'*3 + '\n'
        for ia in geom:
            self._write(fmt.format(*geom.xyz[ia,:]))

        
    def read_sc(self):
        """ Returns `SuperCell` object from the CONTCAR/POSCAR file """
        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_sc()

        # read first line
        self.readline() # LABEL
        # Update scale-factor
        self._scale = float(self.readline())

        # Read cell vectors
        cell = np.empty([3,3],np.float64)
        for i in range(3):
            cell[i,:] = np.fromstring(self.readline(), dtype=float, count=3, sep = ' ')
        cell *= self._scale

        return SuperCell(cell)
    

    def read_geom(self):
        """ Returns Geometry object from the CONTCAR/POSCAR file 
        """
        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_geom()

        sc = self.read_sc()

        # Try and read number of atoms per species 
        # This is one of the two next lines
        nspecies = np.array(self.readline().split(),np.int32)

        # Read whether this is selective or direct
        opt = self.readline()
        direct = True
        if opt[0] in 'Ss':
            direct = False
            opt = self.readline()

        # Check whether this is in fractional or direct
        # coordinates
        cart = False
        if opt[0] in 'CcKk':
            cart = True

        # No matter, we only read the first coordinates
        
        # Number of atoms
        na = np.sum(nspecies)

        xyz = np.empty([na,3],np.float64)
        for ia in range(na):
            xyz[ia,:] = np.fromstring(self.readline(),dtype=float,count=3,sep = ' ')
        if cart:
            # The unit of the coordinates are cartesian
            xyz *= self._scale
        else:
            xyz = np.dot(xyz,sc.cell.T)

        # The POT/CONT-CAR does not contain information on the atomic species
        return Geometry(xyz=xyz,sc=sc)


if __name__ == "__main__":
    # Create geometry
    alat = 3.57
    dist = alat * 3. **.5 / 4
    C = Atom(Z=6,R=dist * 1.01,orbs=2)
    geom = Geometry(np.array([[0,0,0],[1,1,1]],np.float64)*alat/4,
                    atoms = C, sc=SuperCell(np.array([[0,1,1],
                                                      [1,0,1],
                                                      [1,1,0]],np.float64) * alat/2))
    # Write stuff
    print(geom)
    geom.write(POSCARSile('CONTCAR','w'))
    geomr = POSCARSile('CONTCAR','r').read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)
