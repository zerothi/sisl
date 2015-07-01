"""
Sile object for reading/writing TB in/output
"""
from __future__ import print_function

# Import sile objects
from sids.io.sile import *

# Import the geometry object
from sids.geom import Geometry, Atom

import numpy as np

class TBSile(Sile):
    """ Tight-binding file object """

    def read_geom(self):
        """ Reading a geometry in regular TB format """
        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_geom()

        cell = np.zeros([3,3],np.float)
        Z = []
        xyz = []

        nsc = np.zeros([3],np.int)

        def Z2no(i,no):
            try:
                # pure atomic number
                return int(i),no
            except:
                # both atomic number and no
                j = i.replace('[',' ').replace(']',' ').split()
                return int(j[0]),int(j[1])
        
        # The format of the geometry file is
        keys = ['atoms','cell','supercells','nsc']
        for _ in range(len(keys)):
            f, l = self.step_to(keys,case=False)
            l = l.strip()
            if 'supercells' in l.lower() or 'nsc' in l.lower():
                # We have everything in one line
                l = l.split()[1:]
                for i in range(3): nsc[i] = int(l[i])
            elif 'cell' in l.lower():
                if 'begin' in l.lower():
                    for i in range(3):
                        l = self.readline().split()
                        cell[i,0] = float(l[0])
                        cell[i,1] = float(l[1])
                        cell[i,2] = float(l[2])
                    self.readline() # step past the block
                else:
                    # We have everything in one line
                    l = l.split()[1:]
                    for i in range(3):
                        cell[i,i] = float(l[i])
                    # TODO incorporate rotations
            elif 'atoms' in l.lower():
                l = self.readline()
                while not l.startswith('end'):
                    ls = l.split()
                    try:
                        no = int(ls[4])
                    except:
                        no = 1
                    z, no = Z2no(ls[0],no)
                    Z.append({'Z':z,'orbs':no})
                    xyz.append([float(f) for f in ls[1:4]])
                    l = self.readline()
                xyz = np.array(xyz,np.float)
                xyz.shape = (-1,3)
                self.readline() # step past the block

        # Return the geometry
        # Create list of atoms
        geom = Geometry(cell,xyz,atoms=Atom[Z])
        geom.set_supercell(nsc=nsc)

        return geom

    def write_geom(self,geom,**kwargs):
        """
        Writes the geometry to the output file

        Parameters
        ----------
        geom: Geometry
              The geometry we wish to write
        """
        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self as fh:
                fh.write_geom(geom)
            return

        # The format of the geometry file is
        # for now, pretty stringent
        # Get cell_fmt
        cell_fmt = '.5f'
        if 'fmt' in kwargs: cell_fmt = kwargs['fmt']
        if 'cell_fmt' in kwargs: cell_fmt = kwargs['cell_fmt']
        xyz_fmt = '.4e'
        if 'fmt' in kwargs: xyz_fmt = kwargs['fmt']
        if 'xyz_fmt' in kwargs: xyz_fmt = kwargs['xyz_fmt']


        self._write('begin cell\n')
        # Write the cell
        fmt_str = '  {{0:{0}}} {{1:{0}}} {{2:{0}}}\n'.format(cell_fmt)
        for i in range(3):
            self._write(fmt_str.format(*geom.cell[i,:]))
        self._write('end cell\n')

        # Write number of super cells in each direction
        self._write('\nsupercells {0:d} {1:d} {2:d}\n'.format(*(geom.nsc//2)))
        
        # Write all atomic positions along with the specie type
        self._write('\nbegin atoms\n')
        fmt1_str = '  {{0:d}} {{1:{0}}} {{2:{0}}} {{3:{0}}}\n'.format(xyz_fmt)
        fmt2_str = '  {{0:d}}[{{1:d}}] {{2:{0}}} {{3:{0}}} {{4:{0}}}\n'.format(xyz_fmt)

        for ia in geom:
            Z = geom.atoms[ia].Z
            no = geom.atoms[ia].orbs
            if no == 1:
                self._write(fmt1_str.format(Z,*geom.xyz[ia,:]))
            else:
                self._write(fmt2_str.format(Z,no,*geom.xyz[ia,:]))

        self._write('end atoms\n')

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
    print('Writing TBSile')
    geom.write(TBSile('diamond.tb','w'))
    print('Reading TBSile')
    geomr = TBSile('diamond.tb','r').read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)
