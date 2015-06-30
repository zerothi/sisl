"""
Sile object for reading/writing SIESTA binary files
"""
from __future__ import print_function

# Import sile objects
from .sile import *

# Import the geometry object
from sids.geom import Geometry, Atom

import numpy as np

class SIESTASile(NCSile):
    """ SIESTA file object """

    _Bohr = 0.529177

    def read_geom(self):
        """ Returns Geometry object from a SIESTA.nc file 

        NOTE: Interaction range of the Atoms are currently not read.
        """
        with self as nc:
            cell = np.array(nc.variables['cell'][:],np.float64) * self._Bohr
            cell.shape = (3,3)
            xyz = np.array(nc.variables['xa'][:],np.float64) * self._Bohr
            xyz.shape = (-1,3)
            nsc = np.array(nc.variables['nsc'][:],np.int)
            
            if 'BASIS' in nc.groups:
                bg = nc.groups['BASIS']
                # We can actually read the exact basis-information
                b_idx = np.array(bg.variables['basis'][:],np.int)
                
                # Get number of different species
                n_b = len(bg.groups)

                spc = [None] * n_b
                zb = np.zeros([n_b],np.int)
                for basis in bg.groups:
                    # Retrieve index
                    ID = bg.groups[basis].ID
                    atm = dict()
                    atm['Z'] = int(bg.groups[basis].Atomic_number)
                    # We could possibly read in dR, however, that is not so easy?
                    atm['mass'] = float(bg.groups[basis].Mass)
                    atm['tag'] = basis
                    atm['orbs'] = int(bg.groups[basis].Number_of_orbitals)
                    spc[ID-1] = Atom[atm]
                atoms = [None] * len(xyz)
                for ia in xrange(len(xyz)):
                    atoms[ia] = spc[b_idx[ia]-1]
            else:
                atoms = Atom[1]

        # Create and return geometry object
        return Geometry(cell,xyz,atoms=atoms,nsc=nsc)

    def write_geom(self,geom):
        """
        Creates the NetCDF file and writes the geometry information
        """
        sile_raise_write(self)

        def crt_dim(n,name,l):
            if name in n.dimensions: return
            n.createDimension(name,l)
        def crt_var(n,name,*args,**kwargs):
            if name in n.variables: return n.variables[name]
            return n.createVariable(name,*args,**kwargs)

        # Start writing
        with self as nc:
            
            # Create initial dimensions
            crt_dim(nc,'one',1)
            crt_dim(nc,'n_s',np.prod(geom.nsc))
            crt_dim(nc,'xyz',3)
            crt_dim(nc,'no_s',np.prod(geom.nsc)*geom.no)
            crt_dim(nc,'no_u',geom.no)
            crt_dim(nc,'spin',1)
            crt_dim(nc,'na_u',geom.na)

            # Create initial geometry
            v = crt_var(nc,'nsc','i4',('xyz',))
            v.info = 'Number of supercells in each unit-cell direction'
            v = crt_var(nc,'lasto','i4',('na_u',))
            v.info = 'Last orbital of equivalent atom'
            v = crt_var(nc,'Ef','f8',('one',))
            v.info = 'Fermi level'
            v.unit = 'Ry'
            v = crt_var(nc,'xa','f8',('na_u','xyz'))
            v.info = 'Atomic coordinates'
            v.unit = 'Bohr'
            v = crt_var(nc,'cell','f8',('xyz','xyz'))
            v.info = 'Unit cell'
            v.unit = 'Bohr'

            # Create designation of the creation
            nc.method = 'python'

            # Save stuff
            nc.variables['nsc'][:] = geom.nsc
            nc.variables['xa'][:] = geom.xyz / self._Bohr
            nc.variables['cell'][:] = geom.cell / self._Bohr
            nc.variables['Ef'][:] = 0.

            # Create basis group
            if 'BASIS' in nc.groups:
                bs = nc.groups['BASIS']
            else:
                bs = nc.createGroup('BASIS')
                
            # Create variable of basis-indices
            b = crt_var(bs,'basis','i4',('na_u',))
            b.info = "Basis of each atom by ID"

            orbs = np.empty([geom.na],np.int)

            for ia,a,isp in geom.iter_species():
                b[ia] = isp + 1
                orbs[ia] = a.orbs
                if a.tag in bs.groups:
                    # Assert the file sizes
                    if bs.groups[a.tag].Number_of_orbitals != a.orbs:
                        raise ValueError('File ' + self.file + ' has erroneous data in regards of '+
                                         'of the already stored dimensions.')
                else:
                    ba = bs.createGroup(a.tag)
                    ba.ID = np.int32(isp + 1)
                    ba.Atomic_number = np.int32(a.Z)
                    ba.Mass = a.mass
                    ba.Label = a.tag
                    ba.Element = a.symbol
                    ba.Number_of_orbitals = np.int32(a.orbs)

            # Store the lasto variable as the remaining thing to do
            nc.variables['lasto'][:] = np.cumsum(orbs)


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
    geom.write(SIESTASile('diamond.nc','w'))
    io = SIESTASile('diamond.nc','r')
    geomr = io.read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)

    
