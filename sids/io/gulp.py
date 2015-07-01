"""
Sile object for reading/writing GULP in/output
"""
from __future__ import print_function

# Import sile objects
from sids.io.sile import *

# Import the geometry object
from sids.geom import Geometry, Atom

import numpy as np

class GULPSile(NCSile):
    """ GULP file object """

    def read_geom(self,key='Final fractional coordinates'):
        """ Reads a geometry and creates the GULP dynamical geometry """
        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_geom()

        self.step_to('Cartesian lattice vectors')

        # skip 1 line
        self.readline()
        cell = np.zeros([3,3],np.float)
        for i in range(3):
            l = self.readline().split()
            cell[i,0] = float(l[0])
            cell[i,1] = float(l[1])
            cell[i,2] = float(l[2])
            
        # Skip to keyword
        self.step_to(keyword)
                    
        # We skip 5 lines
        for i in range(5): self.readline()
        
        Z = []
        xyz = []
        while True:
            l = self.readline()
            if l[0] == '-': break

            ls = l.split()
            Z.append({'Z':ls[1],'orbs':3})
            xyz.append([float(f) for f in ls[3:6]])

        # Convert to array
        xyz = np.array(xyz,np.float)
        xyz.shape = (-1,3)
        if 'fractional' in key.lower():
            # Correct for fractional coordinates
            xyz[:,0] *= np.sum(cell[:,0])
            xyz[:,1] *= np.sum(cell[:,1])
            xyz[:,2] *= np.sum(cell[:,2])
            
        if len(Z) == 0 or len(xyz) == 0:
            raise ValueError('Could not read in cell information and/or coordinates')

        # Return the geometry
        return Geometry(cell,xyz,atoms=Atom[Z])

    def read_tb(self,**kwargs):
        """ Returns a GULP tight-binding model for the output of GULP """
        if not hasattr(self,'fh'):
            # The file-handle has not been opened
            with self:
                return self.read_tb(**kwargs)

        geom = self.read_geom(**kwargs)

        # Easier for creation of the sparsity pattern
        from scipy.sparse import lil_matrix, diags

        dyn = lil_matrix( (geom.no,geom.no) ,dtype = kwargs.get('dtype',np.float) )
        
        self.step_to('Real Dynamical matrix')

        # skip 1 line
        self.readline()
        
        no = geom.no
        i = 0 ; j = 0
        while True:
            l = self.readline()
            if len(l.strip()) == 0: break

            ls = [float(f) for f in l.split()]
            # add the values (12 values == 3*4)
            for k in range(4):
                dyn[i,j  ] = ls[k*3  ]
                dyn[i,j+1] = ls[k*3+1]
                dyn[i,j+2] = ls[k*3+2]
                j += 3
                if j >= no: 
                    i += 1
                    j = 0
                    if i >= no: break

        # Convert the GULP data to standard units
        dyn = dyn.tocoo()
        dyn.data[:] *= ( 521.469 * 1.23981e-4 ) ** 2

        # Create "fake" overlap matrix
        S = diags([1.],[0],shape=dyn.shape)
        S = S.tocsr()

        return PhononTightBinding.sp2tb(geom,dyn,S)


if __name__ == "__main__":
    pass
