"""
Sile object for reading/writing Wannier90 in/output
"""
from __future__ import print_function

# Import sile objects
from .sile import SileW90
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl.quantity import Hamiltonian

import numpy as np

__all__ = ['winSileW90']


class winSileW90(SileW90):
    """ Wannier seedname output file object """

    def _setup(self):
        """ Setup `winSileW90` after initialization """
        self._seed = self.file.replace('.win', '')


    def _set_file(self, suffix):
        self.file = self._seed + suffix


    @Sile_fh_open
    def _read_sc(self):
        """ Defered routine """

        f, l = self.step_to('unit_cell_cart', case=False)
        if not f:
            raise ValueError("The unit-cell vectors could not be found in the seed-file.")

        # Create the cell
        cell = np.empty([3,3], np.float64)
        for i in [0,1,2]:
            cell[i,:] = [float(x) for x in self.readline().split()]

        return SuperCell(cell)
        
        
    def read_sc(self):
        """ Reads a `SuperCell` and creates the Wannier90 cell """
        self._set_file('.win')

        return self._read_sc()


    @Sile_fh_open
    def _read_geom(self):
        """ Defered routine """

        nc = int(self.readline())

        # Comment
        self.readline()

        na = 0
        sp = [None] * nc
        xyz = np.empty([nc, 3], np.float64)
        for ia in range(nc):
            l = self.readline().split()
            sp[ia] = l.pop(0)
            if sp[ia] == 'X':
                na = ia + 1
            xyz[ia, :] = [float(k) for k in l[:3]]

        return Geometry(xyz[:na,:], atom='H')
        
            
    def read_geom(self, *args, **kwargs):
        """ Reads a `Geometry` and creates the Wannier90 cell """

        # Read in the super-cell
        sc = self.read_sc()
        
        self._set_file('_centres.xyz')

        geom = self._read_geom()
        geom.set_sc(sc)
        
        return geom


    @Sile_fh_open
    def _read_es(self, geom, dtype=np.float64, **kwargs):
        """ Reads a Hamiltonian

        Reads the Hamiltonian model
        """

        cutoff = kwargs.get('cutoff', 0.00001)

        # Rewind to ensure we can read the entire matrix structure
        self.fh.seek(0)

        # Time of creation
        self.readline()

        # Retrieve # of wannier functions
        no = int(self.readline())
        nrpts = int(self.readline())

        # First read across the Wigner-Seitz degeneracy
        # This is formatted with 15 per-line.
        if nrpts % 15 == 0:
            nlines = nrpts
        else:
            nlines = nrpts + 15 - nrpts % 15
        
        ws = []
        for i in range(nlines // 15):
            ws.extend([int(x) for x in self.readline().split()])
        
        # Convert to numpy array
        nws = np.array(ws, np.int32).flatten()
        del ws

        # Figure out the number of supercells
        nsc = [0, 0, 0]
        
        while True:
            l = self.readline()
            if l == '':
                break

            isc = [int(x) for x in l.split()[:3]]
            nsc[0] = max(nsc[0], abs(isc[0]))
            nsc[1] = max(nsc[1], abs(isc[1]))
            nsc[2] = max(nsc[2], abs(isc[2]))

        geom.set_nsc(np.array(nsc, np.int32)*2+1)
        
        # With the geometry in place we can read in the entire matrix
        # Create a new sparse matrix
        from scipy.sparse import lil_matrix
        Hr = lil_matrix((geom.no, geom.no_s), dtype=dtype)
        Hi = lil_matrix((geom.no, geom.no_s), dtype=dtype)
        
        self.fh.seek(0)
        for i in range(nlines // 15 + 3):
            self.readline()
        
        
        while True:
            l = self.readline()
            if l == '':
                break

            ls = l.split()
            
            # Get supercell and wannier functions
            # isc = idx[:3]
            # Hij = idx[3:5]
            idx = [int(x) for x in ls[:5]]

            hr = float(ls[5])
            hi = float(ls[6])

            # Get the offset
            off = geom.sc_index(idx[:3]) * geom.no
            
            if abs(hr) > cutoff:
                Hr[idx[3]-1, idx[4]-1 + off] = hr
            if abs(hi) > cutoff:
                Hi[idx[3]-1, idx[4]-1 + off] = hi

        if np.dtype(dtype).kind == 'c':
            Hr.data[:] = Hr.data[:] + 1j*Hi.data[:]

        return Hamiltonian.sp2HS(geom, Hr)

    
    def read_es(self, *args, **kwargs):
        """ Read the electronic structure of the Wannier90 output 
        
        Parameters
        ----------
        cutoff: (float, 0.00001)
           the cutoff value for the zero Hamiltonian elements
        """

        # Retrieve the geometry...
        geom = self.read_geom()

        # Set file
        self._set_file('_hr.dat')

        return self._read_es(geom, *args, **kwargs)
        
    def ArgumentParser(self, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geom().ArgumentParser(*args, **newkw)


add_sile('win', winSileW90, gzip=True)
