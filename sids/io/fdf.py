"""
Sile object for reading/writing FDF files
"""

from __future__ import print_function

# Import sile objects
from sids.io.sile import *

# Import the geometry object
from sids.geom import Geometry, Atom

import numpy as np

class FDFSile(Sile):
    """ FDF file object """
    # These are the comments
    _comment = ['#','!',';']

    # FDF values for conversions
    _Bohr = 0.529177

    def _read(self,key):
        """ Returns the arguments following the keyword in the FDF file """
        with self as f:
            return f.step_to(key.lower(),case=False)

    def _read_block(self,key,force=False):
        """ Returns the arguments following the keyword in the FDF file """
        k = key.lower()
        with self as fh:
            f,lc = fh.step_to(k,case=False)
            if force and not f:
                # The user requests that the block *MUST* be found
                raise SileError('Requested forced block could not be found: '+str(key) + '.',self)
            if not f: return False,[] # not found
            li = []
            while True:
                l = fh.readline()
                if fh.line_has_key(l.lower(),k,case=False): return True,li
                # Append list
                li.append(l)
        raise SileError('Error on reading block: '+str(key) + ' could not find start/end.')

    def write_geom(self,geom,fmt='.5f'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        with self as f:

            # Write out the cell
            f._write('LatticeConstant 1. Ang\n')
            f._write('%block LatticeVectors\n')
            for i in range(3):
                f._write(' {0} {1} {2}\n'.format(*geom.cell[i,:]))
            f._write('%endblock LatticeVectors\n\n')
            f._write('NumberOfAtoms {0}\n'.format(geom.na))
            f._write('AtomicCoordinatesFormat Ang\n')
            f._write('%block AtomicCoordinatesAndAtomicSpecies\n')
        
            fmt_str = ' {{2:{0}}} {{3:{0}}} {{4:{0}}} {{0}} # {{1}}\n'.format(fmt)
            # Count for the species
            spec = []
            for ia,a,isp in geom.iter_species():
                f._write(fmt_str.format(isp+1,ia+1,*geom.xyz[ia,:]))
                if isp >= len(spec): spec.append(a)
            f._write('%endblock AtomicCoordinatesAndAtomicSpecies\n\n')

            # Write out species
            # First swap key and value
            f._write('NumberOfSpecies {0}\n'.format(len(spec)))
            f._write('%block ChemicalSpeciesLabel\n')
            for i,a in enumerate(spec):
                f._write(' {0} {1} {2}\n'.format(i+1,a.Z,a.tag))
            f._write('%endblock ChemicalSpeciesLabel\n')

            
    def read_geom(self):
        """ Returns Geometry object from the FDF file 

        NOTE: Interaction range of the Atoms are currently not read.
        """
        f,lc = self._read('LatticeConstant')
        s = float(lc.split()[1])
        if 'ang' in lc.lower():
            pass
        elif 'bohr' in lc.lower():
            s /= self._Bohr
        # Read in cell
        cell = np.empty([3,3],np.float64)
        f,lc = self._read_block('LatticeVectors',force=True)
        for i in range(3):
            cell[i,:] = [float(k) for k in lc[i].split()[:3]]
        cell *= s

        # Read atom scaling
        f,lc = self._read('AtomicCoordinatesFormat')
        lc = lc.lower()
        if 'ang' in lc or 'notscaledcartesianang' in lc:
            s = 1.
            pass
        elif 'bohr' in lc or 'notscaledcartesianbohr' in lc:
            s = 1. / self._Bohr
        elif 'scaledcartesian' in lc:
            # the same scaling as the lattice-vectors
            pass

        # Read number of atoms and block
        f, l = self._read('NumberOfAtoms')
        na = 0
        if f: na = int(l.split()[1])
        # Read atom block
        f, atms = self._read_block('AtomicCoordinatesAndAtomicSpecies',force=True)

        # Reduce space if number of atoms specified
        if na > 0: atms = atms[:na]
        na = len(atms)

        # Create array
        xyz = np.empty([na,3],np.float64)
        species = np.empty([na],np.int)
        for ia in xrange(na):
            l = atms[ia].split()
            xyz[ia,:] = [float(k) for k in l[:3]]
            species[ia] = int(l[3]) - 1
        xyz *= s
        
        # Now we read in the species
        f, l = self._read('NumberOfSpecies')
        ns = 0
        if f: ns = int(l.split()[1])

        # Read the block (not strictly needed, if so we simply set all atoms to H)
        f, spcs = self._read_block('ChemicalSpeciesLabel')
        if f:
            if ns > 0: spcs = spcs[:ns]
            ns = len(spcs)
            # Read the species
            sp = []
            for spc in spcs:
                l = spc.split()
                # Create the atom
                sp.append(Atom(Z=int(l[1]),tag=l[2]))

            # Create atoms array with species
            atoms = [None]*na
            for ia in xrange(na):
                atoms[ia] = sp[species[ia]]
        else:
            # Default atom (hydrogen)
            atoms = Atom(1)

        # Create and return geometry object
        return Geometry(cell,xyz,atoms=atoms)


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
    io = FDFSile('diamond.fdf','w')
    io.write_geom(geom)
    io = FDFSile('diamond.fdf','r')
    geomr = io.read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)

    
