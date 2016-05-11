"""
Sile object for reading/writing FDF files
"""

from __future__ import print_function, division

# Import sile objects
from sisl.io.sile import *
from sisl.io._help import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl import Bohr

from os.path import dirname, sep
import numpy as np

__all__ = ['FDFSile']


class FDFSile(Sile):
    """ FDF file object """

    def __init__(self, filename, mode='r', base=None):
        """ Initialize an FDF file from the filename

        By supplying base you can reference files in other directories.
        By default the ``base`` is the directory given in the file name.
        """
        super(self.__class__, self).__init__(filename, mode=mode)
        if base is None:
            # Extract from filename
            self._directory = dirname(filename)
        else:
            self._directory = base
        if len(self._directory) == 0:
            self._directory = '.'

    def _setup(self):
        """ Setup the `FDFSile` after initialization """
        # These are the comments
        self._comment = ['#', '!', ';']

        # List of parent file-handles used while reading
        self._parent_fh = []
        self._directory = '.'

    def readline(self, comment=False):
        """ Reads the next line of the file """
        l = self.fh.readline()
        if comment:
            return l
        while starts_with_list(l, self._comment):
            l = self.fh.readline()
        # In FDF files, %include marks files that progress
        # down in a tree structure
        if '%include' in l:
            # Split for reading tree file
            self._parent_fh.append(self.fh)
            self.fh = open(self._directory + sep + l.split()[1], self._mode)
            # Read the following line in the new file
            return self.readline()
        if len(self._parent_fh) > 0 and l == '':
            # l == '' marks the end of the file
            self.fh.close()
            self.fh = self._parent_fh.pop()
            return self.readline()
        return l

    def _read(self, key):
        """ Returns the arguments following the keyword in the FDF file """
        if hasattr(self, 'fh'):
            return self.step_to(key, case=False)
        with self:
            return self.step_to(key, case=False)

    def _read_block(self, key, force=False):
        """ Returns the arguments following the keyword in the FDF file """
        k = key.lower()
        with self as fh:
            f, lc = fh.step_to(k, case=False)
            if force and not f:
                # The user requests that the block *MUST* be found
                raise SileError(
                    'Requested forced block could not be found: ' +
                    str(key) +
                    '.',
                    self)
            if not f:
                return False, []  # not found
            li = []
            while True:
                l = fh.readline()
                if fh.line_has_key(l, '%endblock', case=False) or \
                   fh.line_has_key(l, k, case=False):
                    return True, li
                # Append list
                li.append(l)
        raise SileError(
            'Error on reading block: ' +
            str(key) +
            ' could not find start/end.')

    def write_geom(self, geom, fmt='.5f'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        if not hasattr(self, 'fh'):
            # The file-handle has not been opened
            with self:
                return self.write_geom(geom, fmt)

        # Write out the cell
        self._write('LatticeConstant 1. Ang\n')
        self._write('%block LatticeVectors\n')
        self._write(' {0} {1} {2}\n'.format(*geom.cell[0, :]))
        self._write(' {0} {1} {2}\n'.format(*geom.cell[1, :]))
        self._write(' {0} {1} {2}\n'.format(*geom.cell[2, :]))
        self._write('%endblock LatticeVectors\n\n')
        self._write('NumberOfAtoms {0}\n'.format(geom.na))
        self._write('AtomicCoordinatesFormat Ang\n')
        self._write('%block AtomicCoordinatesAndAtomicSpecies\n')

        fmt_str = ' {{2:{0}}} {{3:{0}}} {{4:{0}}} {{0}} # {{1}}\n'.format(fmt)
        # Count for the species
        spec = []
        for ia, a, isp in geom.iter_species():
            self._write(fmt_str.format(isp + 1, ia + 1, *geom.xyz[ia, :]))
            if isp >= len(spec):
                spec.append(a)
        self._write('%endblock AtomicCoordinatesAndAtomicSpecies\n\n')

        # Write out species
        # First swap key and value
        self._write('NumberOfSpecies {0}\n'.format(len(spec)))
        self._write('%block ChemicalSpeciesLabel\n')
        for i, a in enumerate(spec):
            self._write(' {0} {1} {2}\n'.format(i + 1, a.Z, a.tag))
        self._write('%endblock ChemicalSpeciesLabel\n')

    def read_sc(self, *args, **kwargs):
        """ Returns `SuperCell` object from the FDF file """
        f, lc = self._read('LatticeConstant')
        s = float(lc.split()[1])
        if 'ang' in lc.lower():
            pass
        elif 'bohr' in lc.lower():
            s /= Bohr

        # Read in cell
        cell = np.empty([3, 3], np.float64)

        f, lc = self._read_block('LatticeVectors')
        if f:
            for i in range(3):
                cell[i, :] = [float(k) for k in lc[i].split()[:3]]
        else:
            f, lc = self._read_block('LatticeParameters')
            tmp = [float(k) for k in lc[0].split()[:6]]
            if f:
                cell = SuperCell.tocell(*tmp)
        if not f:
            # the fdf file contains neither the latticevectors or parameters
            raise SileError(
                'Could not find Vectors or Parameters block in file')
        cell *= s

        return SuperCell(cell)

    def read_geom(self, *args, **kwargs):
        """ Returns Geometry object from the FDF file

        NOTE: Interaction range of the Atoms are currently not read.
        """

        f, lc = self._read('LatticeConstant')
        if not f:
            raise ValueError('Could not find LatticeConstant in fdf file.')
        s = float(lc.split()[1])
        if 'ang' in lc.lower():
            pass
        elif 'bohr' in lc.lower():
            s /= Bohr

        sc = self.read_sc(*args, **kwargs)

        # No fractional coordinates
        is_frac = False

        # Read atom scaling
        f, lc = self._read('AtomicCoordinatesFormat')
        if not f:
            raise ValueError(
                'Could not find AtomicCoordinatesFormat in fdf file.')
        lc = lc.lower()
        if 'ang' in lc or 'notscaledcartesianang' in lc:
            s = 1.
            pass
        elif 'bohr' in lc or 'notscaledcartesianbohr' in lc:
            s = Bohr
        elif 'scaledcartesian' in lc:
            # the same scaling as the lattice-vectors
            pass
        elif 'fractional' in lc or 'scaledbylatticevectors' in lc:
            # no scaling of coordinates as that is entirely
            # done by the latticevectors
            s = 1.
            is_frac = True

        # If the user requests a shifted geometry
        # we correct for this
        origo = np.zeros([3], np.float64)
        run = 'origin' in kwargs
        if run:
            run = kwargs['origin']
        if run:
            f, lor = self._read_block('AtomicCoordinatesOrigin')
            if f:
                origo = np.fromstring(lor[0], count=3, sep=' ') * s
        # Origo cannot be interpreted with fractional coordinates
        # hence, it is not transformed.

        # Read atom block
        f, atms = self._read_block(
            'AtomicCoordinatesAndAtomicSpecies', force=True)
        if not f:
            raise ValueError(
                'Could not find AtomicCoordinatesAndAtomicSpecies in fdf file.')

        # Read number of atoms and block
        f, l = self._read('NumberOfAtoms')
        if not f:
            # We default to the number of elements in the
            # AtomicCoordinatesAndAtomicSpecies block
            na = len(atms)
        else:
            na = int(l.split()[1])

        # Reduce space if number of atoms specified
        if na != len(atms):
            # align number of atoms and atms array
            atms = atms[:na]

        if na == 0:
            raise ValueError(
                'NumberOfAtoms has been determined to be zero, no atoms.')

        # Create array
        xyz = np.empty([na, 3], np.float64)
        species = np.empty([na], np.int32)
        for ia in range(na):
            l = atms[ia].split()
            xyz[ia, :] = [float(k) for k in l[:3]]
            species[ia] = int(l[3]) - 1
        if is_frac:
            xyz = np.dot(xyz, sc.cell)
        xyz *= s
        xyz += origo

        # Now we read in the species
        f, l = self._read('NumberOfSpecies')
        if not f:
            raise ValueError('Could not find NumberOfSpecies in fdf file.')

        ns = 0
        if f:
            ns = int(l.split()[1])

        # Read the block (not strictly needed, if so we simply set all atoms to
        # H)
        f, spcs = self._read_block('ChemicalSpeciesLabel')
        if f:
            # Initialize number of species to
            # the length of the ChemicalSpeciesLabel block
            if ns == 0:
                ns = len(spcs)
            # Pre-allocate the species array
            sp = [None] * ns
            for spc in spcs:
                #  index Z pseudo-tag
                l = spc.split()
                idx = int(l[0]) - 1
                # Insert the atom
                sp[idx] = Atom(Z=int(l[1]), tag=l[2])

            if None in sp:
                idx = sp.index(None) + 1
                raise ValueError(
                    ("Could not populate entire "
                     "species list. "
                     "Please ensure specie with index {} is present".format(idx)))

            # Create atoms array with species
            atoms = [None] * na
            for ia in range(na):
                atoms[ia] = sp[species[ia]]

            if None in atoms:
                idx = atoms.index(None) + 1
                raise ValueError(
                    ("Could not populate entire "
                     "atomic list list. "
                     "Please ensure atom with index {} is present".format(idx)))

        else:
            # Default atom (hydrogen)
            atoms = Atom(1)
            # Force number of species to 1
            ns = 1

        # Create and return geometry object
        return Geometry(xyz, atoms=atoms, sc=sc)


if __name__ == "__main__":
    # Create geometry
    alat = 3.57
    dist = alat * 3. ** .5 / 4
    C = Atom(Z=6, R=dist * 1.01, orbs=2)
    sc = SuperCell(np.array([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 0]], np.float64) * alat / 2)
    geom = Geometry(np.array([[0, 0, 0], [1, 1, 1]], np.float64) * alat / 4,
                    atoms=[C, Atom(Z=6, tag='C_pbe')], sc=sc)
    # Write stuff
    geom.write(FDFSile('diamond.fdf', 'w'))
    geomr = FDFSile('diamond.fdf', 'r').read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)

    with open('tree.fdf', 'w') as fh:
        fh.write('%include diamond.fdf')
    geomr = FDFSile('tree.fdf', 'r').read_geom()
    print(geomr)
    print(geomr.cell)
    print(geomr.xyz)
