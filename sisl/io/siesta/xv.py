from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import SileSiesta
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, Atoms, SuperCell
from sisl.unit.siesta import unit_convert

Bohr2Ang = unit_convert('Bohr', 'Ang')

__all__ = ['xvSileSiesta']


class xvSileSiesta(SileSiesta):
    """ XV file object """

    @Sile_fh_open
    def write_geometry(self, geom, fmt='.9f'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # Write unit-cell
        tmp = np.zeros(6, np.float64)

        # Create format string for the cell-parameters
        fmt_str = ('   ' + ('{:' + fmt + '} ') * 3) * 2 + '\n'
        for i in range(3):
            tmp[0:3] = geom.cell[i, :] / Bohr2Ang
            self._write(fmt_str.format(*tmp))
        self._write('{:12d}\n'.format(geom.na))

        # Create format string for the atomic coordinates
        fmt_str = '{:3d}{:6d} '
        fmt_str += ('{:' + fmt + '} ') * 3 + '   '
        fmt_str += ('{:' + fmt + '} ') * 3 + '\n'
        for ia, a, ips in geom.iter_species():
            tmp[0:3] = geom.xyz[ia, :] / Bohr2Ang
            self._write(fmt_str.format(ips + 1, a.Z, *tmp))

    @Sile_fh_open
    def read_supercell(self):
        """ Returns `SuperCell` object from the XV file """

        cell = np.empty([3, 3], np.float64)
        for i in range(3):
            cell[i, :] = list(map(float, self.readline().split()[:3]))
        cell *= Bohr2Ang

        return SuperCell(cell)

    @Sile_fh_open
    def read_geometry(self, species_Z=False):
        """ Returns a `Geometry` object from the XV file

        Parameters
        ----------
        species_Z : bool, optional
           if ``True`` the atomic numbers are the species indices (useful when
           reading the ChemicalSpeciesLabel block simultaneously).
        """
        sc = self.read_supercell()

        # Read number of atoms
        na = int(self.readline())
        xyz = np.empty([na, 3], np.float64)
        atms = [None] * na
        sp = np.empty([na], np.int32)
        for ia in range(na):
            line = list(map(float, self.readline().split()[:8]))
            sp[ia] = int(line[0])
            if species_Z:
                atms[ia] = Atom(sp[ia])
            else:
                atms[ia] = Atom(int(line[1]))
            xyz[ia, :] = line[2:5]
        xyz *= Bohr2Ang

        # Ensure correct sorting
        max_s = sp.max()
        sp -= 1
        # Ensure we can remove the atom after having aligned them
        atms2 = Atoms(Atom(-150), na=na)
        for i in range(max_s):
            idx = (sp[:] == i).nonzero()[0]
            if len(idx) == 0:
                # Always ensure we have "something" for the unoccupied places
                atms2[idx] = Atom(-150 - i)
            else:
                atms2[idx] = atms[idx[0]]

        return Geometry(xyz, atms2.reduce(), sc=sc)

    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(p, *args, **newkw)


add_sile('XV', xvSileSiesta, gzip=True)
