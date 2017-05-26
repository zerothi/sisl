"""
Sile object for reading/writing ref files from ScaleUp
"""

from __future__ import division, print_function

# Import sile objects
from .sile import SileScaleUp
from .orbocc import orboccSileScaleUp
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl._help import ensure_array
from sisl.units import unit_convert

import numpy as np

__all__ = ['REFSileScaleUp', 'restartSileScaleUp']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ang2Bohr = unit_convert('Ang', 'Bohr')


class REFSileScaleUp(SileScaleUp):
    """ REF file object for ScaleUp """

    @Sile_fh_open
    def read_supercell(self):
        """ Reads a supercell from the Sile """
        # 1st line is number of supercells
        nsc = ensure_array(map(int, self.readline().split()[:3]), np.int32)
        self.readline() # natoms, nspecies
        self.readline() # species
        cell = ensure_array(map(float, self.readline().split()[:9]), np.float64)
        # Typically ScaleUp uses very large unit-cells
        # so supercells will typically be restricted to [3, 3, 3]
        return SuperCell(cell * Bohr2Ang)

    @Sile_fh_open
    def read_geometry(self, primary=False):
        """ Reads a geometry from the Sile """
        # 1st line is number of supercells
        nsc = ensure_array(map(int, self.readline().split()[:3]), np.int32)
        na, ns = map(int, self.readline().split()[:2])
        # Convert species to atom objects
        try:
            species = get_sile(self.file.rsplit('REF', 1)[0] + 'orbocc').read_atom()
        except:
            species = [Atom(s) for s in self.readline().split()[:ns]]

        # Total number of super-cells
        if primary:
            # Only read in the primary unit-cell
            ns = 1
        else:
            ns = np.prod(nsc)

        cell = ensure_array(map(float, self.readline().split()), np.float64)
        try:
            cell.shape = (3, 3)
            if primary:
                cell[0, :] /= nsc[0]
                cell[1, :] /= nsc[1]
                cell[2, :] /= nsc[2]
        except:
            c = np.empty([3, 3], np.float64)
            c[0, 0] = 1. + cell[0]
            c[0, 1] = cell[5] / 2.
            c[0, 2] = cell[4] / 2.
            c[1, 0] = cell[5] / 2.
            c[1, 1] = 1. + cell[1]
            c[1, 2] = cell[3] / 2.
            c[2, 0] = cell[4] / 2.
            c[2, 1] = cell[3] / 2.
            c[2, 2] = 1. + cell[2]
            cell = c * Ang2Bohr
        sc = SuperCell(cell * Bohr2Ang)

        # Create list of coordinates and atoms
        xyz = np.empty([na * ns, 3], np.float64)
        atoms = [None] * na * ns

        # Read the geometry
        for ia in range(na * ns):

            # Retrieve line
            #   ix  iy  iz  ia  is   x  y  z
            line = self.readline().split()

            atoms[ia] = species[int(line[4]) - 1]
            xyz[ia, :] = ensure_array(map(float, line[5:8]), np.float64)

        return Geometry(xyz * Bohr2Ang, atoms, sc=sc)

    @Sile_fh_open
    def write_geometry(self, geom, fmt='18.8e'):
        """ Writes the geometry to the contained file """
        # Check that we can write to the file
        sile_raise_write(self)

        # Number of supercells
        ns = np.prod(geom.sc.nsc // 2 + 1)

        # 1st line is number of supercells
        self._write('{:5d}{:5d}{:5d}\n'.format(*geom.sc.nsc // 2 + 1))
        # natoms, nspecies
        self._write('{:5d}{:5d}\n'.format(len(geom), len(geom.atom.atom)))

        s = ''
        for a, _ in geom.atom:
            # Append the species label
            s += '{:<10}'.format(a.tag)
        self._write(s + '\n')

        fmt_str = '{{:{0}}} '.format(fmt) * 9 + '\n'
        self._write(fmt_str.format(*(geom.cell*Ang2Bohr).reshape(-1)))

        # Create line
        #   ix  iy  iz  ia  is   x  y  z
        line = '{:5d}{:5d}{:5d}{:5d}{:5d}' + '{{:{0}}}'.format(fmt) * 3 + '\n'

        args = [None] * 8
        for i, isc in geom.sc:
            if np.any(isc < 0):
                continue

            # Write the geometry
            for ia in geom:

                args[0] = isc[0]
                args[1] = isc[1]
                args[2] = isc[2]
                args[3] = ia + 1
                args[4] = geom.atom.specie[ia] + 1
                args[5] = geom.xyz[ia, 0] * Ang2Bohr
                args[6] = geom.xyz[ia, 1] * Ang2Bohr
                args[7] = geom.xyz[ia, 2] * Ang2Bohr

                self._write(line.format(*args))

    def ArgumentParser(self, *args, **kwargs):
        """ Returns the arguments that is available for this Sile """
        newkw = Geometry._ArgumentParser_args_single()
        newkw.update(kwargs)
        return self.read_geometry().ArgumentParser(*args, **newkw)


# The restart file is _equivalent_ but with displacements
class restartSileScaleUp(REFSileScaleUp):

    @Sile_fh_open
    def read_geometry(self, *args, **kwargs):
        """ Read geometry of the restart file

        This will also try and read the corresponding .REF file
        such that final coordinates are returned.

        Note that a .restart file from ScaleUp only contains the displacements
        from a .REF file and thus it is not the *actual* atomic coordinates.

        If the .REF file does not exist the returned cell vectors correspond
        to the strain tensor (+1 along the diagonal).
        """

        try:
            ref = get_sile(self.file.rsplit('restart', 1)[0] + 'REF').read_geometry()
        except:
            ref = None

        restart = super(restartSileScaleUp, self).read_geometry()
        if not ref is None:
            restart.sc = SuperCell(np.dot(ref.sc.cell, restart.sc.cell.T),
                                   nsc=restart.nsc)
            restart.xyz += ref.xyz

        return restart


add_sile('REF', REFSileScaleUp, case=False, gzip=True)
add_sile('restart', restartSileScaleUp, case=False, gzip=True)
