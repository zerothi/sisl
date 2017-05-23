"""
Sile object for reading/writing OUT files
"""

from __future__ import print_function, division

import os.path as osp
import numpy as np
import warnings as warn

# Import sile objects
from .sile import SileSiesta
from ..sile import *
from sisl.io._help import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell, Grid

from sisl.utils.cmd import *

from sisl.units import unit_default, unit_group
from sisl.units.siesta import unit_convert

__all__ = ['outSileSiesta']


Bohr2Ang = unit_convert('Bohr', 'Ang')


class outSileSiesta(SileSiesta):
    """ SIESTA output file object 

    This enables reading the output quantities from the SIESTA output.
    """

    def _ensure_species(self, species):
        """ Ensures that the species list is a list with entries (converts `None` to a list). """
        if species is None:
            return [Atom(i) for i in range(150)]
        return species

    @Sile_fh_open
    def read_species(self):
        """ Reads the species from the top of the output file.

        If wanting the species this HAS to be the first routine called.

        It returns an array of `Atom` objects which may easily be indexed.
        """

        line = self.readline()
        while not 'Species number:' in line:
            line = self.readline()
            if line == '':
                # We fake the species by direct atomic number
                return None

        atom = []
        while 'Species number:' in line:
            ls = line.split()
            atom.append(Atom(int(ls[5]), tag=ls[7]))
            line = self.readline()

        return atom

    def _read_supercell_outcell(self):
        """ Wrapper for reading the unit-cell from the outcoor block """

        # Read until outcell is found
        line = self.readline()
        while not 'outcell: Unit cell vectors' in line:
            line = self.readline()

        Ang = 'Ang' in line

        # We read the unit-cell vectors (in Ang)
        cell = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            cell.append([float(x) for x in line[:3]])
            line = self.readline()

        cell = np.array(cell, np.float64)

        if not Ang:
            cell *= Bohr2Ang

        return SuperCell(cell)

    def _read_geometry_outcoor(self, line, last, all, species=None):
        """ Wrapper for reading the geometry as in the outcoor output """
        species = self._ensure_species(species)

        # Now we have outcoor
        scaled = 'scaled' in line
        fractional = 'fractional' in line
        Ang = 'Ang' in line

        # Read in data
        xyz = []
        spec = []
        atom = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            xyz.append([float(x) for x in line[:3]])
            spec.append(line[3])
            try:
                atom.append(line[5])
            except:
                pass
            line = self.readline()

        cell = self._read_supercell_outcell()
        xyz = np.array(xyz, np.float64)

        # Now create the geometry
        if scaled:
            # The output file for siesta does not
            # contain the lattice constant.
            # So... :(
            raise ValueError("Could not read the lattice-constant for the scaled geometry")
        elif fractional:
            xyz = xyz[:, 0] * cell[0, :][None, :] + \
                  xyz[:, 1] * cell[1, :][None, :] + \
                  xyz[:, 2] * cell[2, :][None, :]
        elif not Ang:
            xyz *= Bohr2Ang

        try:
            geom = Geometry(xyz, atom, sc=cell)
        except:
            geom = Geometry(xyz, [species[int(i)-1] for i in spec], sc=cell)

        if all:
            tmp = self._read_geometry_outcoor(last, all, species)
            if tmp is None:
                return [geom]
            return tmp.extend([geom])

        return geom

    def _read_geometry_atomic(self, line, species=None):
        """ Wrapper for reading the geometry as in the outcoor output """
        species = self._ensure_species(species)

        # Now we have outcoor
        Ang = 'Ang' in line

        # Read in data
        xyz = []
        atom = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            xyz.append([float(x) for x in line[1:4]])
            atom.append(species[int(line[4])-1])
            line = self.readline()

        # Retrieve the unit-cell
        cell = self._read_supercell_outcell()
        # Convert xyz
        xyz = np.array(xyz, np.float64)
        if not Ang:
            xyz *= Bohr2Ang

        geom = Geometry(xyz, atom, sc=cell)

        return geom

    @Sile_fh_open
    def read_geometry(self, last=True, all=False):
        """ Reads the geometry from the SIESTA output file

        Parameters
        ----------
        last: bool, True
           only read the last geometry 
        all: bool, False
           return a list of all geometries (like an MD)
           If `True` `last` is ignored
        """

        # The first thing we do is reading the species.
        # Sadly, if this routine is called AFTER some other
        # reading process, it may fail...
        # Perhaps we should rewind to ensure this...
        # But...
        species = self.read_species()

        def type_coord(line):
            if 'outcoor' in line:
                return 1
            elif 'siesta: Atomic coordinates' in line:
                return 2
            # Signal not found
            return 0

        # Read until a coordinate block is found
        line = self.readline()
        while type_coord(line) == 0:
            line = self.readline()
            if line == '':
                break

        coord = type_coord(line)

        if coord == 1:
            return self._read_geometry_outcoor(line, last, all, species)
        elif coord == 2:
            return self._read_geometry_atomic(line, species)

        # Signal not found
        return None

    @Sile_fh_open
    def read_force(self, last=True, all=False):
        """ Reads the forces from the SIESTA output file

        Parameters
        ----------
        last: bool, True
           only read the last force
        all: bool, False
           return a list of all forces (like an MD)
           If `True` `last` is ignored
        """

        # Read until outcoor is found
        line = self.readline()
        while not 'siesta: Atomic forces' in line:
            line = self.readline()
            if line == '':
                return None

        F = []
        line = self.readline()
        while not line.startswith('--'):
            F.append([float(x) for x in line.split()[1:]])
            line = self.readline()

        F = np.array(F)

        if all:
            tmp = self.read_force(last, all)
            if tmp is None:
                return []
            return tmp.extend([F])
        return F

    @Sile_fh_open
    def read_moment(self, orbital=False, quantity='S', last=True, all=False):
        """ Reads the moments from the SIESTA output file
        These will only be present in case of spin-orbit coupling.

        Parameters
        ----------
        orbital: bool, False
           return a table with orbitally resolved
           moments.
        quantity: str, 'S'
           return the spin-moments or the L moments
        last: bool, True
           only read the last force
        all: bool, False
           return a list of all forces (like an MD)
           If `True` `last` is ignored
        """

        # Read until outcoor is found
        line = self.readline()
        while not 'moments: Atomic' in line:
            line = self.readline()
            if line == '':
                return None

        # The moments are printed in SPECIES list
        self.readline() # empty

        na = 0
        # Loop the species
        tbl = []
        # Read the species label
        self.readline() # currently discarded
        while True:
            self.readline() # ""
            self.readline() # Atom    Orb ...
            # Loop atoms in this species list
            while True:
                line = self.readline()
                if line.startswith('Species') or \
                   line.startswith('--'):
                    break
                line = ' '
                atom = []
                ia = 0
                while not line.startswith('--'):
                    line = self.readline().split()
                    if ia == 0:
                        ia = int(line[0])
                    elif ia != int(line[0]):
                        raise ValueError("Error in moments formatting.")
                    # Track maximum number of atoms
                    na = max(ia, na)
                    if quantity == 'S':
                        atom.append([float(x) for x in line[4:7]])
                    elif quantity == 'L':
                        atom.append([float(x) for x in line[7:10]])
                line = self.readline().split() # Total ...
                if not orbital:
                    ia = int(line[0])
                    if quantity == 'S':
                        atom.append([float(x) for x in line[4:7]])
                    elif quantity == 'L':
                        atom.append([float(x) for x in line[8:11]])
                tbl.append((ia, atom))
            if line.startswith('--'):
                break

        # Sort according to the atomic index
        moments = [] * na

        # Insert in the correct atomic
        for ia, atom in tbl:
            moments[ia-1] = atom

        if not all:
            return np.array(moments)
        return moments

    def read_data(self, *args, **kwargs):
        """ Read specific content in the SIESTA out file 

        The currently implemented things are denoted in
        the parameters list.
        Note that the returned quantities are in the order
        of keywords, so:

        >>> read_data(geometry=True, force=True)
        <geom>, <forces>
        >>> read_data(force=True,geometry=True)
        <forces>, <geom>

        Parameters
        ----------
        geom: bool
           return the last geometry in the `outSileSiesta`
        force: bool
           return the last force in the `outSileSiesta`
        moment: bool
           return the last moments in the `outSileSiesta` (only for spin-orbit coupling calculations)
        """
        val = []
        for kw in kwargs:

            if kw == 'geom':
                if kwargs[kw]:
                    val.append(self.read_geometry())

            if kw == 'force':
                if kwargs[kw]:
                    val.append(self.read_force())

            if kw == 'moment':
                if kwargs[kw]:
                    val.append(self.read_moment())

        if len(val) == 0:
            val = None
        elif len(val) == 1:
            val = val[0]
        return val


add_sile('out', outSileSiesta, case=False, gzip=True)
