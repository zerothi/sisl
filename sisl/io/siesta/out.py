from __future__ import print_function, division

import os
import numpy as np

from .sile import SileSiesta
from ..sile import *
from sisl.io._help import *

from sisl import Geometry, Atom, SuperCell
from sisl.utils.cmd import *
from sisl.unit.siesta import unit_convert

__all__ = ['outSileSiesta']


Bohr2Ang = unit_convert('Bohr', 'Ang')


def _ensure_species(species):
    """ Ensures that the species list is a list with entries (converts `None` to a list). """
    if species is None:
        return [Atom(i) for i in range(150)]
    return species


class outSileSiesta(SileSiesta):
    """ Output file from Siesta

    This enables reading the output quantities from the Siesta output.
    """
    _job_completed = False

    def readline(self):
        line = super(outSileSiesta, self).readline()
        if 'Job completed' in line:
            self._job_completed = True
        return line

    readline.__doc__ = SileSiesta.readline.__doc__

    @property
    def job_completed(self):
        """ True if the full file has been read and "Job completed" was found. """
        return self._job_completed

    @sile_fh_open()
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
            if ls[3] == 'Atomic':
                atom.append(Atom(int(ls[5]), tag=ls[7]))
            else:
                atom.append(Atom(int(ls[7]), tag=ls[4]))
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

        cell = np.array(cell)

        if not Ang:
            cell *= Bohr2Ang

        return SuperCell(cell)

    def _read_geometry_outcoor(self, line, species=None):
        """ Wrapper for reading the geometry as in the outcoor output """
        species = _ensure_species(species)

        # Now we have outcoor
        scaled = 'scaled' in line
        fractional = 'fractional' in line
        Ang = 'Ang' in line

        # Read in data
        xyz = []
        spec = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            xyz.append([float(x) for x in line[:3]])
            spec.append(int(line[3]))
            line = self.readline()

        # in outcoor we know it is always just after
        cell = self._read_supercell_outcell()
        xyz = np.array(xyz)

        # Now create the geometry
        if scaled:
            # The output file for siesta does not
            # contain the lattice constant.
            # So... :(
            raise ValueError("Could not read the lattice-constant for the scaled geometry")
        elif fractional:
            xyz = np.dot(xyz, cell.cell)
        elif not Ang:
            xyz *= Bohr2Ang

        # Assign the correct species
        geom = Geometry(xyz, [species[ia - 1] for ia in spec], sc=cell)

        return geom

    def _read_geometry_atomic(self, line, species=None):
        """ Wrapper for reading the geometry as in the outcoor output """
        species = _ensure_species(species)

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

        # Retrieve the unit-cell (but do not skip file-descriptor position)
        # This is because the current unit-cell is not always written.
        pos = self.fh.tell()
        cell = self._read_supercell_outcell()
        self.fh.seek(pos, os.SEEK_SET)

        # Convert xyz
        xyz = np.array(xyz)
        if not Ang:
            xyz *= Bohr2Ang

        return Geometry(xyz, atom, sc=cell)

    @sile_fh_open()
    def read_geometry(self, last=True, all=False):
        """ Reads the geometry from the Siesta output file

        Parameters
        ----------
        last: bool, optional
           only read the last geometry
        all: bool, optional
           return a list of all geometries (like an MD)
           If `True` `last` is ignored

        Returns
        -------
        geometries: list or Geometry or None
             if all is False only one geometry will be returned (or None). Otherwise
             a list of geometries corresponding to the MD-runs.
        """

        # The first thing we do is reading the species.
        # Sadly, if this routine is called AFTER some other
        # reading process, it may fail...
        # Perhaps we should rewind to ensure this...
        # But...
        species = self.read_species()
        if all:
            # force last to be false
            last = False

        def type_coord(line):
            if 'outcoor' in line:
                return 1
            elif 'siesta: Atomic coordinates' in line:
                return 2
            # Signal not found
            return 0

        def next_geom():
            coord = 0
            while coord == 0:
                line = self.readline()
                if line == '':
                    return 0, None
                coord = type_coord(line)

            if coord == 1:
                return 1, self._read_geometry_outcoor(line, species)
            elif coord == 2:
                return 2, self._read_geometry_atomic(line, species)

        # Read until a coordinate block is found
        geom0 = None
        mds = []

        if all or last:
            # we need to read through all things!
            while True:
                coord, geom = next_geom()
                if coord == 0:
                    break
                if coord == 2:
                    geom0 = geom
                else:
                    mds.append(geom)

            # Since the user requests only the MD geometries
            # we only return those
            if last:
                if len(mds) > 0:
                    return mds[-1]
                return geom0
            return mds

        # just read the next geometry we hit
        return next_geom()[1]

    @sile_fh_open()
    def read_force(self, last=True, all=False):
        """ Reads the forces from the Siesta output file

        Parameters
        ----------
        last: bool, optional
           only read the last force
        all: bool, optional
           return a list of all forces (like an MD)
           If `True` `last` is ignored

        Returns
        -------
        numpy.ndarray or None
            returns ``None`` if the forces are not found in the
            output, otherwise forces will be returned
        """
        if all:
            last = False

        # Read until forces are found
        def next_force():
            line = self.readline()
            while not 'siesta: Atomic forces' in line:
                line = self.readline()
                if line == '':
                    return None

            # Now read data
            F = []
            line = self.readline()
            while '---' not in line:
                line = line.split()
                F.append([float(x) for x in line[-3:]])
                line = self.readline()
                if line == '':
                    break

            return np.array(F)

        # list of all forces
        Fs = []

        if all or last:
            while True:
                F = next_force()
                if F is None:
                    break
                Fs.append(F)

            if last:
                return Fs[-1]
            if self.job_completed:
                return Fs[:-1]
            return Fs

        return next_force()

    @sile_fh_open()
    def read_stress(self, key='static', last=True, all=False):
        """ Reads the stresses from the Siesta output file

        Parameters
        ----------
        key : {'static', 'total'}
           which stress to read from the output.
        last: bool, optional
           only read the last stress
        all: bool, optional
           return a list of all stresses (like an MD)
           If `True` `last` is ignored

        Returns
        -------
        numpy.ndarray or None
            returns ``None`` if the stresses are not found in the
            output, otherwise stresses will be returned
        """
        if all:
            last = False

        # Read until stress are found
        def next_stress():
            line = self.readline()
            while not ('siesta: Stress tensor' in line and key in line):
                line = self.readline()
                if line == '':
                    return None

            # Now read data
            S = []
            for _ in range(3):
                line = self.readline().split()
                S.append([float(x) for x in line[-3:]])

            return np.array(S)

        # list of all stresses
        Ss = []

        if all or last:
            while True:
                S = next_stress()
                if S is None:
                    break
                Ss.append(S)

            if last:
                return Ss[-1]
            if self.job_completed and key == 'static':
                return Ss[:-1]
            return Ss

        return next_stress()

    @sile_fh_open()
    def read_moment(self, orbital=False, quantity='S', last=True, all=False):
        """ Reads the moments from the Siesta output file
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
        """ Read specific content in the Siesta out file

        The currently implemented things are denoted in
        the parameters list.
        Note that the returned quantities are in the order
        of keywords, so:

        >>> read_data(geometry=True, force=True)
        <geometry>, <force>
        >>> read_data(force=True, geometry=True)
        <force>, <geometry>

        Parameters
        ----------
        geometry: bool, optional
           read geometry, args are passed to `read_geometry`
        force: bool, optional
           read force, args are passed to `read_force`
        stress: bool, optional
           read stress, args are passed to `read_stress`
        moment: bool, optional
           read moment, args are passed to `read_moment` (only for spin-orbit calculations)
        """
        run = []
        # This loops ensures that we preserve the order of arguments
        # From Py3.6 and onwards the **kwargs is an OrderedDictionary
        for kw in kwargs.keys():
            if kw in ['geometry', 'force', 'moment', 'stress']:
                if kwargs[kw]:
                    run.append(kw)

        # Clean running names
        for name in run:
            kwargs.pop(name)

        val = []
        for name in run:
            val.append(getattr(self, 'read_{}'.format(name.lower()))(*args, **kwargs))

        if len(val) == 0:
            return None
        elif len(val) == 1:
            val = val[0]
        return val


add_sile('out', outSileSiesta, case=False, gzip=True)
