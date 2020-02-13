import os
import numpy as np

from .sile import SileSiesta
from ..sile import *

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
        line = super().readline()
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
        itt = iter(self)
        while not 'moments: Atomic' in next(itt):
            if next(itt) == '':
                return None

        # The moments are printed in SPECIES list
        next(itt) # empty
        next(itt) # empty

        na = 0
        # Loop the species
        tbl = []
        # Read the species label
        while True:
            next(itt) # ""
            next(itt) # Atom    Orb ...
            # Loop atoms in this species list
            while True:
                line = next(itt)
                if line.startswith('Species') or \
                   line.startswith('--'):
                    break
                line = ' '
                atom = []
                ia = 0
                while not line.startswith('--'):
                    line = next(itt).split()
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
                line = next(itt).split() # Total ...
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

    @sile_fh_open()
    def read_scf(self, key='scf', iscf=-1, imd=None):
        r""" Parse SCF information and return a table of SCF information depending on what is requested

        Parameters
        ----------
        key : {'scf', 'ts-scf'}
            parse SCF information from Siesta SCF or TranSiesta SCF
        iscf : int, optional
            which SCF cycle should be stored. If ``-1`` only the final SCF step is stored,
            for None *all* SCF cycles are returned. When `iscf` values queried are not found they
            will be truncated to the nearest SCF step.
        imd: int or None, optional
            whether only a particular MD step is queried, if None, all MD steps are
            parsed and returned. A negative number wraps for the last MD steps.
        """
        def reset_d(d, line):
            if line.startswith('SCF cycle converged'):
                d['_final_iscf'] = len(d['data']) > 0

        if key.lower() == 'scf':
            def parse_next(line, d):
                line = line.strip().replace('*', '0')
                reset_d(d, line)
                if line.startswith('ts-Vha:'):
                    d['ts-Vha'] = float(line.split()[1])
                elif line.startswith('scf:'):
                    d['_found_iscf'] = True
                    if len(line) == 97:
                        data = [int(line[5:9]), float(line[9:25]), float(line[25:41]),
                                float(line[41:57]), float(line[57:67]), float(line[67:77]),
                                float(line[77:87]), float(line[87:97])]
                    elif len(line) == 87:
                        data = [int(line[5:9]), float(line[9:25]), float(line[25:41]),
                                float(line[41:57]), float(line[57:67]), float(line[67:77]),
                                float(line[77:87])]
                    else:
                        # Populate DATA
                        data = line.split()
                        data =  [int(data[1])] + list(map(float, data[2:]))
                    d['data'] = data

        elif key.lower() == 'ts-scf':
            def parse_next(line, d):
                line = line.strip().replace('*', '0')
                reset_d(d, line)
                if line.startswith('ts-Vha:'):
                    d['ts-Vha'] = float(line.split()[1])
                elif line.startswith('ts-q:'):
                    data = line.split()
                    try:
                        d['ts-q'] = list(map(float, data[1:]))
                    except:
                        # We are probably reading a device list
                        pass
                elif line.startswith('ts-scf:'):
                    d['_found_iscf'] = True
                    if len(line) == 100:
                        data = [int(line[8:12]), float(line[12:28]), float(line[28:44]),
                                float(line[44:60]), float(line[60:70]), float(line[70:80]),
                                float(line[80:90]), float(line[90:100]), d['ts-Vha']] + d['ts-q']
                    elif len(line) == 90:
                        data = [int(line[8:12]), float(line[12:28]), float(line[28:44]),
                                float(line[44:60]), float(line[60:70]), float(line[70:80]),
                                float(line[80:90]), d['ts-Vha']] + d['ts-q']
                    else:
                        # Populate DATA
                        data = line.split()
                        data =  [int(data[1])] + list(map(float, data[2:])) + [d['ts-Vha']] + d['ts-q']
                    d['data'] = data

        # A temporary dictionary to hold information while reading the output file
        d = {
            '_found_iscf': False,
            '_final_iscf': False,
            'data': [],
        }
        md = []
        scf = []
        for line in self:
            parse_next(line, d)
            if d['_found_iscf']:
                data = d['data']

                if iscf is None or iscf < 0:
                    scf.append(data)
                elif data.iscf == iscf:
                    scf = data

            if d['_final_iscf']:
                if len(d['data']) == 0:
                    continue

                # First figure out which iscf we should store
                if iscf is None or iscf > 0:
                    # scf is correct
                    pass
                elif iscf < 0:
                    # truncate to 0
                    scf = scf[max(len(scf) + iscf, 0)]

                # Now we know scf is correct

                # Populate md
                md.append(scf)
                # Reset SCF data
                scf = []

                if imd == len(md):
                    return np.array(md[-1])

                d['_final_iscf'] = False

        # Now we know how many MD steps there are
        if imd is None:
            return np.array(md)
        return np.array(md[max(len(md) + imd, 0)])


add_sile('out', outSileSiesta, case=False, gzip=True)
