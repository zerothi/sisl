# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import os
import numpy as np
from functools import lru_cache

from .sile import SileSiesta
from ..sile import add_sile, sile_fh_open, SileError

from sisl._common import Opt
from sisl._internal import set_module
import sisl._array as _a
from sisl import Geometry, Atom, SuperCell
from sisl.utils import PropertyDict
from sisl.utils.cmd import *
from sisl.unit.siesta import unit_convert

__all__ = ['outSileSiesta']


Bohr2Ang = unit_convert('Bohr', 'Ang')


def _ensure_atoms(atoms):
    """ Ensures that the atoms list is a list with entries (converts `None` to a list). """
    if atoms is None:
        return [Atom(i) for i in range(150)]
    elif len(atoms) == 0:
        return [Atom(i) for i in range(150)]
    return atoms


@set_module("sisl.io.siesta")
class outSileSiesta(SileSiesta):
    """ Output file from Siesta

    This enables reading the output quantities from the Siesta output.
    """
    _completed = None

    def readline(self):
        line = super().readline()
        if 'Job completed' in line:
            self._completed = True
        return line

    readline.__doc__ = SileSiesta.readline.__doc__

    @sile_fh_open()
    def completed(self):
        """ True if the full file has been read and "Job completed" was found. """
        if self._completed is None:
            completed = self.step_to("Job completed")[0]
        else:
            completed = self._completed
        if completed:
            self._completed = True
        return completed

    @lru_cache(1)
    @sile_fh_open(True)
    def read_basis(self):
        """ Reads the basis as found in the output file

        This parses 3 things:

        1. At the start of the file there are some initatom output
           specifying which species in the calculation.
        2. Reading the <basis_specs> entries for the masses
        3. Reading the PAO.Basis block output for orbital information
        """
        found, line = self.step_to("Species number:")
        if not found:
            return []

        atoms = {}
        order = []
        while 'Species number:' in line:
            ls = line.split()
            if ls[3] == 'Atomic':
                atoms[ls[7]] = {'Z': int(ls[5]), 'tag': ls[7]}
                order.append(ls[7])
            else:
                atoms[ls[4]] = {'Z': int(ls[7]), 'tag': ls[4]}
                order.append(ls[4])
            line = self.readline()

                # Now go down to basis_specs
        found, line = self.step_to("<basis_specs>")
        while found:
            # =====
            self.readline()
            # actual line
            line = self.readline().split("=")
            tag = line[0].split()[0]
            atoms[tag]["mass"] = float(line[2].split()[0])
            found, line = self.step_to("<basis_specs>", reread=False)

        block = []
        found, line = self.step_to("%block PAO.Basis")
        line = self.readline()
        while not line.startswith("%endblock PAO.Basis"):
            block.append(line)
            line = self.readline()

        from .fdf import fdfSileSiesta
        atom_orbs = fdfSileSiesta._parse_pao_basis(block)
        for atom, orbs in atom_orbs.items():
            atoms[atom]["orbitals"] = orbs

        return [Atom(**atoms[tag]) for tag in order]

    def _r_supercell_outcell(self):
        """ Wrapper for reading the unit-cell from the outcoor block """

        # Read until outcell is found
        found, line = self.step_to("outcell: Unit cell vectors")
        if not found:
            raise ValueError(f"{self.__class__.__name__}._r_supercell_outcell did not find outcell key")

        Ang = 'Ang' in line

        # We read the unit-cell vectors (in Ang)
        cell = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            cell.append([float(x) for x in line[:3]])
            line = self.readline()

        cell = _a.arrayd(cell)

        if not Ang:
            cell *= Bohr2Ang

        return SuperCell(cell)

    def _r_geometry_outcoor(self, line, atoms=None):
        """ Wrapper for reading the geometry as in the outcoor output """
        atoms_order = _ensure_atoms(atoms)
        is_final = 'Relaxed' in line or 'Final (unrelaxed)' in line

        # Now we have outcoor
        scaled = 'scaled' in line
        fractional = 'fractional' in line
        Ang = 'Ang' in line

        # Read in data
        xyz = []
        atoms = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            if len(line) != 6:
                break
            xyz.append(line[:3])
            atoms.append(atoms_order[int(line[3]) - 1])
            line = self.readline()

        # in outcoor we know it is always just after
        # But not if not variable cell.
        # Retrieve the unit-cell (but do not skip file-descriptor position)
        # This is because the current unit-cell is not always written.
        pos = self.fh.tell()
        cell = self._r_supercell_outcell()
        if is_final and self.fh.tell() < pos:
            # we have wrapped around the file
            self.fh.seek(pos, os.SEEK_SET)
        xyz = _a.arrayd(xyz)

        # Now create the geometry
        if scaled:
            # The output file for siesta does not
            # contain the lattice constant.
            # So... :(
            raise ValueError("Could not read the lattice-constant for the scaled geometry")
        elif fractional:
            xyz = xyz.dot(cell.cell)
        elif not Ang:
            xyz *= Bohr2Ang

        return Geometry(xyz, atoms, sc=cell)

    def _r_geometry_atomic(self, line, atoms=None):
        """ Wrapper for reading the geometry as in the outcoor output """
        atoms_order = _ensure_atoms(atoms)

        # Now we have outcoor
        Ang = 'Ang' in line

        # Read in data
        xyz = []
        atoms = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            xyz.append([float(x) for x in line[1:4]])
            atoms.append(atoms_order[int(line[4])-1])
            line = self.readline()

        # Retrieve the unit-cell (but do not skip file-descriptor position)
        # This is because the current unit-cell is not always written.
        pos = self.fh.tell()
        cell = self._r_supercell_outcell()
        self.fh.seek(pos, os.SEEK_SET)

        # Convert xyz
        xyz = _a.arrayd(xyz)
        if not Ang:
            xyz *= Bohr2Ang

        return Geometry(xyz, atoms, sc=cell)

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
        atoms = self.read_basis()
        if all:
            # force last to be false
            last = False

        def func_none(*args, **kwargs):
            """ Wrapper to return None """
            return None

        def next_geom():
            coord, func = 0, func_none
            line = ' '
            while coord == 0 and line != '':
                line = self.readline()
                if 'outcoor' in line and 'coordinates' in line:
                    coord, func = 1, self._r_geometry_outcoor
                elif 'siesta: Atomic coordinates' in line:
                    coord, func = 2, self._r_geometry_atomic
            return coord, func(line, atoms)

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
    def read_force(self, last=True, all=False, total=False, max=False):
        """ Reads the forces from the Siesta output file

        Parameters
        ----------
        last: bool, optional
           only read the last force
        all: bool, optional
           return a list of all forces (like an MD)
           If `True` `last` is ignored
        total: bool, optional
            return the total forces instead of the atomic forces.
        max: bool, optional
            whether only the maximum atomic force should be returned for each step.

            Setting it to `True` is equivalent to `max(outSile.read_force())` in case atomic forces
            are written in the output file (`WriteForces .true.` in the fdf file)

            Note that this is not the same as doing `max(outSile.read_force(total=True))` since
            the forces returned in that case are averages on each axis.

        Returns
        -------
        numpy.ndarray or None
            returns ``None`` if the forces are not found in the
            output, otherwise forces will be returned

            The shape of the array will be different depending on the type of forces requested:
                - atomic (default): (nMDsteps, nAtoms, 3)
                - total: (nMDsteps, 3)
                - max: (nMDsteps, )

            If `all` is `False`, the first dimension does not exist. In the case of max, the returned value
            will therefore be just a float, not an array.

            If `total` and `max` are both `True`, they are returned separately as a tuple: ``(total, max)``
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
            if 'siesta:' in line:
                # This is the final summary, we don't need to read it as it does not contain new information
                # and also it make break things since max forces are not written there
                return None

            # First, we encounter the atomic forces
            while '---' not in line:
                line = line.split()
                if not (total or max):
                    F.append([float(x) for x in line[-3:]])
                line = self.readline()
                if line == '':
                    break

            line = self.readline()
            # Then, the total forces
            if total:
                F = [float(x) for x in line.split()[-3:]]

            line = self.readline()
            #And after that we can read the max force
            if max and len(line.split()) != 0:
                line = self.readline()
                maxF = float(line.split()[1])

                # In case total is also requested, we are going to store it all in the same variable
                # It will be separated later
                if total:
                    F = (*F, maxF)
                else:
                    F = maxF

            return _a.arrayd(F)

        def return_forces(Fs):
            # Handle cases where we can't now if they are found
            if Fs is None: return None
            Fs = _a.arrayd(Fs)
            if max and total:
                return (Fs[..., :-1], Fs[..., -1])
            elif max and not all:
                return Fs.ravel()[0]
            return Fs

        if all or last:
            # list of all forces
            Fs = []
            while True:
                F = next_force()
                if F is None:
                    break
                Fs.append(F)

            if last:
                return return_forces(Fs[-1])

            return return_forces(Fs)

        return return_forces(next_force())

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

            return _a.arrayd(S)

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
            if self.completed() and key == 'static':
                return Ss[:-1]
            return Ss

        return next_stress()

    @sile_fh_open()
    def read_moment(self, orbitals=False, quantity='S', last=True, all=False):
        """ Reads the moments from the Siesta output file

        These will only be present in case of spin-orbit coupling.

        Parameters
        ----------
        orbitals: bool, optional
           return a table with orbitally resolved
           moments.
        quantity: {'S', 'L'}, optional
           return the spin-moments or the L moments
        last: bool, optional
           only read the last force
        all: bool, optional
           return a list of all forces (like an MD)
           If `True` `last` is ignored
        """

        # Read until outcoor is found
        itt = iter(self)
        for line in itt:
            if 'moments: Atomic' in line:
                break
        if not 'moments: Atomic' in line:
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
                if not orbitals:
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
            return _a.arrayd(moments)
        return moments

    @sile_fh_open()
    def read_energy(self):
        """ Reads the final energy distribution

        Currently the energies translated are:

        ``band``
             band structure energy
        ``kinetic``
             electronic kinetic energy
        ``hartree``
             electronic electrostatic Hartree energy
        ``dftu``
             DFT+U energy
        ``spin_orbit``
             spin-orbit energy
        ``extE``
             external field energy
        ``xc``
             exchange-correlation energy
        ``exchange``
             exchange energy
        ``correlation``
             correlation energy
        ``bulkV``
             bulk-bias correction energy
        ``total``
             total energy
        ``negf``
             NEGF energy
        ``fermi``
             Fermi energy
        ``ion.electron``
             ion-electron interaction energy
        ``ion.ion``
             ion-ion interaction energy
        ``ion.kinetic``
             kinetic ion energy


        Any unrecognized key gets added *as is*.

        Examples
        --------
        >>> energies = sisl.get_sile("RUN.out").read_energy()
        >>> ion_energies = energies.ion
        >>> ion_energies.ion # ion-ion interaction energy
        >>> ion_energies.kinetic # ion kinetic energy
        >>> energies.fermi # fermi energy

        Returns
        -------
        PropertyDict : dictionary like lookup table ionic energies are stored in a nested `PropertyDict` at the key ``ion`` (all energies in eV)
        """
        found = self.step_to("siesta: Final energy", reread=False)[0]
        out = PropertyDict()
        out.ion = PropertyDict()
        if not found:
            return out
        itt = iter(self)

        # Read data
        line = next(itt)
        name_conv = {
            "Band Struct.": "band",
            "Kinetic": "kinetic",
            "Hartree": "hartree",
            "Edftu": "dftu",
            "Eldau": "dftu",
            "Eso": "spin_orbit",
            "Ext. field": "extE",
            "Exch.-corr.": "xc",
            "Exch.": "exchange",
            "Corr.": "correlation",
            "Ekinion": "ion.kinetic",
            "Ion-electron": "ion.electron",
            "Ion-ion": "ion.ion",
            "Bulk bias": "bulkV",
            "Total": "total",
            "Fermi": "fermi",
            "Enegf": "negf",
        }
        while len(line.strip()) > 0:
            key, val = line.split("=")
            key = key.split(":")[1].strip()
            key = name_conv.get(key, key)
            if key.startswith("ion."):
                # sub-nest
                out.ion[key[4:]] = float(val)
            else:
                out[key] = float(val)
            line = next(itt)

        return out

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
        energy: bool, optional
           read final energies, args are passed to `read_energy`
        """
        run = []
        # This loops ensures that we preserve the order of arguments
        # From Py3.6 and onwards the **kwargs is an OrderedDictionary
        for kw in kwargs.keys():
            if kw in ['geometry', 'force', 'moment', 'stress', 'energy']:
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
    def read_scf(self, key="scf", iscf=-1, imd=None, as_dataframe=False, ret_header=False):
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
        as_dataframe: boolean, optional
            whether the information should be returned as a `pandas.DataFrame`. The advantage of this
            format is that everything is indexed and therefore you know what each value means.You can also
            perform operations very easily on a dataframe. 
        ret_header: bool, optional
            whether to also return the headers that define each value in the returned array,
            will have no effect if `as_dataframe` is true.
        """

        #These are the properties that are written in SIESTA scf
        props = ["iscf", "Eharris", "E_KS", "FreeEng", "dDmax", "Ef", "dHmax"]

        if not iscf is None:
            if iscf == 0:
                raise ValueError(f"{self.__class__.__name__}.read_scf requires iscf argument to *not* be 0!")
        if not imd is None:
            if imd == 0:
                raise ValueError(f"{self.__class__.__name__}.read_scf requires imd argument to *not* be 0!")
        def reset_d(d, line):
            if line.startswith('SCF cycle converged'):
                if len(d['data']) > 0:
                    d['_final_iscf'] = 1
            elif line.startswith('SCF cycle continued'):
                d['_final_iscf'] = 0

        def common_parse(line, d):
            if line.startswith('ts-Vha:'):
                d['ts-Vha'] = [float(line.split()[1])]
                if 'ts-Vha' not in props:
                    d['order'].append("ts-Vha")
                    props.append("ts-Vha")
            elif line.startswith("spin moment: S"):
                # 4.1 and earlier
                d['S'] = list(map(float, line.split("=")[1].split()[1:]))
                if 'Sx' not in props:
                    d['order'].append("S")
                    props.extend(['Sx', 'Sy', 'Sz'])
            elif line.startswith("spin moment: {S}"):
                # 4.2 and later
                d['S'] = list(map(float, line.split("= {")[1].split()[:3]))
                if 'Sx' not in props:
                    d['order'].append("S")
                    props.extend(['Sx', 'Sy', 'Sz'])
            elif line.startswith('bulk-bias: |v'):
                d['bb-v'] = list(map(float, line.split()[-3:]))
                if 'BB-vx' not in props:
                    d['order'].append("bb-v")
                    props.extend(['BB-vx', 'BB-vy', 'BB-vz'])
            elif line.startswith('bulk-bias: {q'):
                d['bb-q'] = list(map(float, line.split()[-3:]))
                if 'BB-q+' not in props:
                    d['order'].append("bb-q")
                    props.extend(['BB-q+', 'BB-q-', 'BB-q0'])
            else:
                return False
            return True

        if key.lower() == 'scf':
            def parse_next(line, d):
                line = line.strip().replace('*', '0')
                reset_d(d, line)
                if common_parse(line, d):
                    pass
                elif line.startswith('scf:'):
                    d['_found_iscf'] = True
                    if len(line) == 97:
                        # this should be for Efup/dwn
                        # but I think this will fail for as_dataframe (TODO)
                        data = [int(line[5:9]), float(line[9:25]), float(line[25:41]),
                                float(line[41:57]), float(line[57:67]), float(line[67:77]),
                                float(line[77:87]), float(line[87:97])]
                    elif len(line) == 87:
                        data = [int(line[5:9]), float(line[9:25]), float(line[25:41]),
                                float(line[41:57]), float(line[57:67]), float(line[67:77]),
                                float(line[77:87])]
                    else:
                        # Populate DATA by splitting
                        data = line.split()
                        data =  [int(data[1])] + list(map(float, data[2:]))
                    construct_data(d, data)

        elif key.lower() == 'ts-scf':
            def parse_next(line, d):
                line = line.strip().replace('*', '0')
                reset_d(d, line)
                if common_parse(line, d):
                    pass
                elif line.startswith('ts-q:'):
                    data = line.split()[1:]
                    try:
                        d['ts-q'] = list(map(float, data))
                    except:
                        # We are probably reading a device list
                        # ensure that props are appended
                        if data[-1] not in props:
                            props.extend(data)
                        pass
                elif line.startswith('ts-scf:'):
                    d['_found_iscf'] = True
                    if len(line) == 100:
                        data = [int(line[8:12]), float(line[12:28]), float(line[28:44]),
                                float(line[44:60]), float(line[60:70]), float(line[70:80]),
                                float(line[80:90]), float(line[90:100])]
                    elif len(line) == 90:
                        data = [int(line[8:12]), float(line[12:28]), float(line[28:44]),
                                float(line[44:60]), float(line[60:70]), float(line[70:80]),
                                float(line[80:90])]
                    else:
                        # Populate DATA by splitting
                        data = line.split()
                        data = [int(data[1])] + list(map(float, data[2:]))
                    construct_data(d, data)

        # A temporary dictionary to hold information while reading the output file
        d = {
            '_found_iscf': False,
            '_final_iscf': 0,
            'data': [],
            'order': [],
        }
        def construct_data(d, data):
            for key in d["order"]:
                data.extend(d[key])
            d["data"] = data

        md = []
        scf = []
        for line in self:
            parse_next(line, d)
            if d['_found_iscf']:
                d['_found_iscf'] = False
                data = d['data']
                if len(data) == 0:
                    continue

                if iscf is None or iscf < 0:
                    scf.append(data)
                elif data[0] <= iscf:
                    # this ensures we will retain the latest iscf in
                    # case the requested iscf is too big
                    scf = data

            if d['_final_iscf'] == 1:
                d['_final_iscf'] = 2
            elif d['_final_iscf'] == 2:
                d['_final_iscf'] = 0
                data = d['data']
                if len(data) == 0:
                    # this traps the case where we read ts-scf
                    # but find the final scf iteration.
                    # In that case we don't have any data.
                    scf = []
                    continue

                if len(scf) == 0:
                    # this traps cases where final_iscf has
                    # been trickered but we haven't collected anything.
                    # I.e. if key == scf but ts-scf also exists.
                    continue

                # First figure out which iscf we should store
                if iscf is None: # or iscf > 0
                    # scf is correct
                    pass
                elif iscf < 0:
                    # truncate to 0
                    scf = scf[max(len(scf) + iscf, 0)]

                # Populate md
                md.append(np.array(scf))
                # Reset SCF data
                scf = []

                # In case we wanted a given MD step and it's this one, just stop reading
                # We are going to return the last MD (see below)
                if imd == len(md):
                    break

        # Define the function that is going to convert the information of a MDstep to a Dataset
        if as_dataframe:
            import pandas as pd

            def MDstep_dataframe(scf):
                scf = np.atleast_2d(scf)
                return pd.DataFrame(
                    scf[..., 1:],
                    index=pd.Index(scf[..., 0].ravel().astype(np.int32),
                                   name="iscf"),
                    columns=props[1:]
                )

        # Now we know how many MD steps there are

        # We will return stuff based on what the user requested
        # For pandas DataFrame this will be dependent
        #  1. all MD steps requested => imd == index, iscf == column (regardless of iscf==none|int)
        #  2. 1 MD step requested => iscf == index

        if imd is None:
            if as_dataframe:
                if len(md) == 0:
                    # return an empty dataframe (with imd as index)
                    return pd.DataFrame(index=pd.Index([], name="imd"),
                                        columns=props)
                # Regardless of what the user requests we will always have imd == index
                # and iscf a column, a user may easily change this.
                df = pd.concat(map(MDstep_dataframe, md),
                               keys=_a.arangei(1, len(md) + 1), names=["imd"])
                if iscf is not None:
                    df.reset_index("iscf", inplace=True)
                return df

            if iscf is not None:
                # since each MD step may be a different number of SCF steps
                # we can only convert for a specific entry
                md = np.array(md)
            if ret_header:
                return md, props
            return md

        # correct imd to ensure we check against the final size
        imd = min(len(md) - 1, max(len(md) + imd, 0))
        if len(md) == 0:
            # no data collected
            if as_dataframe:
                return pd.DataFrame(index=pd.Index([], name="iscf"),
                                    columns=props[1:])
            md = np.array(md[imd])
            if ret_header:
                return md, props
            return md

        if imd > len(md):
            raise ValueError(f"{self.__class__.__name__}.read_scf could not find requested MD step ({imd}).")

        # If a certain imd was requested, get it
        # Remember that if imd is positive, we stopped reading at the moment we reached it
        scf = np.array(md[imd])
        if as_dataframe:
            return MDstep_dataframe(scf)
        if ret_header:
            return scf, props
        return scf

    @sile_fh_open()
    def read_charge(self, name, iscf=Opt.ANY, imd=Opt.ANY, key_scf="scf", as_dataframe=False):
        r"""Read charges calculated in SCF loop or MD loop (or both)

        Siesta enables many different modes of writing out charges.

        NOTE: currently Mulliken charges are not implemented.

        The below table shows a list of different cases that
        may be encountered, the letters are referred to in the
        return section to indicate what is returned.

        +-----------+-----+-----+--------+-------+------------------+
        | Case      | *A* | *B* | *C*    | *D*   | *E*              |
        +-----------+-----+-----+--------+-------+------------------+
        | Charge    | MD  | SCF | MD+SCF | Final | Orbital resolved |
        +-----------+-----+-----+--------+-------+------------------+
        | Voronoi   | +   | +   | +      | +     | -                |
        +-----------+-----+-----+--------+-------+------------------+
        | Hirshfeld | +   | +   | +      | +     | -                |
        +-----------+-----+-----+--------+-------+------------------+
        | Mulliken  | +   | +   | +      | +     | +                |
        +-----------+-----+-----+--------+-------+------------------+

        Notes
        -----
        Errors will be raised if one requests information not present. I.e.
        passing an integer or `Opt.ALL` for `iscf` will raise an error if
        the SCF charges are not present. For `Opt.ANY` it will return
        the most information, effectively SCF will be returned if present.

        Currently Mulliken is not implemented, any help in reading this would be
        very welcome.

        Parameters
        ----------
        name: {"voronoi", "hirshfeld"}
            the name of the charges that you want to read
        iscf: int or Opt, optional
            index (0-based) of the scf iteration you want the charges for.
            If the enum specifier `Opt.ANY` or `Opt.ALL` are used, then
            the returned quantities depend on what is present.
            If ``None/Opt.NONE`` it will not return any SCF charges.
            If both `imd` and `iscf` are ``None`` then only the final charges will be returned.
        imd: int or Opt, optional
            index (0-based) of the md step you want the charges for.
            If the enum specifier `Opt.ANY` or `Opt.ALL` are used, then
            the returned quantities depend on what is present.
            If ``None/Opt.NONE`` it will not return any MD charges.
            If both `imd` and `iscf` are ``None`` then only the final charges will be returned.
        key_scf : str, optional
            the key lookup for the scf iterations (a ":" will automatically be appended)
        as_dataframe: boolean, optional
            whether charges should be returned as a pandas dataframe.

        Returns
        -------
        numpy.ndarray
            if a specific MD+SCF index is requested (or special cases where output is
            not complete)
        list of numpy.ndarray
            if one both `iscf` or `imd` is different from ``None/Opt.NONE``.
        pandas.DataFrame
            if `as_dataframe` is requested. The dataframe will have multi-indices if multiple
            SCF or MD steps are requested.
        """
        if not hasattr(self, 'fh'):
            with self:
                return read_charge(self, name, iscf, imd, key_scf, as_dataframe)
        namel = name.lower()
        if as_dataframe:
            import pandas as pd
            def _empty_charge():
                # build a fake dataframe with no indices
                return pd.DataFrame(index=pd.Index([], name="atom", dtype=np.int32),
                                    dtype=np.float32)
        else:
            pd = None
            def _empty_charge():
                # return for single value with nan values
                return _a.arrayf([[None]])

        # define helper function for reading voronoi+hirshfeld charges
        def _voronoi_hirshfeld_charges():
            """ Read output from Voronoi/Hirshfeld charges """
            nonlocal pd

            # Expecting something like this:
            # Voronoi Atomic Populations:
            # Atom #     dQatom  Atom pop         S        Sx        Sy        Sz  Species
            #      1   -0.02936   4.02936   0.00000  -0.00000   0.00000   0.00000  C

            # Define the function that parses the charges
            def _parse_charge(line):
                atom_idx, *vals, symbol = line.split()
                # assert that this is a proper line
                # this should catch cases where the following line of charge output
                # is still parseable
                atom_idx = int(atom_idx)
                return list(map(float, vals))

            # first line is the header
            header = (self.readline()
                      .replace("dQatom", "dq") # dQatom in master
                      .replace(" Qatom", " dq") # Qatom in 4.1
                      .replace("Atom pop", "e") # not found in 4.1
                      .split())[2:-1]

            # We have found the header, prepare a list to read the charges
            atom_charges = []
            line = ' '
            while line != "":
                try:
                    line = self.readline()
                    charge_vals = _parse_charge(line)
                    atom_charges.append(charge_vals)
                except:
                    # We already have the charge values and we reached a line that can't be parsed,
                    # this means we have reached the end.
                    break
            if pd is None:
                # not as_dataframe
                return _a.arrayf(atom_charges)

            # determine how many columns we have
            # this will remove atom indices and species, so only inside
            ncols = len(atom_charges[0])
            assert ncols == len(header)

            # the precision is limited, so no need for double precision
            return pd.DataFrame(atom_charges, columns=header, dtype=np.float32,
                                index=pd.RangeIndex(stop=len(atom_charges), name="atom"))

        # define helper function for reading voronoi+hirshfeld charges
        def _mulliken_charges():
            """ Read output from Mulliken charges """
            raise NotImplementedError("Mulliken charges are not implemented currently")

        # Check that a known charge has been requested
        if namel == "voronoi":
            _r_charge = _voronoi_hirshfeld_charges
            charge_keys = ["Voronoi Atomic Populations",
                           "Voronoi Net Atomic Populations"]
        elif namel == "hirshfeld":
            _r_charge = _voronoi_hirshfeld_charges
            charge_keys = ["Hirshfeld Atomic Populations",
                           "Hirshfeld Net Atomic Populations"]
        elif namel == "mulliken":
            _r_charge = _mulliken_charges
            charge_keys = ["mulliken: Atomic and Orbital Populations"]
        else:
            raise ValueError(f"{self.__class__.__name__}.read_charge name argument should be one of {known_charges}, got {name}?")

        # Ensure the key_scf matches exactly (prepend a space)
        key_scf = f" {key_scf.strip()}:"

        # Reading charges may be quite time consuming for large MD simulations.

        # to see if we finished a MD read, we check for these keys
        search_keys = [
            # two keys can signal ending SCF
            "SCF Convergence", "SCF_NOT_CONV",
            "siesta: Final energy",
            key_scf,
            *charge_keys
        ]
        # adjust the below while loop to take into account any additional
        # segments of search_keys
        IDX_SCF_END = [0, 1]
        IDX_FINAL = [2]
        IDX_SCF = [3]
        # the rest are charge keys
        IDX_CHARGE = list(range(len(search_keys) - len(charge_keys),
                                len(search_keys)))

        # state to figure out where we are
        state = PropertyDict()
        state.INITIAL = 0
        state.MD = 1
        state.SCF = 2
        state.CHARGE = 3
        state.FINAL = 4

        # a list of scf_charge
        md_charge = []
        md_scf_charge = []
        scf_charge = []
        final_charge = None

        # signal that any first reads are INITIAL charges
        current_state = state.INITIAL
        charge = _empty_charge()
        FOUND_SCF = False
        FOUND_MD = False
        FOUND_FINAL = False

        # TODO whalrus
        ret = self.step_to(search_keys, case=True, ret_index=True, reread=False)
        while ret[0]:
            if ret[2] in IDX_SCF_END:
                # we finished all SCF iterations
                current_state = state.MD
                md_scf_charge.append(scf_charge)
                scf_charge = []

            elif ret[2] in IDX_SCF:
                current_state = state.SCF
                # collect scf-charges (possibly none)
                scf_charge.append(charge)

            elif ret[2] in IDX_FINAL:
                current_state = state.FINAL
                # don't do anything, this is the final charge construct
                # regardless of where it comes from.

            elif ret[2] in IDX_CHARGE:
                FOUND_CHARGE = True
                # also read charge
                charge = _r_charge()

                if state.INITIAL == current_state or state.CHARGE == current_state:
                    # this signals scf charges
                    FOUND_SCF = True
                    # There *could* be 2 steps if we are mixing H,
                    # this is because it first does
                    # compute H -> compute DM -> compute H
                    # in the first iteration, subsequently we only do
                    # compute compute DM -> compute H
                    # once we hit ret[2] in IDX_SCF we will append
                    scf_charge = []

                elif state.MD == current_state:
                    FOUND_MD = True
                    # we just finished an SCF cycle.
                    # So any output between SCF ending and
                    # a new one beginning *must* be that geometries
                    # charge

                    # Here `charge` may be NONE signalling
                    # we don't have charge in MD steps.
                    md_charge.append(charge)

                    # reset charge
                    charge = _empty_charge()

                elif state.SCF == current_state:
                    FOUND_SCF = True

                elif state.FINAL == current_state:
                    FOUND_FINAL = True
                    # a special state writing out the charges after everything
                    final_charge = charge
                    charge = _empty_charge()
                    scf_charge = []
                    # we should be done and no other charge reads should be found!
                    # should we just break?

                current_state = state.CHARGE

            # step to next entry
            ret = self.step_to(search_keys, case=True, ret_index=True, reread=False)

        if not any((FOUND_SCF, FOUND_MD, FOUND_FINAL)):
            raise SileError(f"{self!s} does not contain any charges ({name})")

        # if the scf-charges are not stored, it means that the MD step finalization
        # has not been read. So correct
        if len(scf_charge) > 0:
            assert False, "this test shouldn't reach here"
            # we must not have read through the entire MD step
            # so this has to be a running simulation
            if charge is not None:
                scf_charge.append(charge)
                charge = _empty_charge()
            md_scf_charge.append(scf_charge)

        # otherwise there is some *parsing* error, so for now we use assert
        assert len(scf_charge) == 0

        if as_dataframe:
            # convert data to proper data structures
            # regardless of user requests. This is an overhead... But probably not that big of a problem.
            if FOUND_SCF:
                md_scf_charge = pd.concat([pd.concat(iscf,
                                                     keys=pd.RangeIndex(1, len(iscf)+1, name="iscf"))
                                           for iscf in md_scf_charge],
                                          keys=pd.RangeIndex(1, len(md_scf_charge)+1, name="imd"))
            if FOUND_MD:
                md_charge = pd.concat(md_charge, keys=pd.RangeIndex(1, len(md_charge)+1, name="imd"))
        else:
            if FOUND_SCF:
                nan_array = _a.emptyf(md_scf_charge[0][0].shape)
                nan_array.fill(np.nan)
                def get_md_scf_charge(scf_charge, iscf):
                    try:
                        return scf_charge[iscf]
                    except:
                        return nan_array
            if FOUND_MD:
                md_charge = np.stack(md_charge)

        # option parsing is a bit *difficult* with flag enums
        # So first figure out what is there, and handle this based
        # on arguments
        def _p(flag, found):
            """ Helper routine to do the following:

            Returns
            -------
            is_opt : bool
                whether the flag is an `Opt`
            flag :
                corrected flag
            """
            if isinstance(flag, Opt):
                # correct flag depending on what `found` is
                # If the values have been found we
                # change flag to None only if flag == NONE
                # If the case has not been found, we
                # change flag to None if ANY or NONE is in flags

                if found:
                    # flag is only NONE, then pass none
                    if not (Opt.NONE ^ flag):
                        flag = None
                else: # not found
                    # we convert flag to none
                    # if ANY or NONE in flag
                    if (Opt.NONE | Opt.ANY) & flag:
                        flag = None

            return isinstance(flag, Opt), flag

        opt_imd, imd = _p(imd, FOUND_MD)
        opt_iscf, iscf = _p(iscf, FOUND_SCF)

        if not (FOUND_SCF or FOUND_MD):
            # none of these are found
            # we request that user does not request any input
            if (opt_iscf or (not iscf is None)) or \
               (opt_imd or (not imd is None)):
                raise SileError(f"{self!s} does not contain MD/SCF charges")

        elif not FOUND_SCF:
            if opt_iscf or (not iscf is None):
                raise SileError(f"{self!s} does not contain SCF charges")

        elif not FOUND_MD:
            if opt_imd or (not imd is None):
                raise SileError(f"{self!s} does not contain MD charges")

        # if either are options they may hold
        if opt_imd and opt_iscf:
            if FOUND_SCF:
                return md_scf_charge
            elif FOUND_MD:
                return md_charge
            elif FOUND_FINAL:
                # I think this will never be reached
                # If neither are found they will be converted to
                # None
                return final_charge

            raise SileError(f"{str(self)} unknown argument for 'imd' and 'iscf'")

        elif opt_imd:
            # flag requested imd
            if not (imd & (Opt.ANY | Opt.ALL)):
                # wrong flag
                raise SileError(f"{str(self)} unknown argument for 'imd'")

            if FOUND_SCF and iscf is not None:
                # this should be handled, i.e. the scf should be taken out
                if as_dataframe:
                    return md_scf_charge.groupby(level=[0, 2]).nth(iscf)
                return np.stack(tuple(get_md_scf_charge(x, iscf) for x in md_scf_charge))

            elif FOUND_MD and iscf is None:
                return md_charge
            raise SileError(f"{str(self)} unknown argument for 'imd' and 'iscf', could not find SCF charges")

        elif opt_iscf:
            # flag requested imd
            if not (iscf & (Opt.ANY | Opt.ALL)):
                # wrong flag
                raise SileError(f"{str(self)} unknown argument for 'iscf'")
            if imd is None:
                # correct imd
                imd = -1
            if as_dataframe:
                md_scf_charge = md_scf_charge.groupby(level=0)
                group = list(md_scf_charge.groups.keys())[imd]
                return md_scf_charge.get_group(group).droplevel(0)
            return np.stack(md_scf_charge[imd])

        elif imd is None and iscf is None:
            if FOUND_FINAL:
                return final_charge
            raise SileError(f"{str(self)} does not contain final charges")

        elif imd is None:
            # iscf is not None, so pass through as though explicitly passed
            imd = -1

        elif iscf is None:
            # we return the last MD step and the requested scf iteration
            if as_dataframe:
                return md_charge.groupby(level=1).nth(imd)
            return md_charge[imd]

        if as_dataframe:
            # first select imd
            md_scf_charge = md_scf_charge.groupby(level=0)
            group = list(md_scf_charge.groups.keys())[imd]
            md_scf_charge = md_scf_charge.get_group(group).droplevel(0)
            return md_scf_charge.groupby(level=1).nth(iscf)
        return md_scf_charge[imd][iscf]


add_sile('out', outSileSiesta, case=False, gzip=True)
