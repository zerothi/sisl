import os
import numpy as np

from .sile import SileSiesta
from ..sile import add_sile, sile_fh_open

from sisl._internal import set_module
import sisl._array as _a
from sisl import Geometry, Atom, SuperCell
from sisl.utils import PropertyDict
from sisl.utils.cmd import *
from sisl.unit.siesta import unit_convert

__all__ = ['outSileSiesta']


Bohr2Ang = unit_convert('Bohr', 'Ang')


def _ensure_species(species):
    """ Ensures that the species list is a list with entries (converts `None` to a list). """
    if species is None:
        return [Atom(i) for i in range(150)]
    return species


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

    @sile_fh_open(True)
    def read_basis_block(self):
        """ Reads the PAO.Basis block that Siesta writes """
        found, line = self.step_to("%block PAO.Basis")
        if not found:
            raise ValueError(f"{self.__class__.__name__}.read_basis_block could not find PAO.Basis in output")

        basis = []
        while not line.startswith("%endblock PAO.Basis"):
            line = self.readline()
            basis.append(line)

        return basis

    def _read_supercell_outcell(self):
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
        xyz = _a.arrayd(xyz)
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
    def read_scf(self, key="scf", iscf=-1, imd=None, as_dataframe=False):
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
                d['ts-Vha'] = float(line.split()[1])
            elif line.startswith('bulk-bias: |v'):
                d['bb-v'] = list(map(float, line.split()[-3:]))
                if 'bb-vx' not in props:
                    props.extend(['BB-vx', 'BB-vy', 'BB-vz'])
            elif line.startswith('bulk-bias: {q'):
                d['bb-q'] = list(map(float, line.split()[-3:]))
                if 'bb-q+' not in props:
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
                    d['data'] = data

        elif key.lower() == 'ts-scf':
            props.append("ts-Vha")
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
                                float(line[80:90]), float(line[90:100]), d['ts-Vha']] + d['ts-q']
                    elif len(line) == 90:
                        data = [int(line[8:12]), float(line[12:28]), float(line[28:44]),
                                float(line[44:60]), float(line[60:70]), float(line[70:80]),
                                float(line[80:90]), d['ts-Vha']] + d['ts-q']
                    else:
                        # Populate DATA by splitting
                        data = line.split()
                        data = [int(data[1])] + list(map(float, data[2:])) + [d['ts-Vha']] + d['ts-q']
                    d['data'] = data

        # A temporary dictionary to hold information while reading the output file
        d = {
            '_found_iscf': False,
            '_final_iscf': 0,
            'data': [],
        }
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

            if iscf is None:
                # since each MD step may be a different number of SCF steps
                # we cannot convert to a dense array
                return md
            return np.array(md)

        # correct imd to ensure we check against the final size
        imd = min(len(md) - 1, max(len(md) + imd, 0))
        if len(md) == 0:
            # no data collected
            if as_dataframe:
                return pd.DataFrame(index=pd.Index([], name="iscf"),
                                    columns=props[1:])
            return np.array(md[imd])

        if imd > len(md):
            raise ValueError(f"{self.__class__.__name__}.read_scf could not find requested MD step ({imd}).")

        # If a certain imd was requested, get it
        # Remember that if imd is positive, we stopped reading at the moment we reached it
        scf = np.array(md[imd])
        if as_dataframe:
            return MDstep_dataframe(scf)
        return scf

    _md_step_last_line = "Target enthalpy"

    @sile_fh_open()
    def read_charge(self, name, iscf=None, imd=None, as_dataframe=False):
        """Reads the net charges printed in the output

        Parameters
        ---------
        name: {"voronoi", "hirshfeld"}
            the name of the charges that you want to read.
        iscf: int, optional
            index of the scf iteration you want the charges for.
            If not specified, all available scf iterations will be returned.
        imd: int
            index of the md step you want the charges for.
            If not specified, all available md steps will be returned.
        as_dataframe: boolean, optional
            whether all the information should be returned as a pandas dataframe.

        Returns
        ---------
        list of numpy.ndarray or numpy.ndarray or pandas.DataFrame
            A list that contains for each MD step, a numpy array of shape (nscf, natoms, columns),
            where the number of columns depends on how much information does SIESTA print about the
            charges.

            Special cases:
              - If a particular scf iteration is requested (`iscf`), the arrays are of shape (natoms, columns)
              - If a particular MD step is requested (`imd`), only one array is returned, instead of a list.

            If `as_dataframe` is set to `True`, a dataframe is returned. The indices of this dataframe are:
              - "atom": The atom index.
              - "iscf": The scf iteration. Not present if a particular scf iteration was requested (`iscf`).
              - "imd": The MD step. Not present if a particular MD step was requested (`imd`).
        """
        # Normalize the "name" argument to lowercase
        if not isinstance(name, str):
            raise TypeError("The 'name' argument should be a string")
        which = name.lower()

        # Check that a known charge has been requested
        known_charges = ("voronoi", "hirshfeld")
        if which not in known_charges:
            raise ValueError(f"'name' should be one of {known_charges}, you passed {which}")

        # Now capitalize, because it is written as capitalized in SIESTA output (see _read below)
        which = which.capitalize()

        # Define the names of the columns that are valid. We could directly read the column names
        # from the header, but some names have spaces so we can not do header_line.split()
        valid_columns = ["dQatom", "Atom pop", "S", "Sx", "Sy", "Sz"]
        # Some information values that will help us with the parsing and processing
        info = {
            "header": "", # The header line for the charges, contains the name of each column
            "read_final": False, # Whether we have read the charges at the end of the file
        }

        # Define a helper function that actually reads the charge values
        def _read(stop_strings=(), just_check=False):
            """Tries to read the charges until a stop sequence or end of file is reached.

            Parameters
            ---------
            stop_strings: array-like of str, optional
                if provided, it will stop attempting to read when any of these strings is found.
            just_check: boolean, optional
                this argument is meant so that you can use the function as a checker to know whether
                charges are present or not, without needing to read them.

                If `True`, returns True when charges are available or the results of the
                self.step_to() call otherwise.

            Returns
            ---------
            list or None
                Returns the list of net charges that has found. If it hasn't found any, returns None. 
            """
            # Try to find the next charges block
            found, line, i_found = self.step_to([f"{which} Atomic Populations",
                                                 f"{which} Net Atomic Populations",
                                                 *stop_strings], ret_index=True)

            # We didn't find a charges block
            if not found or i_found > 1:
                if just_check:
                    return found, i_found - 2, line
                return None

            if just_check:
                return True

            # The next line contains a header with the names of the columns
            # We inform about the header. In the old format, dQatom was called Q atom, we
            # standarize the column names.
            line = self.readline()
            info["header"] = line.replace(" Qatom", " dQatom")

            # We have found the header, prepare a list to read the charges
            atom_charges = []

            # Define the function that parses the charges
            def _parse_charge(line):
                at, *vals, symbol = line.split()
                at = int(line.split()[0])
                return vals

            # Now try to parse each following line until the line can't be parsed
            while line != "":
                try:
                    line = self.readline()
                    charge_vals = _parse_charge(line)
                    atom_charges.append(charge_vals)
                except:
                    # We already have the charge values and we reached a line that can't be parsed,
                    # this means we have reached the end.
                    break

            return atom_charges

        # Perform some checks on the first MD step to know what we are dealing with

        # Here we check if there are charges written for each scf iteration
        info["scf_charges"] = _read(("scf:",), just_check=True) is True

        # This will let us know if there are charges at the end of each MD step
        # Basically, we are going to build an array where True means we found charges
        # and anything else means that we didn't. Checking for the second to last item
        # will tell us if there are charges between the last "scf:" and the end of the MD step.
        found = []
        while True:
            ret = _read(("scf:", self._md_step_last_line), just_check=True)
            found.append(ret)

            if ret is not True:
                if not ret[0]:
                    raise Exception("We can't seem to find the end of a MD step in this file")
                elif ret[1] == 1:
                    break
        info["MD_step_charges"] = found[-2] is True

        # Raise errors if the requested charges are impossible to get given the information
        # written in this file.
        if not info["MD_step_charges"] and imd not in [None, -1]:
            raise ValueError(f"You requested charges for MD step {imd}, but the file does not contain charges for each MD step")
        if not info["scf_charges"] and iscf not in [None, -1]:
            raise ValueError(f"You requested charges for scf iteration {iscf}, but the file does not contain charges for each scf step")

        # Now that we know what we are looking at, we just close and open the file again
        # so that we can read it all from the beggining
        self.fh.close()
        self._open()

        # If there are no scf charges and no md step charges, just go on to read the final ones.
        if not info["MD_step_charges"] and not info["scf_charges"]:
            charges = [_a.arrayd([_read()])]
            if charges[0] is None:
                raise Exception(f"We couldn't find any {which} charges in the file")
            info["read_final"] = True

        # If there are charges inside the MD steps (either md step or scf-wise), loop through
        # all MD steps.
        else:
            charges = []
            md_step = 0
            while True:
                # If a specific md step was requested and this is not it, just go to the next
                if imd is not None and imd >= 0 and md_step != imd:
                    found, _ = self.step_to(self._md_step_last_line, reread=False)
                    if not found:
                        break
                    md_step += 1
                    continue

                # Read all charges in the MD step.
                step_charges = []
                while True:
                    scf_charge = _read((self._md_step_last_line, ))
                    if scf_charge is None:
                        # We have reached the end of the MD step
                        break
                    step_charges.append(scf_charge)

                # If we didn't read any charges for this md step, it basically means that there are
                # no more steps, just leave
                if not step_charges:
                    md_step -=1
                    break

                # Now, do some sanitizing of the charges we have obtained for this step
                if info["scf_charges"]:
                    # There is a first charge printout before the first scf iteration
                    step_charges = step_charges[1:]
                    # Also, if both scf and md charges are turned on, the last printout is repeated
                    if info["MD_step_charges"]:
                        step_charges = step_charges[:-1]

                    # If a specific scf iteration was requested, try to get it
                    if iscf is not None:
                        try:
                            step_charges = [step_charges[iscf]]
                        except IndexError:
                            raise ValueError(f"You are asking for scf iteration {iscf}, but MD step {md_step} has {len(step_charges)} iterations.")

                # Append the MD step charges to the list of charges
                charges.append(_a.arrayd(step_charges))

                # If this was the md step requested, we're done!
                if imd is not None and imd >= 0 and md_step == imd:
                    break

                md_step += 1

            # Now we have read all the MD steps that we needed to read if they were available.
            if imd is not None:
                if imd > md_step or imd < -(md_step + 1):
                    raise ValueError(f"You requested md step number {imd}, but there are only {md_step} steps in this file.")
                elif imd < 0:
                    # If imd was negative, we have obviously needed to read all the md steps,
                    # and now we can retrieve the one that the user needed
                    charges = [charges[imd]]

        # At this point, we have read all the charges, we just need to do some processing before returning them

        # Convert charges list to a dataframe if requested
        if as_dataframe:
            import pandas as pd

            def MD_step_dataframe(step_charges):
                """Given the array of charges for a MD step, returns a dataframe.

                This dataframe is multiindexed ("iscf", "atom") unless a particular iscf was requested,
                in which case, there's only a single index: "atom".
                """
                if step_charges.shape[0] == 1 and iscf is not None:
                    df = pd.DataFrame(step_charges[0])
                    df.index.name = "atom"
                else:
                    if info["read_final"] or (info["MD_step_charges"] and not info["scf_charges"]):
                        # If we read from the final charges and there was no iscf specified,
                        # we inform that the information belongs to the last iscf
                        keys = [-1]
                    else:
                        keys = np.arange(step_charges.shape[0])

                    df = pd.concat([pd.DataFrame(x) for x in step_charges], keys=keys)
                    df.index.names = ["iscf", "atom"]
                return df

            # Get all the dataframes for each MD step
            dfs = [MD_step_dataframe(step_charges) for step_charges in charges]

            print(dfs)

            if len(dfs) == 1 and imd is not None:
                # If the user requested a specific imd, we don't need to add any index
                charges = dfs[0]
            else:
                # Otherwise, we need to concatenate all dfs through a new index: imd
                if info["read_final"]:
                    # If we read from the final charges and there was no imd specified,
                    # we inform that the information belongs to the last imd
                    keys = [-1]
                else:
                    keys = np.arange(len(charges))

                charges = pd.concat(dfs, keys=keys, names=["imd"])

            # Finally, give names to the columns of the dataframe
            charges.columns = [col for col in valid_columns if f" {col} " in info["header"]]
        else:
            # If we are not building a dataframe, we need to also shape the output according to
            # whether the user requested a given imd or iscf
            if iscf is not None:
                charges = [step_charges[0] for step_charges in charges]
            if imd is not None:
                charges = charges[0]

        return charges

add_sile('out', outSileSiesta, case=False, gzip=True)
