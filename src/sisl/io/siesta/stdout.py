# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os
from collections import namedtuple
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Optional

import numpy as np

import sisl._array as _a
from sisl import Atom, Atoms, Geometry, Lattice
from sisl._common import Opt
from sisl._help import voigt_matrix
from sisl._internal import set_module
from sisl.messages import deprecation, warn
from sisl.physics import Spin
from sisl.physics.brillouinzone import MonkhorstPack
from sisl.unit.siesta import unit_convert
from sisl.utils import PropertyDict
from sisl.utils.cmd import *

from .._multiple import SileBinder, postprocess_tuple
from ..sile import SileError, add_sile, sile_fh_open
from .sile import SileSiesta

__all__ = ["stdoutSileSiesta", "outSileSiesta"]


Bohr2Ang = unit_convert("Bohr", "Ang")


def _ensure_atoms(atoms):
    """Ensures that the atoms list is a list with entries (converts `None` to a list)."""
    if atoms is None:
        return [Atom(i) for i in range(150)]
    elif len(atoms) == 0:
        return [Atom(i) for i in range(150)]
    return atoms


def _parse_spin(attr, instance, match):
    """Parse 'redata: Spin configuration *= <value>'"""
    opt = match.string.split("=")[-1].strip()

    if opt.startswith("nambu"):
        return Spin("nambu")
    if opt.startswith("spin-orbit"):
        return Spin("spin-orbit")
    if opt.startswith("collinear") or opt.startswith("colinear"):
        return Spin("polarized")
    if opt.startswith("non-col"):
        return Spin("non-colinear")
    return Spin()


def _read_scf_empty(scf):
    if isinstance(scf, tuple):
        return len(scf[0]) == 0
    return len(scf) == 0


def _read_scf_md_process(scfs):

    if len(scfs) == 0:
        return None

    if not isinstance(scfs, list):
        # single MD request either as:
        #  - np.ndarray
        #  - np.ndarray, tuple
        #  - pd.DataFrame
        return scfs

    has_props = isinstance(scfs[0], tuple)
    if has_props:
        my_len = lambda scf: len(scf[0])
    else:
        my_len = len

    scf_len1 = np.all(_a.fromiterd(map(my_len, scfs)) == 1)
    if isinstance(scfs[0], (np.ndarray, tuple)):

        if has_props:
            props = scfs[0][1]
            scfs = [scf[0] for scf in scfs]

        if scf_len1:
            scfs = np.array(scfs)
        if has_props:
            return scfs, props
        return scfs

    # We are dealing with a dataframe
    import pandas as pd

    df = pd.concat(
        scfs,
        keys=_a.arangei(1, len(scfs) + 1),
        names=["imd"],
    )
    if scf_len1:
        df.reset_index("iscf", inplace=True)
    return df


def _parse_in_dynamics(attr, instance, match):
    """Determines whether we are in the dynamics section or in the *Final* section.

    Basically it returns ``not instance.info._in_final_analysis``.
    """
    return not instance.info._in_final_analysis()


def _parse_version(attr, instance, match):
    opt = match.string.split(":", maxsplit=1)[-1].strip()

    version, *spec = opt.split("-", maxsplit=1)
    try:
        version = tuple(int(v) for v in version.split("."))
    except Exception:
        version = (0, 0, 0)

    # Convert version to a tuple
    Version = namedtuple("Version", "version spec")

    version = Version(version, spec)

    return version


def _in_final(self):
    return self.fh.tell() >= self.info._in_final_analysis_tell


@dataclass
class Toggler:
    """Allows simpler toggling for when a key is found or not.

    When executing `toggle`, then they key is added if not present,
    and return False.
    If the key is present, it will reset the `keys` attribute, add
    the `key`, and return True (because it got re-initialized).
    """

    keys: set[str] = field(init=False, default_factory=set)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return len(self.keys)

    def toggle(self, key: str) -> bool:
        """Toggle a key in the object. Return true if it was already present."""
        ret = False
        if key in self.keys:
            # easier to just reset it!
            self.keys = set()
            ret = True
        self.keys.add(key)
        return ret


@set_module("sisl.io.siesta")
class stdoutSileSiesta(SileSiesta):
    """Output file from Siesta

    This enables reading the output quantities from the Siesta output.
    """

    _info_attributes_ = [
        dict(
            name="version",
            searcher=r"^Version[ ]*: ",
            parser=_parse_version,
            not_found="warn",
        ),
        dict(
            name="na",
            searcher=r"^initatomlists: Number of atoms",
            parser=lambda attr, instance, match: int(match.string.split()[-3]),
            not_found="warn",
        ),
        dict(
            name="no",
            searcher=r"^initatomlists: Number of atoms",
            parser=lambda attr, instance, match: int(match.string.split()[-2]),
            not_found="warn",
        ),
        dict(
            name="nspecies",
            searcher=r"^redata: Number of Atomic Species",
            parser=lambda attr, instance, match: int(match.string.split("=")[-1]),
            not_found="warn",
        ),
        dict(
            name="completed",
            searcher=r".*Job completed",
            parser=lambda attr, instance, match: lambda: True,
            default=lambda: False,
            not_found="warn",
        ),
        dict(
            name="spin",
            searcher=r"^redata: Spin configuration",
            parser=_parse_spin,
        ),
        dict(
            name="_has_forces_in_dynamics",
            searcher=r"^siesta: Atomic forces",
            parser=_parse_in_dynamics,
        ),
        dict(
            name="_has_stress_in_dynamics",
            searcher=r"^siesta: Stress tensor",
            parser=_parse_in_dynamics,
        ),
        dict(
            name="_in_final_analysis_tell",
            searcher=r"^siesta: Final energy",
            parser=lambda attr, instance, match: instance.fh.tell(),
            default=1e12,
        ),
        dict(
            name="_in_final_analysis",
            searcher=None,
            default=_in_final,
            found=True,
        ),
    ]

    @deprecation(
        "stdoutSileSiesta.completed is deprecated in favor of stdoutSileSiesta.info.completed",
        "0.15",
        "0.17",
    )
    def completed(self):
        """True if the full file has been read and "Job completed" was found."""
        return self.info.completed()

    @lru_cache(1)
    @sile_fh_open(True)
    def read_basis(self) -> Atoms:
        """Reads the basis as found in the output file

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
        while "Species number:" in line:
            ls = line.split()
            if ls[3] == "Atomic":
                atoms[ls[7]] = {"Z": int(ls[5]), "tag": ls[7]}
                order.append(ls[7])
            else:
                atoms[ls[4]] = {"Z": int(ls[7]), "tag": ls[4]}
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
            found, line = self.step_to("<basis_specs>", allow_reread=False)

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

        return Atoms([Atom(**atoms[tag]) for tag in order])

    def _r_lattice_outcell(self):
        """Wrapper for reading the unit-cell from the outcoor block"""

        # Read until outcell is found
        found, line = self.step_to("outcell: Unit cell vectors")
        if not found:
            raise ValueError(
                f"{self.__class__.__name__}._r_lattice_outcell did not find outcell key"
            )

        Ang = "Ang" in line

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

        return Lattice(cell)

    def _r_geometry_outcoor(self, line, atoms=None):
        """Wrapper for reading the geometry as in the outcoor output"""
        atoms_order = _ensure_atoms(atoms)
        is_final = "Relaxed" in line or "Final (unrelaxed)" in line

        # Now we have outcoor
        scaled = "scaled" in line
        fractional = "fractional" in line
        Ang = "Ang" in line

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
        cell = self._r_lattice_outcell()
        if is_final and self.fh.tell() < pos:
            # we have wrapped around the file
            self.fh.seek(pos, os.SEEK_SET)
        xyz = _a.arrayd(xyz)

        # Now create the geometry
        if scaled:
            # The output file for siesta does not
            # contain the lattice constant.
            # So... :(
            raise ValueError(
                "Could not read the lattice-constant for the scaled geometry"
            )
        elif fractional:
            xyz = xyz.dot(cell.cell)
        elif not Ang:
            xyz *= Bohr2Ang

        return Geometry(xyz, atoms, lattice=cell)

    def _r_geometry_atomic(self, line, atoms=None):
        """Wrapper for reading the geometry as in the outcoor output"""
        atoms_order = _ensure_atoms(atoms)

        # Now we have outcoor
        Ang = "Ang" in line

        # Read in data
        xyz = []
        atoms = []
        line = self.readline()
        while len(line.strip()) > 0:
            line = line.split()
            xyz.append([float(x) for x in line[1:4]])
            atoms.append(atoms_order[int(line[4]) - 1])
            line = self.readline()

        # Retrieve the unit-cell (but do not skip file-descriptor position)
        # This is because the current unit-cell is not always written.
        pos = self.fh.tell()
        cell = self._r_lattice_outcell()
        self.fh.seek(pos, os.SEEK_SET)

        # Convert xyz
        xyz = _a.arrayd(xyz)
        if not Ang:
            xyz *= Bohr2Ang

        return Geometry(xyz, atoms, lattice=cell)

    @SileBinder()
    @sile_fh_open()
    def read_geometry(self, skip_input: bool = True) -> Geometry:
        """Reads the geometry from the Siesta output file

        Parameters
        ----------
        skip_input :
            the input geometry may be contained as a print-out.
            This is not part of an MD calculation, and hence is by
            default not returned.

        Returns
        -------
        geometries: list or Geometry or None
             if all is False only one geometry will be returned (or None). Otherwise
             a list of geometries corresponding to the MD-runs.
        """
        atoms = self.read_basis()

        def func(*args, **kwargs):
            """Wrapper to return None"""
            return None

        while line := self.readline():
            if "outcoor" in line and "coordinates" in line:
                func = self._r_geometry_outcoor
                break
            elif "siesta: Atomic coordinates" in line and not skip_input:
                func = self._r_geometry_atomic
                break

        return func(line, atoms)

    @SileBinder(postprocess=postprocess_tuple(_a.arrayd))
    @sile_fh_open()
    def read_force(
        self,
        total: bool = False,
        max: bool = False,
        key: Literal["siesta", "ts"] = "siesta",
        skip_final: Optional[bool] = None,
    ):
        """Reads the forces from the Siesta output file

        Parameters
        ----------
        total:
            return the total forces instead of the atomic forces.
        max:
            whether only the maximum atomic force should be returned for each step.

            Setting it to `True` is equivalent to `max(outSile.read_force())` in case atomic forces
            are written in the output file (`WriteForces .true.` in the fdf file)

            Note that this is not the same as doing `max(outSile.read_force(total=True))` since
            the forces returned in that case are averages on each axis.
        key:
            Specifies the indicator string for the forces that are to be read.
            The function will look for a line containing ``f'{key}: Atomic forces'``
            to start reading forces.
        skip_final:
            the final output of the forces is duplicated when the *final*
            output is written.
            By default, this method will return the final forces, but
            only **if** no other forces are found.
            If forces from dynamics are found, then the final forces
            will not be returned, unless explicitly requested through this flag.

        Returns
        -------
        numpy.ndarray or None
            returns ``None`` if the forces are not found in the
            output, otherwise forces will be returned

            The shape of the array will be different depending on the type of forces requested:
                - atomic (default): (nMDsteps, nAtoms, 3)
                - total: (nMDsteps, 3)
                - max: (nMDsteps, )

            If `total` and `max` are both `True`, they are returned separately as a tuple: ``(total, max)``
        """
        # Read until forces are found
        found, line = self.step_to(f"{key}: Atomic forces", allow_reread=False)
        if not found:
            return None

        if skip_final is None:
            # If we have final forces, default to skip the
            # final forces when we have forces in the dynamics
            # sections.
            skip_final = self.info._has_forces_in_dynamics

        # Now read data
        if skip_final and self.info._in_final_analysis():
            # This is the final summary, we don't need to read it as it does not contain new information
            # and also it make break things since max forces are not written there
            return None

        F = []
        # First, we encounter the atomic forces
        while line := self.readline():
            if "---" in line:
                break
            line = line.split()
            if not (total or max):
                F.append([float(x) for x in line[-3:]])

        if not F:
            F = None

        # Parse total forces if requested
        line = self.readline()
        if total and (line := line.split()):
            F = _a.arrayd([float(x) for x in line[-3:]])

        # And after that we can read the max force
        line = self.readline()
        if max and (line := self.readline().split()):
            maxF = _a.arrayd(float(line[1]))

            # In case total is also requested, we are going to
            # store it all in the same variable.
            # It will be separated later
            if total:
                # create tuple
                F = (F, maxF)
            else:
                F = maxF

        return F

    @SileBinder(postprocess=_a.arrayd)
    @sile_fh_open()
    def read_stress(
        self,
        key: Literal["static", "total", "Voigt"] = "static",
        skip_final: Optional[bool] = None,
    ) -> np.ndarray:
        """Reads the stresses from the Siesta output file

        Parameters
        ----------
        key :
           which stress to read from the output.
        skip_final:
            the final output of the stress is duplicated when the *final*
            output is written.
            By default, this method will return the final stress, but
            only **if** no other stresses are found.
            If stresses from dynamics are found, then the final stress
            will not be returned, unless explicitly requested through this flag.

        Returns
        -------
        numpy.ndarray or None
            returns ``None`` if the stresses are not found in the
            output, otherwise stresses will be returned
        """

        is_voigt = key.lower() == "voigt"
        if is_voigt:
            key = "Voigt"
            search = "Stress tensor Voigt"
        else:
            search = "siesta: Stress tensor"

        found = False
        line = " "
        while not found and line != "":
            found, line = self.step_to(search, allow_reread=False)
            found = found and key in line

        if skip_final is None:
            # If we have final stress, default to skip the
            # final stress when we have stress in the dynamics
            # sections.
            skip_final = self.info._has_stress_in_dynamics

        if skip_final and self.info._in_final_analysis():
            # we are in the final section, and don't want to do
            # anything
            return None

        if not found:
            return None

        # Now read data
        if is_voigt:
            Svoigt = _a.arrayd([float(x) for x in line.split()[-6:]])
            S = voigt_matrix(Svoigt, False)
        else:
            S = []
            for _ in range(3):
                line = self.readline().split()
                S.append([float(x) for x in line[-3:]])
            S = _a.arrayd(S)

        return S

    @SileBinder(postprocess=_a.arrayd)
    @sile_fh_open()
    def read_moment(
        self, orbitals: bool = False, quantity: Literal["S", "L"] = "S"
    ) -> np.ndarray:
        """Reads the moments from the Siesta output file

        These will only be present in case of spin-orbit coupling.

        Parameters
        ----------
        orbitals:
           return a table with orbitally resolved
           moments.
        quantity:
           return the spin-moments or the L moments
        """
        # Read until outcoor is found
        if not self.step_to("moments: Atomic", allow_reread=False)[0]:
            return None

        # The moments are printed in SPECIES list
        itt = iter(self)
        next(itt)  # empty
        next(itt)  # empty

        na = 0
        # Loop the species
        tbl = []
        # Read the species label
        while True:
            next(itt)  # ""
            next(itt)  # Atom    Orb ...

            # Loop atoms in this species list
            while True:
                line = next(itt)
                if line.startswith("Species") or line.startswith("--"):
                    break
                line = " "
                atom = []
                ia = 0
                while not line.startswith("--"):
                    line = next(itt).split()
                    if ia == 0:
                        ia = int(line[0])
                    elif ia != int(line[0]):
                        raise ValueError("Error in moments formatting.")

                    # Track maximum number of atoms
                    na = max(ia, na)
                    if quantity == "S":
                        atom.append([float(x) for x in line[4:7]])
                    elif quantity == "L":
                        atom.append([float(x) for x in line[7:10]])

                line = next(itt).split()  # Total ...
                if not orbitals:
                    ia = int(line[0])
                    if quantity == "S":
                        atom.append([float(x) for x in line[4:7]])
                    elif quantity == "L":
                        atom.append([float(x) for x in line[8:11]])
                tbl.append((ia, atom))
            if line.startswith("--"):
                break

        # Sort according to the atomic index
        moments = [] * na

        # Insert in the correct atomic
        for ia, atom in tbl:
            moments[ia - 1] = atom

        return _a.arrayd(moments)

    @sile_fh_open(True)
    def read_energy(self) -> PropertyDict:
        """Reads the final energy distribution

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
        ``basis.enthalpy``
             enthalpy of basis sets, Free + p_basis*V_orbitals


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
        found = self.step_to("siesta: Final energy", allow_reread=False)[0]
        out = PropertyDict()

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
            "(Free)E+ p_basis*V_orbitals": "basis.enthalpy",
            "(Free)E + p_basis*V_orbitals": "basis.enthalpy",  # we may correct the missing space
        }

        def assign(out, key, val):
            key = name_conv.get(key, key)
            try:
                val = float(val)
            except ValueError:
                warn(
                    f"Could not convert energy '{key}' ({val}) to a float, assigning nan."
                )
                val = np.nan

            if "." in key:
                loc, key = key.split(".")
                if not hasattr(out, loc):
                    out[loc] = PropertyDict()
                loc = out[loc]
            else:
                loc = out
            loc[key] = val

        while len(line.strip()) > 0:
            key, val = line.split("=")
            key = key.split(":")[1].strip()
            assign(out, key, val)
            line = next(itt)

        # now skip to the pressure
        found, line = self.step_to(
            ["(Free)E + p_basis*V_orbitals", "(Free)E+ p_basis*V_orbitals"],
            allow_reread=False,
        )
        if found:
            key, val = line.split("=")
            assign(out, key.strip(), val)

        return out

    @SileBinder()
    @sile_fh_open()
    def read_brillouinzone(self, trs: bool = True) -> MonkhorstPack:
        r"""Parses the k-grid section if present, otherwise returns the
        :math:`\Gamma`-point"""

        # store position, read geometry, then reposition file.
        tell = self.fh.tell()
        geom = self.read_geometry()
        self.fh.seek(tell)

        found, line = self.step_to(
            "siesta: k-grid: Number of k-points", allow_reread=False
        )

        if not found:
            return MonkhorstPack(geom, 1, trs=trs)

        nk = int(line.split("=")[-1])

        # default kcell and kdispl
        kcell = np.diag([1, 1, 1])
        kdispl = np.zeros(3)

        # if next line also contains 'siesta: k-grid:' then we can step_to
        line = self.readline()
        if line.startswith("siesta: k-grid:"):
            if self.step_to(
                "siesta: k-grid: Supercell and displacements", allow_reread=False
            )[0]:
                for i in range(3):
                    line = self.readline().split()
                    kdispl[i] = float(line[-1])
                    kcell[i, 2] = int(line[-2])
                    kcell[i, 1] = int(line[-3])
                    kcell[i, 0] = int(line[-4])

        return MonkhorstPack(geom, kcell, displacement=kdispl, trs=trs)

    def read_data(self, *args, **kwargs) -> Any:
        """Read specific content in the Siesta out file

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
            if kw in ("geometry", "force", "moment", "stress", "energy"):
                if kwargs[kw]:
                    run.append(kw)

        # Clean running names
        for name in run:
            kwargs.pop(name)

        slice = kwargs.pop("slice", None)
        val = []
        for name in run:
            if slice is None:
                val.append(getattr(self, f"read_{name.lower()}")(*args, **kwargs))
            else:
                val.append(
                    getattr(self, f"read_{name.lower()}")[slice](*args, **kwargs)
                )

        if len(val) == 0:
            return None
        elif len(val) == 1:
            val = val[0]
        return val

    @SileBinder(
        default_slice=-1, check_empty=_read_scf_empty, postprocess=_read_scf_md_process
    )
    @sile_fh_open()
    def read_scf(
        self,
        key: Literal["scf", "ts-scf"] = "scf",
        iscf: Optional[Union[int, Ellipsis]] = -1,
        as_dataframe: bool = False,
        ret_header: bool = False,
    ):
        r"""Parse SCF information and return a table of SCF information depending on what is requested

        Parameters
        ----------
        key :
            parse SCF information from Siesta SCF or TranSiesta SCF
        iscf :
            which SCF cycle should be stored. If ``-1`` only the final SCF step is stored,
            for `...`/`None` *all* SCF cycles are returned. When `iscf` values queried are not found they
            will be truncated to the nearest SCF step.
        as_dataframe:
            whether the information should be returned as a `pandas.DataFrame`. The advantage of this
            format is that everything is indexed and therefore you know what each value means.You can also
            perform operations very easily on a dataframe.
        ret_header:
            whether to also return the headers that define each value in the returned array,
            will have no effect if `as_dataframe` is true.
        """

        # These are the properties that are written in SIESTA scf
        if iscf is None:
            iscf = Ellipsis
        elif iscf is not Ellipsis:
            if iscf == 0:
                raise ValueError(
                    f"{self.__class__.__name__}.read_scf requires iscf argument to *not* be 0!"
                )

        def reset_d(d, line):
            """Determine if the SCF is done, and signal what it found."""
            if line.startswith("SCF cycle converged") or line.startswith(
                "SCF_NOT_CONV"
            ):
                if d["_toggle"]:  # means it did find some data
                    d["_final_iscf"] = 1

            elif line.startswith("SCF cycle continued"):
                d["_final_iscf"] = 0

        def construct_data(d):
            """Concatenate the data from the `order` and keys"""
            try:
                data = []
                for key in d["order"]:
                    data.extend(d[key])
                    del d[key]
                return data
            except KeyError:
                return None

        def do_finalize(d, key):
            """Check whether we are ready to return data (a similar key is found)."""
            if d["_toggle"].toggle(key):
                return construct_data(d)
            return None

        def add_order(d, key, prop=None):
            if prop is None:
                prop = key

            if key not in d["order"]:
                d["order"].append(key)
                if isinstance(prop, str):
                    d["props"].append(prop)
                else:
                    d["props"].extend(prop)

        def p_scf_fix_spin(line):
            assert len(line) == 97
            data = [
                int(line[5:9]),
                float(line[9:25]),
                float(line[25:41]),
                float(line[41:57]),
                float(line[57:67]),
                float(line[67:77]),
                float(line[77:87]),
                float(line[87:97]),
            ]
            return data

        def p_scf(line):
            assert len(line) == 87
            data = [
                int(line[5:9]),
                float(line[9:25]),
                float(line[25:41]),
                float(line[41:57]),
                float(line[57:67]),
                float(line[67:77]),
                float(line[77:87]),
            ]
            return data

        def p_ts_scf_fix_spin(line):
            assert len(line) == 100
            data = [
                int(line[8:12]),
                float(line[12:28]),
                float(line[28:44]),
                float(line[44:60]),
                float(line[60:70]),
                float(line[70:80]),
                float(line[80:90]),
                float(line[90:100]),
            ]
            return data

        def p_ts_scf(line):
            assert len(line) == 90
            data = [
                int(line[8:12]),
                float(line[12:28]),
                float(line[28:44]),
                float(line[44:60]),
                float(line[60:70]),
                float(line[70:80]),
                float(line[80:90]),
            ]
            return data

        def p_default(line):
            # Populate DATA by splitting
            data = line.split()
            data = [int(data[1])] + list(map(float, data[2:]))
            return data

        # For getting the parser according to len(line)
        scf_parser = {}
        scf_parser[87] = p_scf
        scf_parser[97] = p_scf_fix_spin
        scf_parser[90] = p_ts_scf
        scf_parser[100] = p_ts_scf_fix_spin

        def parse_next(line, d):
            nonlocal key, scf_parser, p_default
            line = line.strip().replace("*", "0")
            reset_d(d, line)
            ret = None

            # Start parsing based on keys
            if line.startswith("ts-Vha:"):
                ret = do_finalize(d, "ts-Vha")
                d["ts-Vha"] = [float(line.split()[1])]
                add_order(d, "ts-Vha")
            elif line.startswith("spin moment: S"):
                ret = do_finalize(d, "S")
                # 4.1 and earlier
                d["S"] = list(map(float, line.split("=")[1].split()[1:]))
                add_order(d, "S", ["Sx", "Sy", "Sz"])
            elif line.startswith("spin moment: {S}"):
                ret = do_finalize(d, "S")
                # 4.2 and later
                d["S"] = list(map(float, line.split("= {")[1].split()[:3]))
                add_order(d, "S", ["Sx", "Sy", "Sz"])
            elif line.startswith("bulk-bias: |v"):
                ret = do_finalize(d, "bb-v")
                # TODO old version should be removed once released
                d["bb-v"] = list(map(float, line.split()[-3:]))
                add_order(d, "bb-v", ["BB-vx", "BB-vy", "BB-vz"])
            elif line.startswith("bulk-bias: {v}"):
                idx = line.index("{v}")
                if line[idx + 3] == "_":
                    # we are in a subset
                    lbl = f"BB-{line[idx + 4:idx + 6]}"
                else:
                    lbl = "BB"

                v = line.split("] {")[1].split()
                v = list(map(float, v[:3]))
                ret = do_finalize(d, lbl)
                d[lbl] = v
                add_order(d, lbl, [f"{lbl}-vx", f"{lbl}-vy", f"{lbl}-vz"])
            elif line.startswith("bulk-bias: dq"):
                ret = do_finalize(d, "BB-q")
                d["BB-q"] = list(map(float, line.split()[-2:]))
                add_order(d, "BB-q", ["BB-dq", "BB-q0"])
            elif line.startswith("ts-q:"):
                ret = do_finalize(d, "ts-q")
                # even for key == scf, this won't be there.
                # So it will just fail...
                data = line.split()[1:]
                try:
                    d["ts-q"] = list(map(float, data))
                except Exception:
                    # We are probably reading a device list
                    add_order(d, "ts-q", data)

            elif line.startswith(key):
                ret = do_finalize(d, "scf")
                parser = scf_parser.get(len(line), p_default)
                d["scf"] = parser(line)
                # we don't need to add_order here, because it starts with scf

            return ret

        # A temporary dictionary to hold information while reading the output file
        d = {
            "_final_iscf": 0,
            "_toggle": Toggler(),
            # The properties defaults to these keys
            # TODO fix for fix-spin calculations (Efup/Efdown)
            "props": ["iscf", "Eharris", "E_KS", "FreeEng", "dDmax", "Ef", "dHmax"],
            # Default the order, start with SCF
            "order": ["scf"],
        }

        # Jump to the `iscf` which is uniquely found for start of SCF cycles.
        _, _ = self.step_to("iscf", allow_reread=False)
        scf = []
        while line := self.readline():
            data = parse_next(line, d)
            if data is None and d["_final_iscf"] == 2:
                # Catch the case where this is the final
                # case, so we can add the data to the list of SCF segments
                data = construct_data(d)

            if data is not None:
                # we have found a new key
                if iscf is Ellipsis or iscf < 0:
                    scf.append(data)

                elif data[0] <= iscf:
                    # this ensures we will retain the latest iscf in
                    # case the requested iscf is too big
                    scf = data

            if d["_final_iscf"] == 1:
                # step to next line!
                d["_final_iscf"] = 2

            elif d["_final_iscf"] == 2:
                # The line after SCF cycle converged
                # Reset, for MD reads
                d["_final_iscf"] = 0

                if len(scf) == 0:
                    # this traps cases where final_iscf has
                    # been trickered but we haven't collected anything.
                    # I.e. if key == scf but ts-scf also exists.
                    continue

                # First figure out which iscf we should store
                if iscf is Ellipsis:  # or iscf > 0
                    # scf is correct
                    pass
                elif iscf < 0:
                    # truncate to 0
                    scf = scf[max(len(scf) + iscf, 0)]

                # found a full MD
                break

        # Define the function that is going to convert the information of a MDstep to a Dataset
        if as_dataframe:
            import pandas as pd

            # Easier
            props = d["props"][1:]

            if len(scf) == 0:
                return pd.DataFrame(index=pd.Index([], name="iscf"), columns=props)

            scf = np.atleast_2d(scf)
            return pd.DataFrame(
                scf[..., 1:],
                index=pd.Index(scf[..., 0].ravel().astype(np.int32), name="iscf"),
                columns=props,
            )

        # Convert to numpy array
        scf = np.array(scf)
        if ret_header:
            return scf, d["props"]
        return scf

    @sile_fh_open(True)
    def read_charge(
        self,
        name: Literal["voronoi", "hirshfeld", "mulliken", "mulliken:<5.2"],
        iscf: Union[Opt, int, Ellipsis] = Opt.ANY,
        imd: Union[Opt, int, Ellipsis] = Opt.ANY,
        key_scf: str = "scf",
        as_dataframe: bool = False,
    ):
        r"""Read charges calculated in SCF loop or MD loop (or both)

        Siesta enables many different modes of writing out charges.

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
        | Mulliken  | +   | +   | +      | +     | -                |
        |   (>=5.2) |     |     |        |       |                  |
        +-----------+-----+-----+--------+-------+------------------+
        | Mulliken  | +   | +   | +      | +     | (+)              |
        |     <5.2  |     |     |        |       |                  |
        +-----------+-----+-----+--------+-------+------------------+

        Notes
        -----
        Errors will be raised if one requests information not present. I.e.
        passing an integer or `Opt.ALL` for `iscf` will raise an error if
        the SCF charges are not present. For `Opt.ANY` it will return
        the most information, effectively SCF will be returned if present.

        Currently orbitally-resolved Mulliken is not implemented, any help in
        reading this would be very welcome.

        Parameters
        ----------
        name:
            the name of the charges that you want to read
        iscf:
            index (0-based) of the scf iteration you want the charges for.
            If the enum specifier `Opt.ANY` or `Opt.ALL`/`...` are used, then
            the returned quantities depend on what is present.
            If ``None/Opt.NONE`` it will not return any SCF charges.
            If both `imd` and `iscf` are ``None`` then only the final charges will be returned.
        imd:
            index (0-based) of the md step you want the charges for.
            If the enum specifier `Opt.ANY` or `Opt.ALL`/`...` are used, then
            the returned quantities depend on what is present.
            If ``None/Opt.NONE`` it will not return any MD charges.
            If both `imd` and `iscf` are ``None`` then only the final charges will be returned.
        key_scf :
            the key lookup for the scf iterations (a ":" will automatically be appended)
        as_dataframe:
            whether charges should be returned as a pandas dataframe.

        Returns
        -------
        numpy.ndarray
            if a specific MD+SCF index is requested (or special cases where output is
            not complete)
        list of numpy.ndarray
            if `iscf` or `imd` is different from ``None/Opt.NONE``.
        pandas.DataFrame
            if `as_dataframe` is requested. The dataframe will have multi-indices if multiple
            SCF or MD steps are requested.
        """
        namel = name.lower()
        if as_dataframe:
            import pandas as pd

            def _empty_charge():
                # build a fake dataframe with no indices
                return pd.DataFrame(
                    index=pd.Index([], name="atom", dtype=np.int32), dtype=np.float32
                )

        else:
            pd = None

            def _empty_charge():
                # return for single value with nan values
                return _a.arrayf([[None]])

        # define helper function for reading voronoi+hirshfeld charges
        def _charges():
            """Read output from Voronoi/Hirshfeld/Mulliken charges"""
            nonlocal pd

            # Expecting something like this (NC/SOC)
            # Voronoi Atomic Populations:
            # Atom #     dQatom  Atom pop         S        Sx        Sy        Sz  Species
            #      1   -0.02936   4.02936   0.00000  -0.00000   0.00000   0.00000  C
            # or (polarized)
            # Voronoi Atomic Populations:
            # Atom #     dQatom  Atom pop        Sz  Species
            #      1   -0.02936   4.02936   0.00000  C
            # In 5.2 this now looks like this:
            # Voronoi Atomic Populations:
            # Atom #  charge [q]  valence [e]        Sz  Species
            #      1    -0.02936      4.02936   0.00000  C

            # first line is the header
            header = (
                self.readline()
                .replace("charge [q]", "dq")  # charge [q] in 5.2
                .replace("valence [e]", "e")  # in 5.2
                .replace("dQatom", "dq")  # dQatom in 5.0
                .replace(" Qatom", " dq")  # Qatom in 4.1
                .replace("Atom pop", "e")  # not found in 4.1
                .split()
            )[2:-1]

            # Define the function that parses the charges
            def _parse_charge(line):
                atom_idx, *vals, symbol = line.split()
                # assert that this is a proper line
                # this should catch cases where the following line of charge output
                # is still parseable
                # atom_idx = int(atom_idx)
                return list(map(float, vals))

            # We have found the header, prepare a list to read the charges
            atom_charges = []
            while (line := self.readline()) != "":
                try:
                    charge_vals = _parse_charge(line)
                    atom_charges.append(charge_vals)
                except Exception:
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
            return pd.DataFrame(
                atom_charges,
                columns=header,
                dtype=np.float32,
                index=pd.RangeIndex(stop=len(atom_charges), name="atom"),
            )

        # define helper function for reading voronoi+hirshfeld charges
        def _mulliken_charges_pre52():
            """Read output from Mulliken charges"""
            nonlocal pd

            # Expecting something like this (NC/SOC)
            # mulliken: Atomic and Orbital Populations:

            # Species: Cl

            # Atom      Orb        Charge      Spin       Svec
            # ----------------------------------------------------------------
            #     1  1 3s         1.75133   0.00566      0.004   0.004  -0.000
            #     1  2 3s         0.09813   0.00658     -0.005  -0.005   0.000
            #     1  3 3py        1.69790   0.21531     -0.161  -0.142   0.018
            #     1  4 3pz        1.72632   0.15770     -0.086  -0.132  -0.008
            #     1  5 3px        1.81369   0.01618     -0.004   0.015  -0.004
            #     1  6 3py       -0.04663   0.02356     -0.017  -0.016  -0.000
            #     1  7 3pz       -0.04167   0.01560     -0.011  -0.011   0.001
            #     1  8 3px       -0.02977   0.00920     -0.006  -0.007   0.000
            #     1  9 3Pdxy      0.00595   0.00054     -0.000  -0.000  -0.000
            #     1 10 3Pdyz      0.00483   0.00073     -0.001  -0.001  -0.000
            #     1 11 3Pdz2      0.00515   0.00098     -0.001  -0.001  -0.000
            #     1 12 3Pdxz      0.00604   0.00039     -0.000  -0.000   0.000
            #     1 13 3Pdx2-y2   0.00607   0.00099     -0.001  -0.001   0.000
            #     1     Total     6.99733   0.41305     -0.288  -0.296   0.007

            # Define the function that parses the charges
            def _parse_charge_total_nc(line):  # non-colinear and soc
                atom_idx, _, *vals = line.split()
                # assert that this is a proper line
                # this should catch cases where the following line of charge output
                # is still parseable
                # atom_idx = int(atom_idx)
                return int(atom_idx), list(map(float, vals))

            def _parse_charge_total(line):  # unpolarized and colinear spin
                atom_idx, val, *_ = line.split()
                return int(atom_idx), float(val)

            # Define the function that parses a single species
            def _parse_species_nc():  # non-colinear and soc
                nonlocal header, atom_idx, atom_charges

                # The mulliken charges are organized per species where the charges
                # for each species are enclosed by dashes (-----)

                # Step to header
                _, line = self.step_to("Atom", allow_reread=False)
                if header is None:
                    header = (
                        line.replace("Charge", "e")
                        .replace("Spin", "S")
                        .replace(
                            "Svec", "Sx Sy Sz"
                        )  # Split Svec into Cartesian components
                        .split()
                    )[2:]

                # Skip over the starting ---- line
                self.readline()

                # Read until closing ---- line
                while "----" not in (line := self.readline()):
                    if "Total" in line:
                        ia, charge_vals = _parse_charge_total_nc(line)
                        atom_idx.append(ia)
                        atom_charges.append(charge_vals)

            def _parse_spin_pol():  # unpolarized and colinear spin
                nonlocal atom_idx, atom_charges

                # The mulliken charges are organized per spin
                # polarization (UP/DOWN). The end of a spin
                # block is marked by a Qtot

                # Read until we encounter "mulliken: Qtot"
                def try_parse_int(s):
                    try:
                        int(s)
                    except ValueError:
                        return False
                    else:
                        return True

                while "mulliken: Qtot" not in (line := self.readline()):
                    words = line.split()
                    if len(words) > 0 and try_parse_int(words[0]):
                        # This should be a line containing the total charge for an atom
                        ia, charge = _parse_charge_total(line)
                        atom_idx.append(ia)
                        atom_charges.append([charge])

            # Determine with which spin type we are dealing
            if self.info.spin.is_unpolarized:  # UNPOLARIZED
                # No spin components so just parse charge
                atom_charges = []
                atom_idx = []
                header = ["e"]
                _parse_spin_pol()

            elif self.info.spin.is_polarized:
                # Parse both spin polarizations
                atom_charges_pol = []
                header = ["e", "Sz"]
                for s in ("UP", "DOWN"):
                    atom_charges = []
                    atom_idx = []
                    self.step_to(f"mulliken: Spin {s}", allow_reread=False)
                    _parse_spin_pol()
                    atom_charges_pol.append(atom_charges)

                # Compute the charge and spin of each atom
                atom_charges_pol_array = _a.arrayf(atom_charges_pol)
                atom_q = (
                    atom_charges_pol_array[0, :, 0] + atom_charges_pol_array[1, :, 0]
                )
                atom_s = (
                    atom_charges_pol_array[0, :, 0] - atom_charges_pol_array[1, :, 0]
                )
                atom_charges[:] = np.stack((atom_q, atom_s), axis=-1)

            elif not self.info.spin.is_diagonal:
                # Parse each species
                atom_charges = []
                atom_idx = []
                header = None
                for _ in range(self.info.nspecies):
                    found, _ = self.step_to("Species:", allow_reread=False)
                    _parse_species_nc()

            else:
                raise NotImplementedError(
                    "Something went wrong... Couldn't parse file."
                )

            # Convert to array and sort in the order of the atoms
            sort_idx = np.argsort(atom_idx)
            atom_charges_array = _a.arrayf(atom_charges)[sort_idx]

            if pd is None:
                # not as_dataframe
                return atom_charges_array

            # determine how many columns we have
            # this will remove atom indices and species, so only inside
            ncols = len(atom_charges[0])
            assert ncols == len(header)

            # the precision is limited, so no need for double precision
            return pd.DataFrame(
                atom_charges_array,
                columns=header,
                dtype=np.float32,
                index=pd.RangeIndex(stop=len(atom_charges), name="atom"),
            )

        # split method to retrieve options
        namel, *opts = namel.split(":")

        # Check that a known charge has been requested
        if namel == "voronoi":
            _r_charge = _charges
            charge_keys = [
                "Voronoi Atomic Populations",
                "Voronoi Net Atomic Populations",
            ]
        elif namel == "hirshfeld":
            _r_charge = _charges
            charge_keys = [
                "Hirshfeld Atomic Populations",
                "Hirshfeld Net Atomic Populations",
            ]
        elif namel == "mulliken":
            _r_charge = _mulliken_charges_pre52
            charge_keys = ["mulliken: Atomic and Orbital Populations"]
            if "<5.2" in opts:
                pass

            else:
                # check if version is understandable
                version = self.info.version
                try:
                    if version.version[:2] >= (5, 2):
                        _r_charge = _charges
                        charge_keys = [
                            "Mulliken Atomic Populations",
                            "Mulliken Net Atomic Populations",
                        ]
                except Exception:
                    pass

        else:
            raise ValueError(
                f"{self.__class__.__name__}.read_charge name argument should be one of [voronoi, hirshfeld, mulliken], got {name}?"
            )

        # Ensure the key_scf matches exactly (prepend a space)
        key_scf = f" {key_scf.strip()}:"

        # Reading charges may be quite time consuming for large MD simulations.

        # to see if we finished a MD read, we check for these keys
        search_keys = [
            # two keys can signal ending SCF
            "SCF Convergence",
            "SCF_NOT_CONV",
            "siesta: Final energy",
            key_scf,
            *charge_keys,
        ]

        # adjust the below while loop to take into account any additional
        # segments of search_keys
        IDX_SCF_END = [0, 1]
        IDX_FINAL = [2]
        IDX_SCF = [3]
        # the rest are charge keys
        IDX_CHARGE = list(range(len(search_keys) - len(charge_keys), len(search_keys)))

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

        while (
            ret := self.step_to(
                search_keys, case=True, ret_index=True, allow_reread=False
            )
        )[0]:
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
                md_scf_charge = pd.concat(
                    [
                        pd.concat(
                            iscf_, keys=pd.RangeIndex(1, len(iscf_) + 1, name="iscf")
                        )
                        for iscf_ in md_scf_charge
                    ],
                    keys=pd.RangeIndex(1, len(md_scf_charge) + 1, name="imd"),
                )
            if FOUND_MD:
                md_charge = pd.concat(
                    md_charge, keys=pd.RangeIndex(1, len(md_charge) + 1, name="imd")
                )
        else:
            if FOUND_SCF:
                nan_array = _a.emptyf(md_scf_charge[0][0].shape)
                nan_array.fill(np.nan)

                def get_md_scf_charge(scf_charge, iscf):
                    try:
                        return scf_charge[iscf]
                    except Exception:
                        return nan_array

            if FOUND_MD:
                md_charge = np.stack(md_charge)

        # option parsing is a bit *difficult* with flag enums
        # So first figure out what is there, and handle this based
        # on arguments
        def _p(flag, found):
            """Helper routine to do the following:

            Returns
            -------
            is_opt : bool
                whether the flag is an `Opt`
            flag :
                corrected flag
            """
            if flag is Ellipsis:
                flag = Opt.ALL

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
                else:  # not found
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
            if (opt_iscf or (not iscf is None)) or (opt_imd or (not imd is None)):
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
                return np.stack(
                    tuple(get_md_scf_charge(x, iscf) for x in md_scf_charge)
                )

            elif FOUND_MD and iscf is None:
                return md_charge
            raise SileError(
                f"{str(self)} unknown argument for 'imd' and 'iscf', could not find SCF charges"
            )

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


outSileSiesta = deprecation(
    "outSileSiesta has been deprecated in favor of stdoutSileSiesta.", "0.15", "0.17"
)(stdoutSileSiesta)

add_sile("siesta.out", stdoutSileSiesta, case=False, gzip=True)
add_sile("out", stdoutSileSiesta, case=False, gzip=True)
