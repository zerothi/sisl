# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl._array import arrayd, arrayi, asarrayi
from sisl._core import Atom, AtomicOrbital, Atoms, Geometry, PeriodicTable
from sisl._help import xml_parse
from sisl._internal import set_module
from sisl.messages import SislWarning, warn
from sisl.unit.siesta import unit_convert
from sisl.utils import (
    collect_action,
    default_ArgumentParser,
    default_namespace,
    direction,
    lstranges,
    run_actions,
    strmap,
)

from ..sile import add_sile, get_sile, sile_fh_open
from .sile import SileSiesta

__all__ = ["pdosSileSiesta"]

Bohr2Ang = unit_convert("Bohr", "Ang")


@set_module("sisl.io.siesta")
class pdosSileSiesta(SileSiesta):
    """Projected DOS file with orbital information

    Data file containing the PDOS as calculated by Siesta.
    """

    def read_geometry(self) -> Geometry:
        """Read the geometry with coordinates and correct orbital counts"""
        return self.read_data()[0]

    @sile_fh_open(True)
    def read_fermi_level(self) -> float:
        """Returns the fermi-level"""
        # Get the element-tree
        root = xml_parse(self.fh).getroot()

        # Try and find the fermi-level
        Ef = root.find("fermi_energy")
        if Ef is None:
            warn(f"{self!s}.read_data could not locate the Fermi-level in the XML tree")
        return Ef

    @sile_fh_open(True)
    def read_data(self, as_dataarray: bool = False):
        r"""Returns data associated with the PDOS file

        For spin-polarized calculations the returned values are up/down, orbitals, energy.
        For non-collinear calculations the returned values are sum/x/y/z, orbitals, energy.

        Parameters
        ----------
        as_dataarray: bool, optional
           If True the returned PDOS is a `xarray.DataArray` with energy, spin
           and orbital information as coordinates in the data.
           The geometry, unit and Fermi level are stored as attributes in the
           DataArray.

        Returns
        -------
        geom : Geometry
            instance with positions, atoms and orbitals.
        E : numpy.ndarray
            the energies at which the PDOS has been evaluated at (if Fermi-level present in file energies are shifted to :math:`E - E_F = 0`).
        PDOS : numpy.ndarray
            an array of DOS with dimensions ``(nspin, geom.no, len(E))`` (with different spin-components) or ``(geom.no, len(E))`` (spin-symmetric).
        all : xarray.DataArray
            if `as_dataarray` is True, only this data array is returned, in this case all data can be post-processed using the `xarray` selection routines.
        """
        # Get the element-tree
        root = xml_parse(self.fh).getroot()

        # Get number of orbitals
        nspin = int(root.find("nspin").text)
        # Try and find the fermi-level
        Ef = root.find("fermi_energy")
        E = arrayd(root.find("energy_values").text.split())
        if Ef is None:
            warn(
                f"{self!s}.read_data could not locate the Fermi-level in the XML tree, using E_F = 0. eV"
            )
        else:
            Ef = float(Ef.text)
            E -= Ef
        ne = len(E)

        # All coordinate, atoms and species data
        xyz = []
        atoms = []
        atom_species = []

        def ensure_size(ia):
            while len(atom_species) <= ia:
                atom_species.append(None)
                xyz.append(None)

        def ensure_size_orb(ia, i):
            while len(atoms) <= ia:
                atoms.append([])
            while len(atoms[ia]) <= i:
                atoms[ia].append(None)

        if nspin == 4:

            def process(D):
                tmp = np.empty(D.shape[0], D.dtype)
                tmp[:] = D[:, 3]
                D[:, 3] = D[:, 0] - D[:, 1]
                D[:, 0] = D[:, 0] + D[:, 1]
                D[:, 1] = D[:, 2]
                D[:, 2] = tmp[:]
                return D

        elif nspin == 2:

            def process(D):
                tmp = D[:, 0] + D[:, 1]
                D[:, 1] = D[:, 0] - D[:, 1]
                D[:, 0] = tmp
                return D

        else:

            def process(D):
                return D

        if as_dataarray:
            import xarray as xr

            if nspin == 1:
                spin = ["sum"]
            elif nspin == 2:
                spin = ["sum", "z"]
            elif nspin == 4:
                spin = ["sum", "x", "y", "z"]

            # Dimensions of the PDOS data-array
            dims = ["E", "spin", "n", "l", "m", "zeta", "polarization"]

            shape = (ne, nspin, 1, 1, 1, 1, 1)

            def to(o, DOS):
                # Coordinates for this dataarray
                coords = [E, spin, [o.n], [o.l], [o.m], [o.zeta], [o.P]]

                return xr.DataArray(
                    data=process(DOS).reshape(shape),
                    dims=dims,
                    coords=coords,
                    name="PDOS",
                )

        else:

            def to(o, DOS):
                return process(DOS)

        D = []

        for orb in root.findall("orbital"):
            # Short-hand function to retrieve integers for the attributes
            def oi(name):
                return int(orb.get(name))

            # Get indices
            ia = oi("atom_index") - 1
            i = oi("index") - 1

            species = orb.get("species")

            # Create the atomic orbital
            try:
                Z = oi("Z")
            except Exception:
                try:
                    Z = PeriodicTable().Z(species)
                except Exception:
                    # Unknown
                    Z = -1

            try:
                P = orb.get("P") == "true"
            except Exception:
                P = False

            ensure_size(ia)
            xyz[ia] = arrayd(orb.get("position").split())
            atom_species[ia] = Z

            # Construct the atomic orbital
            O = AtomicOrbital(n=oi("n"), l=oi("l"), m=oi("m"), zeta=oi("z"), P=P)

            # We know that the index is far too high. However,
            # this ensures a consecutive orbital
            ensure_size_orb(ia, i)
            atoms[ia][i] = O

            # it is formed like : spin-1, spin-2 (however already in eV)
            DOS = arrayd(orb.find("data").text.split()).reshape(-1, nspin)

            D.append(to(O, DOS))

        # Now we need to parse the data
        # First reduce the atom
        atoms = [[o for o in a if o] for a in atoms]
        atoms = Atoms(map(Atom, atom_species, atoms))
        geom = Geometry(arrayd(xyz) * Bohr2Ang, atoms)

        if as_dataarray:
            # Create a new dimension without coordinates (orbital index)
            D = xr.concat(D, "orbital")
            # Add attributes
            D.attrs["geometry"] = geom
            D.attrs["unit"] = "1/eV"
            if Ef is None:
                D.attrs["Ef"] = "Unknown"
            else:
                D.attrs["Ef"] = Ef

            return D

        D = np.moveaxis(np.stack(D, axis=0), 2, 0)
        return geom, E, D

    @default_ArgumentParser(
        description="""
Extract/Plot data from a PDOS/PDOS.xml file

The arguments are parsed as they are passed to the command line; hence order is important.

Consider the following:

   --spin x --atom all --spin y --atom all --plot

This will plot the spin x and y components for all atoms, with no normalization.

   --norm orbital --atom all --spin x --atom 1 --plot

This will normalize the PDOS to the number of projected orbitals in each --atom argument.
The --atom all plots the total DOS (no spin direction), then the 2nd plotted line has only
the x-component of the first atom. Since the normalization in both cases are "orbital" one
can directly compare the values.

One can collect as many curves as necessary, for every --plot/--out argument the data will
be plotted/saved and all prior options will be reset. Hence

   --spin x --atom all --out spin_x_all.dat --spin y --atom all --out spin_y_all.dat

will store the spin x/y components of all atoms in spin_x_all.dat/spin_y_all.dat, respectively.
"""
    )
    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""

        # We limit the import to occur here
        import argparse
        import warnings

        comment = "Fermi-level shifted to 0"
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            geometry, E, PDOS = self.read_data()

            if len(w) > 0:
                if issubclass(w[-1].category, SislWarning):
                    comment = "Fermi-level unknown"

        def norm(geom, orbitals=None, norm="none"):
            r"""Normalization factor depending on the input

            The normalization can be performed in one of the below methods.
            In the following :math:`N` refers to the normalization constant
            that is to be used (i.e. the divisor):

            ``"none"``
               :math:`N=1`
            ``"all"``
               :math:`N` equals the number of orbitals in the total geometry
            ``"atom"``
               :math:`N` equals the total number of orbitals in the selected
               atoms. If `orbitals` is an argument a conversion of `orbitals` to the equivalent
               unique atoms is performed, and subsequently the total number of orbitals on the
               atoms is used. This makes it possible to compare the fraction of orbital DOS easier.
            ``"orbital"``
               :math:`N` is the sum of selected orbitals, if `atoms` is specified, this
               is equivalent to the "atom" option.

            Parameters
            ----------
            orbitals : array_like of int or bool, optional
               only return for a given set of orbitals (default to all)
            norm : {"none", "atom", "orbital", "all"}
               how the normalization of the summed DOS is performed (see `norm` routine)
            """
            # Cast to lower
            norm = norm.lower()
            if norm == "none":
                NORM = 1
            elif norm in ["all", "atom", "orbital"]:
                NORM = geom.no
            else:
                raise ValueError(
                    f"norm error on norm keyword in when requesting normalization!"
                )

            # If the user requests all orbitals
            if orbitals is None:
                return NORM

            # Now figure out what to do
            # Get pivoting indices to average over
            if norm == "orbital":
                NORM = len(orbitals)
            elif norm == "atom":
                a = np.unique(geom.o2a(orbitals))
                # Now sum the orbitals per atom
                NORM = geom.orbitals[a].sum()
            return NORM

        def _sum_filter(PDOS):
            """Default sum is the total DOS, no projection on directions"""
            if PDOS.ndim == 2:
                # non-polarized
                return PDOS
            elif PDOS.shape[0] == 2:
                # polarized
                return PDOS[0]
            return PDOS[0]

        namespace = default_namespace(
            _geometry=geometry,
            _E=E,
            _PDOS=PDOS,
            # The energy range of all data
            _Erng=None,
            _norm="none",
            _PDOS_filter_name="total",
            _PDOS_filter=_sum_filter,
            _data=[],
            _data_description=[],
            _data_header=[],
        )

        def ensure_E(func):
            """This decorater ensures that E is the first element in the _data container"""

            def assign_E(self, *args, **kwargs):
                ns = args[1]
                if len(ns._data) == 0:
                    # We immediately extract the energies
                    ns._data.append(ns._E[ns._Erng].flatten())
                    ns._data_header.append("Energy[eV]")
                return func(self, *args, **kwargs)

            return assign_E

        class ERange(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                E = ns._E
                Emap = strmap(float, value, E.min(), E.max())

                def Eindex(e):
                    return np.abs(E - e).argmin()

                # Convert to actual indices
                E = []
                for begin, end in Emap:
                    if begin is None and end is None:
                        ns._Erng = None
                        return
                    elif begin is None:
                        E.append(range(Eindex(end) + 1))
                    elif end is None:
                        E.append(range(Eindex(begin), len(E)))
                    else:
                        E.append(range(Eindex(begin), Eindex(end) + 1))
                # Issuing unique also sorts the entries
                ns._Erng = np.unique(arrayi(E).flatten())

        p.add_argument(
            "--energy",
            "-E",
            action=ERange,
            help="""Denote the sub-section of energies that are extracted: "-1:0,1:2" [eV]

                       This flag takes effect on all energy-resolved quantities and is reset whenever --plot or --out is called""",
        )

        # The normalization method
        class NormAction(argparse.Action):
            @collect_action
            def __call__(self, parser, ns, value, option_string=None):
                ns._norm = value

        p.add_argument(
            "--norm",
            "-N",
            action=NormAction,
            default="atom",
            choices=["none", "atom", "orbital", "all"],
            help="""Specify the normalization method; "none") no normalization, "atom") total orbitals in selected atoms,
                       "orbital") selected orbitals or "all") all orbitals.

                       Will only take effect on subsequent --atom ranges.

                       This flag is reset whenever --plot or --out is called""",
        )

        if PDOS.ndim == 2:
            # no spin is possible
            pass
        elif PDOS.shape[0] == 2:
            # Add a spin-action
            class Spin(argparse.Action):
                @collect_action
                def __call__(self, parser, ns, value, option_string=None):
                    value = value[0].lower()
                    if value in ("up", "u"):
                        name = "up"

                        def _filter(PDOS):
                            return (PDOS[0] + PDOS[1]) / 2

                    elif value in ("down", "dn", "dw", "d"):
                        name = "down"

                        def _filter(PDOS):
                            return (PDOS[0] - PDOS[1]) / 2

                    elif value in ("sum", "+", "total"):
                        name = "total"

                        def _filter(PDOS):
                            return PDOS[0]

                    elif value in ("z", "spin"):
                        name = "z"

                        def _filter(PDOS):
                            return PDOS[1]

                    else:
                        raise ValueError(
                            f"Wrong argument for --spin [up, down, sum, z], found {value}"
                        )
                    ns._PDOS_filter_name = name
                    ns._PDOS_filter = _filter

            p.add_argument(
                "--spin",
                "-S",
                action=Spin,
                nargs=1,
                help="Which spin-component to store, up/u, down/d, z/spin or sum/+/total",
            )

        elif PDOS.shape[0] == 4:
            # Add a spin-action
            class Spin(argparse.Action):
                @collect_action
                def __call__(self, parser, ns, value, option_string=None):
                    value = value[0].lower()
                    if value in ("sum", "+", "total"):
                        name = "total"

                        def _filter(PDOS):
                            return PDOS[0]

                    else:
                        # the stuff must be a range of directions
                        # so simply put it in
                        idx = list(map(lambda x: direction(x) + 1, value))
                        name = value

                        def _filter(PDOS):
                            return PDOS[idx].sum(0)

                    ns._PDOS_filter_name = name
                    ns._PDOS_filter = _filter

            p.add_argument(
                "--spin",
                "-S",
                action=Spin,
                nargs=1,
                help="Which spin-component to store, sum/+/total, x, y, z or a sum of either of the directions xy, zx etc.",
            )

        def parse_atom_range(geom, value):
            if value.lower() in ("all", ":"):
                return np.arange(geom.no), "all"

            value = ",".join(  # ensure only single commas (no space between them)
                "".join(  # ensure no empty whitespaces
                    ",".join(  # join different lines with a comma
                        value.splitlines()
                    ).split()
                ).split(",")
            )

            # Sadly many shell interpreters does not
            # allow simple [] because they are expansion tokens
            # in the shell.
            # We bypass this by allowing *, [, {
            # * will "only" fail if files are named accordingly, else
            # it will be passed as-is.
            #       {    [    *
            sep = ["c", "b", "*"]
            failed = True
            while failed and len(sep) > 0:
                try:
                    ranges = lstranges(strmap(int, value, 0, len(geom), sep.pop()))
                    failed = False
                except Exception:
                    pass
            if failed:
                print(value)
                raise ValueError("Could not parse the atomic/orbital ranges")

            # we have only a subset of the orbitals
            orbs = []
            no = 0
            for atoms in ranges:
                if isinstance(atoms, list):
                    # Get atoms and orbitals
                    ob = geom.a2o(atoms[0] - 1, True)
                    # We normalize for the total number of orbitals
                    # on the requested atoms.
                    # In this way the user can compare directly the DOS
                    # for same atoms with different sets of orbitals and the
                    # total will add up.
                    no += len(ob)
                    ob = ob[asarrayi(atoms[1]) - 1]
                else:
                    ob = geom.a2o(atoms - 1, True)
                    no += len(ob)
                orbs.append(ob)

            if len(orbs) == 0:
                print("Available atoms:")
                print(f"  1-{len(geometry)}")
                print("Input atoms:")
                print("  ", value)
                raise ValueError(
                    "Atomic/Orbital requests are not fully included in the device region."
                )

            # Add one to make the c-index equivalent to the f-index
            return np.concatenate(orbs).flatten(), value

        # Try and add the atomic specification
        class AtomRange(argparse.Action):
            @collect_action
            @ensure_E
            def __call__(self, parser, ns, value, option_string=None):
                # get which orbitals to extract
                orbs, value = parse_atom_range(ns._geometry, value)
                # calculate the normalization
                scale = norm(ns._geometry, orbs, ns._norm)
                # Calculate PDOS on the selected atoms with the norm
                ns._data.append(ns._PDOS_filter(ns._PDOS)[orbs].sum(0) / scale)
                index = len(ns._data)
                if value == "all":
                    DOS = "DOS"
                else:
                    DOS = "PDOS"

                if ns._PDOS_filter_name is not None:
                    ns._data_header.append(
                        f"{DOS}[spin={ns._PDOS_filter_name}:{value}][1/eV]"
                    )
                    ns._data_description.append(
                        f"Column {index} is the sum of spin={ns._PDOS_filter_name} on atoms[orbs] {value} with normalization 1/{scale}"
                    )
                else:
                    ns._data_header.append(f"{DOS}[{value}][1/eV]")
                    ns._data_description.append(
                        f"Column {index} is the total PDOS on atoms[orbs] {value} with normalization 1/{scale}"
                    )

        p.add_argument(
            "--atom",
            "-a",
            type=str,
            action=AtomRange,
            help="""Limit orbital resolved PDOS to a sub-set of atoms/orbitals: "1-2[3,4]" will yield the 1st and 2nd atom and their 3rd and fourth orbital. Multiple comma-separated specifications are allowed. Note that some shells does not allow [] as text-input (due to expansion), {, [ or * are allowed orbital delimiters.

Multiple options will create a new column/line in output, the --norm and --E should be before any of these arguments""",
        )

        class Out(argparse.Action):
            @run_actions
            def __call__(self, parser, ns, value, option_string=None):
                out = value[0]

                try:
                    # We figure out if the user wants to write
                    # to a geometry
                    obj = get_sile(out, mode="w")
                    if hasattr(obj, "write_geometry"):
                        with obj as fh:
                            fh.write_geometry(ns._geometry)
                        return
                    raise NotImplementedError
                except Exception:
                    pass

                if len(ns._data) == 0:
                    ns._data.append(ns._E)
                    ns._data_header.append("Energy[eV]")
                    ns._data.append(ns._PDOS_filter(ns._PDOS).sum(0))
                    if ns._PDOS_filter_name is not None:
                        ns._data_header.append(
                            f"DOS[spin={ns._PDOS_filter_name}][1/eV]"
                        )
                    else:
                        ns._data_header.append("DOS[1/eV]")

                from sisl.io import tableSile

                tableSile(out, mode="w").write(
                    *ns._data,
                    comment=[comment] + ns._data_description,
                    header=ns._data_header,
                )
                # Clean all data
                ns._norm = "none"
                ns._data = []
                ns._data_header = []
                ns._data_description = []

                ns._PDOS_filter_name = None
                ns._PDOS_filter = _sum_filter
                ns._Erng = None

        p.add_argument(
            "--out",
            "-o",
            nargs=1,
            action=Out,
            help="Store currently collected PDOS (at its current invocation) to the out file.",
        )

        class Plot(argparse.Action):
            @run_actions
            def __call__(self, parser, ns, value, option_string=None):
                if len(ns._data) == 0:
                    ns._data.append(ns._E)
                    ns._data_header.append("Energy[eV]")
                    ns._data.append(ns._PDOS_filter(ns._PDOS).sum(0))
                    if ns._PDOS_filter_name is not None:
                        ns._data_header.append(
                            f"DOS[spin={ns._PDOS_filter_name}][1/eV]"
                        )
                    else:
                        ns._data_header.append("DOS[1/eV]")

                from matplotlib import pyplot as plt

                plt.figure()

                def _get_header(header):
                    header = (
                        header.replace("PDOS", "")
                        .replace("DOS", "")
                        .replace("[1/eV]", "")
                    )
                    if len(header) == 0:
                        return "Total"
                    if header.startswith("["):
                        header = header[1:]
                    if header.endswith("]"):
                        header = header[:-1]
                    return header

                kwargs = {}
                if len(ns._data) > 2:
                    kwargs["alpha"] = 0.6
                for i in range(1, len(ns._data)):
                    plt.plot(
                        ns._data[0],
                        ns._data[i],
                        label=_get_header(ns._data_header[i]),
                        **kwargs,
                    )

                plt.ylabel("DOS [1/eV]")
                if "unknown" in comment:
                    plt.xlabel("E [eV]")
                else:
                    plt.xlabel("E - E_F [eV]")

                plt.legend(loc=8, ncol=3, bbox_to_anchor=(0.5, 1.0))
                if value is None:
                    plt.show()
                else:
                    plt.savefig(value)

                # Clean all data
                ns._norm = "none"
                ns._data = []
                ns._data_header = []
                ns._data_description = []

                ns._PDOS_filter_name = None
                ns._PDOS_filter = _sum_filter
                ns._Erng = None

        p.add_argument(
            "--plot",
            "-p",
            action=Plot,
            nargs="?",
            metavar="FILE",
            help="Plot the currently collected information (at its current invocation).",
        )

        return p, namespace


# PDOS files are:
# They contain the same file (same xml-data)
# However, pdos.xml is preferred because it has higher precision.
#  siesta.PDOS
add_sile("PDOS", pdosSileSiesta, gzip=True)
#  pdos.xml/siesta.PDOS.xml
add_sile("PDOS.xml", pdosSileSiesta, gzip=True)
