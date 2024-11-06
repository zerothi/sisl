# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._array import aranged, arrayd
from sisl._core.atom import Atom
from sisl._core.orbital import SphericalOrbital
from sisl._help import xml_parse
from sisl._internal import set_module
from sisl.unit.siesta import unit_convert
from sisl.utils import PropertyDict, strmap
from sisl.utils.cmd import default_ArgumentParser, default_namespace

from ..sile import add_sile, sile_fh_open
from .sile import SileCDFSiesta, SileSiesta

__all__ = ["ionxmlSileSiesta", "ionncSileSiesta"]


@set_module("sisl.io.siesta")
class ionxmlSileSiesta(SileSiesta):
    """Basis set information in xml format

    Note that the ``ion`` files are equivalent to the ``ion.xml`` files.
    """

    @sile_fh_open(True)
    def read_basis(self) -> Atom:
        """Returns data associated with the ion.xml file"""
        # Get the element-tree
        root = xml_parse(self.fh).getroot()

        # Get number of orbitals
        label = root.find("label").text.strip()
        Z = int(root.find("z").text)  # atomic number, negative for floating
        mass = float(root.find("mass").text)

        # Read in the PAO"s
        paos = root.find("paos")

        # Now loop over all orbitals
        orbital = []

        # All orbital data
        Bohr2Ang = unit_convert("Bohr", "Ang")
        for orb in paos:
            n = int(orb.get("n"))
            l = int(orb.get("l"))
            z = int(orb.get("z"))  # zeta

            q0 = float(orb.get("population"))

            P = not int(orb.get("ispol")) == 0

            # Radial components
            rad = orb.find("radfunc")
            npts = int(rad.find("npts").text)

            # Grid spacing in Bohr (conversion is done later
            # because the normalization is easier)
            delta = float(rad.find("delta").text)

            # Read in data to a list
            dat = arrayd(rad.find("data").text.split())

            # Since the readed data has fewer significant digits we
            # might as well re-create the table of the radial component.
            r = aranged(npts) * delta

            # To get it per Ang**3
            # TODO, check that this is correct.
            # The fact that we have to have it normalized means that we need
            # to convert psi /sqrt(Bohr**3) -> /sqrt(Ang**3)
            # \int psi^\dagger psi == 1
            psi = dat[1::2] * r**l / Bohr2Ang ** (3.0 / 2.0)

            # Get the cutoff radius if it is specified. Otherwise an optimized
            # radius is computed from the radial function by the Orbital class.
            R = rad.find("cutoff")
            if R is not None:
                R = float(R.text) * Bohr2Ang

            # Create the sphericalorbital and then the atomicorbital
            sorb = SphericalOrbital(l, (r * Bohr2Ang, psi), q0, R=R)

            # This will be -l:l (this is the way siesta does it)
            orbital.extend(sorb.toAtomicOrbital(n=n, zeta=z, P=P))

        # Now create the atom and return
        return Atom(Z, orbital, mass=mass, tag=label)


@set_module("sisl.io.siesta")
class ionncSileSiesta(SileCDFSiesta):
    """Basis set information in NetCDF files

    Note that the ``ion.nc`` files are equivalent to the ``ion.xml`` files.
    """

    def read_basis(self) -> Atom:
        """Returns data associated with the ion.xml file"""
        no = len(self._dimension("norbs"))

        # Get number of orbitals
        label = self.Label.strip()
        Z = int(self.Atomic_number)
        mass = float(self.Mass)

        # Retrieve values
        orb_l = self._variable("orbnl_l")[:]  # angular quantum number
        orb_n = self._variable("orbnl_n")[:]  # principal quantum number
        orb_z = self._variable("orbnl_z")[:]  # zeta
        orb_P = self._variable("orbnl_ispol")[:] > 0  # polarization shell, or not
        orb_q0 = self._variable("orbnl_pop")[:]  # q0 for the orbitals
        orb_delta = self._variable("delta")[:]  # delta for the functions
        orb_psi = self._variable("orb")[:, :]
        cutoff = self._variable("cutoff")[:]

        # Now loop over all orbitals
        orbital = []

        # All orbital data
        Bohr2Ang = unit_convert("Bohr", "Ang")
        for io in range(no):
            n = orb_n[io]
            l = orb_l[io]
            z = orb_z[io]
            P = orb_P[io]

            # Grid spacing in Bohr (conversion is done later
            # because the normalization is easier)
            delta = orb_delta[io]

            # Since the readed data has fewer significant digits we
            # might as well re-create the table of the radial component.
            r = aranged(orb_psi.shape[1]) * delta

            # To get it per Ang**3
            # TODO, check that this is correct.
            # The fact that we have to have it normalized means that we need
            # to convert psi /sqrt(Bohr**3) -> /sqrt(Ang**3)
            # \int psi^\dagger psi == 1
            psi = orb_psi[io, :] * r**l / Bohr2Ang ** (3.0 / 2.0)

            R = cutoff[io] * Bohr2Ang

            # Create the sphericalorbital and then the atomicorbital
            sorb = SphericalOrbital(l, (r * Bohr2Ang, psi), orb_q0[io], R=R)

            # This will be -l:l (this is the way siesta does it)
            orbital.extend(sorb.toAtomicOrbital(n=n, zeta=z, P=P))

        # Now create the atom and return
        return Atom(Z, orbital, mass=mass, tag=label)

    @default_ArgumentParser(description="Extracting basis-set information.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """Returns the arguments that is available for this Sile"""
        # limit_args = kwargs.get("limit_arguments", True)
        short = kwargs.get("short", False)

        def opts(*args):
            if short:
                return args
            return [args[0]]

        # We limit the import to occur here
        import argparse

        Bohr2Ang = unit_convert("Bohr", "Ang")
        Ry2eV = unit_convert("Bohr", "Ang")

        # The first thing we do is adding the geometry to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the geometry in a
        # straight forward manner.
        # convert netcdf file to a dictionary
        ion_nc = PropertyDict()
        ion_nc.n = self._variable("orbnl_n")[:]
        ion_nc.l = self._variable("orbnl_l")[:]
        ion_nc.zeta = self._variable("orbnl_z")[:]
        ion_nc.pol = self._variable("orbnl_ispol")[:]
        ion_nc.orbital = self._variable("orb")[:]

        # this gets converted later
        delta = self._variable("delta")[:]
        r = aranged(ion_nc.orbital.shape[1]).reshape(1, -1) * delta.reshape(-1, 1)
        ion_nc.orbital *= r ** ion_nc.l.reshape(-1, 1) / Bohr2Ang * (3.0 / 2.0)
        ion_nc.r = r * Bohr2Ang
        ion_nc.kb = PropertyDict()
        ion_nc.kb.n = self._variable("pjnl_n")[:]
        ion_nc.kb.l = self._variable("pjnl_l")[:]
        ion_nc.kb.e = self._variable("pjnl_ekb")[:] * Ry2eV
        ion_nc.kb.proj = self._variable("proj")[:]
        delta = self._variable("kbdelta")[:]
        r = aranged(ion_nc.kb.proj.shape[1]).reshape(1, -1) * delta.reshape(-1, 1)
        ion_nc.kb.proj *= r ** ion_nc.kb.l.reshape(-1, 1) / Bohr2Ang * (3.0 / 2.0)
        ion_nc.kb.r = r * Bohr2Ang

        vna = self._variable("vna")
        r = aranged(vna[:].size) * vna.Vna_delta
        ion_nc.vna = PropertyDict()
        ion_nc.vna.v = vna[:] * Ry2eV * r / Bohr2Ang**3
        ion_nc.vna.r = r * Bohr2Ang

        # this is charge (not 1/sqrt(charge))
        chlocal = self._variable("chlocal")
        r = aranged(chlocal[:].size) * chlocal.Chlocal_delta
        ion_nc.chlocal = PropertyDict()
        ion_nc.chlocal.v = chlocal[:] * r / Bohr2Ang**3
        ion_nc.chlocal.r = r * Bohr2Ang

        vlocal = self._variable("reduced_vlocal")
        r = aranged(vlocal[:].size) * vlocal.Reduced_vlocal_delta
        ion_nc.vlocal = PropertyDict()
        ion_nc.vlocal.v = vlocal[:] * r / Bohr2Ang**3
        ion_nc.vlocal.r = r * Bohr2Ang

        if "core" in self.variables:
            # this is charge (not 1/sqrt(charge))
            core = self._variable("core")
            r = aranged(core[:].size) * core.Core_delta
            ion_nc.core = PropertyDict()
            ion_nc.core.v = core[:] * r / Bohr2Ang**3
            ion_nc.core.r = r * Bohr2Ang

        d = {
            "_data": ion_nc,
            "_kb_proj": False,
            "_l": True,
            "_n": True,
        }
        namespace = default_namespace(**d)

        # l-quantum number
        class lRange(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                value = (
                    value.replace("s", 0)
                    .replace("p", 1)
                    .replace("d", 2)
                    .replace("f", 3)
                    .replace("g", 4)
                )
                ns._l = strmap(int, value)[0]

        p.add_argument(
            "-l",
            action=lRange,
            help="Denote the sub-section of l-shells that are plotted: 's,f'",
        )

        # n quantum number
        class nRange(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._n = strmap(int, value)[0]

        p.add_argument(
            "-n",
            action=nRange,
            help="Denote the sub-section of n quantum numbers that are plotted: '2-4,6'",
        )

        class Plot(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                import matplotlib.pyplot as plt

                # Retrieve values
                data = ns._data

                # We have these plots:
                #  - orbitals
                #  - projectors
                #  - chlocal
                #  - vna
                #  - vlocal
                #  - core (optional)

                # We'll plot them like this:
                #  orbitals | projectors
                #  vna + vlocal | chlocal + core
                #
                # Determine different n, l
                fig, axs = plt.subplots(2, 2)

                # Now plot different orbitals
                for n, l, zeta, pol, r, orb in zip(
                    data.n, data.l, data.zeta, data.pol, data.r, data.orbital
                ):
                    if pol == 1:
                        pol = "P"
                    else:
                        pol = ""
                    axs[0][0].plot(r, orb, label=f"n{n}l{l}Z{zeta}{pol}")
                axs[0][0].set_title("Orbitals")
                axs[0][0].set_xlabel("Distance [Ang]")
                axs[0][0].set_ylabel("Value [a.u.]")
                axs[0][0].legend()

                # plot projectors
                for n, l, e, r, proj in zip(
                    data.kb.n, data.kb.l, data.kb.e, data.kb.r, data.kb.proj
                ):
                    axs[0][1].plot(r, proj, label=f"n{n}l{l} e={e:.5f}")
                axs[0][1].set_title("KB projectors")
                axs[0][1].set_xlabel("Distance [Ang]")
                axs[0][1].set_ylabel("Value [a.u.]")
                axs[0][1].legend()

                axs[1][0].plot(data.vna.r, data.vna.v, label="Vna")
                axs[1][0].plot(data.vlocal.r, data.vlocal.v, label="Vlocal")
                axs[1][0].set_title("Potentials")
                axs[1][0].set_xlabel("Distance [Ang]")
                axs[1][0].set_ylabel("Potential [eV]")
                axs[1][0].legend()

                axs[1][1].plot(data.chlocal.r, data.chlocal.v, label="Chlocal")
                if "core" in data:
                    axs[1][1].plot(data.core.r, data.core.v, label="core")
                axs[1][1].set_title("Charge")
                axs[1][1].set_xlabel("Distance [Ang]")
                axs[1][1].set_ylabel("Charge [Ang^3]")
                axs[1][1].legend()

                if value is None:
                    plt.show()
                else:
                    plt.savefig(value)

        p.add_argument(
            *opts("--plot", "-p"),
            action=Plot,
            nargs="?",
            metavar="FILE",
            help="Plot the content basis set file, possibly saving plot to a file.",
        )

        return p, namespace


add_sile("ion.xml", ionxmlSileSiesta, gzip=True)
add_sile("ion.nc", ionncSileSiesta)
