# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional

import numpy as np

from sisl._core.geometry import Geometry
from sisl._internal import set_module
from sisl.messages import deprecation
from sisl.typing import UnitsVar
from sisl.unit import serialize_units_arg, unit_convert
from sisl.utils import PropertyDict

from .._multiple import SileBinder
from ..sile import add_sile, sile_fh_open
from .sile import SileORCA

__all__ = ["txtSileORCA"]


@set_module("sisl.io.orca")
class txtSileORCA(SileORCA):
    """Output from the ORCA property.txt file"""

    _info_attributes_ = [
        dict(
            name="na",
            searcher=r".*Number of atoms:",
            parser=lambda attr, instance, match: int(match.string.split()[-1]),
            not_found="error",
        ),
        dict(
            name="no",
            searcher=r".*number of basis functions:",
            parser=lambda attr, instance, match: int(match.string.split()[-1]),
            not_found="error",
        ),
        dict(
            name="vdw_correction",
            searcher=r".*\$ VdW_Correction",
            parser=lambda attr, instance, match: True,
            default=False,
            not_found="ignore",
        ),
    ]

    @property
    @deprecation(
        "txtSileORCA.na is deprecated in favor of txtSileORCA.info.na", "0.15", "0.17"
    )
    def na(self):
        """Number of atoms"""
        return self.info.na

    @property
    @deprecation(
        "txtSileORCA.no is deprecated in favor of txtSileORCA.info.no", "0.15", "0.17"
    )
    def no(self):
        """Number of orbitals (basis functions)"""
        return self.info.no

    @SileBinder(postprocess=np.array)
    @sile_fh_open()
    def read_electrons(self) -> Optional[tuple[float, float]]:
        """Read number of electrons (alpha, beta)

        Returns
        -------
        out: numpy.ndarray or list of numpy.ndarray
            alpha and beta electrons
        """
        f = self.step_to("Number of Alpha Electrons", allow_reread=False)
        if f[0]:
            alpha = float(f[1].split()[-1])
            beta = float(self.readline().split()[-1])
        else:
            return None
        return alpha, beta

    @SileBinder()
    @sile_fh_open()
    def read_energy(self, units: UnitsVar = "eV") -> PropertyDict:
        """Reads the energy blocks

        Parameters
        ----------
        units :
            selects units in the returned data

        Notes
        -----
        Energies written by ORCA have units of Ha.

        Returns
        -------
        PropertyDict or list of PropertyDict
            all data from the "DFT_Energy" and "VdW_Correction" blocks
        """
        # read the DFT_Energy block
        f = self.step_to("$ DFT_Energy", allow_reread=False)[0]
        if not f:
            return None

        units = serialize_units_arg(units)
        Ha2unit = unit_convert("Ha", units["energy"])

        self.readline()  # description
        self.readline()  # geom. index
        self.readline()  # prop. index

        E = PropertyDict()

        line = self.readline()
        while "----" not in line:
            v = line.split()
            value = float(v[-1]) * Ha2unit
            if v[0] == "Exchange":
                E["exchange"] = value
            elif v[0] == "Correlation":
                if v[2] == "NL":
                    E["correlation_nl"] = value
                else:
                    E["correlation"] = value
            elif v[0] == "Exchange-Correlation":
                E["xc"] = value
            elif v[0] == "Embedding":
                E["embedding"] = value
            elif v[1] == "DFT":
                E["total"] = value
            line = self.readline()

        line = self.readline()
        if self.info.vdw_correction:
            v = self.step_to("Van der Waals Correction:")[1].split()
            E["vdw"] = float(v[-1]) * Ha2unit

        return E

    @SileBinder()
    @sile_fh_open()
    def read_geometry(self) -> Geometry:
        """Reads the geometry from ORCA property.txt file

        Returns
        -------
        Geometry or list of Geometry
            the geometries contained
        """
        # Read the Geometry block
        f = self.step_to("!GEOMETRY!", allow_reread=False)[0]
        if not f:
            return None

        line = self.readline()
        na = int(line.split()[-1])
        self.readline()  # skip Geometry index
        self.readline()  # skip Coordinates line

        atoms = []
        xyz = np.empty([na, 3], np.float64)
        for ia in range(na):
            l = self.readline().split()
            atoms.append(l[1])
            xyz[ia] = l[2:5]

        return Geometry(xyz, atoms)

    @sile_fh_open()
    def read_gtensor(self) -> PropertyDict:
        r"""Reads electronic g-tensor data from the ``EPRNMR_GTensor`` block

        Returns
        -------
        PropertyDict
            Electronic g-tensor
        """
        G = PropertyDict()
        f, line = self.step_to("EPRNMR_GTensor", allow_reread=False)
        if not f:
            return None

        for _ in range(4):  # skip 4 lines
            self.readline()

        G["multiplicity"] = int(self.readline().split()[-1])
        self.readline()
        self.readline()

        tensor = np.empty([3, 3], np.float64)
        for i in range(3):
            v = self.readline().split()
            tensor[i] = v[1:4]
        G["tensor"] = tensor  # raw (total) tensor

        self.readline()  # skip g tensor line
        self.readline()  # skip header

        vectors = np.empty([3, 3], np.float64)
        for i in range(3):
            v = self.readline().split()
            vectors[i] = v[1:4]
        G["vectors"] = vectors  # g-tensor eigenvectors

        self.readline()  # skip eigenvalue line
        self.readline()  # skip header
        v = self.readline().split()
        G["eigenvalues"] = np.array(v[1:4], np.float64)

        return G

    @sile_fh_open()
    def read_hyperfine_coupling(self, units: UnitsVar = "eV") -> list[PropertyDict]:
        r"""Reads hyperfine couplings from the ``EPRNMR_ATensor`` block

        For a nucleus :math:`k`, the hyperfine interaction is usually
        written in terms of the symmetric :math:`3\times 3` hyperfine
        tensor :math:`\mathbf A^{(k)}` such that

        .. math::

           H_{\mathrm{hfi}} = \mathbf{S} \cdot \mathbf A^{(k)} \mathbf{I}^{(k)}

        where :math:`\mathbf{S}` and :math:`\mathbf{I}^{(k)}`
        represent the electron and nuclear spin operators, respectively.

        For a study of hyperfine coupling in nanographenes using ORCA
        see :cite:`Sengupta2023`.

        Parameters
        ----------
        units :
            selects units in the returned data

        Notes
        -----
        Hyperfine tensors written by ORCA have units of MHz.

        Currently the fields of each `PropertyDict` contains:

        * ``ia``: atomic index
        * ``species``: species for atom
        * ``isotope``: the atomic isotope
        * ``spin``: spin multiplicity
        * ``prefactor``: prefactor defined in output
        * ``tensor``: the :math:`\mathbf A^{(k)}` tensor
        * ``vectors``: eigenvectors
        * ``eigenvalues``: eigenvalues
        * ``iso``: Fermi contact

        Returns
        -------
        list of PropertyDict
            Hyperfine coupling data
        """
        f, line = self.step_to("EPRNMR_ATensor", allow_reread=False)
        if not f:
            return None

        units = serialize_units_arg(units)
        MHz2unit = unit_convert("MHz", units["energy"])

        def read_A():
            A = PropertyDict()
            # Read the EPRNMR_ATensor block
            f, line = self.step_to("Nucleus:", allow_reread=False)
            if not f:
                return None

            v = line.split()
            A["ia"] = int(v[1])
            A["species"] = v[2]

            v = self.readline().split()
            A["isotope"] = int(v[-1])

            v = self.readline().split()
            A["spin"] = float(v[-1])

            v = self.readline().split()
            A["prefactor"] = float(v[-1]) * MHz2unit

            self.readline()  # skip Raw line
            self.readline()  # skip header

            tensor = np.empty([3, 3], np.float64)
            for i in range(3):
                v = self.readline().split()
                tensor[i] = v[1:4]
            A["tensor"] = tensor * MHz2unit  # raw A_total tensor

            self.readline()  # skip eigenvector line
            self.readline()  # skip header

            vectors = np.empty([3, 3], np.float64)
            for i in range(3):
                v = self.readline().split()
                vectors[i] = v[1:4]
            A["vectors"] = vectors

            self.readline()  # skip eigenvalue line
            self.readline()  # skip header

            v = self.readline().split()
            A["eigenvalues"] = (
                np.array(v[1:4], np.float64) * MHz2unit
            )  # eigenvalues of A_total

            v = self.readline().split()
            A["iso"] = float(v[1]) * MHz2unit  # Fermi contact A_FC

            return A

        for _ in range(3):
            self.readline()
        stored_nuclei = int(self.readline().split()[-1])

        return [read_A() for k in range(stored_nuclei)]


add_sile("txt", txtSileORCA, gzip=True)
