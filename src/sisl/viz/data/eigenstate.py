# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# TODO when forward refs with local variables
# from __future__ import annotations

from pathlib import Path
from typing import Literal

import sisl
from sisl.io import fdfSileSiesta, wfsxSileSiesta

from .._single_dispatch import singledispatchmethod
from ..data_sources import FileDataSIESTA
from .data import Data


class EigenstateData(Data):
    """Wavefunction data class"""

    @singledispatchmethod
    @classmethod
    def new(cls, data):
        return cls(data)

    @new.register
    @classmethod
    def from_eigenstate(cls, eigenstate: sisl.EigenstateElectron):
        return cls(eigenstate)

    @new.register
    @classmethod
    def from_path(cls, path: Path, *args, **kwargs):
        """Creates a sile from the path and tries to read the eigenstates from it.

        Parameters
        ----------
        path:
            The path to the file to read the eigenstates from.

            Depending of the sile extracted from the path, the corresponding `EigenstateData` constructor
            will be called.
        **kwargs:
            Extra arguments to be passed to the EigenstateData constructor.
        """
        return cls.new(sisl.get_sile(path), *args, **kwargs)

    @new.register
    @classmethod
    def from_string(cls, string: str, *args, **kwargs):
        """Converts the string to a path and calls the `from_path` method.

        Parameters
        ----------
        string:
            The string to be converted to a path.
        **kwargs:
            Extra arguments directly passed to the `from_path` method.

        See Also
        --------
        from_path: The arguments are passed to this method.
        """
        return cls.new(Path(string), *args, **kwargs)

    @new.register
    @classmethod
    def from_fdf(
        cls,
        fdf: fdfSileSiesta,
        source: Literal["wfsx", "hamiltonian"] = "wfsx",
        k: tuple[float, float, float] = (0, 0, 0),
        spin: int = 0,
    ):
        """Gets eigenstate coefficients using the fdf file as the root.

        Parameters
        ----------
        fdf:
            The FDF file to be used as the root.
        source:
            The source of the eigenstates. It can be either from the WFSX file
            or from the Hamiltonian, in which case the eigenstates need to be
            calculated.
        k:
            The k-point for which the eigenstate coefficients are desired.
        spin:
            The spin for which the eigenstate coefficients are desired.
        """
        if source == "wfsx":
            sile = FileDataSIESTA(fdf=fdf, cls=wfsxSileSiesta)

            assert isinstance(sile, wfsxSileSiesta)

            geometry = fdf.read_geometry(output=True)

            return cls.new(sile, geometry=geometry, k=k, spin=spin)
        elif source == "hamiltonian":
            H = fdf.read_hamiltonian()

            return cls.new(H, k=k, spin=spin)

    @new.register
    @classmethod
    def from_siesta_wfsx(
        cls,
        wfsx_file: wfsxSileSiesta,
        geometry: sisl.Geometry,
        k: tuple[float, float, float] = (0, 0, 0),
        spin: int = 0,
    ):
        """Gets eigenstate coefficients from the WFSX file.

        Parameters
        ----------
        fdf:
            The FDF file to be used as the root.
        geometry:
            The geometry to be used for the Hamiltonian.
        k:
            The k-point for which the eigenstate coefficients are desired.
        spin:
            The spin for which the eigenstate coefficients are desired.
        """

        """Reads the wavefunction coefficients from a SIESTA WFSX file"""
        # Get the WFSX file. If not provided, it is inferred from the fdf.
        if not wfsx_file.file.is_file():
            raise ValueError(f"File '{wfsx_file.file}' does not exist.")

        sizes = wfsx_file.read_sizes()
        H = sisl.Hamiltonian(geometry, dim=sizes.nspin)

        wfsx = sisl.get_sile(wfsx_file.file, parent=H)

        # Try to find the eigenstate that we need
        eigenstate = wfsx.read_eigenstate(k=k, spin=spin)
        if eigenstate is None:
            # We have not found it.
            raise ValueError(f"A state with k={k} was not found in file {wfsx.file}.")

        return cls.new(eigenstate)

    @new.register
    @classmethod
    def from_hamiltonian(
        cls,
        H: sisl.Hamiltonian,
        k: tuple[float, float, float] = (0, 0, 0),
        spin: int = 0,
    ):
        """Calculates the eigenstates from a Hamiltonian and then generates the wavefunctions.

        Parameters
        ----------
        H:
            The Hamiltonian to be used to calculate the eigenstates.
        k:
            The k-point for which the eigenstate coefficients are desired.
        spin:
            The spin for which the eigenstate coefficients are desired.
        """
        return cls.new(H.eigenstate(k, spin=spin))

    def __getitem__(self, key):
        return self._data[key]
