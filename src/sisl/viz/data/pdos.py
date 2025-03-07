# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# TODO when forward refs work with locals
# from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
from xarray import DataArray

import sisl
from sisl._core.geometry import Geometry
from sisl.io import fdfSileSiesta, pdosSileSiesta, tbtncSileTBtrans, wfsxSileSiesta
from sisl.physics import Hamiltonian, Spin
from sisl.physics.distribution import get_distribution

from .._single_dispatch import singledispatchmethod
from ..data_sources import FileDataSIESTA
from ..processors.spin import get_spin_options
from .xarray import OrbitalData


class PDOSData(OrbitalData):
    """Holds PDOS Data in a custom xarray DataArray.

    The point of this class is to normalize the data coming from different sources
    so that functions can use it without worrying where the data came from.
    """

    def sanity_check(
        self,
        na: Optional[int] = None,
        no: Optional[int] = None,
        n_spin: Optional[int] = None,
        atom_tags: Optional[Sequence[str]] = None,
        dos_checksum: Optional[float] = None,
    ):
        """Check that the dataarray satisfies the requirements to be treated as PDOSData."""
        super().sanity_check()

        array = self._data
        geometry = array.attrs["geometry"]
        assert isinstance(geometry, Geometry)

        if na is not None:
            assert geometry.na == na
        if no is not None:
            assert geometry.no == no
        if atom_tags is not None:
            assert (
                len(set(atom_tags) - set([atom.tag for atom in geometry.atoms.atom]))
                == 0
            )

        for k in ("spin", "orb", "E"):
            assert (
                k in array.dims
            ), f"'{k}' dimension missing, existing dimensions: {array.dims}"

        # Check if we have the correct number of spin channels
        if n_spin is not None:
            assert len(array.spin) == n_spin
        # Check if we have the correct number of orbitals
        assert len(array.orb) == geometry.no

        # Check if the checksum of the DOS is correct
        if dos_checksum is not None:
            this_dos_checksum = float(array.sum())
            assert np.allclose(
                this_dos_checksum, dos_checksum
            ), f"Checksum of the DOS is incorrect. Expected {dos_checksum} but got {this_dos_checksum}"

    @classmethod
    def toy_example(
        cls,
        geometry: Optional[Geometry] = None,
        spin: Union[str, int, Spin] = "",
        nE: int = 100,
    ):
        """Creates a toy example of a bands data array"""

        if geometry is None:
            orbitals = [
                sisl.AtomicOrbital("2s"),
                sisl.AtomicOrbital("2px"),
                sisl.AtomicOrbital("2py"),
                sisl.AtomicOrbital("2pz"),
            ]

            geometry = sisl.geom.graphene(atoms=sisl.Atom(Z=6, orbitals=orbitals))

        PDOS = np.random.rand(geometry.no, nE)

        spin = Spin(spin)

        if spin.is_polarized:
            PDOS = np.array([PDOS / 2, PDOS / 2])
        elif not spin.is_diagonal:
            PDOS = np.array([PDOS, PDOS, np.zeros_like(PDOS), np.zeros_like(PDOS)])

        return cls.new(PDOS, geometry, np.arange(nE), spin=spin)

    @singledispatchmethod
    @classmethod
    def new(cls, data: DataArray) -> "PDOSData":
        return cls(data)

    @new.register
    @classmethod
    def from_numpy(
        cls,
        PDOS: np.ndarray,
        geometry: Geometry,
        E: Sequence[float],
        E_units: str = "eV",
        spin: Optional[Union[sisl.Spin, str, int]] = None,
        extra_attrs: dict = {},
    ):
        """

        Parameters
        ----------
        PDOS: numpy.ndarray of shape ([nSpin], nE, nOrb)
            The Projected Density Of States, orbital resolved. The array can have 2 or 3 dimensions,
            since the spin dimension is optional. The spin class of the calculation that produced the data
            is inferred from the spin dimension:
                If there is no spin dimension or nSpin == 1, the calculation is spin unpolarized.
                If nSpin == 2, the calculation is spin polarized. It is expected that [total, z] is
                provided, not [spin0, spin1].
                If nSpin == 4, the calculation is assumed to be with noncolinear spin.
        geometry: sisl.Geometry
            The geometry to which the data corresponds. It must have as many orbitals as the PDOS data.
        E:
            The energies to which the data corresponds.
        E_units:
            The units of the energy. Defaults to 'eV'.
        extra_attrs:
            A dictionary of extra attributes to be added to the DataArray. One of the attributes that
        """
        # Understand what the spin class is for this data.
        data_spin = sisl.Spin.UNPOLARIZED
        if PDOS.squeeze().ndim == 3:
            data_spin = {
                1: sisl.Spin.UNPOLARIZED,
                2: sisl.Spin.POLARIZED,
                4: sisl.Spin.NONCOLINEAR,
            }[PDOS.shape[0]]
        data_spin = sisl.Spin(data_spin)

        # If no spin specification was passed, then assume the spin is what we inferred from the data.
        # Otherwise, make sure the spin specification is consistent with the data.
        if spin is None:
            spin = data_spin
        else:
            spin = sisl.Spin(spin)
            if data_spin.is_diagonal:
                assert spin == data_spin
            else:
                assert not spin.is_diagonal

        if PDOS.ndim == 2:
            # Add an extra axis for spin at the beggining if the array only has dimensions for orbitals and energy.
            PDOS = PDOS[None, ...]

        # Check that the number of orbitals in the geometry and the data match.
        orb_dim = PDOS.ndim - 2
        if geometry is not None:
            if geometry.no != PDOS.shape[orb_dim]:
                raise ValueError(
                    f"The geometry provided contains {geometry.no} orbitals, while we have PDOS information of {PDOS.shape[orb_dim]}."
                )

        # Build the standardized dataarray, with everything needed to understand it.
        E_units = extra_attrs.pop("E_units", "eV")

        if spin.is_polarized:
            spin_coords = ["total", "z"]
        elif not spin.is_diagonal:
            spin_coords = get_spin_options(spin)
        else:
            spin_coords = ["total"]

        coords = [
            ("spin", spin_coords),
            ("orb", range(PDOS.shape[orb_dim])),
            ("E", E, {"units": E_units}),
        ]

        attrs = {
            "spin": spin,
            "geometry": geometry,
            "units": f"1/{E_units}",
            **extra_attrs,
        }

        return cls.new(DataArray(PDOS, coords=coords, name="PDOS", attrs=attrs))

    @new.register
    @classmethod
    def from_path(cls, path: Path, *args, **kwargs):
        """Creates a sile from the path and tries to read the PDOS from it.

        Parameters
        ----------
        path:
            The path to the file to read the PDOS from.

            Depending of the sile extracted from the path, the corresponding `PDOSData` constructor
            will be called.
        **kwargs:
            Extra arguments to be passed to the PDOSData constructor.
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
        source: Literal["pdos", "tbtnc", "wfsx", "hamiltonian"] = "pdos",
        **kwargs,
    ):
        """Gets the PDOS from the fdf file.

        It uses the fdf file as the pivoting point to find the rest of files needed.

        Parameters
        ----------
        fdf:
            The fdf file to read the PDOS from.
        source:
            The source to read the PDOS data from.
        **kwargs
            Extra arguments to be passed to the PDOSData constructor, which depends
            on the source requested.

            One should check ``PDOSData.from_*`` for details for each PDOS retrieval.
        """
        if source == "pdos":
            sile = FileDataSIESTA(fdf=fdf, cls=pdosSileSiesta)

            assert isinstance(sile, pdosSileSiesta)

            return cls.new(sile)
        elif source == "tbtnc":
            sile = FileDataSIESTA(fdf=fdf, cls=tbtncSileTBtrans)

            assert isinstance(sile, tbtncSileTBtrans)

            geometry = fdf.read_geometry(output=True)

            return cls.new(sile, geometry=geometry, **kwargs)
        elif source == "wfsx":
            sile = FileDataSIESTA(fdf=fdf, cls=wfsxSileSiesta)

            assert isinstance(sile, wfsxSileSiesta)

            geometry = fdf.read_geometry(output=True)

            return cls.new(sile, geometry=geometry)
        elif source == "hamiltonian":
            H = fdf.read_hamiltonian()

            return cls.new(H, **kwargs)

    @new.register
    @classmethod
    def from_siesta_pdos(cls, pdos_file: pdosSileSiesta):
        """Gets the PDOS from a SIESTA PDOS file.

        Parameters
        ----------
        pdos_file:
            The PDOS file to read the PDOS from.
        """
        # Get the info from the .PDOS file
        geometry, E, PDOS = pdos_file.read_data()

        return cls.new(PDOS, geometry, E)

    @new.register
    @classmethod
    def from_tbtrans(
        cls,
        tbt_nc: tbtncSileTBtrans,
        geometry: Union[Geometry, None] = None,
        elec: Union[int, str, None] = None,
    ):
        """Reads the PDOS from a *.TBT.nc file coming from a TBtrans run.

        Parameters
        ----------
        tbt_nc:
            The TBtrans file to read the PDOS from.
        geometry:
            Full geometry of the system (including scattering and electrode regions).
            Right now only used to get the basis of each atom, which is not
            stored in the TBT.nc file.
        elec:
            which electrode to get the PDOS from. Can be None for the Green function,
            otherwise the specified. Otherwise it is the index/name of an electrode
            so that one gets the ADOS from that electrode.
        """
        if elec is None:
            elec = "Gf"
            PDOS = tbt_nc.DOS(sum=False).T
        else:
            PDOS = tbt_nc.ADOS(elec, sum=False).T
            elec = f"A{elec}"
        E = tbt_nc.E

        read_geometry_kwargs = {}
        if geometry is not None:
            read_geometry_kwargs["atoms"] = geometry.atoms

        # Read the geometry from the TBT.nc file and get only the device part
        geometry = tbt_nc.read_geometry(**read_geometry_kwargs).sub(tbt_nc.a_dev)

        return cls.new(PDOS, geometry, E, extra_attrs={"elec": elec})

    @new.register
    @classmethod
    def from_hamiltonian(
        cls,
        H: Hamiltonian,
        kgrid: Tuple[int, int, int] = None,
        kgrid_displ: Tuple[float, float, float] = (0, 0, 0),
        Erange: Tuple[float, float] = (-2, 2),
        E0: float = 0,
        nE: int = 100,
        distribution=get_distribution("gaussian"),
    ):
        """Calculates the PDOS from a sisl Hamiltonian.

        Parameters
        ----------
        H:
            The Hamiltonian from which to calculate the PDOS.
        kgrid:
            Number of kpoints in each reciprocal space direction. A Monkhorst-pack grid
            will be generated from this specification. The PDOS will be averaged over the
            whole k-grid.
        kgrid_displ:
            Displacement of the Monkhorst-Pack grid.
        Erange:
            Energy range (min and max) for the PDOS calculation.
        E0:
            Energy shift for the PDOS calculation.
        nE:
            Number of energy points for the PDOS calculation.
        distribution:
            The distribution to use for smoothing the PDOS along the energy axis.
            Each state will be broadened by this distribution.
        """

        # Get the kgrid or generate a default grid by checking the interaction between cells
        # This should probably take into account how big the cell is.
        kgrid = kgrid
        if kgrid is None:
            kgrid = [3 if nsc > 1 else 1 for nsc in H.geometry.nsc]

        Erange = Erange
        if Erange is None:
            raise ValueError(
                "You need to provide an energy range to calculate the PDOS from the Hamiltonian"
            )

        E = np.linspace(Erange[0], Erange[-1], nE) + E0

        bz = sisl.MonkhorstPack(H, kgrid, kgrid_displ)

        # Define the available spins
        spin_indices = [0]
        if H.spin.is_polarized:
            spin_indices = [0, 1]

        # Calculate the PDOS for all available spins
        PDOS = []
        for spin in spin_indices:
            with bz.apply as parallel:
                spin_PDOS = parallel.average.eigenstate(
                    spin=spin, wrap=lambda eig: eig.PDOS(E, distribution=distribution)
                )

            PDOS.append(spin_PDOS)

        if len(spin_indices) == 1:
            PDOS = PDOS[0]
        else:
            # Convert from spin components to total and z contributions.
            total = PDOS[0] + PDOS[1]
            z = PDOS[0] - PDOS[1]

            PDOS = np.concatenate([total, z])

        PDOS = np.array(PDOS)

        return cls.new(PDOS, H.geometry, E, spin=H.spin, extra_attrs={"bz": bz})

    @new.register
    @classmethod
    def from_wfsx(
        cls,
        wfsx_file: wfsxSileSiesta,
        H: Hamiltonian,
        geometry: Union[Geometry, None] = None,
        Erange=(-2, 2),
        nE: int = 100,
        E0: float = 0,
        distribution=get_distribution("gaussian"),
    ):
        """Generates the PDOS values from a file containing eigenstates.

        Parameters
        ----------
        wfsx_file:
            The file containing the eigenstates.
        H:
            The Hamiltonian to which the eigenstates correspond. Used to get the overlap
            matrix and the spin type.
        geometry:
            The geometry of the system. If not provided, it is extracted from the Hamiltonian.
        Erange:
            The energy range in which to calculate the PDOS.
        nE:
            The number of energy points for which to calculate the PDOS.
        E0:
            Reference energy to take as 0. Energy levels will be shifted by `-E0`.
        distribution:
            The distribution to use for smoothing the PDOS along the energy axis.
            Each state will be broadened by this distribution.
        """
        if geometry is None:
            geometry = getattr(H, "geometry", None)

        # Get the wfsx file
        wfsx_sile = FileDataSIESTA(path=wfsx_file, cls=sisl.io.wfsxSileSiesta, parent=H)

        # Read the sizes of the file, which contain the number of spin channels
        # and the number of orbitals and the number of k points.
        sizes = wfsx_sile.read_sizes()
        # Check that spin sizes of hamiltonian and wfsx file match
        assert (
            H.spin.size(H.dtype) == sizes.nspin
        ), f"Hamiltonian has spin size {H.spin.size(H.dtype)} while file has spin size {sizes.nspin}"
        # Get the size of the spin channel. The size returned might be 8 if it is a spin-orbit
        # calculation, but we need only 4 spin channels (total, x, y and z), same as with non-colinear
        nspin = min(4, sizes.nspin)

        # Get the energies for which we need to calculate the PDOS.
        Erange = Erange
        E = np.linspace(Erange[0], Erange[-1], nE) + E0

        # Initialize the PDOS array
        PDOS = np.zeros((nspin, sizes.no_u, E.shape[0]), dtype=np.float64)

        # Loop through eigenstates in the WFSX file and add their contribution to the PDOS.
        # Note that we pass the hamiltonian as the parent here so that the overlap matrix
        # for each point can be calculated by eigenstate.PDOS()
        for eigenstate in wfsx_sile.yield_eigenstate():
            spin = eigenstate.info.get("spin", 0)
            if nspin == 4:
                spin = slice(None)

            PDOS[spin] += eigenstate.PDOS(
                E, distribution=distribution
            ) * eigenstate.info.get("weight", 1)

        return cls.new(PDOS, geometry, E, spin=H.spin)
