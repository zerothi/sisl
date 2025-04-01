# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Ideally this code should also use annotaions,
# TODO when forward refs work with annotations
# from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

import sisl
from sisl.io import bandsSileSiesta, fdfSileSiesta, wfsxSileSiesta
from sisl.physics.brillouinzone import BrillouinZone
from sisl.physics.spin import Spin

from .._single_dispatch import singledispatchmethod
from ..data_sources import FileDataSIESTA, HamiltonianDataSource
from .xarray import XarrayData

try:
    from aiida import orm

    Aiida_node = orm.Node
    AIIDA_AVAILABLE = True
except ModuleNotFoundError:

    class Aiida_node:
        pass

    AIIDA_AVAILABLE = False


class BandsData(XarrayData):
    def sanity_check(
        self,
        n_spin: Optional[int] = None,
        nk: Optional[int] = None,
        nbands: Optional[int] = None,
        klabels: Optional[Sequence[str]] = None,
        kvals: Optional[Sequence[float]] = None,
    ):
        """Check that the dataarray satisfies the requirements to be treated as PDOSData."""
        super().sanity_check()

        array = self._data

        for k in ("k", "band"):
            assert (
                k in array.dims
            ), f"'{k}' dimension missing, existing dimensions: {array.dims}"

        spin = array.attrs["spin"]
        assert isinstance(spin, Spin)

        if n_spin is not None:
            if n_spin == 1:
                assert (
                    spin.is_unpolarized
                ), f"Spin in the data is {spin}, but n_spin=1 was expected"
            elif n_spin == 2:
                assert (
                    spin.is_polarized
                ), f"Spin in the data is {spin}, but n_spin=2 was expected"
            elif n_spin == 4:
                assert (
                    not spin.is_diagonal
                ), f"Spin in the data is {spin}, but n_spin=4 was expected"

        # Check if we have the correct number of spin channels
        if spin.is_polarized:
            assert (
                "spin" in array.dims
            ), f"'spin' dimension missing for polarized spin, existing dimensions: {array.dims}"
            if n_spin is not None:
                assert len(array.spin) == n_spin
        else:
            assert (
                "spin" not in array.dims
            ), f"'spin' dimension present for spin different than polarized, existing dimensions: {array.dims}"
            assert (
                "spin" not in array.coords
            ), f"'spin' coordinate present for spin different than polarized, existing dimensions: {array.dims}"

        # Check shape of bands
        if nk is not None:
            assert len(array.k) == nk
        if nbands is not None:
            if not spin.is_diagonal:
                assert len(array.band) == nbands * 2
            else:
                assert len(array.band) == nbands

        # Check if k ticks match the expected ones
        if klabels is not None:
            assert "axis" in array.k.attrs, "No axis specification for the k dimension."
            assert (
                "ticktext" in array.k.attrs["axis"]
            ), "No ticks were found for the k dimension"
            assert tuple(array.k.attrs["axis"]["ticktext"]) == tuple(
                klabels
            ), f"Expected labels {klabels} but found {array.k.attrs['axis']['ticktext']}"
        if kvals is not None:
            assert "axis" in array.k.attrs, "No axis specification for the k dimension."
            assert (
                "tickvals" in array.k.attrs["axis"]
            ), "No ticks were found for the k dimension"
            assert np.allclose(
                array.k.attrs["axis"]["tickvals"], kvals
            ), f"Expected label values {kvals} but found {array.k.attrs['axis']['tickvals']}"

    @classmethod
    def toy_example(
        cls,
        spin: Union[str, int, Spin] = "",
        n_states: int = 20,
        nk: int = 30,
        gap: Optional[float] = None,
    ):
        """Creates a toy example of a bands data array"""

        spin = Spin(spin)

        n_bands = n_states if spin.is_diagonal else n_states * 2

        if spin.is_polarized:
            polynoms_shape = (2, n_bands)
            dims = ("spin", "k", "band")
            shift = np.tile(np.arange(0, n_bands), 2).reshape(2, -1)
        else:
            polynoms_shape = (n_bands,)
            dims = ("k", "band")
            shift = np.arange(0, n_bands)

        # Create some random coefficients for degree 2 polynomials that will be used to generate the bands
        random_polinomials = np.random.rand(*polynoms_shape, 3)
        random_polinomials[..., 0] *= 10  # Bigger curvature
        random_polinomials[
            ..., : n_bands // 2, 0
        ] *= -1  # Make the curvature negative below the gap
        random_polinomials[
            ..., 2
        ] += shift  # Shift each polynomial so that bands stack on top of each other

        # Compute bands
        x = np.linspace(0, 1, nk)
        y = (
            np.outer(x**2, random_polinomials[..., 0])
            + np.outer(x, random_polinomials[..., 1])
            + random_polinomials[..., 2].ravel()
        )

        y = y.reshape(nk, *polynoms_shape)

        if spin.is_polarized:
            # Make sure that the top of the valence band and bottom of the conduction band
            # are the same spin (to facilitate computation of the gap).
            VB_spin = y[..., : n_bands // 2].argmin() // (nk * n_bands)
            CB_spin = y[..., n_bands // 2 :].argmax() // (nk * n_bands)

            if VB_spin != CB_spin:
                y[..., n_bands // 2 :] = np.flip(y[..., n_bands // 2 :], axis=0)

            y = y.transpose(1, 0, 2)

        # Compute gap limits
        top_VB = y[..., : n_bands // 2].max()
        bottom_CB = y[..., n_bands // 2 :].min()

        # Correct the gap if some specific value was requested
        generated_gap = bottom_CB - top_VB
        if gap is not None:
            add_shift = gap - generated_gap
            y[..., n_bands // 2 :] += add_shift
            bottom_CB += add_shift

        # Compute fermi level
        fermi = (top_VB + bottom_CB) / 2

        # Create the dataarray
        data = xr.DataArray(
            y - fermi,
            coords={
                "k": x,
                "band": np.arange(0, n_bands),
            },
            dims=dims,
        )

        data = xr.Dataset({"E": data})

        # Add spin moments if the spin is not diagonal
        if not spin.is_diagonal:
            spin_moments = np.random.rand(nk, n_bands, 3) * 2 - 1
            data["spin_moments"] = xr.DataArray(
                spin_moments,
                coords={"k": x, "band": np.arange(0, n_bands), "axis": ["x", "y", "z"]},
                dims=("k", "band", "axis"),
            )

        # Add the spin class of the data
        data.attrs["spin"] = spin

        # Inform of where to place the ticks
        data.k.attrs["axis"] = {
            "tickvals": [0, x[-1]],
            "ticktext": ["Gamma", "X"],
        }

        return cls.new(data)

    @singledispatchmethod
    @classmethod
    def new(cls, bands_data):
        return cls(bands_data)

    @new.register
    @classmethod
    def from_dataset(cls, bands_data: xr.Dataset):
        """Creates a bands plot from an xarray ``Dataset``.

        Parameters
        ----------
        bands_data:
            The dataset containing the bands data. It should have at least an
            energy variable named 'E' and coordinates 'k' and 'band'. Optionally,
            it can have a 'spin' coordinate.

            Coordinates can have an 'axis' attribute that will be used to determine
            the layout of the corresponding axis in a plot.

            The geometry of the system can be passed as the 'geometry' attribute
            of the dataset.
        """
        old_attrs = bands_data.attrs

        # Check if there's a spin attribute
        spin = old_attrs.get("spin", None)

        # If not, guess it
        if spin is None:
            if "spin" not in bands_data:
                spin = Spin(Spin.UNPOLARIZED)
            else:
                spin = {
                    1: Spin.UNPOLARIZED,
                    2: Spin.POLARIZED,
                    4: Spin.NONCOLINEAR,
                }[bands_data.spin.shape[0]]

                spin = Spin(spin)

        # Remove the spin coordinate if the data is not spin polarized
        if "spin" in bands_data and not spin.is_polarized:
            bands_data = bands_data.isel(spin=0).drop_vars("spin")

        if spin.is_polarized:
            spin_options = [0, 1]
            bands_data["spin"] = ("spin", spin_options, bands_data.spin.attrs)
        # elif not spin.is_diagonal:
        #     spin_options = get_spin_options(spin)
        #     bands_data['spin'] = ('spin', spin_options, bands_data.spin.attrs)

        # If the energy variable doesn't have units, set them as eV
        if "E" in bands_data and "units" not in bands_data.E.attrs:
            bands_data.E.attrs["units"] = "eV"
        # Same with the k coordinate, which we will assume are 1/Ang
        if "k" in bands_data and "units" not in bands_data.k.attrs:
            bands_data.k.attrs["units"] = "1/Ang"
        # If there are ticks, show the grid.
        if (
            "axis" in bands_data.k.attrs
            and bands_data.k.attrs["axis"].get("ticktext") is not None
        ):
            bands_data.k.attrs["axis"] = {
                "showgrid": True,
                **bands_data.k.attrs.get("axis", {}),
            }

        bands_data.attrs = {**old_attrs, "spin": spin}

        if "geometry" not in bands_data.attrs:
            if "parent" in bands_data.attrs:
                parent = bands_data.attrs["parent"]
                if hasattr(parent, "geometry"):
                    bands_data.attrs["geometry"] = parent.geometry

        return cls(bands_data)

    @new.register
    @classmethod
    def from_dataarray(cls, bands_data: xr.DataArray):
        """Creates a ``BandsData`` object from an xarray ``DataArray``.

        Parameters
        ----------
        bands_data: xr.DataArray
            The dataarray containing the band energies.

        See Also
        --------
        from_dataset:
            Called after the dataarray is wrapped in a dataset.
            It contains documentation about the expected structure of the dataarray.
            Attributes are transferred from the dataarray to the dataset.
        """
        bands_data_ds = xr.Dataset({"E": bands_data})
        bands_data_ds.attrs.update(bands_data.attrs)

        return cls.new(bands_data_ds)

    @new.register
    @classmethod
    def from_path(cls, path: Path, *args, **kwargs):
        """Creates a sile from the path and tries to read the bands from it.

        Parameters
        ----------
        path:
            The path to the file to read the bands from.

            Depending of the sile extracted from the path, the corresponding `BandsData` constructor
            will be called.
        **kwargs:
            Extra arguments to be passed to the `BandsData` constructor.
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
        cls, fdf: fdfSileSiesta, bands_file: Union[str, bandsSileSiesta, None] = None
    ):
        """Gets the bands data from a SIESTA .bands file.

        Parameters
        ----------
        fdf:
            The fdf file that was used to run the calculation.
        bands_file:
            Path to the bands file. If `None`, it will be assumed that the bands file
            is in the same directory as the fdf file and has the name `<SystemLabel>.bands`,
            with SystemLabel being retrieved from the fdf file.
        """
        bands_file = FileDataSIESTA(
            fdf=fdf, path=bands_file, cls=sisl.io.bandsSileSiesta
        )

        assert isinstance(bands_file, bandsSileSiesta)

        return cls.new(bands_file)

    @new.register
    @classmethod
    def from_siesta_bands(cls, bands_file: bandsSileSiesta):
        """Gets the bands data from a SIESTA .bands file

        Parameters
        ----------
        bands_file:
            The bands file to read the data from.
        """

        bands_data = bands_file.read_data(as_dataarray=True)
        bands_data.k.attrs["axis"] = {
            "tickvals": bands_data.attrs.pop("ticks"),
            "ticktext": bands_data.attrs.pop("ticklabels"),
        }

        return cls.new(bands_data)

    @new.register
    @classmethod
    def from_hamiltonian(
        cls,
        bz: sisl.BrillouinZone,
        H: Union[sisl.Hamiltonian, None] = None,
        extra_vars: Sequence[Union[dict, str]] = (),
    ):
        """Uses a sisl's `BrillouinZone` object to calculate the bands.

        It computes the eigenvalues of the Hamiltonian at each k point in the Brillouin zone

        Parameters
        ----------
        bz:
            The Brillouin zone object containing the k points to use for calculating
            the bands. This will most likely be a `BandStructure` object.
        H:
            The Hamiltonian to use for the calculations. If `None`, the parent of the
            Brillouin zone will be used, which is typically what you want!
        extra_vars:
            Additional variables to calculate for each eigenstate, apart from their energy.
            Each item of the list should be a dictionary with the following keys:
            * 'name', str: The name of the variable.
            * 'getter', callable: A function that gets 3 arguments: eigenstate, plot and
              spin index, and returns the values of the variable in a numpy array. This
              function will be called for each eigenstate object separately. That is, once
              for each (k-point, spin) combination.
            * 'coords', tuple of str: The names of the  dimensions of the returned array.
              The number of coordinates should match the number of dimensions.
            * 'coords_values', dict: If this variable introduces a new coordinate, you should
              pass the values for that coordinate here. If the coordinates were already defined
              by another variable, they will already have values. If you are unsure that the
              coordinates are new, just pass the values for them, they will get overwritten.

            Each item can also be a string indicating the name of a known variable: 'norm2', 'spin_moment', 'ipr'.
        """
        if bz is None:
            raise ValueError("No band structure (k points path) was provided")

        if not isinstance(getattr(bz, "parent", None), sisl.Hamiltonian):
            H = HamiltonianDataSource(H=H)
            bz.set_parent(H)
        else:
            H = bz.parent

        # Define the spin class of this calculation.
        spin = H.spin

        if isinstance(bz, sisl.BandStructure):
            ticks = bz.lineartick()
            kticks = bz.lineark()
        else:
            ticks = (None, None)
            kticks = np.arange(0, len(bz))

        # Get the wrapper function that we should call on each eigenstate.
        # This also returns the coordinates and names to build the final dataset.
        bands_wrapper, all_vars, coords_values = _get_eigenstate_wrapper(
            kticks, spin, extra_vars=extra_vars
        )

        # Get a dataset with all values for all spin indices
        spin_datasets = []
        coords = [var["coords"] for var in all_vars]
        name = [var["name"] for var in all_vars]
        for spin_index in coords_values["spin"]:
            # Non collinear routines don't accept the keyword argument "spin"
            spin_kwarg = {"spin": spin_index}
            if not spin.is_diagonal:
                spin_kwarg = {}

            with bz.apply.renew(zip=True) as parallel:
                spin_bands = parallel.dataarray.eigenstate(
                    wrap=partial(bands_wrapper, spin_index=spin_index),
                    **spin_kwarg,
                    coords=coords,
                    name=name,
                )

            spin_datasets.append(spin_bands)

        # Merge everything into a single dataset with a spin dimension
        bands_data = xr.concat(spin_datasets, "spin").assign_coords(coords_values)

        # If the band structure contains discontinuities, we will copy the dataset
        # adding the discontinuities.
        if isinstance(bz, sisl.BandStructure) and len(bz._jump_idx) > 0:
            old_coords = bands_data.coords
            coords = {
                name: (
                    bz.insert_jump(old_coords[name])
                    if name == "k"
                    else old_coords[name].values
                )
                for name in old_coords
            }

            def _add_jump(array):
                if "k" in array.coords:
                    array = array.transpose("k", ...)
                    return (array.dims, bz.insert_jump(array))
                else:
                    return array

            bands_data = xr.Dataset(
                {name: _add_jump(bands_data[name]) for name in bands_data},
                coords=coords,
                attrs=bands_data.attrs,
            )

        # Add the spin class of the data
        bands_data.attrs["spin"] = spin

        # Inform of where to place the ticks
        bands_data.k.attrs["axis"] = {
            "tickvals": ticks[0],
            "ticktext": ticks[1],
        }

        return cls.new(bands_data)

    @new.register
    @classmethod
    def from_wfsx(
        cls, wfsx_file: wfsxSileSiesta, fdf: str, extra_vars=(), need_H=False
    ):
        """Plots bands from the eigenvalues contained in a WFSX file.

        It also needs to get a geometry.

        Parameters
        ----------
        wfsx_file:
            The WFSX file to read the eigenstates from.
        fdf:
            Path to the fdf file used to run the calculation. Needed to gather
            information about the geometry and the hamiltonian/overlap if needed.
        extra_vars:
            Additional variables to calculate for each eigenstate, apart from their energy.

            Each item of the list should be a dictionary with the following keys:
            * 'name', str: The name of the variable.
            * 'getter', callable: A function that gets 3 arguments: eigenstate, plot and
              spin index, and returns the values of the variable in a numpy array. This
              function will be called for each eigenstate object separately. That is, once
              for each (k-point, spin) combination.
            * 'coords', tuple of str: The names of the  dimensions of the returned array.
              The number of coordinates should match the number of dimensions.
            * 'coords_values', dict: If this variable introduces a new coordinate, you should
              pass the values for that coordinate here. If the coordinates were already defined
              by another variable, they will already have values. If you are unsure that the
              coordinates are new, just pass the values for them, they will get overwritten.

            Each item can also be a string indicating the name of a known variable: 'norm2', 'spin_moment', 'ipr'.
        need_H:
            Whether the Hamiltonian is needed to read the WFSX file.
        """
        if need_H:
            H = HamiltonianDataSource(H=fdf)
            if H is None:
                raise ValueError(
                    "Hamiltonian was not setup, and it is needed for the calculations"
                )
            parent = H
            geometry = parent.geometry
        else:
            # Get the fdf sile
            fdf = FileDataSIESTA(path=fdf)
            # Read the geometry from the fdf sile
            geometry = fdf.read_geometry(output=True)
            parent = geometry

        # Get the wfsx file
        wfsx_sile = FileDataSIESTA(
            fdf=fdf, path=wfsx_file, cls=sisl.io.wfsxSileSiesta, parent=parent
        )

        # Now read all the information of the k points from the WFSX file
        k, weights, nwfs = wfsx_sile.read_info()
        # Get the number of wavefunctions in the file while performing a quick check
        nwf = np.unique(nwfs)
        if len(nwf) > 1:
            raise ValueError(
                f"File {wfsx_sile.file} contains different number of wavefunctions in some k points"
            )
        nwf = nwf[0]
        # From the k values read in the file, build a brillouin zone object.
        # We will use it just to get the linear k values for plotting.
        bz = BrillouinZone(geometry, k=k, weight=weights)

        # Read the sizes of the file, which contain the number of spin channels
        # and the number of orbitals and the number of k points.
        nspin, nou, nk, _ = wfsx_sile.read_sizes()

        # Find out the spin class of the calculation.
        spin = Spin(
            {
                1: Spin.UNPOLARIZED,
                2: Spin.POLARIZED,
                4: Spin.NONCOLINEAR,
                8: Spin.SPINORBIT,
            }[nspin]
        )
        # Now find out how many spin channels we need. Note that if there is only
        # one spin channel there will be no "spin" dimension on the final dataset.
        nspin = 2 if spin.is_polarized else 1

        # Determine whether spin moments will be calculated.
        spin_moments = False
        if not spin.is_diagonal:
            # We need to set the parent
            try:
                H = sisl.get_sile(fdf).read_hamiltonian()
                if H is not None:
                    # We could read a hamiltonian, set it as the parent of the wfsx sile
                    wfsx_sile = FileDataSIESTA(
                        path=wfsx_sile.file, kwargs=dict(parent=parent)
                    )
                    spin_moments = True
            except:
                pass

        # Get the wrapper function that we should call on each eigenstate.
        # This also returns the coordinates and names to build the final dataset.
        bands_wrapper, all_vars, coords_values = _get_eigenstate_wrapper(
            sisl.physics.linspace_bz(bz),
            extra_vars=extra_vars,
            spin_moments=spin_moments,
            spin=spin,
        )
        # Make sure all coordinates have values so that we can assume the shape
        # of arrays below.
        coords_values["band"] = np.arange(0, nwf)
        coords_values["orb"] = np.arange(0, nou)

        # Initialize all the arrays. For each quantity we will initialize
        # an array of the needed shape.
        arrays = {}
        for var in all_vars:
            # These are all the extra dimensions of the quantity. Note that a
            # quantity does not need to have extra dimensions.
            extra_shape = [len(coords_values[coord]) for coord in var["coords"]]
            # First two dimensions will always be the spin channel and the k index.
            # Then add potential extra dimensions.
            shape = (nspin, len(bz), *extra_shape)
            # Initialize the array.
            arrays[var["name"]] = np.empty(shape, dtype=var.get("dtype", np.float64))

        # Loop through eigenstates in the WFSX file and add their contribution to the bands
        ik = -1
        for eigenstate in wfsx_sile.yield_eigenstate():
            i_spin = eigenstate.info.get("spin", 0)
            # Every time we encounter spin 0, we are in a new k point.
            if i_spin == 0:
                ik += 1
                if ik == 0:
                    # If this is the first eigenstate we read, get the wavefunction
                    # indices. We will assume that ALL EIGENSTATES have the same indices.
                    # Note that we already checked previously that they all have the same
                    # number of wfs, so this is a fair assumption.
                    coords_values["band"] = eigenstate.info["index"]

            # Get all the values for this eigenstate.
            returns = bands_wrapper(eigenstate, spin_index=i_spin)
            # And store them in the respective arrays.
            for var, vals in zip(all_vars, returns):
                arrays[var["name"]][i_spin, ik] = vals

        # Now that we have all the values, just build the dataset.
        bands_data = xr.Dataset(
            data_vars={
                var["name"]: (("spin", "k", *var["coords"]), arrays[var["name"]])
                for var in all_vars
            }
        ).assign_coords(coords_values)

        bands_data.attrs = {"parent": bz, "spin": spin, "geometry": geometry}

        return cls.new(bands_data)

    @new.register
    @classmethod
    def from_aiida(cls, aiida_bands: Aiida_node):
        """Creates the bands plot reading from an aiida BandsData node.

        Parameters
        ----------
        aiida_bands:
            The aiida node containing the bands data (a BandsData aiida node).
        """
        plot_data = aiida_bands._get_bandplot_data(cartesian=True)
        bands = plot_data["y"]

        # Expand the bands array to have an extra dimension for spin
        if bands.ndim == 2:
            bands = np.expand_dims(bands, 0)

        # Get the info about where to put the labels
        tick_info = defaultdict(list)
        for tick, label in plot_data["labels"]:
            tick_info["tickvals"].append(tick)
            tick_info["ticktext"].append(label)

        # Construct the dataarray
        data = xr.DataArray(
            bands,
            coords={
                "spin": np.arange(0, bands.shape[0]),
                "k": ("k", plot_data["x"], {"axis": tick_info}),
                "band": np.arange(0, bands.shape[2]),
            },
            dims=("spin", "k", "band"),
        )

        return cls.new(data)


def _get_eigenstate_wrapper(
    k_vals, spin, extra_vars: Sequence[Union[dict, str]] = (), spin_moments: bool = True
):
    """Helper function to build the function to call on each eigenstate.

    Parameters
    ----------
    k_vals: array_like of shape (nk,)
        The (linear) values of the k points. This will be used for plotting
        the bands.
    extra_vars: array-like of dict, optional
        This argument determines the extra quantities that should be included
        in the final dataset of the bands. Energy and spin moments (if available)
        are already included, so no need to pass them here.
        Each item of the array defines a new quantity and should contain a dictionary
        with the following keys:
        * 'name', str: The name of the quantity.
        * 'getter', callable: A function that gets 3 arguments: eigenstate, plot and
          spin index, and returns the values of the quantity in a numpy array. This
          function will be called for each eigenstate object separately. That is, once
          for each (k-point, spin) combination.
        * 'coords', tuple of str: The names of the  dimensions of the returned array.
          The number of coordinates should match the number of dimensions.
        * 'coords_values', dict: If this variable introduces a new coordinate, you should
          pass the values for that coordinate here. If the coordinates were already defined
          by another variable, they will already have values. If you are unsure that the
          coordinates are new, just pass the values for them, they will get overwritten.
    spin_moments:
        Whether to add, if the spin is not diagonal, spin moments.

    Returns
    --------
    function:
        The function that should be called for each eigenstate and will return a tuple of size
        n_vars with the values for each variable.
    tuple of dicts:
        A tuple containing the dictionaries that define all variables. Exactly the same as
        the passed `extra_vars`, but with the added Energy and spin moment (if available) variables.
    dict:
        Dictionary containing the values for each coordinate involved in the dataset.
    """
    # In case it is a non_colinear or spin-orbit calculation we will get the spin moments
    if spin_moments and not spin.is_diagonal:
        extra_vars = ("spin_moment", *extra_vars)

    # Define the available spin indices. Notice that at the end the spin dimension
    # is removed from the dataset unless the calculation is spin polarized. So having
    # spin_indices = [0] is just for convenience.
    spin_indices = [0]
    if spin.is_polarized:
        spin_indices = [0, 1]

    # Add a variable to get the eigenvalues.
    all_vars = (
        {
            "coords": ("band",),
            "coords_values": {"spin": spin_indices, "k": k_vals},
            "name": "E",
            "getter": lambda eigenstate, spin, spin_index: eigenstate.eig,
        },
        "ipr",
        *extra_vars,
    )

    # Convert known variable keys to actual variables.
    all_vars = tuple(
        _KNOWN_EIGENSTATE_VARS[var] if isinstance(var, str) else var for var in all_vars
    )

    # Now build the function that will be called for each eigenstate and will
    # return the values for each variable.
    def bands_wrapper(eigenstate, spin_index):
        return tuple(var["getter"](eigenstate, spin, spin_index) for var in all_vars)

    # Finally get the values for all coordinates involved.
    coords_values = {}
    for var in all_vars:
        coords_values.update(var.get("coords_values", {}))

    return bands_wrapper, all_vars, coords_values


def _norm2_getter(eigenstate, spin, spin_index):
    norm2 = eigenstate.norm2(projection="orbital")

    if not spin.is_diagonal:
        # If it is a non-colinear or spin orbit calculation, we have two weights for each
        # orbital (one for each spin component of the state), so we just pair them together
        # and sum their contributions to get the weight of the orbital.
        norm2 = norm2.reshape(len(norm2), -1, 2).sum(2)

    return norm2.real


def _spin_moment_getter(eigenstate, spin, spin_index):
    return eigenstate.spin_moment().real


def _ipr_getter(eigenstate, spin, spin_index):
    return eigenstate.ipr()


_KNOWN_EIGENSTATE_VARS = {
    "norm2": {
        "coords": ("band", "orb"),
        "name": "norm2",
        "getter": _norm2_getter,
    },
    "spin_moment": {
        "coords": ("axis", "band"),
        "coords_values": dict(axis=["x", "y", "z"]),
        "name": "spin_moments",
        "getter": _spin_moment_getter,
    },
    "ipr": {
        "coords": ("band",),
        "name": "ipr",
        "getter": _ipr_getter,
    },
}
