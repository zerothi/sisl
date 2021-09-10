# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

import sisl
from ..plot import entry_point
from .bands import BandsPlot
from ..plotutils import random_color
from ..input_fields import OrbitalQueries, TextInput, SwitchInput, ColorPicker, FloatInput, SileInput


class FatbandsPlot(BandsPlot):
    """Colorful representation of orbital weights in bands.

    Parameters
    -------------
    wfsx_file: wfsxSileSiesta, optional
        The WFSX file to get the weights of the different orbitals in the
        bands.             In standard SIESTA nomenclature, this should be
        the *.bands.WFSX file, as it is the one             that contains the
        weights that correspond to the bands.                          This
        file is only meaningful (and required) if fatbands are plotted from
        the .bands file.             Otherwise, the bands and weights will be
        generated from the hamiltonian by sisl.             If the *.bands
        file is provided but the wfsx one isn't, we will try to find it.
        If `bands_file` is SystemLabel.bands, we will look for
        SystemLabel.bands.WFSX
    scale: float, optional
        The factor by which the width of all fatbands should be multiplied.
        Note that each group has an additional individual factor that you can
        also tweak.
    groups: array-like of dict, optional
        The different groups that are displayed in the fatbands   Each item
        is a dict. Structure of the expected dicts:{         'name':
        'species':          'atoms':          'orbitals':          'spin':
        'normalize':          'color':          'scale':  }
    bands_file: bandsSileSiesta, optional
        This parameter explicitly sets a .bands file. Otherwise, the bands
        file is attempted to read from the fdf file
    band_structure: BandStructure, optional
        A band structure. it can either be provided as a sisl.BandStructure
        object or         as a list of points, which will be parsed into a
        band structure object.            Each item is a dict. Structure of
        the expected dicts:{         'x':          'y':          'z':
        'divisions':          'name': Tick that should be displayed at this
        corner of the path. }
    aiida_bands:  optional
        An aiida BandsData node.
    eigenstate_map:  optional
        This function receives the eigenstate object for each k value when
        the bands are being extracted from a hamiltonian.             You can
        do whatever you want with it, the point of this function is to avoid
        running the diagonalization process twice.
    add_band_data:  optional
        This function receives each band and should return a dictionary with
        additional arguments              that are passed to the band drawing
        routine. It also receives the plot as the second argument.
        See the docs of `sisl.viz.backends.templates.Backend.draw_line` to
        understand what are the supported arguments             to be
        returned. Notice that the arguments that the backend is able to
        process can be very framework dependant.
    Erange: array-like of shape (2,), optional
        Energy range where the bands are displayed.
    E0: float, optional
        The energy to which all energies will be referenced (including
        Erange).
    bands_range: array-like of shape (2,), optional
        The bands that should be displayed. Only relevant if Erange is None.
    spin:  optional
        Determines how the different spin configurations should be displayed.
        In spin polarized calculations, it allows you to choose between spin
        0 and 1.             In non-colinear spin calculations, it allows you
        to ask for a given spin texture,             by specifying the
        direction.
    spin_texture_colorscale: str, optional
        The plotly colorscale to use for the spin texture (if displayed)
    gap: bool, optional
        Whether the gap should be displayed in the plot
    direct_gaps_only: bool, optional
        Whether to show only gaps that are direct, according to the gap
        tolerance
    gap_tol: float, optional
        The difference in k that must exist to consider to gaps
        different.             If two gaps' positions differ in less than
        this, only one gap will be drawn.             Useful in cases
        where there are degenerated bands with exactly the same values.
    gap_color: str, optional
        Color to display the gap
    custom_gaps: array-like of dict, optional
        List of all the gaps that you want to display.   Each item is a dict.
        Structure of the expected dicts:{         'from': K value where to
        start measuring the gap.                      It can be either the
        label of the k-point or the numeric value in the plot.         'to':
        K value where to end measuring the gap.                      It can
        be either the label of the k-point or the numeric value in the plot.
        'color': The color with which the gap should be displayed
        'spin': The spin components where the gap should be calculated. }
    bands_width: float, optional
        Width of the lines that represent the bands
    bands_color: str, optional
        Choose the color to display the bands.  This will be used for the
        spin up bands if the calculation is spin polarized
    spindown_color: str, optional
        Choose the color for the spin down bands.Only used if the
        calculation is spin polarized.
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    backend:  optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    _plot_type = 'Fatbands'

    _update_methods = {
        "read_data": [],
        "set_data": ["_draw_gaps", "_get_groups_weights"],
        "get_figure": []
    }

    _parameters = (

        SileInput(key='wfsx_file', name='Path to WFSX file',
            dtype=sisl.io.siesta.wfsxSileSiesta,
            default=None,
            help="""The WFSX file to get the weights of the different orbitals in the bands.
            In standard SIESTA nomenclature, this should be the *.bands.WFSX file, as it is the one
            that contains the weights that correspond to the bands.
            
            This file is only meaningful (and required) if fatbands are plotted from the .bands file.
            Otherwise, the bands and weights will be generated from the hamiltonian by sisl.
            If the *.bands file is provided but the wfsx one isn't, we will try to find it.
            If `bands_file` is SystemLabel.bands, we will look for SystemLabel.bands.WFSX
            """
        ),

        FloatInput(key='scale', name='Scale factor',
            default=None,
            help="""The factor by which the width of all fatbands should be multiplied.
            Note that each group has an additional individual factor that you can also tweak."""
            # Probably scale should not multiply but normalize everything relative to the energy range!
        ),

        OrbitalQueries(
            key="groups", name="Fatbands groups",
            default=None,
            help="""The different groups that are displayed in the fatbands""",
            queryForm=[

                TextInput(
                    key="name", name="Name",
                    default="Group",
                    width="s100% m50% l20%",
                    params={
                        "placeholder": "Name of the line..."
                    },
                ),

                'species', 'atoms', 'orbitals', 'spin',

                SwitchInput(
                    key="normalize", name="Normalize",
                    default=True,
                    params={
                        "offLabel": "No",
                        "onLabel": "Yes"
                    }
                ),

                ColorPicker(
                    key="color", name="Color",
                    default=None,
                ),

                FloatInput(
                    key="scale", name="Scale factor",
                    default=1,
                ),
            ]
        ),

    )

    @property
    def weights(self):
        return self.bands_data["weight"]

    @entry_point("siesta output")
    def _read_siesta_output(self, wfsx_file, bands_file, root_fdf):
        """Generates fatbands from SIESTA output.

        Uses the `.bands` file to read the bands and a `.wfsx` file to
        retrieve the wavefunctions coefficients. 
        """
        from xarray import DataArray, Dataset

        # Try to get the wfsx file either by user input or by guessing it
        # from bands_file
        bands_file = self.get_sile(bands_file or "bands_file").file
        if wfsx_file is None:
            wfsx_file = bands_file.with_suffix(bands_file.suffix + ".WFSX")

        # We will need the overlap matrix from the hamiltonian to get the correct
        # weights.
        # If there is no root_fdf we will try to guess it from bands_file
        if root_fdf is None and not hasattr(self, "H"):
            possible_fdf = bands_file.with_suffix(".fdf")
            print(f"We are assuming that the fdf associated to {bands_file} is {possible_fdf}."+
            ' If it is not, please provide a "root_fdf" by using the update_settings method.')
            self.update_settings(root_fdf=root_fdf, run_updates=False)

        self.setup_hamiltonian()

        # If the wfsx doesn't exist, we will not even bother to read the bands
        if not wfsx_file.exists():
            raise ValueError(f"We did not find a WFSX file in the location {wfsx_file}")

        # Otherwise we will make BandsPlot read the bands
        super()._read_siesta_output()

        # And then read the weights from the wfsx file
        wfsx_sile = self.get_sile(wfsx_file)

        weights = []
        for i, state in enumerate(wfsx_sile.yield_eigenstate(self.H)):
            # Each eigenstate represents all the states for a given k-point

            # Get the band indices to which these states correspond
            if i == 0:
                bands = state.info["indices"]

            # Get the weights for this eigenstate
            weights.append(state.norm2(sum=False))

        weights = np.array(weights).real

        # Finally, build the weights dataarray so that it can be used by _set_data
        weights = DataArray(
            weights,
            coords={
                "k": self.bands.k,
                "band": bands,
                "orb": np.arange(0, weights.shape[2]),
            },
            dims=("k", "band", "orb")
        )

        # Add the spin dimension so that the weights array is normalized,
        # even though spin is not yet supported by this entrypoint
        weights = weights.expand_dims("spin")

        # Merge everything into a dataset
        attrs = self.bands_data.attrs
        self.bands_data = Dataset({"E": self.bands_data, "weight": weights})
        self.bands_data.attrs = attrs

        # Set up the options for the 'groups' setting based on the plot's associated geometry
        self._set_group_options()

    @entry_point("hamiltonian")
    def _read_from_H(self):
        """
        Calculates the fatbands from a sisl hamiltonian.
        """
        # Define the function that will "catch" each eigenstate and
        # build the weights array. See BandsPlot._read_from_H to understand where
        # this will go exactly
        def _weights_from_eigenstate(eigenstate, plot, spin_index):

            weights = eigenstate.norm2(sum=False)

            if not plot.spin.is_diagonal:
                # If it is a non-colinear or spin orbit calculation, we have two weights for each
                # orbital (one for each spin component of the state), so we just pair them together
                # and sum their contributions to get the weight of the orbital.
                weights = weights.reshape(len(weights), -1, 2).sum(2)

            return weights.real

        # We make bands plot read the bands, which will also populate the weights
        # thanks to the above step
        bands_read = False; err = None
        try:
            super()._read_from_H(extra_vars=[{"coords": ("band", "orb"), "name": "weight", "getter": _weights_from_eigenstate}])
            bands_read = True
        except Exception as e:
            # Let's keep this error, we are going to at least set the group options so that the
            # user knows what can they choose (specially important for the GUI)
            err = e

        self._set_group_options()
        if not bands_read:
            raise err

    def _set_group_options(self):

        # Try to find a geometry if there isn't already one
        if not hasattr(self, "geometry"):

            # From the hamiltonian
            band_struct = self.get_setting("band_structure")
            if band_struct is not None:
                self.geometry = band_struct.parent.geometry

        if getattr(self, "H", None) is not None:
            spin = self.H.spin
        else:
            # There is yet no spin support reading from bands.WFSX
            spin = sisl.Spin.UNPOLARIZED

        self.get_param('groups').update_options(self.geometry, spin)

    def _set_data(self):
        # We get the information that the Bandsplot wants to send to the drawer
        from_bandsplot = super()._set_data()

        # And add some extra information related to the weights.
        return {
            **from_bandsplot,
            **self._get_groups_weights()
        }

    def _get_groups_weights(self, groups, E0, bands_range, scale):
        """Returns a dictionary with information about all the weights that have been requested
        The return of this function is expected to be passed to the drawers.
        """
        # We get the bands range that is going to be plotted
        # Remember that the BandsPlot will have updated this setting accordingly,
        # so it's safe to use it directly
        min_band, max_band = bands_range

        # Get the weights that matter
        plot_weights = self.weights.sel(band=slice(min_band, max_band))

        if groups is None:
            groups = ()

        if scale is None:
            # Probably we can calculate a more suitable scale
            scale = 1

        groups_weights = {}
        groups_metadata = {}
        # Here we get the values of the weights for each group of orbitals.
        for i, group in enumerate(groups):
            group = {**group}

            # Use only the active requests
            if not group.get("active", True):
                continue

            # Give a name to the request in case it didn't have one.
            if group.get("name") is None:
                group["name"] = f"Group {i}"

            # Multiply the groups' scale by the global scale
            group["scale"] = group.get("scale", 1) * scale

            # Get the weight values for the request and store them to send to the drawer
            self._get_group_weights(group, plot_weights, values_storage=groups_weights, metadata_storage=groups_metadata)

        return {"groups_weights": groups_weights, "groups_metadata": groups_metadata}

    def _get_group_weights(self, group, weights=None, values_storage=None, metadata_storage=None):
        """Extracts the weight values that correspond to a specific fatbands request.
        Parameters
        --------------
        group: dict
            the request to process.
        weights: DataArray, optional
            the part of the weights dataarray that falls in the energy range that we want to draw.
            If not provided, the full weights data stored in `self.weights` is used.
        values_storage: dict, optional
            a dictionary where the weights values will be stored using the request's name as the key.
        metadata_storage: dict, optional
            a dictionary where metadata for the request will be stored using the request's name as the key.
        Returns
        ----------
        xarray.DataArray
            The weights resulting from the request. They are indexed by spin, band and k value.
        """

        if weights is None:
            weights = self.weights
        if "spin" not in weights.coords:
            weights = weights.expand_dims("spin")

        groups_param = self.get_param("groups")

        group = groups_param.complete_query(group)

        orb = groups_param.get_orbitals(group)

        # Get the weights for the requested orbitals
        weights = weights.sel(orb=orb)

        # Now get a particular spin component if the user wants it
        if group["spin"] is not None:
            weights = weights.sel(spin=group["spin"])

        if group["normalize"]:
            weights = weights.mean("orb")
        else:
            weights = weights.sum("orb")

        if group["color"] is None:
            group["color"] = random_color()

        group_name = group["name"]
        values = weights.transpose("spin", "band", "k") * group["scale"]

        if values_storage is not None:
            if group_name in values_storage:
                raise ValueError(f"There are multiple groups that are named '{group_name}'")
            values_storage[group_name] = values

        if metadata_storage is not None:
            # Build the dictionary that contains metadata for this group.
            metadata = {
                "style": {
                    "line": {"color": group["color"]}
                }
            }

            metadata_storage[group_name] = metadata

        return values

    # -------------------------------------
    #         Convenience methods
    # -------------------------------------

    def split_groups(self, on="species", only=None, exclude=None, clean=True, colors=(), **kwargs):
        """
        Builds groups automatically to draw their contributions.
        Works exactly the same as `PdosPlot.split_DOS`
        Parameters
        --------
        on: str, {"species", "atoms", "Z", "orbitals", "n", "l", "m", "zeta", "spin"} or list of str
            the parameter to split along.
            Note that you can combine parameters with a "+" to split along multiple parameters
            at the same time. You can get the same effect also by passing a list.
        only: array-like, optional
            if desired, the only values that should be plotted out of
            all of the values that come from the splitting.
        exclude: array-like, optional
            values that should not be plotted
        clean: boolean, optional
            whether the plot should be cleaned before drawing.
            If False, all the groups that come from the method will
            be drawn on top of what is already there.
        colors: array-like, optional
            A list of colors to be used. There can be more colors than
            needed, or less. If there are less colors than groups, the colors
            will just be repeated.
        **kwargs:
            keyword arguments that go directly to each request.
            This is useful to add extra filters. For example:
            `plot.split_groups(on="orbitals", species=["C"])`
            will split the PDOS on the different orbitals but will take
            only those that belong to carbon atoms.
        Examples
        -----------
        >>> plot = H.plot.fatbands()
        >>>
        >>> # Split the fatbands in n and l but show only the fatbands from Au
        >>> # Also use "Au $ns" as a template for the name, where $n will
        >>> # be replaced by the value of n.
        >>> plot.split_groups(on="n+l", species=["Au"], name="Au $ns")
        """
        groups = self.get_param('groups')._generate_queries(
            on=on, only=only, exclude=exclude, **kwargs)

        if len(colors) > 0:
            # Repeat the colors in case there are more groups than colors
            colors = np.tile(colors, len(groups) // len(colors) + 1)

            # Asign colors
            for i, _ in enumerate(groups):
                groups[i]['color'] = colors[i]

        # If the user doesn't want to clean the plot, we will just add the groups to the existing ones
        if not clean:
            groups = [*self.get_setting("groups"), *groups]

        return self.update_settings(groups=groups)

    def scale_fatbands(self, factor, from_current=False):
        """Scales all bands by a given factor.
        Basically, it updates 'scale' setting.
        Parameters
        -----------
        factor: float
            the factor that should be used to scale.
        from_current: boolean, optional
            whether 'factor' is meant to multiply the current scaling factor.
            If False, it will just replace the current factor.
        """

        if from_current:
            scale = self.get_setting('scale') * factor
        else:
            scale = factor

        return self.update_settings(scale=scale)
