from collections import defaultdict

import numpy as np
from pathlib import Path
from plotly.colors import DEFAULT_PLOTLY_COLORS

import sisl
from ..plot import Plot, entry_point
from .bands import BandsPlot
from ..plotutils import random_color
from ..input_fields import OrbitalQueries, TextInput, DropdownInput, SwitchInput, ColorPicker, FloatInput, SileInput
from ..input_fields.range import ErangeInput


class FatbandsPlot(BandsPlot):
    """
    Colorful representation of orbital weights in bands.

    Parameters
    -------------
    wfsx_file: wfsxSileSiesta, optional
        The WFSX file to get the weights of the different orbitals in the
        bands.             In standard SIESTA nomenclature, this should be
        the *.bands.WFSX file, as it is the one             that contains the
        weights that correspond to the bands.                          This
        file is only meaningful (and required) if fatbands are plotted from
        the .bands file.             Otherwise, the bands and weights will be
        generated from the hamiltonian by sisl.              If the *.bands
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
        The BandStructure object to be used.
    aiida_bands:  optional
                     An aiida BandsData node.
    add_band_trace_data:  optional
        A function that receives each band (as a DataArray) and adds data to
        the trace. It also recieves the plot object.              The
        returned data may even overwrite the existing one, therefore it can
        be useful to fully customize your bands plot (individual style for
        each band if you want).
    eigenstate_map:  optional
        This function receives the eigenstate object for each k value when
        the bands are being extracted from a hamiltonian.             You can
        do whatever you want with it, the point of this function is to avoid
        running the diagonalization process twice.
    Erange: array-like of shape (2,), optional
        Energy range where the bands are displayed.
    E0: float, optional
        The energy to which all energies will be referenced (including
        Erange).
    bands_range: array-like of shape (2,), optional
        The bands that should be displayed. Only relevant if Erange is None.
    path: array-like of dict, optional
        Path along which bands are drawn in units of reciprocal lattice
        vectors.             Note that if you want to provide a path
        programatically you can do it more easily with the `band_structure`
        setting   Each item is a dict. Structure of the expected dicts:{
        'x':          'y':          'z':          'divisions':
        'tick': Tick that should be displayed at this corner of the path. }
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
    """

    _plot_type = 'Fatbands'

    _update_methods = {
        "read_data": [],
        "set_data": ["_draw_gaps", "_draw_fatbands"],
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

    @entry_point("siesta output")
    def _read_siesta_output(self, wfsx_file, bands_file, root_fdf):
        """
        Generates fatbands from SIESTA output.

        Uses the `.bands` file to read the bands and a `.wfsx` file to
        retrieve the wavefunctions coefficients. 
        """
        from xarray import DataArray

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
        self.weights = DataArray(
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
        self.weights = self.weights.expand_dims("spin")

        # Set up the options for the 'groups' setting based on the plot's associated geometry
        self._set_group_options()

    @entry_point("hamiltonian")
    def _read_from_H(self):
        """
        Calculates the fatbands from a sisl hamiltonian.
        """
        from xarray import DataArray

        self.weights = [[], []]

        # Define the function that will "catch" each eigenstate and
        # build the weights array. See BandsPlot._read_from_H to understand where
        # this will go exactly
        def _weights_from_eigenstate(eigenstate, plot, spin_index):

            weights = eigenstate.norm2(sum=False)

            if plot.spin.spins > 2:
                # If it is a non-colinear or spin orbit calculation, we have two weights for each
                # orbital (one for each spin component of the state), so we just pair them together
                # and sum their contributions to get the weight of the orbital.
                weights = weights.reshape(len(weights), -1, 2).sum(2)

            plot.weights[spin_index].append(weights)

        # We make bands plot read the bands, which will also populate the weights
        # thanks to the above step
        bands_read = False; err = None
        try:
            super()._read_from_H(eigenstate_map=_weights_from_eigenstate)
            bands_read = True
        except Exception as e:
            # Let's keep this error, we are going to at least set the group options so that the
            # user knows what can they choose (specially important for the GUI)
            err = e

        self._set_group_options()
        if not bands_read:
            raise err

        # If there was only one spin component then we just take the first item in self.weights
        if not self.weights[1]:
            self.weights = [self.weights[0]]

        # Then we just convert the weights to a DataArray
        self.weights = np.array(self.weights).real

        self.weights = DataArray(
            self.weights,
            coords={
                "k": self.bands.k,
                "spin": np.arange(self.weights.shape[0]),
                "band": np.arange(self.weights.shape[2]),
                "orb": np.arange(self.weights.shape[3]),
            },
            dims=("spin", "k", "band", "orb")
        )

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

        # We let the bands plot draw the bands
        super()._set_data(draw_before_bands=self._draw_fatbands,
                         # Avoid bands being displayed in the legend individually (it would be a mess)
                         add_band_trace_data=lambda band, plot: {'showlegend': False}
        )

    def _draw_fatbands(self, groups, E0, bands_range, scale):

        # We get the bands range that is going to be plotted
        # Remember that the BandsPlot will have updated this setting accordingly,
        # so it's safe to use it directly
        plotted_bands = bands_range

        # If we don't have weights for all plotted bands (specially possible if we
        # have read from the WFSX file), reduce the range of the bands. Note that this
        # does not affect the bands displayed, just the "fatbands".
        plotted_bands[0] = max(self.weights.band.values.min(), plotted_bands[0])
        plotted_bands[1] = min(self.weights.band.values.max(), plotted_bands[1])

        # Get the bands that matter
        plot_eigvals = self.bands.sel(band=np.arange(*plotted_bands)) - E0
        # Get the weights that matter
        plot_weights = self.weights.sel(band=np.arange(*plotted_bands))

        # If the user didn't provide groups and didn't specify that groups is an
        # empty list, we are going to build the default groups, which is to split by species
        # in case there is more than one species or else, by orbitals
        if groups is None:
            if len(self.geometry.atoms.atom) > 1:
                group_by = 'species'
            else:
                group_by = 'orbitals'

            return self.split_groups(group_by)

        # We are going to need a trace that goes forward and then back so that
        # it is self-fillable
        xs = self.bands.k.values
        area_xs = [*xs, *np.flip(xs)]

        prev_traces = len(self.data)

        groups_param = self.get_param("groups")
        if scale is None:
            # Probably we can calculate a more suitable scale
            scale = 1

        # Let's plot each group of orbitals as an area that surrounds each band
        for i, group in enumerate(groups):

            group = groups_param.complete_query(group, name=f"Group {i+1}")

            #Use only the active requests
            if not group.get("active", True):
                continue

            orb = groups_param.get_orbitals(group)

            # Get the weights for the requested orbitals
            weights = plot_weights.sel(orb=orb)

            # Now get a particular spin component if the user wants it
            if group["spin"] is not None:
                weights = weights.sel(spin=group["spin"])

            if group["normalize"]:
                weights = weights.mean("orb")
            else:
                weights = weights.sum("orb")

            if group["color"] is None:
                group["color"] = random_color()

            for ispin, (spin_eigvals, spin_weights) in enumerate(zip(plot_eigvals.transpose("spin", "band", "k"), weights.transpose("spin", "band", "k"))):

                self.add_traces([{
                    "type": "scatter",
                    "mode": "lines",
                    "x": area_xs,
                    "y": [*(band + band_weights*scale*group["scale"]), *np.flip(band - band_weights*scale*group["scale"])],
                    "line":{"width": 0, "color": group["color"]},
                    "showlegend": i == 0 and ispin == 0,
                    "name": group["name"],
                    "legendgroup":group["name"],
                    "fill": "toself"
                } for i, (band, band_weights) in enumerate(zip(spin_eigvals, spin_weights))])

    # -------------------------------------
    #         Convenience methods
    # -------------------------------------

    def split_groups(self, on="species", only=None, exclude=None, clean=True, colors=DEFAULT_PLOTLY_COLORS, **kwargs):
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
        """
        Scales all bands by a given factor.

        Basically, it updates 'scale' setting.

        Parameters
        -----------
        factor: float
            the factor that should be used to scale.
        from_current: boolean, optional
            whether 'factor' is meant to multiply the current scaling factor.
            If False, it will just replace the current factor.
        """
        scale = self.get_setting('scale')

        if from_current:
            scale *= factor
        else:
            scale = factor

        return self.update_settings(scale=scale)
