from collections import defaultdict
from functools import partial
import itertools

import numpy as np
import plotly.express as px

import sisl
from ..plot import Plot, entry_point
from ..plotutils import find_files
from ..input_fields import (
    TextInput, SwitchInput, ColorPicker, DropdownInput,
    IntegerInput, FloatInput, RangeInput, RangeSlider,
    QueriesInput, ProgramaticInput, FunctionInput, SileInput,
    PlotableInput, SpinSelect, AiidaNodeInput
)
from ..input_fields.range import ErangeInput


class BandsPlot(Plot):
    """
    Plot representation of the bands.

    Parameters
    -------------
    bands_file: bandsSileSiesta, optional
        This parameter explicitly sets a .bands file. Otherwise, the bands
        file is attempted to read from the fdf file
    band_structure: BandStructure, optional
        The BandStructure object to be used.
    aiida_bands : aiida.BandsData, optional
        the bands from an Aiida BandsData node
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

    _plot_type = "Bands"

    _parameters = (

        SileInput(key = "bands_file", name = "Path to bands file",
            width = "s100% m50% l33%",
            dtype=sisl.io.siesta.bandsSileSiesta,
            group="dataread",
            params = {
                "placeholder": "Write the path to your bands file here...",
            },
            help = """This parameter explicitly sets a .bands file. Otherwise, the bands file is attempted to read from the fdf file """
        ),

        PlotableInput(key = "band_structure", name = "Band structure object",
            default= None,
            dtype=sisl.BandStructure,
            help = "The BandStructure object to be used."
        ),

        AiidaNodeInput(key="aiida_bands", name="Aiida BandsData node",
            default=None,
            help="""
            An aiida BandsData node.
            """
        ),

        FunctionInput(key="add_band_trace_data", name="Additional data for band traces",
            default=None,
            positional=["band", "plot"],
            returns=[dict],
            help="""A function that receives each band (as a DataArray) and adds data to the trace. It also recieves the plot object. 
            The returned data may even overwrite the existing one, therefore it can be useful to fully customize your bands plot (individual style for each band if you want)."""
        ),

        FunctionInput(key="eigenstate_map", name="Eigenstate map function",
            default=None,
            positional=["eigenstate", "plot"],
            returns=[],
            help="""This function receives the eigenstate object for each k value when the bands are being extracted from a hamiltonian.
            You can do whatever you want with it, the point of this function is to avoid running the diagonalization process twice."""
        ),

        ErangeInput(key="Erange",
            help = "Energy range where the bands are displayed."
        ),

        FloatInput(key="E0", name="Reference energy",
            default=0,
            help="""The energy to which all energies will be referenced (including Erange)."""
        ),

        RangeSlider(key = "bands_range", name = "Bands range",
            default = None,
            width = "s90%",
            params = {
                'step': 1,
            },
            help = "The bands that should be displayed. Only relevant if Erange is None."
        ),

        QueriesInput(key = "path", name = "Bands path",
            default = [],
            help="""Path along which bands are drawn in units of reciprocal lattice vectors.<br>
            Note that if you want to provide a path programatically you can do it more easily with the `band_structure` setting""",
            queryForm=[

                FloatInput(
                    key="x", name="X",
                    width = "s50% m20% l10%",
                    default=0,
                    params={
                        "step": 0.01
                    }
                ),

                FloatInput(
                    key="y", name="Y",
                    width = "s50% m20% l10%",
                    default=0,
                    params={
                        "step": 0.01
                    }
                ),

                FloatInput(
                    key="z", name="Z",
                    width="s50% m20% l10%",
                    default=0,
                    params={
                        "step": 0.01
                    }
                ),

                IntegerInput(
                    key="divisions", name="Divisions",
                    width="s50% m20% l10%",
                    default=50,
                    params={
                        "min": 0,
                        "step": 10
                    }
                ),

                TextInput(
                    key="tick", name="Tick",
                    width = "s50% m20% l10%",
                    default=None,
                    params = {
                        "placeholder": "Tick..."
                    },
                    help = "Tick that should be displayed at this corner of the path."
                )

            ]
        ),

        SpinSelect(key="spin", name="Spin",
            default=None,
            help="""Determines how the different spin configurations should be displayed.
            In spin polarized calculations, it allows you to choose between spin 0 and 1.
            In non-colinear spin calculations, it allows you to ask for a given spin texture,
            by specifying the direction."""
        ),

        TextInput(key="spin_texture_colorscale", name="Spin texture colorscale",
            default=None,
            help="The plotly colorscale to use for the spin texture (if displayed)"
        ),

        SwitchInput(key="gap", name="Show gap",
            default=False,
            params={
                'onLabel': 'Yes',
                'offLabel': 'No'
            },
            help="Whether the gap should be displayed in the plot"
        ),

        SwitchInput(key="direct_gaps_only", name="Only direct gaps",
            default=False,
            params={
                'onLabel': 'Yes',
                'offLabel': 'No'
            },
            help="Whether to show only gaps that are direct, according to the gap tolerance"
        ),

        FloatInput(key="gap_tol", name="Gap tolerance",
            default=0.01,
            params={
                'step': 0.001
            },
            help="""The difference in k that must exist to consider to gaps different.<br>
            If two gaps' positions differ in less than this, only one gap will be drawn.<br>
            Useful in cases where there are degenerated bands with exactly the same values."""
        ),

        ColorPicker(key="gap_color", name="Gap color",
            default=None,
            help="Color to display the gap"
        ),

        QueriesInput(key="custom_gaps", name="Custom gaps",
            default=[],
            help="""List of all the gaps that you want to display.""",
            queryForm=[

                TextInput(
                    key="from", name="From",
                    help="""K value where to start measuring the gap. 
                    It can be either the label of the k-point or the numeric value in the plot.""",
                    width="s50% m20% l10%",
                    default="0",
                ),

                TextInput(
                    key="to", name="To",
                    help="""K value where to end measuring the gap. 
                    It can be either the label of the k-point or the numeric value in the plot.""",
                    width="s50% m20% l10%",
                    default="0",
                ),

                ColorPicker(
                    key="color", name="Line color",
                    help="The color with which the gap should be displayed",
                    width="s50% m20% l10%",
                    default=None,
                ),

                SpinSelect(
                    key="spin", name="Spin",
                    help="The spin components where the gap should be calculated.",
                    default=None,
                    only_if_polarized=True,
                ),

            ]
        ),

        FloatInput(key="bands_width", name="Band lines width",
            default=1,
            help="Width of the lines that represent the bands"
        ),

        ColorPicker(key = "bands_color", name = "No spin/spin up line color",
            default = "black",
            help = "Choose the color to display the bands. <br> This will be used for the spin up bands if the calculation is spin polarized"
        ),

        ColorPicker(key = "spindown_color", name = "Spin down line color",
            default = "blue",
            help = "Choose the color for the spin down bands.<br>Only used if the calculation is spin polarized."
        ),

    )

    _update_methods = {
        "read_data": [],
        "set_data": ["_draw_gaps"],
        "get_figure": []
    }

    _layout_defaults = {
        'xaxis_title': 'K',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'xaxis_showgrid': True,
        'yaxis_title': 'Energy [eV]'
    }

    @classmethod
    def _default_animation(cls, wdir=None, frame_names=None, **kwargs):
        """
        Defines the default animation, which is to look for all .bands files in wdir.
        """
        bands_files = find_files(wdir, "*.bands", sort = True)

        def _get_frame_names(self):

            return [childPlot.get_setting("bands_file").name for childPlot in self.child_plots]

        return cls.animated("bands_file", bands_files, frame_names = _get_frame_names, wdir = wdir, **kwargs)

    def _after_init(self):
        self.spin = sisl.Spin("")

        self.add_shortcut("g", "Toggle gap", self.toggle_gap)

    @entry_point('aiida bands')
    def _read_aiida_bands(self, aiida_bands):
        """
        Creates the bands plot reading from an aiida BandsData node.
        """
        import xarray as xr

        plot_data = aiida_bands._get_bandplot_data(cartesian=True)
        bands = plot_data["y"]

        # Expand the bands array to have an extra dimension for spin
        if bands.ndim == 2:
            bands = np.expand_dims(bands, 0)

        # Get the info about where to put the labels
        tick_info = defaultdict(list)
        for tick, label in plot_data["labels"]:
            tick_info["ticks"].append(tick)
            tick_info["ticklabels"].append(label)

        # Construct the dataarray
        self.bands = xr.DataArray(
            bands,
            coords={
                "spin": np.arange(0, bands.shape[0]),
                "k": plot_data["x"],
                "band": np.arange(0, bands.shape[2]),
            },
            dims=("spin", "k", "band"),
            attrs={**tick_info}
        )

    @entry_point('band structure')
    def _read_from_band_structure(self, band_structure, eigenstate_map):
        """
        Uses a sisl's `BandStructure` object to calculate the bands.
        """
        import xarray as xr

        if band_structure is None:
            raise ValueError("No band structure (k points path) was provided")

        if not isinstance(getattr(band_structure, "parent", None), sisl.Hamiltonian):
            self.setup_hamiltonian()
            band_structure.set_parent(self.H)
        else:
            self.H = band_structure.parent

        # Define the spin class of this calculation.
        self.spin = self.H.spin

        self.ticks = band_structure.lineartick()
        self.kPath = band_structure._k

        # We define a wrapper to get the values out of the eigenstates
        # to give the possibility to the user to do something inbetween
        # NOTE THAT THIS IS USED BY FAT BANDS TO GET THE WEIGHTS SIMULTANEOUSLY
        eig_map = eigenstate_map

        # Also, in this wrapper we will get the spin moments in case it is a non_colinear
        # calculation
        if self.spin.is_noncolinear:
            self.spin_moments = []
        elif hasattr(self, "spin_moments"):
            del self.spin_moments

        def bands_wrapper(eigenstate, spin_index):
            if callable(eig_map):
                eig_map(eigenstate, self, spin_index)
            if hasattr(self, "spin_moments"):
                self.spin_moments.append(eigenstate.spin_moment())
            return eigenstate.eig

        # Define the available spins
        spin_indices = [0]
        if self.spin.is_polarized:
            spin_indices = [0, 1]

        # Get the eigenstates for all the available spin components
        bands_arrays = []
        for spin_index in spin_indices:

            # Non collinear routines don't accept the keyword argument "spin"
            spin_kwarg = {"spin": spin_index}
            if self.spin.is_noncolinear:
                spin_kwarg = {}

            spin_bands = band_structure.apply.dataarray.eigenstate(
                wrap=partial(bands_wrapper, spin_index=spin_index),
                **spin_kwarg,
                coords=('band',),
            )

            bands_arrays.append(spin_bands)

        # Merge everything into a single dataarray with a spin dimension
        self.bands = xr.concat(bands_arrays, "spin").assign_coords({"spin": spin_indices}).transpose("k", "spin", "band")

        self.bands['k'] = band_structure.lineark()
        self.bands.attrs = {"ticks": self.ticks[0], "ticklabels": self.ticks[1], **bands_arrays[0].attrs}

        if hasattr(self, "spin_moments"):
            self.spin_moments = xr.DataArray(
                self.spin_moments,
                coords={
                    "k": self.bands.k,
                    "band": self.bands.band,
                    "axis": ["x", "y", "z"]
                },
                dims=("k", "band", "axis")
            )

    @entry_point('path')
    def _read_from_H(self, path, band_structure, eigenstate_map=None):
        """
        This entry point just generates a band structure from the path and
        then the band_structure entry point takes it from there.

        (it seems stupid now, but this is because of how things were at the beggining)

        Therefore it should be removed. Only one of "path" or "band_structure"
        inputs should exist. And then it should be always parsed into a sisl.BandStructure.
        """
        #Get the requested path
        self.path = path
        if not self.path and band_structure:
            return self._read_from_band_structure(eigenstate_map=eigenstate_map)
        if self.path and len(self.path) > 1:
            self.path = [point for point in path if point.get("active", True)]
        else:
            raise ValueError(f"You need to provide at least 2 points of the path to draw the bands. Please update the 'path' setting. The current path is: {self.path}")

        band_struct = sisl.BandStructure(
            None, # The band structure entry_point will take care of this
            point=np.array([[
                    point.get("x", None) or 0, point.get("y", None) or 0, point.get("z", None) or 0
                ] for point in self.path], dtype=float),
            division=np.array([point["divisions"] for point in self.path[1:]], dtype=int),
            name=np.array([point.get("tick", '') for point in self.path])
        )

        self._read_from_band_structure(band_structure=band_struct, eigenstate_map=eigenstate_map)

    @entry_point('bands file')
    def _read_siesta_output(self, bands_file, path, band_structure):
        """
        Reads the bands information from a SIESTA bands file.
        """
        #Get the info from the bands file
        self.path = path

        if self.path and self.path != getattr(self, "siesta_path", None) or band_structure:
            raise ValueError("A path was provided, therefore we can not use the .bands file even if there is one")

        self.bands = self.get_sile(bands_file or "bands_file").read_data(as_dataarray=True)

        # Inform of the path that it's being used if we can
        # THIS IS ONLY WORKING PROPERLY FOR FRACTIONAL UNITS OF THE BAND POINTS RN
        if hasattr(self, "fdf_sile") and self.fdf_sile.get("BandLines"):

            try:
                self.siesta_path = []
                points = self.fdf_sile.get("BandLines")

                for i, point in enumerate(points):

                    divisions, x, y, z, *others = point.split()
                    divisions = int(divisions) - int(points[i-1].split()[0]) if i > 0 else None
                    tick = others[0] if len(others) > 0 else None

                    self.siesta_path.append({"active": True, "x": float(x), "y": float(y), "z": float(z), "divisions": divisions, "tick": tick})

                    self.update_settings(path=self.siesta_path, run_updates=False, no_log=True)
            except Exception as e:
                print(f"Could not correctly read the bands path from siesta.\n Error {e}")

        # Define the spin class of the results we have retrieved
        if len(self.bands.spin.values) == 2:
            self.spin = sisl.Spin("p")

    def _after_read(self):

        # Inform the spin input of what spin class are we handling
        self.get_param("spin").update_options(self.spin)
        self.get_param("custom_gaps").get_param("spin").update_options(self.spin)

        self._calculate_gaps()

        # Make sure that the bands_range control knows which bands are available
        i_bands = self.bands.band.values

        if len(i_bands) > 30:
            i_bands = i_bands[np.linspace(0, len(i_bands)-1, 20, dtype=int)]

        self.modify_param('bands_range', 'inputField.params', {
            **self.get_param('bands_range')["inputField"]["params"],
            "min": min(i_bands),
            "max": max(i_bands),
            "allowCross": False,
            "marks": {int(i): str(i) for i in i_bands},
        })

    def _set_data(self, Erange, E0, bands_range, spin, bands_width, bands_color, spindown_color, add_band_trace_data, draw_before_bands=None):
        """
        Converts the bands dataframe into a data object for plotly.

        It stores the data under self.data, so that it can be accessed by posterior methods.

        Returns
        ---------
        self.data: list of dicts
            contains a dictionary for each bandStruct with all its information.
        """

        # Shift all the bands to the reference
        filtered_bands = self.bands - E0

        # Get the bands that matter for the plot
        if Erange is None:

            if bands_range is None:
            # If neither E range or bands_range was provided, we will just plot the 15 bands below and above the fermi level
                CB = int(filtered_bands.where(filtered_bands <= 0).argmax('band').max())
                bands_range = [int(max(filtered_bands["band"].min(), CB - 15)), int(min(filtered_bands["band"].max() + 1, CB + 16))]

            i_bands = np.arange(*bands_range)
            filtered_bands = filtered_bands.where(filtered_bands.band.isin(i_bands), drop=True)
            self.update_settings(
                run_updates=False,
                Erange=np.array([float(f'{val:.3f}') for val in [float(filtered_bands.min() - 0.01), float(filtered_bands.max() + 0.01)]]),
                bands_range=bands_range, no_log=True)
        else:
            Erange = np.array(Erange)
            filtered_bands = filtered_bands.where((filtered_bands <= Erange[1]) & (filtered_bands >= Erange[0])).dropna("band", "all")
            self.update_settings(run_updates=False, bands_range=[int(filtered_bands['band'].min()), int(filtered_bands['band'].max())], no_log=True)

        # Let's treat the spin if the user requested it
        self.spin_texture = False
        if spin is not None and len(spin) > 0:
            if isinstance(spin[0], int):
                filtered_bands = filtered_bands.sel(spin=spin)
            elif isinstance(spin[0], str):
                if not hasattr(self, "spin_moments"):
                    raise ValueError(f"You requested spin texture ({spin[0]}), but spin moments have not been calculated. The spin class is {self.spin.kind}")
                self.spin_texture = True

        if not callable(add_band_trace_data):
            add_band_trace_data = lambda *args, **kwargs: {}

        # Give the oportunity to draw before bands are drawn (used by Fatbands, for example)
        if callable(draw_before_bands):
            draw_before_bands()

        if self.spin_texture:

            def scatter_additions(band, spin_index):

                return {
                    "mode": "markers",
                    "marker": {"color": self.spin_moments.sel(band=band, axis=spin[0]).values, "size": bands_width, "showscale": True, "coloraxis": "coloraxis"},
                    "showlegend": False
                }
        else:

            def scatter_additions(band, spin_index):

                return {
                    "mode": "lines",
                    'line': {"color": [bands_color, spindown_color][spin_index], 'width': bands_width},
                }

        #Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
        self.add_traces(np.ravel([[{
                        'type': 'scatter',
                        'x': band.k.values,
                        'y': (band).values,
                        'mode': 'lines',
                        'name': "{} spin {}".format(band.band.values, ["up", "down"][spin]) if self.spin.is_polarized else str(band.band.values),
                        **scatter_additions(band.band.values, spin),
                        'hoverinfo':'name',
                        "hovertemplate": '%{y:.2f} eV',
                        **add_band_trace_data(band, self)
                        } for band in spin_bands] for spin_bands, spin in zip(filtered_bands.transpose('spin', 'band', 'k'), filtered_bands.spin.values)]).tolist())

        self._draw_gaps()

    def _after_get_figure(self, Erange, spin, spin_texture_colorscale):
        #Add the ticks
        self.figure.layout.xaxis.tickvals = getattr(self.bands, "ticks", None)
        self.figure.layout.xaxis.ticktext = getattr(self.bands, "ticklabels", None)
        self.figure.layout.yaxis.range = Erange
        self.figure.layout.xaxis.range = self.bands.k.values[[0, -1]]

        # If we are showing spin textured bands, customize the colorbar
        if self.spin_texture:
            self.layout.coloraxis.colorbar = {"title": f"Spin texture ({spin[0]})"}
            self.update_layout(coloraxis = {"cmin": -1, "cmax": 1, "colorscale": spin_texture_colorscale})

    def _calculate_gaps(self):
        """
        Calculates the gap (or gaps) assuming 0 is the fermi level.

        It creates the attributes `gap` and `gap_info`
        """
        # Calculate the band gap to store it
        above_fermi = self.bands.where(self.bands > 0)
        below_fermi = self.bands.where(self.bands < 0)
        CBbot = above_fermi.min()
        VBtop = below_fermi.max()

        CB = above_fermi.where(above_fermi==CBbot, drop=True).squeeze()
        VB = below_fermi.where(below_fermi==VBtop, drop=True).squeeze()

        self.gap = float(CBbot - VBtop)

        self.gap_info = {
            'k': (VB["k"].values, CB['k'].values),
            'bands': (VB["band"].values, CB["band"].values),
            'spin': (VB["spin"].values, CB["spin"].values) if self.spin.is_polarized else (0, 0),
            'Es': [float(VBtop), float(CBbot)]
        }

    def _draw_gaps(self, gap, gap_tol, gap_color, direct_gaps_only, custom_gaps):
        """
        Draws the calculated gaps and the custom gaps in the plot
        """
        # Draw gaps
        if gap:

            gapKs = [np.atleast_1d(k) for k in self.gap_info['k']]

            # Remove "equivalent" gaps
            def clear_equivalent(ks):
                if len(ks) == 1:
                    return ks

                uniq = [ks[0]]
                for k in ks[1:]:
                    if abs(min(np.array(uniq) - k)) > gap_tol:
                        uniq.append(k)
                return uniq

            all_gapKs = itertools.product(*[clear_equivalent(ks) for ks in gapKs])

            for gap_ks in all_gapKs:

                if direct_gaps_only and abs(gap_ks[1] - gap_ks[0]) > gap_tol:
                    continue

                self.draw_gap(*gap_ks, color=gap_color, true_gap=True)

        # Draw the custom gaps. These are gaps that do not necessarily represent
        # the maximum and the minimum of the VB and CB.
        for custom_gap in custom_gaps:

            requested_spin = custom_gap.get("spin", None)
            if requested_spin is None:
                requested_spin = [0, 1]

            for spin in self.bands.spin:
                if spin in requested_spin:
                    self.draw_gap(custom_gap["from"], custom_gap["to"], color=custom_gap.get("color", None), gap_spin=spin)

    def _sanitize_k(self, k):
        """Returns the float value of a k point in the plot.

        Parameters
        ------------
        k: float or str
            The k point that you want to sanitize.
            If it can be parsed into a float, the result of `float(k)` will be returned.
            If it is a string and it is a label of a k point, the corresponding k value for that
            label will be returned

        Returns
        ------------
        float
            The sanitized k value.
        """
        san_k = None

        try:
            san_k = float(k)
        except ValueError:
            if k in self.bands.attrs["ticklabels"]:
                i_tick = self.bands.attrs["ticklabels"].index(k)
                san_k = self.bands.attrs["ticks"][i_tick]
            else:
                pass
                # raise ValueError(f"We can not interpret {k} as a k-location in the current bands plot")
                # This should be logged instead of raising the error

        return san_k

    def draw_gap(self, from_k, to_k=None, color=None, true_gap=False, gap_spin=0, save=False, **kwargs):
        """
        Function that draws a given gap in the plot.

        Please note that if you want gaps to persist through updates you should
        use the "custom_gaps" setting or indicate save=True in this function.

        Parameters
        -----------
        from_k: float or str
            The k value where you want the gap to start (bottom limit).
            If "to_k" is not provided, it will be interpreted also as the top limit.

            If a k-value is a float, it will be directly interpreted
            as the position in the graph's k axis.

            If a k-value is a string, it will be attempted to be parsed
            into a float. If not possible, it will be interpreted as a label
            (e.g. "Gamma").
        to_k: float or str, optional
            same as "from_k" but in this case represents the top limit.

            If not provided, "from_k" will be used.
        color: str, optional
            the color with which the gap should be represented.
        true_gap: bool, optional
            whether it is the true gap of the structure. You should probably
            never use this, it is just used internally to save time.
        gap_spin: int, optional
            the spin component where you want to draw the gap.
        save: bool, optional
            whether you want this gap to persist through updates.
        **kwargs:
            keyword arguments that are passed directly to the new trace.
        """
        if to_k is None:
            to_k = from_k

        if save:
            return self.update_settings(custom_gaps=[*self.settings["custom_gaps"], {"from": from_k, "to": to_k, "color": color, "spin": [gap_spin]}])

        ks = [None, None]
        # Parse the names of the kpoints into their numeric values
        # if a string was provided.
        for i, val in enumerate((from_k, to_k)):
            ks[i] = self._sanitize_k(val)

        if true_gap:
            # Take the energies from the gap information
            Es = self.gap_info["Es"]
            name = "Gap"
        else:
            name = f"Gap ({from_k}-{to_k})"
            VB, CB = self.gap_info["bands"]
            Es = [self.bands.sel(k=k, band=band, spin=gap_spin, method="nearest") for k, band in zip(ks, (VB, CB))]
            # Get the real values of ks that have been obtained
            # because we might not have exactly the ks requested
            ks = [np.ravel(E.k)[0] for E in Es]
            Es = [np.ravel(E)[0] for E in Es]

        self.add_trace({
            'type': 'scatter',
            'mode': 'lines+markers+text',
            'x': ks,
            'y': Es,
            'text': [f'Gap: {Es[1] - Es[0]:.3f} eV', ''],
            'marker': {'color': color},
            'line': {'color': color},
            'name': name,
            'textposition': 'top right',
            **kwargs
        })

        return self

    def toggle_gap(self):
        """
        If the gap was being displayed, hide it. Else, show it.
        """
        return self.update_settings(gap= not self.settings["gap"])

    def plot_Ediff(self, band1, band2):
        """
        Plots the energy difference between two bands.

        Parameters
        ----------
        band1, band2: int
            the indices of the two bands you want to get the difference for.

        Returns
        ---------
        Plot
            a new plot with the plotted information.
        """
        two_bands = self.bands.sel(band=[band1, band2]).squeeze().values

        diff = two_bands[:, 1] - two_bands[:, 0]

        fig = px.line(x=self.bands.k.values, y=diff)

        plt = super().from_plotly(fig)

        plt.update_layout({**self.layout.to_plotly_json(), "title": f"Energy difference between bands {band1} and {band2}", "yaxis_range": [np.min(diff), np.max(diff)]})

        return plt

    def _plot_Kdiff(self, band1, band2, E=None, offsetE=False):
        """
        ONLY WORKING FOR A PAIR OF BANDS THAT ARE ALWAYS INCREASING OR ALWAYS DECREASING
        AND ARE ISOLATED (sorry)

        Plots the k difference between two bands.

        Parameters
        -----------
        band1, band2: int
            the indices of the two bands you want to get the difference for.
        E: array-like, optional
            the energy values for which we want the K difference between the two bands
        offsetE: boolean
            whether the energy should be referenced to the minimum of the first band

        Returns
        ---------
        Plot
            a new plot with the plotted information.
        """
        b1, b2 = self.bands.sel(band=[band1, band2]).squeeze().values.T
        ks = self.bands.k.values

        if E is None:
            #Interpolate the values of K for band2 that correspond to band1's energies.
            b2Ks_for_b1Es = np.interp(b1, b2, ks)

            E = b1
            diff = ks - b2Ks_for_b1Es

        else:
            if offsetE:
                E += np.min(b1)

            diff = np.interp(E, b1, ks) - \
                np.interp(E, b2, ks)

        E -= np.min(b1) if offsetE else 0

        fig = px.line(x=diff, y=E)

        plt = super().from_plotly(fig)

        plt.update_layout({"title": f"Delta K between bands {band1} and {band2}", 'xaxis_title': 'Delta k', 'yaxis_title': 'Energy [eV]'})

        return plt

    def effective_mass(self, band, k, k_direction, band_spin=0, n_points=10):
        """Calculates the effective mass from the curvature of a band in a given k point.

        It works by fitting the band to a second order polynomial.

        Notes
        -----
        Only valid if there are no band-crossings in the fitted range.  
        The effective mass may be highly dependent on the `k_direction` parameter, as well as the
        number of points fitted.

        Parameters
        -----------
        band: int
            The index of the band that we want to fit
        k: float or str
            The k value where we want to find the curvature of the band to calculate the effective mass.
        band_spin: int, optional
            The spin value for which we want the effective mass.
        n_points: int
            The number of points that we want to use for the polynomial fit.
        k_direction: {"symmetric", "right", "left"}, optional
            Indicates in which direction -starting from `k`- should the band be fitted. 
            "left" and "right" mean that the fit will only be done in one direction, while
            "symmetric" indicates that points from both sides will be used.

        Return
        -----------
        float
            The efective mass, in atomic units.
        """
        from sisl.unit.base import units

        # Get the band that we want to fit
        band_vals = self.bands.sel(band=band, spin=band_spin)

        # Sanitize k to a float
        k = self._sanitize_k(k)
        # Find the index of the requested k
        k_index = abs(self.bands.k -k).values.argmin()

        # Determine which slice of the band will we take depending on k_direction and n_points
        if k_direction == "symmetric":
            sel_slice = slice(k_index - n_points // 2, k_index + n_points // 2 + 1)
        elif k_direction == "left":
            sel_slice = slice(k_index - n_points + 1, k_index + 1)
        elif k_direction == "right":
            sel_slice = slice(k_index, k_index + n_points)
        else:
            raise ValueError(f"k_direction must be one of ['symmetric', 'left', 'right'], {k_direction} was passed")

        # Grab the slice of the band that we are going to fit
        sel_band = band_vals[sel_slice] * units("eV", "Hartree")
        sel_k = self.bands.k[sel_slice] - k

        # Fit the band to a second order polynomial
        polyfit = np.polynomial.Polynomial.fit(sel_k, sel_band, 2)

        # Get the coefficient for the second order term
        coeff_2 = polyfit.convert().coef[2]

        # Calculate the effective mass from the dispersion relation.
        # Note that hbar = m_e = 1, since we are using atomic units.
        eff_m = 1 / (2 * coeff_2)

        return eff_m
