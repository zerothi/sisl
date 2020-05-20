import numpy as np
import xarray as xr
import itertools
import plotly.express as px

import os

import sisl
from ..plot import Plot, PLOTS_CONSTANTS
from ..plotutils import find_files
from ..input_fields import TextInput, FilePathInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeInput, RangeSlider, QueriesInput, ProgramaticInput, FunctionInput
from ..input_fields.range import ErangeInput

class BandsPlot(Plot):
    '''
    Plot representation of the bands.

    Parameters
    -------------
    bands_file: str, optional
        This parameter explicitly sets a .bands file. Otherwise, the bands
        file is attempted to read from the fdf file
    band_structure: None, optional
        The bandStruct structure object to be used.
    add_band_trace_data: None, optional
        A function that receives each band (as a DataArray) and adds data to
        the trace. It also recieves the plot object.              The
        returned data may even overwrite the existing one, therefore it can
        be useful to fully customize your bands plot (individual style for
        each band if you want).
    eigenstate_map: None, optional
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
        setting
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
    bands_width: float, optional
        Width of the lines that represent the bands
    bands_color: str, optional
        Choose the color to display the bands.  This will be used for the
        spin up bands if the calculation is spin polarized
    spindown_color: str, optional
        Choose the color for the spin down bands.Only used if the
        calculation is spin polarized.
    reading_order: None, optional
        Order in which the plot tries to read the data it needs.
    root_fdf: str, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    '''

    _plot_type = "Bands"
    
    _requirements = {
        "siesOut": {
            "files": ["$struct$.bands", "*.bands"]
        }
    }
    
    _parameters = (

        FilePathInput(key = "bands_file", name = "Path to bands file",
            width = "s100% m50% l33%",
            group="dataread",
            params = {
                "placeholder": "Write the path to your bands file here...",
            },
            help = '''This parameter explicitly sets a .bands file. Otherwise, the bands file is attempted to read from the fdf file '''
        ),

        ProgramaticInput(key = "band_structure", name = "bandStruct structure object",
            default= None,
            help = "The bandStruct structure object to be used."
        ),

        FunctionInput(key="add_band_trace_data", name="Additional data for band traces",
            default=None,
            positional=["band", "plot"],
            returns=[dict],
            help='''A function that receives each band (as a DataArray) and adds data to the trace. It also recieves the plot object. 
            The returned data may even overwrite the existing one, therefore it can be useful to fully customize your bands plot (individual style for each band if you want).'''
        ),

        FunctionInput(key="eigenstate_map", name="Eigenstate map function",
            default=None,
            positional=["eigenstate", "plot"],
            returns=[],
            help='''This function receives the eigenstate object for each k value when the bands are being extracted from a hamiltonian.
            You can do whatever you want with it, the point of this function is to avoid running the diagonalization process twice.'''
        ),

        ErangeInput(key="Erange",
            help = "Energy range where the bands are displayed."
        ),

        FloatInput(key="E0", name="Reference energy",
            default=0,
            help='''The energy to which all energies will be referenced (including Erange).'''
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
            help='''Path along which bands are drawn in units of reciprocal lattice vectors.<br>
            Note that if you want to provide a path programatically you can do it more easily with the `band_structure` setting''',
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
            help='''The difference in k that must exist to consider to gaps different.<br>
            If two gaps' positions differ in less than this, only one gap will be drawn.<br>
            Useful in cases where there are degenerated bands with exactly the same values.'''
        ),

        ColorPicker(key = "gap_color", name = "Gap color",
            default = None,
            help = "Color to display the gap"
        ),

        FloatInput(key = "bands_width", name = "Band lines width",
            default = 1,
            help = "Width of the lines that represent the bands"
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

    _layout_defaults = {
        'xaxis_title': 'K',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'xaxis_showgrid': True,
        'yaxis_title': 'Energy (eV)'
    }

    @classmethod
    def _default_animation(cls, wdir = None, frame_names=None, **kwargs):
        
        bands_files = find_files(wdir, "*.bands", sort = True)

        def _get_frame_names(self):

            return [os.path.basename( childPlot.setting("bands_file")) for childPlot in self.childPlots]

        return cls.animated("bands_file", bands_files, frame_names = _get_frame_names, wdir = wdir, **kwargs)

    def _after_init(self):

        self.add_shortcut("g", "Toggle gap", self.toggle_gap)

    def _read_from_H(self, eigenstate_map=None):

        bandStruct = self.setting("band_structure")
        
        #If no bandStruct structure is provided, then build it ourselves
        if bandStruct is None:

            if not hasattr(self, "H"):
                self.setup_hamiltonian()

            #Get the requested path
            self.path = self.setting('path')
            if self.path and len(self.path) > 1:
                self.path = [point for point in self.setting("path") if point["active"]]
            else:
                raise Exception(f"You need to provide at least 2 points of the path to draw the bands. Please update the 'path' setting. The current path is: {self.path}")

            bandStruct = sisl.BandStructure(
                self.H,
                point=np.array([[point["x"] or 0, point["y"] or 0, point["z"] or 0] for point in self.path], dtype=float),
                division=np.array([point["divisions"] for point in self.path][1:], dtype=int) ,
                name=np.array([point["tick"] for point in self.path])
            )

        else:

            if not hasattr(bandStruct, "H"):
                self.setup_hamiltonian()
                bandStruct.set_parent(self.H)
            
        self.ticks = bandStruct.lineartick()
        self.kPath = bandStruct._k

        # We define a wrapper to get the values out of the eigenstates
        # to give the possibility to the user to do something inbetween
        # NOTE THAT THIS IS USED BY FAT BANDS TO GET THE WEIGHTS SIMULTANEOUSLY
        eig_map = eigenstate_map or self.setting('eigenstate_map')
        def bands_wrapper(eigenstate):
            if callable(eig_map):
                eig_map(eigenstate, self)
            return eigenstate.eig

        # THIS DOES NOT SUPPORT SPIN!!!!!!!!!!!!!!!! (I think)
        self.bands = bandStruct.apply.dataarray.eigenstate(
            wrap=bands_wrapper,
            coords=('band',),
        )

        self.bands = self.bands.expand_dims('spin', axis=1)
        self.bands['k'] = bandStruct.lineark()

        self.bands.attrs = {"ticks": self.ticks[0], "ticklabels": self.ticks[1]}

    def _read_siesta_output(self):
        
        #Get the info from the bands file
        self.path = self.setting("path")

        if self.path and self.path != getattr(self, "siestaPath", None) or self.setting("band_structure"):
            raise Exception("A path was provided, therefore we can not use the .bands file even if there is one")

        bands_file = self.setting("bands_file") or self.requiredFiles[0]

        self.bands = self.get_sile(bands_file).read_data(as_dataarray=True)

        # Inform of the path that it's being used if we can
        # THIS IS ONLY WORKING PROPERLY FOR FRACTIONAL UNITS OF THE BAND POINTS RN
        if hasattr(self, "fdfSile") and self.fdfSile.get("BandLines"):

            try:
                self.siestaPath = []
                points = self.fdfSile.get("BandLines")

                for i, point in enumerate(points):

                    divisions, x, y, z, *others = point.split()
                    divisions = int(divisions) - int(points[i-1].split()[0]) if i > 0 else None
                    tick = others[0] if len(others) > 0 else None

                    self.siestaPath.append({"active": True, "x": float(x), "y": float(y), "z": float(z), "divisions": divisions, "tick": tick})
                    
                    self.update_settings(path=self.siestaPath, run_updates=False, no_log=True)
            except Exception as e:
                print(f"Could not correctly read the bands path from siesta.\n Error {e}")
    
    def _after_read(self):

        self.isSpinPolarized = len(self.bands.spin.values) == 2
        self._calculate_gaps()

        # Make sure that the bands_range control knows which bands are available
        iBands = self.bands.band.values

        if len(iBands) > 30:
            iBands = iBands[np.linspace(0, len(iBands)-1, 20, dtype=int)]

        self.modify_param('bands_range', 'inputField.params', {
            **self.get_param('bands_range')["inputField"]["params"],
            "min": min(iBands),
            "max": max(iBands),
            "allowCross": False,
            "marks": { int(i): str(i) for i in iBands },
        })
    
    def _set_data(self, draw_before_bands=None, add_band_trace_data=None):
        
        '''
        Converts the bands dataframe into a data object for plotly.

        It stores the data under self.data, so that it can be accessed by posterior methods.

        Returns
        ---------
        self.data: list of dicts
            contains a dictionary for each bandStruct with all its information.
        '''

        Erange = self.setting('Erange')
        E0 = self.setting('E0')

        # Shift all the bands to the reference
        filtered_bands = self.bands - E0

        # Get the bands that matter for the plot
        if Erange is None:
            bands_range = self.setting("bands_range")

            if bands_range is None:
                # If neither E range or bands_range was provided, we will just plot the 15 bands below and above the fermi level
                CB = int(filtered_bands.where(filtered_bands <= 0).argmax('band').max())
                bands_range = [int(max(filtered_bands["band"].min(), CB - 15)), int(min(filtered_bands["band"].max() + 1, CB + 16))]

            iBands = np.arange(*bands_range)
            filtered_bands = filtered_bands.where(filtered_bands.band.isin(iBands), drop=True)
            self.update_settings(
                run_updates=False,
                Erange=np.array([float(f'{val:.3f}') for val in [float(filtered_bands.min()), float(filtered_bands.max())]]),
                bands_range=bands_range, no_log=True)
        else:
            Erange = np.array(Erange)
            filtered_bands = filtered_bands.where( (filtered_bands <= Erange[1]) & (filtered_bands >= Erange[0])).dropna("band", "all")
            self.update_settings(run_updates=False, bands_range=[int(filtered_bands['band'].min()), int(filtered_bands['band'].max())], no_log=True)

        add_band_trace_data = add_band_trace_data or self.setting("add_band_trace_data")
        if not callable(add_band_trace_data):
            add_band_trace_data = lambda * args, **kwargs: {}
        
        # Give the oportunity to draw before bands are drawn (used by Fatbands)
        if callable(draw_before_bands):
            draw_before_bands()

        #Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
        self.add_traces(np.ravel([[{
                        'type': 'scatter',
                        'x': band.k.values,
                        'y': (band).values,
                        'mode': 'lines', 
                        'name': "{} spin {}".format( band.band.values, PLOTS_CONSTANTS["spins"][spin]) if self.isSpinPolarized else str(band.band.values) , 
                        'line': {"color": [self.setting("bands_color"),self.setting("spindown_color")][spin], 'width' : self.setting("bands_width")},
                        'hoverinfo':'name',
                        "hovertemplate": '%{y:.2f} eV',
                        **add_band_trace_data(band, self)
                        } for band in spin_bands] for spin_bands, spin in zip(filtered_bands.transpose('spin', 'band', 'k'), filtered_bands.spin.values)]).tolist())
        
        self._draw_gaps()

    def _after_get_figure(self):

        #Add the ticks
        self.figure.layout.xaxis.tickvals = getattr(self.bands, "ticks", None)
        self.figure.layout.xaxis.ticktext = getattr(self.bands, "ticklabels", None)
        self.figure.layout.yaxis.range = np.array(self.setting("Erange"))
    
    def _calculate_gaps(self):

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
            'spin': (VB["spin"].values, CB["spin"].values) if self.isSpinPolarized else (0,0),
            'Es': [float(VBtop), float(CBbot)]
        }

    def _draw_gaps(self):

        # Draw gaps
        if self.setting("gap"):

            gap_tolerance = self.setting('gap_tol')
            gap_color = self.setting('gap_color')
            only_direct = self.setting('direct_gaps_only')

            gapKs = [np.atleast_1d(k) for k in self.gap_info['k']]

            # Remove "equivalent" gaps
            def clear_equivalent(ks):
                if len(ks) == 1:
                    return ks

                uniq = [ks[0]]
                for k in ks[1:]:
                    if abs(min(np.array(uniq) - k)) > gap_tolerance:
                        uniq.append(k)
                return uniq

            all_gapKs = itertools.product(*[clear_equivalent(ks) for ks in gapKs])

            for gapKs in all_gapKs:

                if only_direct and abs(gapKs[1] - gapKs[0]) > gap_tolerance:
                    continue

                self.add_trace({
                    'type': 'scatter',
                    'mode': 'lines+markers+text',
                    'x': gapKs,
                    'y': self.gap_info["Es"],
                    'text': [f'Gap: {self.gap:.3f} eV', ''],
                    'marker':{'color': gap_color },
                    'line': {'color': gap_color},
                    'name': 'Gap',
                    'textposition': 'top right',
                })

    def toggle_gap(self):

        self.update_settings(gap= not self.settings["gap"])
    
    def plot_Ediff(self, band1, band2):
        '''
        Plots the energy difference between two bands.

        Parameters
        ----------
        band1, band2: int
            the indices of the two bands you want to get the difference for.

        Returns
        ---------
        Plot
            a new plot with the plotted information.
        '''

        two_bands = self.bands.sel(band=[band1, band2]).squeeze().values

        diff = two_bands[:, 1] - two_bands[:, 0]

        fig = px.line(x=self.bands.k.values ,y=diff)

        plt = Plot.from_plotly(fig)

        plt.update_layout({ **self.layout.to_plotly_json(), "title": f"Energy difference between bands {band1} and {band2}", "yaxis_range": [np.min(diff), np.max(diff)]})

        return plt

    def _plot_Kdiff(self, band1, band2, E=None, offsetE=False):
        '''
        ONLY WORKING FOR A PAIR OF BANDS THAT ARE ALWAYS INCREASING OR ALWAYS DECREASING
        
        Plots the energy difference between two bands.

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
        '''

        b1, b2 = self.bands.sel(band=[band1, band2]).squeeze().values.T
        ks = self.bands.k.values

        if E is None:
            #Interpolate the values of K for band2 that correspond to band1's energies.
            b2Ks_for_b1Es = np.interp(b1, b2, ks)

            E = b1
            diff = ks - b2Ks_for_b1Es

        else:
            diff = np.interp(E, b1, ks) - \
                np.interp(E, b2, ks)

        E += np.min(b1) if offsetE else 0

        fig = px.line(x=diff, y=E)

        plt = Plot.from_plotly(fig)

        plt.update_layout({"title": f"Delta K between bands {band1} and {band2}"})

        return plt
