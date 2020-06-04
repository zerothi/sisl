from collections import defaultdict

import numpy as np
import os
import pandas as pd
from  plotly.colors import DEFAULT_PLOTLY_COLORS
from xarray import DataArray

import sisl
from ..plot import Plot
from .bands import BandsPlot
from ..plotutils import random_color
from ..input_fields import OrbitalQueries, TextInput, DropdownInput, SwitchInput, ColorPicker, FloatInput, FilePathInput
from ..input_fields.range import ErangeInput

class FatbandsPlot(BandsPlot):
    '''
    Colorful representation of orbital weights in bands.

    Parameters
    -------------
    groups: array-like of dict, optional
        The different groups that are displayed in the fatbands
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

    _plot_type = 'Fatbands'

    _parameters = (

        FilePathInput(key='wfsx_file', name='Path to WFSX file',
            default=None,
            help='''The WFSX file to get the weights of the different orbitals in the bands.
            In standard SIESTA nomenclature, this should be the *.bands.WFSX file, as it is the one
            that contains the weights that correspond to the bands.
            
            This file is only meaningful (and required) if fatbands are plotted from the .bands file.
            Otherwise, the bands and weights will be generated from the hamiltonian by sisl.

            If the *.bands file is provided but the wfsx one isn't, we will try to find it.
            If `bands_file` is SystemLabel.bands, we will look for SystemLabel.bands.WFSX
            '''
        ),

        OrbitalQueries(
            key="groups", name="Fatbands groups",
            default=None,
            help='''The different groups that are displayed in the fatbands''',
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
                    key="factor", name="Width factor",
                    default=1,
                ),
            ]
        ),

    )

    def _read_siesta_output(self):

        # Try to get the wfsx file either by user input or by guessing it
        # from bands_file
        wfsx_file = self.setting('wfsx_file')
        bands_file = self.setting("bands_file") or self.requiredFiles[0]
        if wfsx_file is None:
            wfsx_file = f'{bands_file}.WFSX'

        # We will need the overlap matrix from the hamiltonian to get the correct
        # weights. 
        # If there is no root_fdf we will try to guess it from bands_file 
        root_fdf = self.setting('root_fdf')
        if root_fdf is None and not hasattr(self, 'H'):
            possible_fdf = f'{os.path.splitext(bands_file)[0]}.fdf'
            print(f'We are assuming that the fdf associated to {bands_file} is {possible_fdf}.'+
            ' If it is not, please provide a "root_fdf" by using the update_settings method.')
            self.set_files(root_fdf=possible_fdf)
        
        self.setup_hamiltonian()

        # If the wfsx doesn't exist, we will not even bother to read the bands
        if not os.path.exists(wfsx_file):
            raise Exception(f"We didn't find a WFSX file in the location {wfsx_file}")

        # Otherwise we will make BandsPlot read the bands
        BandsPlot._read_siesta_output(self)

        # And then read the weights from the wfsx file
        wfsx_sile = self.get_sile(wfsx_file)

        weights = []
        for i, state in enumerate(wfsx_sile.yield_eigenstate(self.H)):
            # Each eigenstate represents all the states for a given k-point

            # Get the band indices to which these states correspond
            if i == 0:
                bands = state.info['indices']

            # Get the weights for this eigenstate
            weights.append(state.norm2(sum=False))

        weights = np.array(weights).real

        # Finally, build the weights dataarray so that it can be used by _set_data
        self.weights = DataArray(
            weights,
            coords={
                'k': self.bands.k,
                'band': bands,
                'orb': np.arange(0, weights.shape[2]),
            },
            dims=('k', 'band', 'orb')
        )

        # Set up the options for the 'groups' setting based on the plot's associated geometry
        self._set_group_options()

    def _read_from_H(self):

        self.weights = []

        # Define the function that will "catch" each eigenstate and 
        # build the weights array. See BandsPlot._read_from_H to understand where
        # this will go exactly
        def _weights_from_eigenstate(eigenstate, plot):
            plot.weights.append(eigenstate.norm2(sum=False))

        # We make bands plot read the bands, which will also populate the weights
        # thanks to the above step
        bands_read = False; err = None
        try:
            BandsPlot._read_from_H(self, eigenstate_map=_weights_from_eigenstate)
            bands_read = True
        except Exception as e:
            # Let's keep this error, we are going to at least set the group options so that the
            # user knows what can they choose (specially important for the GUI)
            err = e

        self._set_group_options()
        if not bands_read:
            raise err
        
        # Then we just convert the weights to a DataArray
        self.weights = np.array(self.weights).real

        self.weights = DataArray(
            self.weights,
            coords={
                'k': self.bands.k,
                'band': np.arange(0, self.weights.shape[1]),
                'orb': np.arange(0, self.weights.shape[2]),
            },
            dims=('k', 'band', 'orb')
        )
    
    def _set_group_options(self):

        # Try to find a geometry if there isn't already one
        if not hasattr(self, "geom"):

            # From the hamiltonian
            band_struct = self.setting("band_structure")
            if band_struct is not None:
                self.geom = band_struct.parent.geom
        
        self.get_param('groups').update_options(self.geom)

    def _set_data(self):

        # We let the bands plot draw the bands
        BandsPlot._set_data(
            self, draw_before_bands=self._draw_fatbands,
            # Avoid bands being displayed in the legend individually (it would be a mess)
            add_band_trace_data=lambda band, plot: {'showlegend': False}
        )
    
    def _draw_fatbands(self):

        E0 = self.setting('E0')

        # We get the bands range that is going to be plotted
        # Remember that the BandsPlot will have updated this setting accordingly,
        # so it's safe to use it directly
        plotted_bands = self.setting("bands_range")
        
        # If we don't have weights for all plotted bands (specially possible if we
        # have read from the WFSX file), reduce the range of the bands. Note that this
        # does not affect the bands displayed, just the "fatbands".
        plotted_bands[0] = max(self.weights.band.values.min(), plotted_bands[0])
        plotted_bands[1] = min(self.weights.band.values.max(), plotted_bands[1])
        
        #Get the bands that matter (spin polarization currently not implemented)
        plot_eigvals = self.bands.sel(band=np.arange(*plotted_bands), spin=0) - E0
        # Get the weights that matter
        plot_weights = self.weights.sel(band=np.arange(*plotted_bands))
        
        # Get the groups of orbitals whose bands are requested
        groups = self.setting('groups')

        # If the user didn't provide groups and didn't specify that groups is an
        # empty list, we are going to build the default groups, which is to split by species
        # in case there is more than one species or else, by orbitals
        if groups is None:
            if len(self.geom.atoms.atom) > 1:
                group_by = 'species'
            else:
                group_by = 'orbitals'

            return self.build_groups(group_by)

        # We are going to need a trace that goes forward and then back so that
        # it is self-fillable
        xs = self.bands.k.values
        area_xs = [*xs, *np.flip(xs)]

        prev_traces = len(self.data)

        groups_param = self.get_param("groups")

        # Let's plot each group of orbitals as an area that surrounds each band
        for i ,group in enumerate(groups):

            group = groups_param.complete_query(group, name=f"Group {i+1}")

            #Use only the active requests
            if not group.get("active", True):
                continue

            orb = groups_param.get_orbitals(group)
            
            weights = plot_weights.sel(orb=orb)
            if group["normalize"]:
                weights = weights.mean("orb")
            else:
                weights = weights.sum("orb")

            if group["color"] is None:
                group["color"] = random_color()

            self.add_traces([{
                'type': 'scatter',
                'mode': 'lines',
                'x': area_xs,
                'y': [*(band + band_weights*group["factor"]), *np.flip(band - band_weights*group["factor"])],
                'line':{'width': 0, "color": group["color"]},
                'showlegend': i == 0,
                'name': group["name"],
                'legendgroup':group["name"],
                'fill': 'toself'
            } for i, (band, band_weights) in enumerate(zip(plot_eigvals.transpose('band', 'k'), weights.transpose('band', 'k')))])

    # -------------------------------------
    #         Convenience methods
    # -------------------------------------

    def build_groups(self, on="species", only=None, exclude=None, clean=True, colors=DEFAULT_PLOTLY_COLORS, **kwargs):
        '''
        Builds groups automatically to draw their contributions.

        Works exactly the same as `PdosPlot.split_DOS`

        Parameters
        --------
        on: str, {"species", "atoms", "orbitals", "spin"}
            the parameter to split along
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
            `plot.build_groups(on="orbitals", species=["C"])`
            will split the PDOS on the different orbitals but will take
            only those that belong to carbon atoms.
        '''

        groups = self.get_param('groups')._generate_queries(
            on=on, only=only, exclude=exclude, **kwargs)

        # Repeat the colors in case there are more groups than colors
        colors = np.tile(colors, len(groups) // len(colors) + 1)

        # Asign colors
        for i, _ in enumerate(groups):
            groups[i]['color'] = colors[i]

        # If the user doesn't want to clean the plot, we will just add the groups to the existing ones
        if not clean:
            groups = [*self.setting("groups"), *groups]

        return self.update_settings(groups=groups)

    def scale_fatbands(self, factor, from_current=False):
        '''
        Scales all bands by a given factor.

        Basically, it updates the 'factor' key of all the groups provided
        in the group setting.

        Parameters
        -----------
        factor: float
            the factor that should be used to scale.
        from_current: boolean, optional
            whether 'factor' is meant to multiply the current scaling factor.
            If False, it will just replace the current factor.
        '''

        groups = self.setting('groups')

        # Asign colors
        for i, _ in enumerate(groups):
            if from_current:
                groups[i]['factor'] *= factor
            else:
                groups[i]['factor'] = factor

        return self.update_settings(groups=groups)
