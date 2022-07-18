# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections import defaultdict
from functools import partial
import itertools
from typing import Union, Literal

import numpy as np

from sisl.viz.nodes.plotters.plotter import PlotterNode
from ..data_sources import DataSource, FileDataSIESTA, HamiltonianDataSource 
from sisl.viz.nodes.node import Node
import xarray as xr

import sisl
from sisl.physics.brillouinzone import BrillouinZone
from sisl.physics.spin import Spin
from ...input_fields import SpinIndexSelect

try:
    import pathos
    _do_parallel_calc = True
except:
    _do_parallel_calc = False

class BandsData:

    def __init__(self, bands_data: Union[xr.DataArray, xr.Dataset]):

        old_attrs = bands_data.attrs

        if isinstance(bands_data, xr.DataArray):
            bands_data = xr.Dataset({"E": bands_data})
        
        # Check if there's a spin attribute
        spin = old_attrs.get("spin", None)

        # If not, guess it
        if spin is None:
            if 'spin' not in bands_data:
                spin = Spin(Spin.UNPOLARIZED)
            else:
                spin = {
                    1: Spin.UNPOLARIZED,
                    2: Spin.POLARIZED,
                    4: Spin.NONCOLINEAR,
                }[bands_data.spin.shape[0]]

                spin = Spin(spin)

        # Remove the spin coordinate if the data is spin unpolarized
        if 'spin' in bands_data and spin.is_unpolarized:
            bands_data = bands_data.sum("spin")

        if not spin.is_unpolarized:
            spin_options = SpinIndexSelect.get_spin_options(spin)
            bands_data['spin'] = ('spin', spin_options, bands_data.spin.attrs)

        # If the energy variable doesn't have units, set them as eV
        if 'E' in bands_data and 'units' not in bands_data.E.attrs:
            bands_data.E.attrs['units'] = 'eV'
        # Same with the k coordinate, which we will assume are 1/Ang
        if 'k' in bands_data and 'units' not in bands_data.k.attrs:
            bands_data.k.attrs['units'] = '1/Ang'
        # If there are ticks, show the grid.
        if 'axis' in bands_data.k.attrs and bands_data.k.attrs['axis'].get('ticktext') is not None:
            bands_data.k.attrs['axis'] = {"showgrid": True, **bands_data.k.attrs.get('axis', {})}

        bands_data.attrs = {
            **old_attrs, "spin": spin
        }

        self._data = bands_data

class BandsDataNode(DataSource):
    @classmethod
    def register(cls, func):
        return cls.from_func(func)

    _get = BandsData

@BandsDataNode.register
def BandsDataSIESTA(fdf=None, bands_file=None):
    """Gets the bands data from a SIESTA .bands file"""
    bands_file = FileDataSIESTA(fdf=fdf, path=bands_file, cls=sisl.io.bandsSileSiesta)
    
    bands_data = bands_file.read_data(as_dataarray=True)
    bands_data.k.attrs['axis'] = {
        'tickvals': bands_data.attrs.pop('ticks'),
        'ticktext': bands_data.attrs.pop('ticklabels')
    }

    return BandsData(bands_data)._data

@BandsDataNode.register
def BandsDataAiida(aiida_bands):
    """
    Creates the bands plot reading from an aiida BandsData node.
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
            "k": ('k', plot_data["x"], {"axis": tick_info}),
            "band": np.arange(0, bands.shape[2]),
        },
        dims=("spin", "k", "band"),
    )

    return BandsData(data)._data

def _get_eigenstate_wrapper(k_vals, spin, extra_vars=(), spin_moments=True):
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
            - 'name', str: The name of the quantity.
            - 'getter', callable: A function that gets 3 arguments: eigenstate, plot and
            spin index, and returns the values of the quantity in a numpy array. This
            function will be called for each eigenstate object separately. That is, once
            for each (k-point, spin) combination.
            - 'coords', tuple of str: The names of the  dimensions of the returned array.
            The number of coordinates should match the number of dimensions.
            of 
            - 'coords_values', dict: If this variable introduces a new coordinate, you should
            pass the values for that coordinate here. If the coordinates were already defined
            by another variable, they will already have values. If you are unsure that the
            coordinates are new, just pass the values for them, they will get overwritten.
    spin_moments: bool, optional
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
        def _spin_moment_getter(eigenstate, spin, spin_index):
            return eigenstate.spin_moment().real

        extra_vars = ({
            "coords": ("band", "axis"), "coords_values": dict(axis=["x", "y", "z"]),
            "name": "spin_moments", "getter": _spin_moment_getter},
        *extra_vars)

    # Define the available spin indices. Notice that at the end the spin dimension
    # is removed from the dataset unless the calculation is spin polarized. So having
    # spin_indices = [0] is just for convenience.
    spin_indices = [0]
    if spin.is_polarized:
        spin_indices = [0, 1]

    # Add a variable to get the eigenvalues.
    all_vars = ({
        "coords": ("band",), "coords_values": {"spin": spin_indices, "k": k_vals},
        "name": "E", "getter": lambda eigenstate, spin, spin_index: eigenstate.eig},
        *extra_vars
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

@BandsDataNode.register
def BandsDataH(band_structure, H=None, extra_vars=()):
    """
    Uses a sisl's `BandStructure` object to calculate the bands.
    """
    if band_structure is None:
        raise ValueError("No band structure (k points path) was provided")

    if not isinstance(getattr(band_structure, "parent", None), sisl.Hamiltonian):
        H = HamiltonianDataSource(H=H)
        band_structure.set_parent(H)
    else:
        H = band_structure.parent

    # Define the spin class of this calculation.
    spin = H.spin

    ticks = band_structure.lineartick()

    # Get the wrapper function that we should call on each eigenstate.
    # This also returns the coordinates and names to build the final dataset.
    bands_wrapper, all_vars, coords_values= _get_eigenstate_wrapper(
        band_structure.lineark(), spin, extra_vars=extra_vars
    )

    # Get a dataset with all values for all spin indices
    spin_datasets = []
    coords = [var['coords'] for var in all_vars]
    name = [var['name'] for var in all_vars]
    for spin_index in coords_values['spin']:

        # Non collinear routines don't accept the keyword argument "spin"
        spin_kwarg = {"spin": spin_index}
        if not spin.is_diagonal:
            spin_kwarg = {}

        with band_structure.apply(pool=_do_parallel_calc, zip=True) as parallel:
            spin_bands = parallel.dataarray.eigenstate(
                wrap=partial(bands_wrapper, spin_index=spin_index),
                **spin_kwarg,
                coords=coords, name=name,
            )

        spin_datasets.append(spin_bands)

    # Merge everything into a single dataset with a spin dimension
    bands_data = xr.concat(spin_datasets, "spin").assign_coords(coords_values)

    # If the band structure contains discontinuities, we will copy the dataset
    # adding the discontinuities.
    if len(band_structure._jump_idx) > 0:

        old_coords = bands_data.coords
        coords = {
            name: band_structure.insert_jump(old_coords[name]) if name == "k" else old_coords[name].values
            for name in old_coords
        }

        def _add_jump(array):
            if "k" in array.coords:
                array = array.transpose("k", ...)
                return (array.dims, band_structure.insert_jump(array))
            else:
                return array

        bands_data = xr.Dataset(
            {name: _add_jump(bands_data[name]) for name in bands_data},
            coords=coords
        )

    # Inform of where to place the ticks
    bands_data.k.attrs["axis"] = {
        "tickvals": ticks[0], 
        "ticktext": ticks[1],
    }

    return BandsData(bands_data)._data

@BandsDataNode.register
def BandsDataWFSX(fdf, wfsx_file, extra_vars=(), need_H=False):
    """Plots bands from the eigenvalues contained in a WFSX file.

    It also needs to get a geometry.
    """
    if need_H:
        H = HamiltonianDataSource(H=fdf)
        if H is None:
            raise ValueError("Hamiltonian was not setup, and it is needed for the calculations")
        parent = H
        geometry = parent.geometry
    else:
        # Get the fdf sile
        fdf = FileDataSIESTA(path=fdf)
        # Read the geometry from the fdf sile
        geometry = fdf.read_geometry(output=True)
        parent = geometry

    # Get the wfsx file
    wfsx_sile = FileDataSIESTA(fdf=fdf, path=wfsx_file, cls=sisl.io.wfsxSileSiesta, parent=parent)

    # Now read all the information of the k points from the WFSX file
    k, weights, nwfs = wfsx_sile.read_info()
    # Get the number of wavefunctions in the file while performing a quick check
    nwf = np.unique(nwfs)
    if len(nwf) > 1:
        raise ValueError(f"File {wfsx_sile.file} contains different number of wavefunctions in some k points")
    nwf = nwf[0]
    # From the k values read in the file, build a brillouin zone object.
    # We will use it just to get the linear k values for plotting.
    bz = BrillouinZone(geometry, k=k, weight=weights)

    # Read the sizes of the file, which contain the number of spin channels
    # and the number of orbitals and the number of k points.
    nspin, nou, nk, _ = wfsx_sile.read_sizes()

    # Find out the spin class of the calculation.
    spin = Spin({
        1: Spin.UNPOLARIZED, 2: Spin.POLARIZED,
        4: Spin.NONCOLINEAR, 8: Spin.SPINORBIT
    }[nspin])
    # Now find out how many spin channels we need. Note that if there is only
    # one spin channel there will be no "spin" dimension on the final dataset.
    nspin = 2 if spin.is_polarized else 1

    # Determine whether spin moments will be calculated.
    spin_moments = False
    if not spin.is_diagonal:
        # We need to set the parent
        try:
            H = HamiltonianDataSource(H=fdf)
            if H is not None:
                # We could read a hamiltonian, set it as the parent of the wfsx sile
                wfsx_sile = FileDataSIESTA(path=wfsx_sile.file, kwargs=dict(parent=parent))
                spin_moments = True
        except:
            pass

    # Get the wrapper function that we should call on each eigenstate.
    # This also returns the coordinates and names to build the final dataset.
    bands_wrapper, all_vars, coords_values = _get_eigenstate_wrapper(
        sisl.physics.linspace_bz(bz), extra_vars=extra_vars,
        spin_moments=spin_moments, spin=spin
    )
    # Make sure all coordinates have values so that we can assume the shape
    # of arrays below.
    coords_values['band'] = np.arange(0, nwf)
    coords_values['orb'] = np.arange(0, nou)

    # Initialize all the arrays. For each quantity we will initialize
    # an array of the needed shape.
    arrays = {}
    for var in all_vars:
        # These are all the extra dimensions of the quantity. Note that a
        # quantity does not need to have extra dimensions.
        extra_shape = [len(coords_values[coord]) for coord in var['coords']]
        # First two dimensions will always be the spin channel and the k index.
        # Then add potential extra dimensions.
        shape = (nspin, len(bz), *extra_shape)
        # Initialize the array.
        arrays[var['name']] = np.empty(shape, dtype=var.get('dtype', np.float64))

    # Loop through eigenstates in the WFSX file and add their contribution to the bands
    ik = -1
    for eigenstate in wfsx_sile.yield_eigenstate():
        spin = eigenstate.info.get("spin", 0)
        # Every time we encounter spin 0, we are in a new k point.
        if spin == 0:
            ik +=1
            if ik == 0:
                # If this is the first eigenstate we read, get the wavefunction
                # indices. We will assume that ALL EIGENSTATES have the same indices.
                # Note that we already checked previously that they all have the same
                # number of wfs, so this is a fair assumption.
                coords_values['band'] = eigenstate.info['index']

        # Get all the values for this eigenstate.
        returns = bands_wrapper(eigenstate, spin_index=spin)
        # And store them in the respective arrays.
        for var, vals in zip(all_vars, returns):
            arrays[var['name']][spin, ik] = vals

    # Now that we have all the values, just build the dataset.
    bands_data = xr.Dataset(
        data_vars={
            var['name']: (("spin", "k", *var['coords']), arrays[var['name']])
            for var in all_vars
        }
    ).assign_coords(coords_values)

    bands_data.attrs = {"parent": bz}

    return BandsData(bands_data)._data

class BandsProcessor(Node):
    pass

@BandsProcessor.from_func
def SpinMoments(bands_data=None, axis="x"):
    spin_moms = bands_data.spin_moments.sel(axis=axis)
    spin_moms = spin_moms.rename(f'spin_moments_{axis}')
    return spin_moms

@BandsProcessor.from_func
def filter_bands(bands_data, Erange=None, E0=0, bands_range=None, spin=None):
    filtered_bands = bands_data.copy()
    # Shift the energies according to the reference energy, while keeping the
    # attributes (which contain the units, amongst other things)
    filtered_bands['E'] = bands_data.E - E0
    continous_bands = filtered_bands.dropna("k", how="all")

    # Get the bands that matter for the plot
    if Erange is None:
        if bands_range is None:
            # If neither E range or bands_range was provided, we will just plot the 15 bands below and above the fermi level
            CB = int(continous_bands.E.where(continous_bands.E <= 0).argmax('band').max())
            bands_range = [int(max(continous_bands["band"].min(), CB - 15)), int(min(continous_bands["band"].max() + 1, CB + 16))]

        filtered_bands = filtered_bands.sel(band=slice(*bands_range))
        continous_bands = filtered_bands.dropna("k", how="all")

        # This is the new Erange
        # Erange = np.array([float(f'{val:.3f}') for val in [float(continous_bands.E.min() - 0.01), float(continous_bands.E.max() + 0.01)]])
    else:
        filtered_bands = filtered_bands.where((filtered_bands <= Erange[1]) & (filtered_bands >= Erange[0])).dropna("band", "all")
        continous_bands = filtered_bands.dropna("k", how="all")

        # This is the new bands range
        #bands_range = [int(continous_bands['band'].min()), int(continous_bands['band'].max())]

    # Give the filtered bands the same attributes as the full bands
    filtered_bands.attrs = bands_data.attrs

    filtered_bands.E.attrs = bands_data.E.attrs
    filtered_bands.E.attrs['E0'] = filtered_bands.E.attrs.get('E0', 0) + E0

    # Let's treat the spin if the user requested it
    if not isinstance(spin, (int, type(None))):
        if len(spin) > 0:
            spin = spin[0]
        else:
            spin = None

    if spin is not None:
        # Only use the spin setting if there is a spin index
        if "spin" in filtered_bands.coords:
            filtered_bands = filtered_bands.sel(spin=spin)

    return filtered_bands

@BandsProcessor.from_func
def style_bands(bands_data, bands_style={"color": "black", "width": 1}, 
    spindown_style={"color": "blue", "width": 1}):

    bands_style = {'color': 'black', 'width': 1, 'opacity': 1, **bands_style}
    for key in bands_style:
        if callable(bands_style[key]):
            bands_style[key] = bands_style[key](bands_data=bands_data)

    if 'spin' in bands_data.coords:
        spindown_style = {**bands_style, **spindown_style}
        style_arrays = {}
        for key in ['color', 'width', 'opacity']:
            style_arrays[key] = xr.DataArray([bands_style[key], spindown_style[key]], dims=['spin'])
    else:
        style_arrays = {}
        for key in ['color', 'width', 'opacity']:
            style_arrays[key] = xr.DataArray(bands_style[key])

    return bands_data.assign(style_arrays)

@BandsProcessor.from_func
def calculate_gap(bands_data):
    bands_E = bands_data['E']
    # Calculate the band gap to store it
    shifted_bands = bands_E
    above_fermi = bands_E.where(shifted_bands > 0)
    below_fermi = bands_E.where(shifted_bands < 0)
    CBbot = above_fermi.min()
    VBtop = below_fermi.max()

    CB = above_fermi.where(above_fermi == CBbot, drop=True).squeeze()
    VB = below_fermi.where(below_fermi == VBtop, drop=True).squeeze()

    gap = float(CBbot - VBtop)

    return {
        'gap': gap,
        'k': (VB["k"].values, CB['k'].values),
        'bands': (VB["band"].values, CB["band"].values),
        'spin': (VB["spin"].values, CB["spin"].values) if bands_data.attrs['spin'].is_polarized else (0, 0),
        'Es': [float(VBtop), float(CBbot)]
    }

@BandsProcessor.from_func
def sanitize_k(bands_data, k):
    """Returns the float value of a k point in the plot.

    Parameters
    ------------
    bands_data: xr.Dataset
        The dataset containing bands energy information.
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
        if 'axis' in bands_data.k.attrs and bands_data.k.attrs['axis'].get('ticktext') is not None:
            ticktext = bands_data.k.attrs['axis']['ticktext']
            tickvals = bands_data.k.attrs['axis']['tickvals']
            if k in ticktext:
                i_tick = ticktext.index(k)
                san_k = tickvals[i_tick]
            else:
                pass
                # raise ValueError(f"We can not interpret {k} as a k-location in the current bands plot")
                # This should be logged instead of raising the error

    return san_k

@BandsProcessor.from_func
def get_gap_coords(bands_data, gap_bands, from_k, to_k=None, gap_spin=0):
    """
    Calculates the coordinates of a gap given some k values.
    Parameters
    -----------
    bands_data: xr.Dataset
        The dataset containing bands energy information.
    gap_bands: array-like of int
        Length 2 array containing the band indices of the gap.
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
    gap_spin: int, optional
        the spin component where you want to draw the gap.

    Returns
    -----------
    tuple
        A tuple containing (k_values, E_values)
    """
    if to_k is None:
        to_k = from_k

    ks = [None, None]
    # Parse the names of the kpoints into their numeric values
    # if a string was provided.
    for i, val in enumerate((from_k, to_k)):
        ks[i] = sanitize_k(bands_data, val)

    VB, CB = gap_bands
    spin_bands = bands_data.E.sel(spin=gap_spin) if "spin" in bands_data.coords else bands_data.E
    Es = [spin_bands.dropna("k", "all").sel(k=k, band=band, method="nearest") for k, band in zip(ks, (VB, CB))]
    # Get the real values of ks that have been obtained
    # because we might not have exactly the ks requested
    ks = [np.ravel(E.k)[0] for E in Es]
    Es = [np.ravel(E)[0] for E in Es]

    return ks, Es

@BandsProcessor.from_func
def draw_gaps(bands_data, gap, gap_info, gap_tol, gap_color, gap_marker, direct_gaps_only, custom_gaps, E_axis: Literal["x", "y"]):
    """
    Draws the calculated gaps and the custom gaps in the plot
    """
    draw_actions = []

    # Draw gaps
    if gap:

        gapKs = [np.atleast_1d(k) for k in gap_info['k']]

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

            ks, Es = get_gap_coords(bands_data, gap_info['bands'], *gap_ks, gap_spin=gap_info.get('spin', [0])[0])
            name = "Gap"

            draw_actions.append(
                draw_gap(ks, Es, color=gap_color, name=name, marker=gap_marker, E_axis=E_axis)
            )

    # Draw the custom gaps. These are gaps that do not necessarily represent
    # the maximum and the minimum of the VB and CB.
    for custom_gap in custom_gaps:

        requested_spin = custom_gap.get("spin", None)
        if requested_spin is None:
            requested_spin = [0, 1]

        avail_spins = bands_data.get("spin", [0])

        for spin in avail_spins:
            if spin in requested_spin:
                from_k = custom_gap["from"]
                to_k = custom_gap["to"]
                color = custom_gap.get("color", None)
                name = f"Gap ({from_k}-{to_k})"
                ks, Es = get_gap_coords(bands_data, gap_info['bands'], from_k, to_k, gap_spin=spin)

                draw_actions.append(
                    draw_gap(ks, Es, color=color, name=name, marker=custom_gap.get("marker", {}), E_axis=E_axis)
                )

    return draw_actions

@BandsProcessor.from_func
def draw_gap(ks, Es, color=None, marker={}, name="Gap", E_axis: Literal["x", "y"] = "y"):
    if E_axis == "x":
        coords = {"x": Es, "y": ks}
    elif E_axis == "y":
        coords = {"y": Es, "x": ks}
    else:
        raise ValueError(f"E_axis must be either 'x' or 'y', but was {E_axis}")

    return PlotterNode.draw_line(**{
        **coords,
        'text': [f'Gap: {Es[1] - Es[0]:.3f} eV', ''],
        'name': name,
        'textposition': 'top right',
        'marker': {"size": 7, 'color': color, **marker},
        'line': {'color': color},
    })[0]