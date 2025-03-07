# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# TODO when forward refs work with annotations
# from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np
import xarray as xr

from ..plotters import plot_actions


def filter_bands(
    bands_data: xr.Dataset,
    Erange: Optional[tuple[float, float]] = None,
    E0: float = 0,
    bands_range: Optional[tuple[int, int]] = None,
    spin: Optional[int] = None,
) -> xr.Dataset:
    filtered_bands = bands_data.copy()
    # Shift the energies according to the reference energy, while keeping the
    # attributes (which contain the units, amongst other things)
    filtered_bands["E"] = bands_data.E - E0

    # Get the bands that matter for the plot
    if Erange is None:
        if bands_range is None:
            continous_bands = filtered_bands.dropna("k", how="all", subset=["E"])

            # If neither E range or bands_range was provided, we will just plot the 15 bands below and above the fermi level
            CB = int(
                continous_bands.E.where(continous_bands.E <= 0).argmax("band").max()
            )
            bands_range = [
                int(max(continous_bands["band"].min(), CB - 15)),
                int(min(continous_bands["band"].max() + 1, CB + 16)),
            ]

        filtered_bands = filtered_bands.sel(band=slice(*bands_range))

        # This is the new Erange
        # continous_bands = filtered_bands.dropna("k", how="all", subset=["E"])
        # Erange = np.array([float(f'{val:.3f}') for val in [float(continous_bands.E.min() - 0.01), float(continous_bands.E.max() + 0.01)]])
    else:
        filtered_bands = filtered_bands.where(
            (filtered_bands <= Erange[1]) & (filtered_bands >= Erange[0])
        ).dropna("band", how="all", subset=["E"])

        # This is the new bands range
        # continous_bands = filtered_bands.dropna("k", how="all", subset=["E"])
        # bands_range = [int(continous_bands['band'].min()), int(continous_bands['band'].max())]

    # Give the filtered bands the same attributes as the full bands
    filtered_bands.attrs = bands_data.attrs

    filtered_bands.E.attrs = bands_data.E.attrs
    filtered_bands.E.attrs["E0"] = filtered_bands.E.attrs.get("E0", 0) + E0

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


def style_bands(
    bands_data: xr.Dataset,
    bands_style: dict = {"color": "black", "width": 1},
    spindown_style: dict = {"color": "blue", "width": 1},
    group_legend: bool = True,
) -> xr.Dataset:
    """Returns the bands dataset, with the style information added to it.

    Parameters
    ------------
    bands_data:
        The dataset containing bands energy information.
    bands_style:
        Dictionary containing the style information for the bands.
    spindown_style:
        Dictionary containing the style information for the spindown bands.
        Any style that is not present in this dictionary will be taken from
        the "bands_style" dictionary.
    group_legend:
        Whether the bands will be grouped in the legend. This will determine
        how the names of each band are set
    """

    # If the user provided a styler function, apply it.
    if bands_style.get("styler") is not None:
        if callable(bands_style["styler"]):
            bands_data = bands_style["styler"](data=bands_data)

    # Include default styles in bands_style, only if they are not already
    # present in the bands dataset (e.g. because the styler included them)
    default_styles = {"color": "black", "width": 1, "opacity": 1, "dash": "solid"}
    for key in default_styles:
        if key not in bands_data.data_vars and key not in bands_style:
            bands_style[key] = default_styles[key]

    # If some key in bands_style is a callable, apply it
    for key in bands_style:
        if callable(bands_style[key]):
            bands_style[key] = bands_style[key](data=bands_data)

    # Build the style dataarrays
    if "spin" in bands_data.dims:
        spindown_style = {**bands_style, **spindown_style}
        style_arrays = {}
        for key in ["color", "width", "opacity", "dash"]:
            if isinstance(bands_style[key], xr.DataArray):
                if not isinstance(spindown_style[key], xr.DataArray):
                    down_style = bands_style[key].copy(deep=True)
                    down_style.values[:] = spindown_style[key]
                    spindown_style[key] = down_style

                style_arrays[key] = xr.concat(
                    [bands_style[key], spindown_style[key]], dim="spin"
                )
            else:
                style_arrays[key] = xr.DataArray(
                    [bands_style[key], spindown_style[key]], dims=["spin"]
                )

        # Determine the names of the bands
        if group_legend:
            style_arrays["line_name"] = xr.DataArray(
                ["Spin up Bands", "Spin down Bands"], dims=["spin"]
            )
        else:
            names = []
            for s in bands_data.spin:
                spin_string = "UP" if s == 0 else "DOWN"
                for iband in bands_data.band.values:
                    names.append(f"{spin_string}_{iband}")

            style_arrays["line_name"] = xr.DataArray(
                np.array(names).reshape(2, -1),
                coords=[
                    ("spin", bands_data.spin.values),
                    ("band", bands_data.band.values),
                ],
            )
    else:
        style_arrays = {}
        for key in ["color", "width", "opacity", "dash"]:
            style_arrays[key] = xr.DataArray(bands_style[key])

        # Determine the names of the bands
        if group_legend:
            style_arrays["line_name"] = xr.DataArray("Bands")
        else:
            style_arrays["line_name"] = bands_data.band

    # Merge the style arrays with the bands dataset and return the styled dataset
    return bands_data.assign(style_arrays)


def calculate_gap(bands_data: xr.Dataset) -> dict:
    bands_E = bands_data.E
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
        "gap": gap,
        "k": (VB["k"].values, CB["k"].values),
        "bands": (VB["band"].values, CB["band"].values),
        "spin": (
            (VB["spin"].values, CB["spin"].values)
            if bands_data.attrs["spin"].is_polarized
            else (0, 0)
        ),
        "Es": (float(VBtop), float(CBbot)),
    }


def sanitize_k(bands_data: xr.Dataset, k: Union[float, str]) -> Optional[float]:
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
        if (
            "axis" in bands_data.k.attrs
            and bands_data.k.attrs["axis"].get("ticktext") is not None
        ):
            ticktext = bands_data.k.attrs["axis"]["ticktext"]
            tickvals = bands_data.k.attrs["axis"]["tickvals"]
            if k in ticktext:
                i_tick = ticktext.index(k)
                san_k = tickvals[i_tick]
            else:
                pass
                # raise ValueError(f"We can not interpret {k} as a k-location in the current bands plot")
                # This should be logged instead of raising the error

    return san_k


def get_gap_coords(
    bands_data: xr.Dataset,
    bands: tuple[int, int],
    from_k: Union[float, str],
    to_k: Optional[Union[float, str]] = None,
    spin: int = 0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calculates the coordinates of a gap given some k values.

    Parameters
    -----------
    bands_data: xr.Dataset
        The dataset containing bands energy information.
    bands: array-like of int
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
    spin: int, optional
        the spin component where you want to draw the gap. Has no effect
        if the bands are not spin-polarized.

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

    VB, CB = bands
    spin_bands = (
        bands_data.E.sel(spin=spin) if "spin" in bands_data.coords else bands_data.E
    )
    Es = [
        spin_bands.dropna("k", how="all").sel(k=k, band=band, method="nearest")
        for k, band in zip(ks, (VB, CB))
    ]
    # Get the real values of ks that have been obtained
    # because we might not have exactly the ks requested
    ks = tuple(np.ravel(E.k)[0] for E in Es)
    Es = tuple(np.ravel(E)[0] for E in Es)

    return ks, Es


def draw_gaps(
    bands_data: xr.Dataset,
    gap: bool,
    gap_info: dict,
    gap_tol: float,
    gap_color: Optional[str],
    gap_marker: Optional[dict],
    direct_gaps_only: bool,
    custom_gaps: Sequence[dict],
    E_axis: Literal["x", "y"],
) -> list[dict]:
    """Returns the drawing actions to draw gaps.

    Parameters
    ------------
    bands_data :
        The dataset containing bands energy information.
    gap :
        Whether to draw the minimum gap passed as gap_info or not.
    gap_info :
        Dictionary containing the information of the minimum gap,
        as returned by `calculate_gap`.
    gap_tol :
        Tolerance in k to consider that two gaps are the same.
    gap_color :
        Color of the line that draws the gap.
    gap_marker :
        Marker specification of the limits of the gap.
    direct_gaps_only :
        Whether to draw the minimum gap only if it is a direct gap.
    custom_gaps :
        List of custom gaps to draw. Each dict can contain the keys:
        - "from": the k value where the gap starts.
        - "to": the k value where the gap ends. If not present, equal to "from".
        - "spin": For which spin component do you want to draw the gap
        (has effect only if spin is polarized). Optional. If None and the bands
        are polarized, the gap will be drawn for both spin components.
        - "color": Color of the line that draws the gap. Optional.
        - "marker": Marker specification for the limits of the gap. Optional.
    E_axis:
        Axis where the energy is plotted.
    """
    draw_actions = []

    # Draw gaps
    if gap:
        gapKs = [np.atleast_1d(k) for k in gap_info["k"]]

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

            ks, Es = get_gap_coords(
                bands_data,
                gap_info["bands"],
                *gap_ks,
                spin=gap_info.get("spin", [0])[0],
            )
            name = "Gap"

            draw_actions.append(
                draw_gap(
                    ks, Es, color=gap_color, name=name, marker=gap_marker, E_axis=E_axis
                )
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
                to_k = custom_gap.get("to", from_k)
                color = custom_gap.get("color", None)
                name = f"Gap ({from_k}-{to_k})"
                ks, Es = get_gap_coords(
                    bands_data, gap_info["bands"], from_k, to_k, spin=spin
                )

                draw_actions.append(
                    draw_gap(
                        ks,
                        Es,
                        color=color,
                        name=name,
                        marker=custom_gap.get("marker", {}),
                        E_axis=E_axis,
                    )
                )

    return draw_actions


def draw_gap(
    ks: tuple[float, float],
    Es: tuple[float, float],
    color: Optional[str] = None,
    marker: dict = {},
    name: str = "Gap",
    E_axis: Literal["x", "y"] = "y",
) -> dict:
    """Returns the drawing action to draw a gap.

    Parameters
    ------------
    ks: tuple of float
        The k values where the gap starts and ends.
    Es: tuple of float
        The energy values where the gap starts and ends.
    color: str or None
        Color of the line that draws the gap.
    marker: dict
        Marker specification for the limits of the gap.
    name: str
        Name to give to the line that draws the gap.
    E_axis: Literal["x", "y"]
        Axis where the energy is plotted.
    """
    if E_axis == "x":
        coords = {"x": Es, "y": ks}
    elif E_axis == "y":
        coords = {"y": Es, "x": ks}
    else:
        raise ValueError(f"E_axis must be either 'x' or 'y', but was {E_axis}")

    return plot_actions.draw_line(
        **{
            **coords,
            "text": [f"Gap: {Es[1] - Es[0]:.3f} eV", ""],
            "name": name,
            "textposition": "top right",
            "marker": {"size": 7, "color": color, **marker},
            "line": {"color": color},
        }
    )
