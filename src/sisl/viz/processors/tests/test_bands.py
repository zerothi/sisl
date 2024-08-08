# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sisl import Spin
from sisl.viz.data import BandsData
from sisl.viz.processors.bands import (
    calculate_gap,
    draw_gap,
    draw_gaps,
    filter_bands,
    get_gap_coords,
    sanitize_k,
    style_bands,
)

pytestmark = [pytest.mark.viz, pytest.mark.processors]


@pytest.fixture(
    scope="module", params=["unpolarized", "polarized", "noncolinear", "spinorbit"]
)
def spin(request):
    return Spin(request.param)


@pytest.fixture(scope="module")
def gap():
    return 2.5


@pytest.fixture(scope="module")
def bands_data(spin, gap):
    return BandsData.toy_example(spin=spin, gap=gap)


@pytest.fixture(scope="module", params=["x", "y"])
def E_axis(request):
    return request.param


def test_filter_bands(bands_data):
    spin = bands_data.attrs["spin"]

    # Check that it works without any arguments
    filtered_bands = filter_bands(bands_data)

    # Test filtering by band index.
    filtered_bands = filter_bands(bands_data, bands_range=[0, 5])
    assert np.all(filtered_bands.band == np.arange(0, 6))

    # Test filtering by energy. First check that we actually
    # have bands beyond the energy range that we want to test.
    assert bands_data.E.min() <= -5
    assert bands_data.E.max() >= 5

    filtered_bands = filter_bands(bands_data, Erange=[-5, 5])

    assert filtered_bands.E.min() >= -5
    assert filtered_bands.E.max() <= 5

    if spin.is_polarized:
        filtered_bands = filter_bands(bands_data, Erange=[-5, 5], spin=1)

        assert filtered_bands.E.min() >= -5
        assert filtered_bands.E.max() <= 5

        assert filtered_bands.spin == 1


def test_calculate_gap(bands_data, gap):
    spin = bands_data.attrs["spin"]

    gap_info = calculate_gap(bands_data)

    # Check that the gap value is correct
    assert gap_info["gap"] == gap

    # Check also that the position of the gap is in the information
    assert isinstance(gap_info["k"], tuple) and len(gap_info["k"]) == 2

    VB = len(bands_data.band) // 2 - 1
    assert isinstance(gap_info["bands"], tuple) and len(gap_info["bands"]) == 2
    assert gap_info["bands"][0] < gap_info["bands"][1]

    assert isinstance(gap_info["spin"], tuple) and len(gap_info["spin"]) == 2
    if not spin.is_polarized:
        assert gap_info["spin"] == (0, 0)

    assert isinstance(gap_info["Es"], tuple) and len(gap_info["Es"]) == 2
    assert np.allclose(gap_info["Es"], (-gap / 2, gap / 2))


def test_sanitize_k(bands_data):
    assert sanitize_k(bands_data, "Gamma") == 0
    assert sanitize_k(bands_data, "X") == 1


def test_get_gap_coords(bands_data):
    spin = bands_data.attrs["spin"]

    vb = len(bands_data.band) // 2 - 1

    # We can get the gamma gap by specifying both origin and destination or
    # just origin. Check also Gamma to X.
    for to_k in ["Gamma", None, "X"]:
        k, E = get_gap_coords(
            bands_data, (vb, vb + 1), from_k="Gamma", to_k=to_k, spin=1
        )

        kval = 1 if to_k == "X" else 0

        # Check that the K is correct
        assert k[0] == 0
        assert k[1] == kval

        # Check that E is correct.
        if spin.is_polarized:
            bands_E = bands_data.E.sel(spin=1)
        else:
            bands_E = bands_data.E

        assert E[0] == bands_E.sel(band=vb, k=0)
        assert E[1] == bands_E.sel(band=vb + 1, k=kval)


def test_draw_gap(E_axis):
    ks = (0, 0.5)
    Es = (0, 1)

    if E_axis == "x":
        x, y = Es, ks
    else:
        x, y = ks, Es

    gap_action = draw_gap(ks, Es, color="red", name="test", E_axis=E_axis)

    assert isinstance(gap_action, dict)
    assert gap_action["method"] == "draw_line"

    action_kwargs = gap_action["kwargs"]

    assert action_kwargs["name"] == "test"
    assert action_kwargs["line"]["color"] == "red"
    assert action_kwargs["marker"]["color"] == "red"
    assert action_kwargs["x"] == x
    assert action_kwargs["y"] == y


@pytest.mark.parametrize("display_gap", [True, False])
def test_draw_gaps(bands_data, E_axis, display_gap):
    spin = bands_data.attrs["spin"]

    gap_info = calculate_gap(bands_data)

    # Run the function only to draw the minimum gap.
    gap_actions = draw_gaps(
        bands_data,
        gap=display_gap,
        gap_info=gap_info,
        gap_tol=0.3,
        gap_color="red",
        gap_marker={},
        direct_gaps_only=False,
        custom_gaps=[],
        E_axis=E_axis,
    )

    assert isinstance(gap_actions, list)
    assert len(gap_actions) == (1 if display_gap else 0)

    if display_gap:
        assert isinstance(gap_actions[0], dict)
        assert gap_actions[0]["method"] == "draw_line"

        action_kwargs = gap_actions[0]["kwargs"]
        assert action_kwargs["line"]["color"] == "red"
        assert action_kwargs["marker"]["color"] == "red"

    # Now run the function with a custom gap.
    gap_actions = draw_gaps(
        bands_data,
        gap=display_gap,
        gap_info=gap_info,
        gap_tol=0.3,
        gap_color="red",
        gap_marker={},
        direct_gaps_only=False,
        custom_gaps=[{"from": "Gamma", "to": "X", "color": "blue"}],
        E_axis=E_axis,
    )

    assert isinstance(gap_actions, list)
    assert len(gap_actions) == (2 if display_gap else 1) + (
        1 if spin.is_polarized else 0
    )

    # Check the minimum gap
    if display_gap:
        assert isinstance(gap_actions[0], dict)
        assert gap_actions[0]["method"] == "draw_line"

        action_kwargs = gap_actions[0]["kwargs"]
        assert action_kwargs["line"]["color"] == "red"
        assert action_kwargs["marker"]["color"] == "red"

    # Check the custom gap
    assert isinstance(gap_actions[-1], dict)
    assert gap_actions[-1]["method"] == "draw_line"

    action_kwargs = gap_actions[-1]["kwargs"]
    assert action_kwargs["line"]["color"] == "blue"
    assert action_kwargs["marker"]["color"] == "blue"
    assert action_kwargs["x" if E_axis == "y" else "y"] == (0, 1)


def test_style_bands(bands_data):
    spin = bands_data.attrs["spin"]

    # Check basic styles
    styled_bands = style_bands(
        bands_data,
        {"color": "red", "width": 3},
        spindown_style={"opacity": 0.5, "color": "blue"},
    )

    assert isinstance(styled_bands, xr.Dataset)

    for k in ("color", "width", "opacity"):
        assert k in styled_bands.data_vars

    if not spin.is_polarized:
        assert styled_bands.color == "red"
        assert styled_bands.width == 3
        assert styled_bands.opacity == 1
    else:
        assert np.all(styled_bands.color == ["red", "blue"])
        assert np.all(styled_bands.width == [3, 3])
        assert np.all(styled_bands.opacity == [1, 0.5])

    # Check function as style
    def color(data):
        return xr.DataArray(
            np.where(data.band < 5, "red", "blue"), coords=[("band", data.band.values)]
        )

    styled_bands = style_bands(
        bands_data,
        {"color": color, "width": 3},
        spindown_style={"opacity": 0.5, "color": "blue"},
    )

    assert isinstance(styled_bands, xr.Dataset)

    for k in ("color", "width", "opacity"):
        assert k in styled_bands.data_vars

    assert "band" in styled_bands.color.coords
    if spin.is_polarized:
        bands_color = styled_bands.color.sel(spin=0)
        assert np.all(styled_bands.color.sel(spin=1) == "blue")
    else:
        bands_color = styled_bands.color
    assert np.all((styled_bands.band < 5) == (bands_color == "red"))
