from typing import Literal

from sisl.viz.nodes.plotters.plotter import CompositePlotterNode, PlotterNode

from sisl.viz.nodes.processors.bands import calculate_gap, draw_gaps
from ..plotters import PlotterNodeXY

from .plot import Plot

from ..processors import BandsData, filter_bands, style_bands
from ..processors.axes import get_axis_var

class BandsPlot(Plot):

    @staticmethod
    def _workflow(bands_data: BandsData = None, 
        Erange=[-2, 2], E0=0, E_axis: Literal["x", "y"] = "y", bands_range=None, spin=None, 
        bands_style={'color': 'black', 'width': 1, "opacity": 1}, spindown_style={"color": "blue", "width": 1},
        gap=False, gap_tol=0.01, gap_color="red", gap_marker={"size": 7}, direct_gaps_only=False, custom_gaps=[],
        bands_mode: Literal["line", "scatter", "area_line"] = "area_line"
    ):
        if bands_data is None:
            raise ValueError("You need to provide a bands data source in `bands_data`")

        # Filter the bands
        filtered_bands = filter_bands(bands_data, Erange=Erange, E0=E0, bands_range=bands_range, spin=spin)

        # Add the styles
        styled_bands = style_bands(filtered_bands, bands_style=bands_style, spindown_style=spindown_style)

        # Determine what goes on each axis
        x = get_axis_var(axis="x", var="E", var_axis=E_axis, other_var="k")
        y = get_axis_var(axis="y", var="E", var_axis=E_axis, other_var="k")
        
        # Get the actions to plot lines
        bands_plottings = PlotterNodeXY(data=styled_bands, x=x, y=y, set_axrange=True, what=bands_mode)

        # Gap calculation
        gap_info = calculate_gap(filtered_bands)
        # Plot it if the user has asked for it.
        gaps_plottings = draw_gaps(bands_data, gap, gap_info, gap_tol, gap_color, gap_marker, direct_gaps_only, custom_gaps, E_axis=E_axis)

        return CompositePlotterNode(bands_plottings, gaps_plottings, composite_method=None)