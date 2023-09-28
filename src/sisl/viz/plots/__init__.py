"""
Module containing all sisl-provided plots, both in a functional form and as Workflows.
"""

from .bands import BandsPlot, FatbandsPlot, bands_plot, fatbands_plot
from .geometry import GeometryPlot, SitesPlot, geometry_plot, sites_plot
from .grid import GridPlot, WavefunctionPlot, grid_plot, wavefunction_plot
from .merged import merge_plots
from .pdos import PdosPlot, pdos_plot
