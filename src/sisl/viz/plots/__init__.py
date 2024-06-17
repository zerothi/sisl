# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Module containing all sisl-provided plots, both in a functional form and as Workflows.
"""

from .bands import BandsPlot, FatbandsPlot, bands_plot, fatbands_plot
from .geometry import GeometryPlot, SitesPlot, geometry_plot, sites_plot
from .grid import GridPlot, WavefunctionPlot, grid_plot, wavefunction_plot
from .matrix import AtomicMatrixPlot, atomic_matrix_plot
from .merged import animation, merge_plots, subplots
from .pdos import PdosPlot, pdos_plot
