# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
This file defines all the classes that are plotable.

It does so by patching them accordingly
"""
import sisl
import sisl.io.siesta as siesta
# import sisl.io.tbtrans as tbtrans
from sisl.io.sile import BaseSile, get_siles

from ._plotables import register_data_source, register_plotable, register_sile_method
from .data import *
from .plots import *

# from .old_plot import Plot
# from .plotutils import get_plot_classes


__all__ = []


# -----------------------------------------------------
#               Register data sources
# -----------------------------------------------------

# This will automatically register as plotable everything that 
# the data source can digest

register_data_source(PDOSData, PdosPlot, "pdos_data", default=[siesta.pdosSileSiesta])
register_data_source(BandsData, BandsPlot, "bands_data", default=[siesta.bandsSileSiesta])
register_data_source(BandsData, FatbandsPlot, "bands_data", data_source_init_kwargs={"extra_vars": ("norm2", )})
register_data_source(EigenstateData, WavefunctionPlot,  "eigenstate", default=[sisl.EigenstateElectron])

# -----------------------------------------------------
#               Register plotable siles
# -----------------------------------------------------

register = register_plotable

for GeomSile in get_siles(attrs=["read_geometry"]):
    register_sile_method(GeomSile, "read_geometry", GeometryPlot, 'geometry')

for GridSile in get_siles(attrs=["read_grid"]):
    register_sile_method(GridSile, "read_grid", GridPlot, 'grid', default=True)

# # -----------------------------------------------------
# #           Register plotable sisl objects
# # -----------------------------------------------------

# # Geometry
register(sisl.Geometry, GeometryPlot, 'geometry', default=True)

# # Grid
register(sisl.Grid, GridPlot, 'grid', default=True)

# Brilloiun zone
register(sisl.BrillouinZone, SitesPlot, 'sites_obj')

sisl.BandStructure.plot.set_default("bands")
