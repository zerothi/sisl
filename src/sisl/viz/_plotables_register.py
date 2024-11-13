# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
This file defines all the classes that are plotable.

It does so by patching them accordingly
"""
import sisl
import sisl.io.siesta as siesta
from sisl.io.sile import get_siles

from ._plotables import register_data_source, register_plotable, register_sile_method
from .data import *
from .plots import *

# ======================
#     IMPORTANT NOTE
# ======================
# If you register a new plotable class, make sure it is included in the
# lazy_viz_classes list in sisl/_lazy_viz.py, so that a placeholder is set for
# its plot attribute.

__all__ = []

register = register_plotable

# # -----------------------------------------------------
# #           Register plotable sisl objects
# # -----------------------------------------------------

# Matrices
register(sisl.SparseCSR, AtomicMatrixPlot, "matrix", default=True)
register(sisl.SparseOrbital, AtomicMatrixPlot, "matrix", default=True)
register(sisl.SparseAtom, AtomicMatrixPlot, "matrix", default=True)

# # Geometry
register(sisl.Geometry, GeometryPlot, "geometry", default=True)

# # Grid
register(sisl.Grid, GridPlot, "grid", default=True)

# Brilloiun zone
register(sisl.BrillouinZone, SitesPlot, "sites_obj")

# -----------------------------------------------------
#               Register data sources
# -----------------------------------------------------

# This will automatically register as plotable everything that
# the data source can digest

register_data_source(PDOSData, PdosPlot, "pdos_data", default=[siesta.pdosSileSiesta])
register_data_source(
    BandsData, BandsPlot, "bands_data", default=[siesta.bandsSileSiesta]
)
register_data_source(
    BandsData,
    FatbandsPlot,
    "bands_data",
    data_source_defaults={"extra_vars": ("norm2",)},
)
register_data_source(
    EigenstateData, WavefunctionPlot, "eigenstate", default=[sisl.EigenstateElectron]
)

# -----------------------------------------------------
#               Register plotable siles
# -----------------------------------------------------

for GeomSile in get_siles(attrs=["read_geometry"]):
    register_sile_method(GeomSile, "read_geometry", GeometryPlot, "geometry")

for GridSile in get_siles(attrs=["read_grid"]):
    register_sile_method(GridSile, "read_grid", GridPlot, "grid", default=True)


sisl.BandStructure.plot.set_default("bands")
sisl.Hamiltonian.plot.set_default("atomicmatrix")
