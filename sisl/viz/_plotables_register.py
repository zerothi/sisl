# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
This file defines all the classes that are plotable.

It does so by patching them accordingly
"""
import sisl
import sisl.io.siesta as siesta
import sisl.io.tbtrans as tbtrans
from sisl.io.sile import get_siles, BaseSile

from .plots import *
from .plot import Plot
from .plotutils import get_plot_classes

from ._plotables import register_plotable

__all__ = []

# -----------------------------------------------------
#               Register plotable siles
# -----------------------------------------------------

register = register_plotable

for GridSile in get_siles(attrs=["read_grid"]):
    register(GridSile, GridPlot, 'grid_file', default=True)

for GeomSile in get_siles(attrs=["read_geometry"]):
    register(GeomSile, GeometryPlot, 'geom_file', default=True)
    register(GeomSile, BondLengthMap, 'geom_file')

for HSile in get_siles(attrs=["read_hamiltonian"]):
    register(HSile, WavefunctionPlot, 'H', default=HSile != siesta.fdfSileSiesta)
    register(HSile, PdosPlot, "H")
    register(HSile, BandsPlot, "H")
    register(HSile, FatbandsPlot, "H")

for cls in get_plot_classes():
    register(siesta.fdfSileSiesta, cls, "root_fdf", overwrite=True)

# register(siesta.outSileSiesta, ForcesPlot, 'out_file', default=True)

register(siesta.bandsSileSiesta, BandsPlot, 'bands_file', default=True)
register(siesta.bandsSileSiesta, FatbandsPlot, 'bands_file')

register(siesta.pdosSileSiesta, PdosPlot, 'pdos_file', default=True)
register(tbtrans.tbtncSileTBtrans, PdosPlot, 'tbt_out', default=True)

# -----------------------------------------------------
#           Register plotable sisl objects
# -----------------------------------------------------

# Geometry
register(sisl.Geometry, GeometryPlot, 'geometry', default=True)
register(sisl.Geometry, BondLengthMap, 'geometry')

# Grid
register(sisl.Grid, GridPlot, 'grid', default=True)

# Hamiltonian
register(sisl.Hamiltonian, WavefunctionPlot, 'H', default=True)
register(sisl.Hamiltonian, PdosPlot, "H")
register(sisl.Hamiltonian, BandsPlot, "H")
register(sisl.Hamiltonian, FatbandsPlot, "H")

# Band structure
register(sisl.BandStructure, BandsPlot, "band_structure", default=True)
register(sisl.BandStructure, FatbandsPlot, "band_structure")

# Eigenstate
register(sisl.EigenstateElectron, WavefunctionPlot, 'eigenstate', default=True)
