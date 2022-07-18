import xarray as xr
import pytest

import sisl
from sisl import Spin
from sisl.viz.nodes import lazy_context
from sisl.viz.nodes.processors.bands import BandsDataSIESTA, BandsDataH, BandsDataWFSX

def _check_bands_data(bands_data):
    assert isinstance(bands_data, xr.Dataset)
    assert "k" in bands_data.dims
    assert "E" in bands_data
    assert "k" in bands_data.coords

@pytest.mark.parametrize("filename, spin", [
    ("SrTiO3.bands", Spin.UNPOLARIZED),
])
@lazy_context(nodes=False)
def test_siesta_bands_file(sisl_files, filename, spin):
    """Tests reading and normalizing PDOS data from .PDOS siesta file."""
    
    bands_file = sisl_files(f"sisl/io/siesta/{filename}")
    bands_data = BandsDataSIESTA(bands_file=bands_file)
    
    _check_bands_data(bands_data)

@pytest.mark.parametrize("spin", [Spin.UNPOLARIZED, Spin.POLARIZED, Spin.NONCOLINEAR, Spin.SPINORBIT])
@lazy_context(nodes=False)
def test_H(spin):
    """Tests computing and normalizing PDOS data from a sisl hamiltonian"""
    gr = sisl.geom.graphene()
    H = sisl.Hamiltonian(gr)
    H.construct([(0.1, 1.44), (0, -2.7)])

    H = H.transform(spin=spin)

    bz = sisl.BandStructure(H, [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]], 6, ["Gamma", "M", "K"])

    data = BandsDataH(band_structure=bz)

    _check_bands_data(data)

@lazy_context(nodes=False)
def test_siesta_WFSX_file(sisl_files):
    """Tests computing and normalizing PDOS data from a .WFSX siesta file."""
    fdf = sisl_files(f"sisl/io/siesta/bi2se3_3ql.fdf")
    wfsx = sisl_files(f"sisl/io/siesta/bi2se3_3ql.bands.WFSX")

    data = BandsDataWFSX(fdf=fdf, wfsx_file=wfsx)

    _check_bands_data(data)

