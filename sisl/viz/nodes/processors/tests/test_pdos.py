import xarray as xr
import pytest

import sisl
from sisl import Spin, Geometry
from sisl.viz.nodes import lazy_context
from sisl.viz.nodes.processors.pdos import PDOSDataH, PDOSDataSIESTA, PDOSDataTBTrans, PDOSDataWFSX

def _check_pdos_data(pdos_data, spin, no, nE, Emin, Emax):
    assert isinstance(pdos_data, xr.DataArray)
    assert "orb" in pdos_data.dims
    assert "E" in pdos_data.dims

    assert "geometry" in pdos_data.attrs

    geometry = pdos_data.attrs["geometry"]
    assert isinstance(geometry, Geometry)

    assert geometry.no == no == len(pdos_data.orb)
    assert nE == len(pdos_data.E)
    assert Emin == pytest.approx(pdos_data.E[0], abs=1e-3)
    assert Emax == pytest.approx(pdos_data.E[-1], abs=1e-3)

    assert "spin" in pdos_data.attrs
    assert pdos_data.attrs["spin"].kind == spin.kind

    if spin.is_unpolarized:
        assert "spin" not in pdos_data.dims
    else:
        assert "spin" in pdos_data.dims
        n_spin = 2 if spin.is_polarized else 4
        assert len(pdos_data.spin) == n_spin

@pytest.mark.parametrize("filename, spin", [
    ("SrTiO3.PDOS", Spin.UNPOLARIZED),
    ("SrTiO3_polarized.PDOS", Spin.POLARIZED),
    ("SrTiO3_noncollinear.PDOS", Spin.NONCOLINEAR),
])
@lazy_context(nodes=False)
def test_siesta_PDOS_file(sisl_files, filename, spin):
    """Tests reading and normalizing PDOS data from .PDOS siesta file."""
    delta_E = 0.025
    nE = 3000
    Emin = -64.097182
    Emax = Emin + nE * delta_E
    
    pdos_file = sisl_files(f"sisl/io/siesta/{filename}")
    pdos_data = PDOSDataSIESTA(pdos_file=pdos_file)
    
    _check_pdos_data(pdos_data._data, spin=Spin(spin), no=72, nE=nE, Emin=Emin, Emax=Emax)

@pytest.mark.parametrize("spin", [Spin.UNPOLARIZED, Spin.POLARIZED, Spin.NONCOLINEAR, Spin.SPINORBIT])
@lazy_context(nodes=False)
def test_H(spin):
    """Tests computing and normalizing PDOS data from a sisl hamiltonian"""
    gr = sisl.geom.graphene()
    H = sisl.Hamiltonian(gr)
    H.construct([(0.1, 1.44), (0, -2.7)])

    H = H.transform(spin=spin)

    nE = 200
    Erange = [-10, 10]

    data = PDOSDataH(H=H, nE=nE, Erange=Erange)._data

    _check_pdos_data(data, spin=Spin(spin), no=gr.no, nE=nE, Emin=Erange[0], Emax=Erange[-1])

@lazy_context(nodes=False)
def test_siesta_WFSX_file(sisl_files):
    """Tests computing and normalizing PDOS data from a .WFSX siesta file."""
    wfsx = sisl.get_sile(sisl_files(f"sisl/io/siesta/bi2se3_3ql.bands.WFSX"))
    geometry = sisl.get_sile(sisl_files(f"sisl/io/siesta/bi2se3_3ql.fdf")).read_geometry()
    geometry = sisl.Geometry(geometry.xyz, atoms=wfsx.read_basis())

    # Since there is no hamiltonian for bi2se3_3ql.fdf, we create a dummy one
    H = sisl.Hamiltonian(geometry, dim=4)

    nE = 200
    Erange = [-10, 10]

    data = PDOSDataWFSX(H=H, wfsx_file=wfsx, nE=nE, Erange=Erange)._data

    _check_pdos_data(data, spin=Spin(Spin.NONCOLINEAR), no=geometry.no, nE=nE, Emin=Erange[0], Emax=Erange[-1])

@lazy_context(nodes=False)
def test_tbtrans_TBT_nc(sisl_files):
    """Tests computing and normalizing PDOS data from a .WFSX siesta file."""

    tbt_nc = sisl_files("sisl/io/tbtrans/1_graphene_all.TBT.nc")
    data = PDOSDataTBTrans(tbt_nc=tbt_nc)._data

    _check_pdos_data(data, spin=Spin(Spin.UNPOLARIZED), no=36, nE=400, Emin=-1.995, Emax=1.995)

