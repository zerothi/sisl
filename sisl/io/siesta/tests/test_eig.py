import pytest
import os.path as osp
import sisl
from sisl.io.siesta.fdf import *
from sisl.io.siesta.eig import *
import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_si_pdos_kgrid_eig(sisl_files):
    f = sisl_files(_dir, 'si_pdos_kgrid.EIG')
    eig = eigSileSiesta(f).read_data()

    # nspin, nk, nb
    assert np.all(eig.shape == (1, 32, 26))


def test_si_pdos_kgrid_eig_ArgumentParser(sisl_files, sisl_tmp):
    pytest.importorskip("matplotlib", reason="matplotlib not available")
    png = sisl_tmp('si_pdos_kgrid.EIG.png', _dir)
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.EIG'))
    p, ns = si.ArgumentParser()
    p.parse_args([], namespace=ns)
    p.parse_args(['--energy', ' -2:2'], namespace=ns)
    p.parse_args(['--energy', ' -2:2', '--plot', png], namespace=ns)


def test_soc_pt2_xx_eig(sisl_files):
    f = sisl_files(_dir, 'SOC_Pt2_xx.EIG')
    eig = eigSileSiesta(f).read_data()

    # nspin, nk, nb
    # Since SO/NC mixes spin-channels it makes no sense
    # to have them separately
    assert np.all(eig.shape == (1, 1, 60))


def test_soc_pt2_xx_eig_fermi_level(sisl_files):
    f = sisl_files(_dir, 'SOC_Pt2_xx.EIG')
    ef = eigSileSiesta(f).read_fermi_level()
    fdf = sisl_files(_dir, 'SOC_Pt2_xx.fdf')
    ef1 = fdfSileSiesta(fdf).read_fermi_level(order='EIG')
    assert ef == pytest.approx(ef1)
    # This should prefer the TSHS
    ef2 = fdfSileSiesta(fdf).read_fermi_level(order='TSHS')
    # since we are using a different conversion in sisl
    # vs. siesta we have to make this.
    # once https://gitlab.com/siesta-project/siesta/-/merge_requests/30
    # is merged
    assert ef == pytest.approx(ef2, abs=1e-5)
