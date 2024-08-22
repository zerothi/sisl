# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl.io.siesta.eig import *
from sisl.io.siesta.fdf import *

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def _convert45(unit):
    """Convert from legacy units to CODATA2018"""
    from sisl.unit.siesta import units, units_legacy

    return units(unit) / units_legacy(unit)


def test_si_pdos_kgrid_eig(sisl_files):
    f = sisl_files("siesta", "Si_pdos_k", "Si_pdos.EIG")
    eig = eigSileSiesta(f).read_data()

    # nspin, nk, nb
    assert np.all(eig.shape == (1, 63, 18))


def test_si_pdos_kgrid_eig_ArgumentParser(sisl_files, sisl_tmp):
    pytest.importorskip("matplotlib", reason="matplotlib not available")
    png = sisl_tmp("si_pdos_kgrid.EIG.png")
    si = eigSileSiesta(sisl_files("siesta", "Si_pdos_k", "Si_pdos.EIG"))
    p, ns = si.ArgumentParser()
    p.parse_args([], namespace=ns)
    p.parse_args(["--energy", " -2:2"], namespace=ns)
    p.parse_args(["--energy", " -2:2", "--plot", png], namespace=ns)


def test_si_pdos_kgrid_eig_ArgumentParser_de(sisl_files, sisl_tmp):
    dat = sisl_tmp("si_pdos_kgrid.EIG.dat")
    kp = sisl_files("siesta", "Si_pdos_k", "Si_pdos.KP")
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.EIG"))
    p, ns = si.ArgumentParser()
    assert ns._dos_args[0] == pytest.approx(0.005)
    assert ns._dos_args[2] == "gaussian"
    assert ns._weights is not None
    ns._weights = None
    p.parse_args(["-kp", str(kp)], namespace=ns)
    assert ns._weights is not None
    p.parse_args(["--dos", "dE=1meV", "kT=20meV"], namespace=ns)
    assert ns._weights is not None
    assert ns._dos_args[0] == pytest.approx(1e-3)
    assert ns._dos_args[1] == pytest.approx(20e-3)
    assert ns._dos_args[2] == "gaussian"
    p.parse_args(["--dos", "kT=10meV", "dist=lorentzian"], namespace=ns)
    assert ns._dos_args[0] == pytest.approx(1e-3)
    assert ns._dos_args[1] == pytest.approx(10e-3)
    assert ns._dos_args[2] == "lorentzian"
    assert len(ns._data) == 3
    p.parse_args(["--dos", "1meV", "5meV", "gaussian"], namespace=ns)
    assert ns._dos_args[0] == pytest.approx(1e-3)
    assert ns._dos_args[1] == pytest.approx(5e-3)
    assert ns._dos_args[2] == "gaussian"
    assert len(ns._data) == 4
    p.parse_args(["--out", dat], namespace=ns)


def test_soc_pt2_xx_eig(sisl_files):
    f = sisl_files("siesta", "Pt2_soc", "Pt2_xx.EIG")
    eig = eigSileSiesta(f).read_data()

    # nspin, nk, nb
    # Since SO/NC mixes spin-channels it makes no sense
    # to have them separately
    assert np.all(eig.shape == (1, 1, 76))


def test_soc_pt2_xx_eig_fermi_level(sisl_files):
    f = sisl_files("siesta", "Pt2_soc", "Pt2_xx.EIG")
    ef = eigSileSiesta(f).read_fermi_level()
    fdf = sisl_files("siesta", "Pt2_soc", "Pt2.fdf")
    ef1 = fdfSileSiesta(fdf).read_fermi_level(order="EIG")
    assert ef == pytest.approx(ef1)
    # This should prefer the TSHS
    ef2 = fdfSileSiesta(fdf).read_fermi_level(order="TSHS")

    # This test is for 5, with codata changes.
    assert ef == pytest.approx(ef2, abs=1e-5)
