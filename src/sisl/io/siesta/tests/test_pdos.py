# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" pytest test configures """


import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_si_pdos_gamma(sisl_files):
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_gamma", "Si_pdos.PDOS.xml"))
    geom, E, pdos = si.read_data()
    assert len(geom) == 2
    assert len(E) == 200
    assert pdos.shape == (1, geom.no, 200)
    assert si.read_fermi_level() == pytest.approx(-4.48376650463)


def test_si_pdos_info(sisl_files, sisl_tmp, capsys):
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_gamma", "Si_pdos.PDOS.xml"))
    p, ns = si.ArgumentParser()
    p.parse_args(["--info"], namespace=ns)
    out = capsys.readouterr().out
    assert "Spin: unpolarized" in out
    assert "200 points" in out
    assert "relative to E_F = -4.48377 eV" in out
    assert "  1-2: Si [1: 3sZ1, 2: 3pyZ1," in out
    assert "9: 3dx2-y2Z1P]" in out

    # the printed indices select all orbitals: their sum is the total DOS
    f = sisl_tmp("info.dat")
    p, ns = si.ArgumentParser()
    p.parse_args(["--norm", "none", "--atom", "1-2[1-9]", "--out", f], namespace=ns)
    E, dos = sisl.get_sile(f).read_data()
    _, E_ref, pdos = si.read_data()
    assert np.allclose(E, E_ref)
    assert np.allclose(dos, pdos[0].sum(0))


def test_si_pdos_gamma_xarray(sisl_files):
    pytest.importorskip("xarray", reason="xarray not available")
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_gamma", "Si_pdos.PDOS.xml"))
    X = si.read_data(as_dataarray=True)
    assert len(X.geometry) == 2
    assert len(X.E) == 200
    assert len(X.spin) == 1
    assert X.spin[0] == "sum"
    size = np.prod(X.shape[2:])
    assert size >= X.geometry.no


def test_si_pdos_kgrid(sisl_files):
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.PDOS.xml"))
    geom, E, pdos = si.read_data()
    assert len(geom) == 2
    assert len(E) == 200
    assert pdos.shape == (1, geom.no, 200)


def test_si_pdos_kgrid_xarray(sisl_files):
    pytest.importorskip("xarray", reason="xarray not available")
    si = sisl.get_sile(sisl_files("siesta", "Si_pdos_k", "Si_pdos.PDOS.xml"))
    X = si.read_data(as_dataarray=True)
    assert len(X.geometry) == 2
    assert len(X.E) == 200
    assert len(X.spin) == 1
    assert X.spin[0] == "sum"
    size = np.prod(X.shape[2:])
    assert size >= X.geometry.no
