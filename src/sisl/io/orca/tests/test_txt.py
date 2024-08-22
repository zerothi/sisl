# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl.io.orca.txt import *

pytestmark = [pytest.mark.io, pytest.mark.orca]


def test_tags(sisl_files):
    f = sisl_files("orca", "nitric_oxide", "molecule_property.txt")
    out = txtSileORCA(f)
    assert out.info.na == 2


def test_read_electrons(sisl_files):
    f = sisl_files("orca", "nitric_oxide", "molecule_property.txt")
    out = txtSileORCA(f)
    N = out.read_electrons[:]()
    assert N[0, 0] == pytest.approx(7.9999985377)
    assert N[0, 1] == pytest.approx(6.9999989872)
    N = out.read_electrons[-1]()
    assert N[0] == pytest.approx(7.9999985377)
    assert N[1] == pytest.approx(6.9999989872)


def test_read_energy(sisl_files):
    f = sisl_files("orca", "nitric_oxide", "molecule_property.txt")
    out = txtSileORCA(f)
    E = out.read_energy[:]()
    assert len(E) == 2
    E = out.read_energy[-1]()
    assert pytest.approx(E.total) == pytest.approx(-3532.4797529097723)


def test_read_energy_vdw(sisl_files):
    f = sisl_files("orca", "carbon_monoxide", "molecule_property.txt")
    out = txtSileORCA(f)
    E = out.read_energy[:]()
    assert len(E) == 2
    assert pytest.approx(E[0].total) == pytest.approx(-3081.2651523095283)
    assert pytest.approx(E[1].total) == pytest.approx(-3081.2651523149702)
    assert pytest.approx(E[1].vdw) == pytest.approx(-0.011180550414138613)
    E = out.read_energy[-1]()
    assert pytest.approx(E.total) == pytest.approx(-3081.2651523149702)
    assert pytest.approx(E.vdw) == pytest.approx(-0.011180550414138613)


def test_read_geometry(sisl_files):
    f = sisl_files("orca", "nitric_oxide", "molecule_property.txt")
    out = txtSileORCA(f)
    G = out.read_geometry[:]()
    assert G[0].xyz[0, 0] == pytest.approx(0.421218019838, abs=1e-6)
    assert G[0].xyz[1, 0] == pytest.approx(1.578781980162, abs=1e-6)
    assert G[1].xyz[0, 0] == pytest.approx(0.421218210279, abs=1e-6)
    assert G[1].xyz[1, 0] == pytest.approx(1.578781789721, abs=1e-6)
    G = out.read_geometry[-1]()
    assert G.xyz[0, 0] == pytest.approx(0.421218210279, abs=1e-6)
    assert G.xyz[1, 0] == pytest.approx(1.578781789721, abs=1e-6)
    assert G.xyz[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert G.xyz[1, 1] == pytest.approx(0.0, abs=1e-6)
    assert G.atoms[0].tag == "N"
    assert G.atoms[1].tag == "O"


def test_multiple_calls(sisl_files):
    f = sisl_files("orca", "nitric_oxide", "molecule_property.txt")
    out = txtSileORCA(f)
    N = out.read_electrons[:]()
    assert len(N) == 2
    E = out.read_energy[:]()
    assert len(E) == 2
    G = out.read_geometry[:]()
    assert len(G) == 2
    N = out.read_electrons[:]()
    assert len(N) == 2


def test_info_no(sisl_files):
    f = sisl_files("orca", "phenalenyl", "molecule_property.txt")
    out = txtSileORCA(f)
    assert out.info.no == 284


def test_gtensor(sisl_files):
    # file without g-tensor
    f = sisl_files("orca", "nitric_oxide", "molecule_property.txt")
    out = txtSileORCA(f)
    assert out.read_gtensor() is None

    # file with g-tensor
    f = sisl_files("orca", "phenalenyl", "molecule_property.txt")
    out = txtSileORCA(f)
    G = out.read_gtensor()

    assert G.multiplicity == 2
    assert G.tensor[0, 0] == pytest.approx(2.002750)
    assert G.vectors[0, 1] == pytest.approx(-0.009799, abs=1e-4)
    for i in range(3):
        v = G.vectors[i]
        assert v.dot(v) == pytest.approx(1)
    assert G.eigenvalues[0] == pytest.approx(2.002127)


def test_hyperfine_coupling(sisl_files):
    # file without hyperfine_coupling tensors
    f = sisl_files("orca", "nitric_oxide", "molecule_property.txt")
    out = txtSileORCA(f)
    assert out.read_hyperfine_coupling(units="MHz") is None

    # file with hyperfine_coupling tensors
    f = sisl_files("orca", "phenalenyl", "molecule_property.txt")
    out = txtSileORCA(f)
    A = out.read_hyperfine_coupling(units="MHz")
    assert len(A) == 22
    assert A[0].iso == pytest.approx(-23.380794)
    assert A[1].ia == 1
    assert A[1].species == "C"
    assert A[1].isotope == 13
    assert A[1].spin == 0.5
    assert A[1].prefactor == pytest.approx(134.190303)
    assert A[1].tensor[0, 1] == pytest.approx(-0.320129)
    assert A[1].tensor[2, 2] == pytest.approx(68.556557)
    assert A[1].vectors[1, 0] == pytest.approx(0.407884)
    for i in range(3):
        v = A[1].vectors[i]
        assert v.dot(v) == pytest.approx(1)
    assert A[1].eigenvalues[1] == pytest.approx(5.523380)
    assert A[1].iso == pytest.approx(26.247902)
    assert A[12].species == "C"
    assert A[13].species == "H"


def test_hyperfine_coupling_units(sisl_files):
    f = sisl_files("orca", "phenalenyl", "molecule_property.txt")
    out = txtSileORCA(f)
    A = out.read_hyperfine_coupling()
    B = out.read_hyperfine_coupling(units="eV")
    C = out.read_hyperfine_coupling(units={"energy": "eV"})

    assert A[0].iso == B[0].iso == C[0].iso

    A = out.read_hyperfine_coupling(units=("MHz", "Ang"))
    B = out.read_hyperfine_coupling(units={"energy": "MHz"})

    assert A[0].iso == B[0].iso
    assert A[0].iso != C[0].iso
