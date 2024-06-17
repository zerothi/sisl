# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os.path as osp
import sys

import numpy as np
import pytest

import sisl
from sisl.io.orca.stdout import *

pytestmark = [pytest.mark.io, pytest.mark.orca]
_dir = osp.join("sisl", "io", "orca")


def test_tags(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)
    assert out.info.na == 2
    assert out.info.no == 62
    assert out.completed()


def test_read_electrons(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)

    N = out.read_electrons[:]()
    assert N[0, 0] == 7.999998537730
    assert N[0, 1] == 6.999998987205

    N = out.read_electrons[-1]()
    assert N[0] == 7.999998537734
    assert N[1] == 6.999998987209


def test_charge_name(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)
    for name in ["mulliken", "MULLIKEN", "loewdin", "Lowdin", "LÖWDIN"]:
        assert out.read_charge(name=name) is not None


def test_charge_mulliken_atom(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)

    C = out.read_charge[:](name="mulliken", projection="atom")
    S = out.read_charge[:](name="mulliken", projection="atom", spin=True)
    assert len(C) == 2
    assert C[0][0] == 0.029160
    assert S[0][0] == 0.687779
    assert C[0][1] == -0.029160
    assert S[0][1] == 0.312221
    assert C[1][0] == 0.029158
    assert S[1][0] == 0.687793
    assert C[1][1] == -0.029158
    assert S[1][1] == 0.312207

    C = out.read_charge[-1](name="mulliken", projection="atom")
    S = out.read_charge[-1](name="mulliken", projection="atom", spin=True)
    assert C[0] == 0.029158
    assert S[0] == 0.687793
    assert C[1] == -0.029158
    assert S[1] == 0.312207


def test_lowedin_atom(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)

    C = out.read_charge[:](name="loewdin", projection="atom")
    S = out.read_charge[:](name="loewdin", projection="atom", spin=True)
    assert len(C) == 2
    assert C[0][0] == -0.111221
    assert S[0][0] == 0.660316
    assert C[0][1] == 0.111221
    assert S[0][1] == 0.339684
    assert C[1][0] == -0.111223
    assert S[1][0] == 0.660327
    assert C[1][1] == 0.111223
    assert S[1][1] == 0.339673

    C = out.read_charge[-1](name="loewdin", projection="atom")
    S = out.read_charge[-1](name="loewdin", projection="atom", spin=True)
    assert C[0] == -0.111223
    assert S[0] == 0.660327
    assert C[1] == 0.111223
    assert S[1] == 0.339673


def test_charge_mulliken_reduced(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)

    C = out.read_charge[:](name="mulliken", projection="orbital")
    S = out.read_charge[:](name="mulliken", projection="orbital", spin=True)
    assert len(C) == 2
    # first charge block
    assert C[0][(0, "s")] == 3.915850
    assert C[0][(0, "pz")] == 0.710261
    assert C[0][(1, "dz2")] == 0.004147
    assert C[0][(1, "p")] == 4.116068
    # first spin block
    assert S[0][(0, "dx2y2")] == 0.001163
    assert S[0][(1, "f+2")] == -0.000122
    # last charge block
    assert C[1][(0, "pz")] == 0.710263
    assert C[1][(0, "f0")] == 0.000681
    assert C[1][(1, "s")] == 3.860487
    # last spin block
    assert S[1][(0, "p")] == 0.685743
    assert S[1][(1, "dz2")] == -0.000163

    C = out.read_charge[-1](name="mulliken", projection="orbital")
    S = out.read_charge[-1](name="mulliken", projection="orbital", spin=True)
    # last charge block
    assert C[(0, "pz")] == 0.710263
    assert C[(0, "f0")] == 0.000681
    assert C[(1, "s")] == 3.860487
    # last spin block
    assert S[(0, "p")] == 0.685743
    assert S[(1, "dz2")] == -0.000163

    C = out.read_charge[:](name="mulliken", projection="orbital", orbitals="pz")
    assert C[0][0] == 0.710261

    S = out.read_charge[:](
        name="mulliken", projection="orbital", orbitals="f+2", spin=True
    )
    assert S[0][1] == -0.000122

    S = out.read_charge[-1](
        name="mulliken", projection="orbital", orbitals="p", spin=True
    )
    assert S[0] == 0.685743


def test_charge_loewdin_reduced(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)

    C = out.read_charge[:](name="loewdin", projection="orbital")
    S = out.read_charge[:](name="loewdin", projection="orbital", spin=True)
    assert len(S) == 2
    assert C[0][(0, "s")] == 3.553405
    assert C[0][(0, "pz")] == 0.723111
    assert C[1][(0, "pz")] == 0.723113
    assert S[1][(1, "pz")] == -0.010829

    C = out.read_charge[-1](name="loewdin", projection="orbital")
    S = out.read_charge[-1](name="loewdin", projection="orbital", spin=True)
    assert C[(0, "f-3")] == 0.017486
    assert S[(1, "pz")] == -0.010829
    assert C[(0, "pz")] == 0.723113
    assert S[(1, "pz")] == -0.010829

    C = out.read_charge[:](name="loewdin", projection="orbital", orbitals="s")
    assert C[0][0] == 3.553405

    C = out.read_charge[-1](name="loewdin", projection="orbital", orbitals="f-3")
    assert C[0] == 0.017486

    C = out.read_charge[-1](name="loewdin", projection="orbital", orbitals="pz")
    assert C[0] == 0.723113


def test_charge_mulliken_full(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)

    C = out.read_charge[:](name="mulliken", projection="orbital", reduced=False)
    S = out.read_charge[:](
        name="mulliken", projection="orbital", reduced=False, spin=True
    )
    assert len(C) == 2
    assert C[0][0] == 0.821857
    assert S[0][0] == -0.000020
    assert C[0][32] == 1.174653
    assert S[0][32] == -0.000200
    assert C[1][8] == 0.313072
    assert S[1][8] == 0.006429

    C = out.read_charge[-1](name="mulliken", projection="orbital", reduced=False)
    S = out.read_charge[-1](
        name="mulliken", projection="orbital", reduced=False, spin=True
    )
    assert C[8] == 0.313072
    assert S[8] == 0.006429


def test_charge_loewdin_full(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)

    C = out.read_charge[:](name="loewdin", projection="orbital", reduced=False)
    S = out.read_charge[:](
        name="loewdin", projection="orbital", reduced=False, spin=True
    )
    assert len(S) == 2
    assert C[0][0] == 0.894846
    assert S[0][0] == 0.000337
    assert C[0][61] == 0.006054
    assert S[0][61] == 0.004362
    assert C[1][8] == 0.312172
    assert S[1][8] == 0.005159

    C = out.read_charge[-1](name="loewdin", projection="orbital", reduced=False)
    S = out.read_charge[-1](
        name="loewdin", projection="orbital", reduced=False, spin=True
    )
    assert C[8] == 0.312172
    assert S[8] == 0.005159


def test_charge_atom_unpol(sisl_files):
    f = sisl_files(_dir, "carbon_monoxide", "molecule.output")
    out = stdoutSileORCA(f)

    C = out.read_charge[:](name="mulliken", projection="atom")
    S = out.read_charge[:](name="mulliken", projection="atom", spin=True)
    assert len(C) == 2
    assert S is None
    assert C[0][0] == -0.037652

    C = out.read_charge[-1](name="mulliken", projection="atom")
    S = out.read_charge[-1](name="mulliken", projection="atom", spin=True)
    assert C[0] == -0.037652
    assert S is None

    C = out.read_charge[-1](name="loewdin", projection="atom")
    S = out.read_charge[-1](name="loewdin", projection="atom", spin=True)
    assert C[0] == -0.259865
    assert S is None


def test_charge_orbital_reduced_unpol(sisl_files):
    f = sisl_files(_dir, "carbon_monoxide", "molecule.output")
    out = stdoutSileORCA(f)

    C = out.read_charge[:](name="mulliken", projection="orbital")
    S = out.read_charge[:](name="mulliken", projection="orbital", spin=True)
    assert len(C) == 2
    assert S is None
    assert C[0][(0, "py")] == 0.534313
    assert C[1][(1, "px")] == 1.346363

    C = out.read_charge[-1](name="mulliken", projection="orbital")
    S = out.read_charge[-1](name="mulliken", projection="orbital", spin=True)
    assert C[(0, "px")] == 0.954436
    assert S is None

    C = out.read_charge[-1](name="mulliken", projection="orbital", orbitals="px")
    S = out.read_charge[-1](
        name="mulliken", projection="orbital", orbitals="px", spin=True
    )
    assert C[0] == 0.954436
    assert S is None

    C = out.read_charge[-1](name="loewdin", projection="orbital")
    S = out.read_charge[-1](name="loewdin", projection="orbital", spin=True)
    assert C[(0, "d")] == 0.315910
    assert S is None

    C = out.read_charge[-1](name="loewdin", projection="orbital", orbitals="d")
    S = out.read_charge[-1](
        name="loewdin", projection="orbital", orbitals="d", spin=True
    )
    assert C[0] == 0.315910
    assert S is None


def test_charge_orbital_full_unpol(sisl_files):
    f = sisl_files(_dir, "carbon_monoxide", "molecule.output")
    out = stdoutSileORCA(f)

    C = out.read_charge[-1](name="mulliken", projection="orbital", reduced=False)
    S = out.read_charge[-1](
        name="mulliken", projection="orbital", reduced=False, spin=True
    )
    assert C is None
    assert S is None


def test_read_energy(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)

    E = out.read_energy[:]()
    assert len(E) == 2
    assert E[0].total != 0

    E = out.read_energy[-1]()
    assert pytest.approx(E.total) == -3532.4784695729268


def test_read_energy_vdw(sisl_files):
    f = sisl_files(_dir, "carbon_monoxide", "molecule.output")
    out = stdoutSileORCA(f)

    E = out.read_energy[-1]()
    assert E.vdw != 0
    assert pytest.approx(E.total) == -3081.2640328972802


def test_read_orbital_energies(sisl_files):
    f = sisl_files(_dir, "nitric_oxide", "molecule.output")
    out = stdoutSileORCA(f)

    E = out.read_orbital_energies[:]()
    assert pytest.approx(E[0][0, 0]) == -513.8983
    assert pytest.approx(E[0][0, 1]) == -513.6538
    assert pytest.approx(E[0][61, 0]) == 1173.4258
    assert pytest.approx(E[0][61, 1]) == 1173.6985
    assert pytest.approx(E[1][0, 0]) == -513.8983
    assert pytest.approx(E[1][0, 1]) == -513.6538
    assert pytest.approx(E[1][61, 0]) == 1173.4259
    assert pytest.approx(E[1][61, 1]) == 1173.6985

    E = out.read_orbital_energies[-1]()
    assert E.shape == (out.info.no, 2)
    assert pytest.approx(E[61, 0]) == 1173.4259


def test_read_orbital_energies_unpol(sisl_files):
    f = sisl_files(_dir, "carbon_monoxide", "molecule.output")
    out = stdoutSileORCA(f)

    E = out.read_orbital_energies[:]()
    assert pytest.approx(E[0][0]) == -513.0978
    assert pytest.approx(E[0][61]) == 1171.5965
    assert pytest.approx(E[1][0]) == -513.0976
    assert pytest.approx(E[1][61]) == 1171.5967

    E = out.read_orbital_energies[-1]()
    assert E.shape == (out.info.no,)
    assert pytest.approx(E[0]) == -513.0976
    assert pytest.approx(E[61]) == 1171.5967


def test_multiple_calls(sisl_files):
    f = sisl_files(_dir, "carbon_monoxide", "molecule.output")
    out = stdoutSileORCA(f)

    N = out.read_electrons[:]()
    assert len(N) == 2

    E = out.read_orbital_energies[:]()
    assert len(E) == 2

    E = out.read_energy[:]()
    assert len(E) == 2

    C = out.read_charge[:](name="mulliken", projection="atom")
    assert len(C) == 2

    N = out.read_electrons[:]()
    assert len(N) == 2
