# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os.path as osp

import numpy as np
import pytest

import sisl
from sisl.io.vasp.outcar import outcarSileVASP

pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join("sisl", "io", "vasp")


def test_diamond_outcar_energies(sisl_files):
    f = sisl_files(_dir, "diamond", "OUTCAR")
    f = outcarSileVASP(f)

    with pytest.warns(sisl.SislDeprecation, match=r"no longer returns the last entry"):
        E0 = f.read_energy()

        E = f.read_energy[-1]()
        Eall = f.read_energy[:]()

    assert E0.sigma0 == 0.8569373  # first block
    assert E.sigma0 == -18.18677613  # last block

    assert E0 == Eall[0]
    assert E == Eall[-1]
    assert len(Eall) > 1
    assert f.info.completed()

    EHa = f.read_energy[-1](units="Ha")
    assert EHa.sigma0 != E.sigma0


def test_diamond_outcar_cputime(sisl_files):
    f = sisl_files(_dir, "diamond", "OUTCAR")
    f = outcarSileVASP(f)

    assert f.cpu_time() > 0.0
    assert f.info.completed()


def test_diamond_outcar_completed(sisl_files):
    f = sisl_files(_dir, "diamond", "OUTCAR")
    f = outcarSileVASP(f)

    assert f.info.completed()


def test_diamond_outcar_trajectory(sisl_files):
    f = sisl_files(_dir, "diamond", "OUTCAR")
    f = outcarSileVASP(f)

    step = f.read_trajectory()

    assert step.xyz[0, 1] == 0.0
    assert step.xyz[1, 1] == 0.89250
    assert step.force[0, 1] == 0.0
    assert step.force[1, 0] == 0.0

    traj = f.read_trajectory[:]()
    assert len(traj) == 1


def test_graphene_relax_outcar_trajectory(sisl_files):
    f = sisl_files(_dir, "graphene_relax", "OUTCAR")
    f = outcarSileVASP(f)

    step = f.read_trajectory[9]()
    assert step.cell[0, 0] == 2.462060590
    assert step.cell[1, 0] == -1.231030295
    assert step.cell[2, 2] == 9.804915686
    assert step.force[0, 2] == -0.006138
    assert step.force[1, 2] == 0.006138

    traj = f.read_trajectory[:]()
    assert len(traj) == 10
    assert traj[0].cell[0, 0] == 2.441046239
    assert traj[0].xyz[0, 2] == 0.00000
    assert traj[0].xyz[1, 2] == 0.20000
    assert traj[0].force[0, 2] == 3.448038
    assert traj[0].force[1, 2] == -3.448038
    assert traj[-1].cell[2, 2] == 9.804915686
    assert traj[-1].xyz[1, 2] == -0.00037
    assert traj[-1].force[0, 2] == -0.006138
    assert traj[-1].force[1, 2] == 0.006138


def test_graphene_md_outcar_trajectory(sisl_files):
    f = sisl_files(_dir, "graphene_md", "OUTCAR")
    f = outcarSileVASP(f)

    step = f.read_trajectory[99]()
    assert step.xyz[0, 0] == 0.09703
    assert step.force[0, 2] == -0.278082

    traj = f.read_trajectory[:]()
    assert len(traj) == 100
    assert traj[0].xyz[0, 0] == 0.12205
    assert traj[0].xyz[1, 0] == 0.09520
    assert traj[0].force[0, 1] == -0.017766
    assert traj[0].force[1, 1] == 0.017766
    assert traj[-1].xyz[0, 0] == 0.09703
    assert traj[-1].force[0, 2] == -0.278082
