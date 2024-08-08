# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

import sisl
from sisl.io.vasp.outcar import outcarSileVASP

pytestmark = [pytest.mark.io, pytest.mark.vasp]


def test_diamond_outcar_energies(sisl_files):
    f = sisl_files("vasp", "diamond", "OUTCAR")
    f = outcarSileVASP(f)

    with pytest.warns(sisl.SislDeprecation, match=r"no longer returns the last entry"):
        E0 = f.read_energy()

        E = f.read_energy[-1]()
        Eall = f.read_energy[:]()

    assert E0.sigma0 == 38.32164289  # first block
    assert E.sigma0 == -18.16472605  # last block

    assert E0 == Eall[0]
    assert E == Eall[-1]
    assert len(Eall) > 1
    assert f.info.completed()

    EHa = f.read_energy[-1](units="Ha")
    assert EHa.sigma0 != E.sigma0


def test_diamond_outcar_cputime(sisl_files):
    f = sisl_files("vasp", "diamond", "OUTCAR")
    f = outcarSileVASP(f)

    assert f.cpu_time() > 0.0
    assert f.info.completed()


def test_diamond_outcar_completed(sisl_files):
    f = sisl_files("vasp", "diamond", "OUTCAR")
    f = outcarSileVASP(f)

    assert f.info.completed()


def test_diamond_outcar_trajectory(sisl_files):
    f = sisl_files("vasp", "diamond", "OUTCAR")
    f = outcarSileVASP(f)

    step = f.read_trajectory()

    assert step.xyz[0, 1] == 0.0
    assert step.xyz[1, 1] == 0.89250
    assert step.force[0, 1] == 0.0
    assert step.force[1, 0] == 0.0

    traj = f.read_trajectory[:]()
    assert len(traj) == 1


def test_graphene_relax_outcar_trajectory(sisl_files):
    f = sisl_files("vasp", "graphene_relax", "OUTCAR")
    f = outcarSileVASP(f)

    step = f.read_trajectory[9]()
    assert step.cell[0, 0] == 2.472466059
    assert step.cell[1, 0] == -1.236233029
    assert step.cell[2, 2] == 9.789010603
    assert step.force[0, 2] == -0.03229
    assert step.force[1, 2] == 0.03229

    traj = f.read_trajectory[:]()
    assert len(traj) == 11
    assert traj[0].cell[0, 0] == 2.441046239
    assert traj[0].xyz[0, 2] == 0.00000
    assert traj[0].xyz[1, 2] == 0.20000
    assert traj[0].force[0, 2] == 3.353225
    assert traj[0].force[1, 2] == -3.353225
    assert traj[-1].cell[2, 2] == 9.791779146
    assert traj[-1].xyz[1, 2] == 9.79061
    assert traj[-1].force[0, 2] == -0.019886
    assert traj[-1].force[1, 2] == 0.019886


def test_graphene_md_outcar_trajectory(sisl_files):
    f = sisl_files("vasp", "graphene_md", "OUTCAR")
    f = outcarSileVASP(f)

    step = f.read_trajectory[99]()
    assert step.xyz[0, 0] == 0.09669
    assert step.force[0, 2] == -0.196454

    traj = f.read_trajectory[:]()
    assert len(traj) == 100
    assert traj[0].xyz[0, 0] == 0.12205
    assert traj[0].xyz[1, 0] == 0.09520
    assert traj[0].force[0, 1] == -0.022846
    assert traj[0].force[1, 1] == 0.022846
    assert traj[-1].xyz[0, 0] == 0.09669
    assert traj[-1].force[0, 2] == -0.196454
