# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import units
from sisl.io.wannier90 import *

pytestmark = [pytest.mark.io, pytest.mark.wannier90, pytest.mark.w90]


@pytest.mark.parametrize("unit", ["", "Ang\n", "ang\n", "bohr\n"])
def test_seedname_read_frac(sisl_tmp, unit):
    f = sisl_tmp("read_frac.win")
    with open(f, "w") as fh:
        fh.write(
            """
begin unit_cell_cart
  {} 2. 0. 0.
  0. 2. 0
  0 0 2.
end unit_cell_cart

begin atoms_frac
  C 0.5 0.5 0.5
end
""".format(
                unit
            )
        )
    g = winSileWannier90(f).read_geometry(order=["win"])

    if len(unit) == 0:
        unit = "ang"
    unit = units(unit.strip().capitalize(), "Ang")

    assert np.allclose(g.cell, np.identity(3) * 2 * unit)
    assert np.allclose(g.xyz, [1 * unit] * 3)


@pytest.mark.parametrize("unit_sc", ["", "Ang\n", "ang\n", "bohr\n"])
@pytest.mark.parametrize("unit", ["", "Ang\n", "ang\n", "bohr\n"])
def test_seedname_read_coord(sisl_tmp, unit_sc, unit):
    f = sisl_tmp("read_coord.win")
    with open(f, "w") as fh:
        fh.write(
            """
begin unit_cell_cart
  {} 2. 0. 0.
  0. 2. 0
  0 0 2.
end unit_cell_cart

begin atoms_cart
  {} C 0.5 0.5 0.5
end
""".format(
                unit_sc, unit
            )
        )
    g = winSileWannier90(f).read_geometry(order=["win"])

    if len(unit) == 0:
        unit = "ang"
    unit = units(unit.strip().capitalize(), "Ang")

    if len(unit_sc) == 0:
        unit_sc = "ang"
    unit_sc = units(unit_sc.strip().capitalize(), "Ang")

    assert np.allclose(g.cell, np.identity(3) * 2 * unit_sc)
    assert np.allclose(g.xyz, [0.5 * unit] * 3)


@pytest.mark.parametrize("frac", [True, False])
def test_seedname_write_read(sisl_tmp, sisl_system, frac):
    f = sisl_tmp("write_read.win")
    sile = winSileWannier90(f, "w")
    sile.write_geometry(sisl_system.g, frac=frac)

    g = winSileWannier90(f).read_geometry(order=["win"])
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)


def test_seedname_read_ham(sisl_files):
    f = winSileWannier90(sisl_files("wannier90", "silicon", "silicon.win"))

    ham = {}
    for key in ["hr", "tb"]:
        ham[key] = f.read_hamiltonian(cutoff=1e-4, order=[key])
        if not key == "hr":
            assert ham["hr"].spsame(ham[key])


def test_seedname_read_lattice(sisl_files):
    f = winSileWannier90(sisl_files("wannier90", "silicon", "silicon.win"))

    lat1 = f.read_lattice(order="tb")
    lat2 = f.read_lattice(order="win")
    assert np.allclose(lat1.cell, lat2.cell)
