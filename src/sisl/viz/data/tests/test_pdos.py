# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

import sisl
from sisl.viz.data import PDOSData

pytestmark = [pytest.mark.viz, pytest.mark.data]


@pytest.mark.parametrize(
    "spin", ["unpolarized", "polarized", "noncolinear", "spinorbit"]
)
def test_pdos_from_sisl_H(spin):
    gr = sisl.geom.graphene()
    H = sisl.Hamiltonian(gr)
    H.construct([(0.1, 1.44), (0, -2.7)])

    n_spin, H = {
        "unpolarized": (1, H),
        "polarized": (2, H.transform(spin=sisl.Spin.POLARIZED)),
        "noncolinear": (4, H.transform(spin=sisl.Spin.NONCOLINEAR)),
        "spinorbit": (4, H.transform(spin=sisl.Spin.SPINORBIT)),
    }[spin]

    data = PDOSData.new(H, Erange=(-5, 5))

    checksum = 17.599343960516066
    if n_spin > 1:
        checksum = checksum * 2

    data.sanity_check(
        na=2, no=2, n_spin=n_spin, atom_tags=("C",), dos_checksum=checksum
    )


@pytest.mark.parametrize("spin", ["unpolarized", "polarized", "noncolinear"])
def test_pdos_from_siesta_PDOS(spin, siesta_test_files):
    n_spin, filename = {
        "unpolarized": (1, "SrTiO3.PDOS"),
        "polarized": (2, "SrTiO3_polarized.PDOS"),
        "noncolinear": (4, "SrTiO3_noncollinear.PDOS"),
    }[spin]

    file = siesta_test_files(filename)

    data = PDOSData.new(file)

    checksum = 1240.0012709612743
    if n_spin > 1:
        checksum = checksum * 2

    data.sanity_check(
        na=5, no=72, n_spin=n_spin, atom_tags=("Sr", "Ti", "O"), dos_checksum=checksum
    )


@pytest.mark.parametrize("spin", ["noncolinear"])
def test_pdos_from_siesta_wfsx(spin, siesta_test_files):
    n_spin, filename = {
        "noncolinear": (4, "bi2se3_3ql.bands.WFSX"),
    }[spin]

    # From a siesta .WFSX file
    # Since there is no hamiltonian for bi2se3_3ql.fdf, we create a dummy one
    wfsx = sisl.get_sile(siesta_test_files(filename))

    geometry = sisl.get_sile(siesta_test_files("bi2se3_3ql.fdf")).read_geometry()
    geometry = sisl.Geometry(geometry.xyz, atoms=wfsx.read_basis())

    H = sisl.Hamiltonian(geometry, dim=4)

    data = PDOSData.new(wfsx, H=H)

    # For now, the checksum is 0 because we have no overlap matrix.
    checksum = 0
    if n_spin > 1:
        checksum = checksum * 2

    data.sanity_check(
        na=15, no=195, n_spin=n_spin, atom_tags=("Bi", "Se"), dos_checksum=checksum
    )


@pytest.mark.parametrize(
    "spin", ["unpolarized", "polarized", "noncolinear", "spinorbit"]
)
def test_toy_example(spin):
    data = PDOSData.toy_example(spin=spin)

    n_spin = {"unpolarized": 1, "polarized": 2, "noncolinear": 4, "spinorbit": 4}[spin]

    data.sanity_check(
        n_spin=n_spin, na=data.geometry.na, no=data.geometry.no, atom_tags=["C"]
    )
