# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

import sisl
from sisl.viz.data import BandsData

pytestmark = [pytest.mark.viz, pytest.mark.data]


@pytest.mark.parametrize(
    "spin", ["unpolarized", "polarized", "noncolinear", "spinorbit"]
)
def test_bands_from_sisl_H(spin):
    gr = sisl.geom.graphene()
    H = sisl.Hamiltonian(gr)
    H.construct([(0.1, 1.44), (0, -2.7)])

    n_spin, H = {
        "unpolarized": (1, H),
        "polarized": (2, H.transform(spin=sisl.Spin.POLARIZED)),
        "noncolinear": (4, H.transform(spin=sisl.Spin.NONCOLINEAR)),
        "spinorbit": (4, H.transform(spin=sisl.Spin.SPINORBIT)),
    }[spin]

    bz = sisl.BandStructure(
        H, [[0, 0, 0], [2 / 3, 1 / 3, 0], [1 / 2, 0, 0]], 6, ["Gamma", "M", "K"]
    )

    data = BandsData.new(bz)

    data.sanity_check(
        n_spin=n_spin,
        nk=6,
        nbands=2,
        klabels=["Gamma", "M", "K"],
        kvals=[0.0, 1.70309799, 2.55464699],
    )


def test_bands_from_siesta_bands(sisl_files):
    n_spin = 1
    file = sisl_files("siesta", "SrTiO3", "unpolarized", "SrTiO3.bands")

    data = BandsData.new(file)

    data.sanity_check(
        n_spin=n_spin,
        nk=150,
        nbands=72,
        klabels=("Gamma", "X", "M", "Gamma", "R", "X"),
        kvals=[0.0, 0.429132, 0.858265, 1.465149, 2.208428, 2.815313],
    )


@pytest.mark.parametrize("spin", ["noncolinear"])
def test_bands_from_siesta_wfsx(spin, sisl_files):
    n_spin = 4
    dirs = ("siesta", "Bi2Se3_3layer")

    # From a siesta .WFSX file
    # Since there is no hamiltonian for bi2se3_3ql.fdf, we create a dummy one
    wfsx = sisl.get_sile(sisl_files(*dirs, "Bi2Se3.bands.WFSX"))

    fdf = sisl_files(*dirs, "Bi2Se3.fdf")

    data = BandsData.new(wfsx, fdf=fdf)

    data.sanity_check(n_spin=n_spin, nk=16, nbands=4)


@pytest.mark.parametrize(
    "spin", ["unpolarized", "polarized", "noncolinear", "spinorbit"]
)
def test_toy_example(spin):
    nk = 15
    n_states = 28

    data = BandsData.toy_example(spin=spin, nk=nk, n_states=n_states)

    n_spin = {"unpolarized": 1, "polarized": 2, "noncolinear": 4, "spinorbit": 4}[spin]

    data.sanity_check(
        n_spin=n_spin, nk=nk, nbands=n_states, klabels=["Gamma", "X"], kvals=[0, 1]
    )

    if n_spin == 4:
        assert "spin_moments" in data.data_vars
