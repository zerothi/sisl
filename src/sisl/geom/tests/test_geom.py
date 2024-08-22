# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools
from functools import partial

import numpy as np
import pytest

from sisl import Atom, Lattice, SislError
from sisl._math_small import cross3, dot3
from sisl.geom import *

pytestmark = [pytest.mark.geom]


class CellDirect(Lattice):
    @property
    def volume(self):
        return dot3(self.cell[0, :], cross3(self.cell[1, :], self.cell[2, :]))


def is_right_handed(geometry):
    sc = CellDirect(geometry.lattice.cell)
    return sc.volume > 0.0


def has_vacuum(geometry, vacuum):
    fxyz = geometry.fxyz
    # print(fxyz.max(0) - fxyz.min(0), geometry.lattice.length)
    vac = (1 - (fxyz.max(0) - fxyz.min(0))) * geometry.lattice.length
    vac = vac[~geometry.lattice.pbc]
    print(vac, vacuum)
    return np.allclose(vac, vacuum)


def test_basic():
    a = sc(2.52, Atom["Fe"])
    assert is_right_handed(a)
    a = bcc(2.52, Atom["Fe"])
    assert is_right_handed(a)
    a = bcc(2.52, Atom["Fe"], orthogonal=True)
    a = fcc(2.52, Atom["Au"])
    assert is_right_handed(a)
    a = fcc(2.52, Atom["Au"], orthogonal=True)
    a = hcp(2.52, Atom["Au"])
    assert is_right_handed(a)
    a = hcp(2.52, Atom["Au"], orthogonal=True)
    a = rocksalt(5.64, ["Na", "Cl"])
    assert is_right_handed(a)
    a = rocksalt(5.64, [Atom("Na", R=3), Atom("Cl", R=4)], orthogonal=True)


@pytest.mark.parametrize(
    "func, orthogonal, vacuum",
    itertools.product([graphene, goldene], [True, False], [10, 20]),
)
def test_flat(func, orthogonal, vacuum):
    a = func(orthogonal=orthogonal, vacuum=vacuum)
    assert is_right_handed(a)
    assert has_vacuum(a, vacuum)
    a = func(atoms="C", orthogonal=orthogonal, vacuum=vacuum)
    assert is_right_handed(a)
    assert has_vacuum(a, vacuum)


@pytest.mark.parametrize(
    "vacuum",
    [10, (20, 30, 10)],
)
def test_flat_flakes(vacuum):
    g = graphene_flake(shells=0, bond=1.42, vacuum=vacuum)
    assert g.na == 6
    # All atoms are close to the center
    assert len(g.close(g.center(), 1.44)) == g.na
    # All atoms have two neighbors
    assert len(g.axyz(AtomNeighbors(min=2, max=2, R=1.44))) == g.na
    assert is_right_handed(g)
    assert has_vacuum(g, vacuum)

    g = graphene_flake(shells=1, bond=1.42, vacuum=vacuum)
    assert g.na == 24
    assert len(g.close(g.center(), 4)) == g.na
    assert len(g.axyz(AtomNeighbors(min=2, max=2, R=1.44))) == 12
    assert len(g.axyz(AtomNeighbors(min=3, max=3, R=1.44))) == 12
    assert is_right_handed(g)
    assert has_vacuum(g, vacuum)

    bn = honeycomb_flake(shells=1, atoms=["B", "N"], bond=1.42, vacuum=vacuum)
    assert bn.na == 24
    assert np.allclose(bn.xyz, g.xyz)
    # Check that atoms are alternated.
    assert len(bn.axyz(AtomZ(5) & AtomNeighbors(min=1, R=1.44, neighbor=AtomZ(5)))) == 0
    assert is_right_handed(bn)
    assert has_vacuum(bn, vacuum)


@pytest.mark.parametrize(
    "vacuum",
    [10, (20, 30, 10)],
)
def test_triangulene(vacuum):
    g = triangulene(3, vacuum=vacuum)
    assert g.na == 22
    assert is_right_handed(g)
    assert has_vacuum(g, vacuum)

    g = triangulene(3, atoms=["B", "N"], vacuum=vacuum)
    assert g.atoms.nspecies == 2
    assert is_right_handed(g)
    assert has_vacuum(g, vacuum)


@pytest.mark.parametrize(
    "vacuum",
    [10, (30, 10)],
)
def test_nanotube(vacuum):
    a = nanotube(1.42, vacuum=vacuum)
    assert is_right_handed(a)
    assert has_vacuum(a, vacuum)
    a = nanotube(1.42, chirality=(3, 5), vacuum=vacuum)
    assert is_right_handed(a)
    assert has_vacuum(a, vacuum)
    a = nanotube(1.42, chirality=(6, -3), vacuum=vacuum)
    assert is_right_handed(a)
    assert has_vacuum(a, vacuum)


def test_diamond():
    a = diamond()
    assert is_right_handed(a)


@pytest.mark.parametrize(
    "vacuum",
    [10, 30],
)
def test_bilayer(vacuum):
    a = bilayer(1.42, vacuum=vacuum)
    assert is_right_handed(a)
    assert has_vacuum(a, vacuum)


def test_bilayer_arguments():
    # Just test arguments
    bilayer(1.42, stacking="AA")
    bilayer(1.42, stacking="BA")
    bilayer(1.42, stacking="AB")
    for m in range(7):
        bilayer(1.42, twist=(m, m + 1))
    bilayer(1.42, twist=(6, 7), layer="bottom")
    bilayer(1.42, twist=(6, 7), layer="TOP")
    bilayer(1.42, bottom_atoms=(Atom["B"], Atom["N"]), twist=(6, 7))
    bilayer(1.42, top_atoms=(Atom(5), Atom(7)), twist=(6, 7))
    _, _ = bilayer(1.42, twist=(6, 7), ret_angle=True)

    with pytest.raises(ValueError):
        bilayer(1.42, twist=(6, 7), layer="undefined")

    with pytest.raises(ValueError):
        bilayer(1.42, twist=(6, 7), stacking="undefined")

    with pytest.raises(ValueError):
        bilayer(1.42, twist=("str", 7), stacking="undefined")


@pytest.mark.parametrize(
    "vacuum",
    [10, (30, 10)],
)
def test_nanoribbon(vacuum):
    for w in range(0, 5):
        a = nanoribbon(w, 1.42, (Atom(5), Atom(7)), kind="zigzag", vacuum=vacuum)
        assert is_right_handed(a)
        assert has_vacuum(a, vacuum)


def test_nanoribbon_arguments():
    for w in range(0, 5):
        nanoribbon(w, 1.42, Atom(6), kind="armchair")
        nanoribbon(w, 1.42, Atom(6), kind="zigzag")
        nanoribbon(w, 1.42, Atom(6), kind="chiral")
        nanoribbon(w, 1.42, Atom(6), kind="chiral", chirality=(2, 2))
        nanoribbon(w, 1.42, (Atom(5), Atom(7)), kind="armchair")

    with pytest.raises(ValueError):
        nanoribbon(6, 1.42, (Atom(5), Atom(7)), kind="undefined")

    with pytest.raises(ValueError):
        nanoribbon("str", 1.42, (Atom(5), Atom(7)), kind="undefined")


def test_graphene_nanoribbon():
    graphene_nanoribbon(6, kind="armchair")


@pytest.mark.parametrize(
    "vacuum",
    [10, (30, 10)],
)
def test_agnr(vacuum):
    a = agnr(5, vacuum=vacuum)
    assert is_right_handed(a)
    assert has_vacuum(a, vacuum)


@pytest.mark.parametrize(
    "vacuum",
    [10, (30, 10)],
)
def test_zgnr(vacuum):
    a = zgnr(5, vacuum=vacuum)
    assert is_right_handed(a)
    assert has_vacuum(a, vacuum)


@pytest.mark.parametrize(
    "vacuum",
    [10, (30, 10)],
)
def test_cgnr(vacuum):
    a = cgnr(6, (3, 1), atoms=["B", "N"], vacuum=vacuum)
    assert is_right_handed(a)
    assert has_vacuum(a, vacuum)


@pytest.mark.parametrize(
    "W, invert_first",
    itertools.product(range(3, 20), [True, False]),
)
def test_heteroribbon_one_unit(W, invert_first):
    # Check that all ribbon widths are properly cut into one unit
    geometry = heteroribbon(
        [(W, 1)], bond=1.42, atoms=Atom(6, 1.43), invert_first=invert_first
    )

    assert geometry.na == W


def test_heteroribbon():
    """Runs the heteroribbon builder for all possible combinations of
    widths and asserts that they are always properly aligned.
    """
    # Build combinations
    combinations = itertools.product([7, 8, 9, 10, 11], [7, 8, 9, 10, 11])
    L = itertools.repeat(2)

    for Ws in combinations:
        geom = heteroribbon(
            zip(Ws, L), bond=1.42, atoms=Atom(6, 1.43), align="auto", shift_quantum=True
        )

        # Assert no dangling bonds.
        assert len(geom.asc2uc({"neighbors": 1})) == 0


def test_graphene_heteroribbon():
    a = graphene_heteroribbon([(7, 2), (9, 2)])


def test_graphene_heteroribbon_errors():
    # 7-open with 9 can only be perfectly aligned.
    graphene_heteroribbon([(7, 1), (9, 1)], align="center", on_lone_atom="raise")
    with pytest.raises(SislError):
        graphene_heteroribbon(
            [(7, 1), (9, 1, -1)], align="center", on_lone_atom="raise"
        )
    # From the bottom
    graphene_heteroribbon([(7, 1), (9, 1, -1)], align="bottom", on_lone_atom="raise")
    with pytest.raises(SislError):
        graphene_heteroribbon([(7, 1), (9, 1, 0)], align="bottom", on_lone_atom="raise")
    # And from the top
    graphene_heteroribbon([(7, 1), (9, 1, 1)], align="top", on_lone_atom="raise")
    with pytest.raises(SislError):
        graphene_heteroribbon([(7, 1), (9, 1, -1)], align="top", on_lone_atom="raise")

    grap_heteroribbon = partial(graphene_heteroribbon, align="auto", shift_quantum=True)

    # Odd section with open end
    with pytest.raises(SislError):
        grap_heteroribbon([(7, 3), (5, 2)])

    # Shift limits are imposed correctly
    # In this case -2 < shift < 1
    grap_heteroribbon([(7, 3), (11, 2, 0)])
    grap_heteroribbon([(7, 3), (11, 2, -1)])
    with pytest.raises(SislError):
        grap_heteroribbon([(7, 3), (11, 2, 1)])
    with pytest.raises(SislError):
        grap_heteroribbon([(7, 3), (11, 2, -2)])

    # Periodic boundary conditions work properly
    # grap_heteroribbon([[10, 2], [8, 1, 0]], pbc=False)
    # with pytest.raises(ValueError):
    #     grap_heteroribbon([[10, 2], [8, 1, 0]], pbc=True)

    # Even ribbons should only be shifted towards the center
    grap_heteroribbon([(10, 2), (8, 2, -1)])
    with pytest.raises(SislError):
        grap_heteroribbon([(10, 2), (8, 2, 1)])
    grap_heteroribbon(
        [(10, 1), (8, 2, 1)],
    )  # pbc=False)
    with pytest.raises(SislError):
        grap_heteroribbon(
            [(10, 1), (8, 2, -1)],
        )  # pbc=False)


def test_fcc_slab():
    for o in [True, False]:
        fcc_slab(alat=4.08, atoms="Au", miller=(1, 0, 0), orthogonal=o)
        fcc_slab(4.08, "Au", 100, orthogonal=o)
        fcc_slab(4.08, "Au", 110, orthogonal=o)
        fcc_slab(4.08, "Au", 111, orthogonal=o)
        fcc_slab(4.08, 79, "100", layers=5, vacuum=None, orthogonal=o)
        fcc_slab(4.08, 79, "110", layers=5, orthogonal=o)
        fcc_slab(4.08, 79, "111", layers=5, start=1, orthogonal=o)
        fcc_slab(4.08, 79, "111", layers=5, start="C", orthogonal=o)
        fcc_slab(4.08, 79, "111", layers=5, end=2, orthogonal=o)
        a = fcc_slab(4.08, 79, "111", layers=5, end="B", orthogonal=o)
        assert is_right_handed(a)

    with pytest.raises(ValueError):
        fcc_slab(4.08, "Au", 100, start=0, end=0)
    with pytest.raises(ValueError):
        fcc_slab(4.08, "Au", 1000)
    with pytest.raises(NotImplementedError):
        fcc_slab(4.08, "Au", 200)
    assert not np.allclose(
        fcc_slab(5.64, "Au", 100, end=1, layers="BABAB").xyz,
        fcc_slab(5.64, "Au", 100, end=1, layers=" BABAB ").xyz,
    )
    assert np.allclose(
        fcc_slab(5.64, "Au", 100, layers=" AB AB BA ", vacuum=2).xyz,
        fcc_slab(
            5.64, "Au", 100, layers=(None, 2, 2, 2, None), vacuum=2, start=(0, 0, 1)
        ).xyz,
    )
    assert np.allclose(
        fcc_slab(5.64, "Au", 100, layers=" AB AB BA ", vacuum=(2, 1)).xyz,
        fcc_slab(
            5.64,
            "Au",
            100,
            layers=(None, 2, " ", 2, None, 2, None),
            vacuum=(2, 1),
            end=(1, 1, 0),
        ).xyz,
    )

    # example in documentation
    assert np.allclose(
        fcc_slab(
            4.0, "Au", 100, layers=(" ", 3, 5, 3), start=(0, 1, 0), vacuum=(10, 1, 2)
        ).xyz,
        fcc_slab(4.0, "Au", 100, layers=" ABA BABAB ABA", vacuum=(10, 1, 2)).xyz,
    )

    assert np.allclose(
        fcc_slab(
            4.0, "Au", 100, layers=(" ", 3, 5, 3), start=(1, 0), vacuum=(10, 1, 2)
        ).xyz,
        fcc_slab(4.0, "Au", 100, layers=" BAB ABABA ABA", vacuum=(10, 1, 2)).xyz,
    )


def test_bcc_slab():
    for o in [True, False]:
        bcc_slab(alat=4.08, atoms="Au", miller=(1, 0, 0), orthogonal=o)
        bcc_slab(4.08, "Au", 100, orthogonal=o)
        bcc_slab(4.08, "Au", 110, orthogonal=o)
        bcc_slab(4.08, "Au", 111, orthogonal=o)
        bcc_slab(4.08, 79, "100", layers=5, vacuum=None, orthogonal=o)
        bcc_slab(4.08, 79, "110", layers=5, orthogonal=o)
        assert bcc_slab(4.08, 79, "111", layers=5, start=1, orthogonal=o).equal(
            bcc_slab(4.08, 79, "111", layers="BCABC", orthogonal=o)
        )
        bcc_slab(4.08, 79, "111", layers="BCABC", start="B", orthogonal=o)
        bcc_slab(4.08, 79, "111", layers="BCABC", start=1, orthogonal=o)
        bcc_slab(4.08, 79, "111", layers=5, start="C", orthogonal=o)
        bcc_slab(4.08, 79, "111", layers=5, end=2, orthogonal=o)
        a = bcc_slab(4.08, 79, "111", layers=5, end="B", orthogonal=o)
        assert is_right_handed(a)

    with pytest.raises(ValueError):
        bcc_slab(4.08, "Au", 100, start=0, end=0)
    with pytest.raises(ValueError):
        bcc_slab(4.08, "Au", 1000)
    with pytest.raises(NotImplementedError):
        bcc_slab(4.08, "Au", 200)
    assert not np.allclose(
        bcc_slab(5.64, "Au", 100, end=1, layers="BABAB").xyz,
        bcc_slab(5.64, "Au", 100, end=1, layers=" BABAB ").xyz,
    )


def test_rocksalt_slab():
    rocksalt_slab(5.64, [Atom(11, R=3), Atom(17, R=4)], 100)
    assert rocksalt_slab(5.64, ["Na", "Cl"], 100, layers=5).equal(
        rocksalt_slab(5.64, ["Na", "Cl"], 100, layers="ABABA")
    )
    rocksalt_slab(5.64, ["Na", "Cl"], 110, vacuum=None)
    rocksalt_slab(5.64, ["Na", "Cl"], 111, orthogonal=False)
    rocksalt_slab(5.64, ["Na", "Cl"], 111, orthogonal=True)
    a = rocksalt_slab(5.64, "Na", 100)
    assert is_right_handed(a)

    with pytest.raises(ValueError):
        rocksalt_slab(5.64, ["Na", "Cl"], 100, start=0, end=0)
    with pytest.raises(ValueError):
        rocksalt_slab(5.64, ["Na", "Cl"], 1000)
    with pytest.raises(NotImplementedError):
        rocksalt_slab(5.64, ["Na", "Cl"], 200)
    with pytest.raises(ValueError):
        rocksalt_slab(5.64, ["Na", "Cl"], 100, start=0, layers="BABAB")
    rocksalt_slab(5.64, ["Na", "Cl"], 100, start=1, layers="BABAB")
    with pytest.raises(ValueError):
        rocksalt_slab(5.64, ["Na", "Cl"], 100, end=0, layers="BABAB")
    rocksalt_slab(5.64, ["Na", "Cl"], 100, end=1, layers="BABAB")
    assert not np.allclose(
        rocksalt_slab(5.64, ["Na", "Cl"], 100, end=1, layers="BABAB").xyz,
        rocksalt_slab(5.64, ["Na", "Cl"], 100, end=1, layers=" BABAB ").xyz,
    )
