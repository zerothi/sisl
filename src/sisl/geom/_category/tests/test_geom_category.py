# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Cuboid, Geometry, PeriodicTable
from sisl.geom import *

pytestmark = [pytest.mark.geom, pytest.mark.category, pytest.mark.geom_category]


def test_geom_category_kw_and_call():
    # Check that categories can be built indistinctively using the kw builder
    # or directly calling them.
    cat1 = AtomCategory.kw(odd={})
    cat2 = AtomCategory(odd={})
    assert cat1 == cat2


def test_geom_category():
    hBN = honeycomb(1.42, [Atom(5, R=1.43), Atom(7, R=1.43)]) * (10, 11, 1)

    B = AtomZ([5, 6])
    B2 = AtomNeighbors(2, neighbor=B)
    N = AtomZ(7)
    N2 = AtomNeighbors(2, neighbor=N)
    B3 = AtomNeighbors(3, neighbor=B)
    N3 = AtomNeighbors(3, neighbor=N)

    assert B != N

    n2 = AtomNeighbors(2)
    n3 = AtomNeighbors(3)
    n3n2 = AtomNeighbors(1, 3, neighbor=n2) & n3
    n3n3n2 = AtomNeighbors(1, 3, neighbor=n3n2) & n3

    category = (B & B2) ^ (N & N2) ^ (B & B3) ^ (N & N3) ^ n2

    cat = category.categorize(hBN)
    cat1 = category.categorize(hBN, atoms=[2, 3])
    assert len(cat1) == 2
    assert cat[2] == cat1[0]
    assert cat[3] == cat1[1]


def test_geom_category_no_r():
    hBN = honeycomb(1.42, Atom[5, 7]) * (10, 11, 1)

    B = AtomZ(5)
    B2 = AtomNeighbors(2, neighbor=B, R=1.43)
    N = AtomZ(7)
    N2 = AtomNeighbors(min=2, max=2, neighbor=N, R=(0.01, 1.43))
    PT = PeriodicTable()
    B3 = AtomNeighbors(3, neighbor=B, R=lambda atom: (0.01, PT.radius(atom.Z)))
    N3 = AtomNeighbors(3, neighbor=N, R=1.43)
    Nabove3 = AtomNeighbors(min=3, neighbor=N, R=1.43)

    assert B != N2
    assert N2 != N3

    n2 = AtomNeighbors(2, R=1.43)

    category = (B & B2) ^ (N & N2) ^ (B & B3) ^ (N & N3) ^ n2

    cat = category.categorize(hBN)


def test_geom_category_even_odd():
    hBN = honeycomb(1.42, Atom[5, 7]) * (4, 5, 1)

    odd = AtomOdd()
    even = AtomEven()

    assert odd != even
    assert even != odd

    cat = (odd | even).categorize(hBN)
    for i, c in enumerate(cat):
        if i % 2 == 0:
            assert c == even
        else:
            assert c == odd


def test_geom_category_index():
    hBN = honeycomb(1.42, Atom[5, 7]) * (4, 5, 1)

    first_and_third = AtomIndex([0, 2])
    assert set([0, 2]) == set(hBN.asc2uc(first_and_third))

    odd = AtomIndex(mod=2)
    even = ~odd
    str(odd)
    str(even)

    assert odd != even
    assert even != odd

    cat = (odd | even).categorize(hBN)
    for i, c in enumerate(cat):
        if i % 2 == 0:
            assert c == even
        else:
            assert c == odd


def test_geom_category_seq():
    geom = honeycomb(1.42, Atom[5, 7]) * (4, 5, 1)

    cat = AtomSeq("")
    assert len(geom.asc2uc(cat)) == 0

    cat = AtomSeq("0:")
    assert len(geom.asc2uc(cat)) == geom.na

    cat = AtomSeq(":")
    assert len(geom.asc2uc(cat)) == geom.na

    cat = AtomSeq(":-2")
    assert len(geom.asc2uc(cat)) == geom.na - 1

    cat = AtomSeq("-2:")
    assert len(geom.asc2uc(cat)) == 2

    cat = AtomSeq("-2:2:")
    assert len(geom.asc2uc(cat)) == 1

    cat = AtomSeq("-2")
    assert set(geom.asc2uc(cat)) == set([len(geom) - 1 - 2])

    cat = AtomSeq("0,3,5")
    assert set(geom.asc2uc(cat)) == set([0, 3, 5])

    cat = AtomSeq(":3,5")
    assert set(geom.asc2uc(cat)) == set([0, 1, 2, 3, 5])


def test_geom_category_tag():
    atoms = [
        Atom(Z=6, tag="C1"),
        Atom(Z=6, tag="C2"),
        Atom(Z=6, tag="C3"),
        Atom(Z=1, tag="H"),
    ]
    geom = Geometry([[0, 0, 0]] * 4, atoms=atoms)

    cat = AtomTag("")
    assert len(geom.asc2uc(cat)) == 4

    cat = AtomTag("C")
    assert set(geom.asc2uc(cat)) == set([0, 1, 2])

    cat = AtomTag("C1")
    assert set(geom.asc2uc(cat)) == set([0])

    cat = AtomTag("C$")
    assert len(geom.asc2uc(cat)) == 0

    cat = AtomTag("C[13]")
    assert set(geom.asc2uc(cat)) == set([0, 2])

    cat = AtomTag("[CH]")
    assert set(geom.asc2uc(cat)) == set([0, 1, 2, 3])

    cat = AtomTag("[CH]$")
    assert set(geom.asc2uc(cat)) == set([3])


def test_geom_category_frac_site():
    hBN_gr = bilayer(1.42, Atom[5, 7], Atom[6]) * (4, 5, 1)
    mid_layer = np.average(hBN_gr.xyz[:, 2])

    A_site = AtomFracSite(graphene())
    bottom = AtomXYZ(z_lt=mid_layer, z=(None, mid_layer))
    top = ~AtomCategory(xyz={"z_lt": mid_layer}, z=(None, mid_layer))

    bottom_A = A_site & bottom
    top_A = A_site & top

    cat = (bottom_A | top_A).categorize(hBN_gr)
    str(cat)
    cnull = 0
    for i, c in enumerate(cat):
        if c == NullCategory():
            cnull += 1
        elif hBN_gr.xyz[i, 2] < mid_layer:
            assert c == bottom_A
        else:
            assert False
    assert cnull == len(hBN_gr) // 4 * 3


def test_geom_category_frac_A_B_site():
    gr = graphene() * (4, 5, 1)

    A_site = AtomFracSite(graphene())
    B_site = AtomFracSite(graphene(), foffset=(-1 / 3, -1 / 3, 0))

    cat = (A_site | B_site).categorize(gr)
    for i, c in enumerate(cat):
        if i % 2 == 0:
            assert c == A_site
        else:
            assert c == B_site


def test_geom_category_shape():
    hBN_gr = bilayer(1.42, Atom[5, 7], Atom[6]) * (4, 5, 1)
    mid_layer = np.average(hBN_gr.xyz[:, 2])

    A_site = AtomFracSite(graphene())
    bottom = AtomXYZ(Cuboid([10000, 10000, mid_layer], [-100, -100, 0]))
    top = ~bottom

    bottom_A = A_site & bottom
    top_A = A_site & top

    cat = (bottom_A | top_A).categorize(hBN_gr)
    cnull = 0
    for i, c in enumerate(cat):
        if c == NullCategory():
            cnull += 1
        elif hBN_gr.xyz[i, 2] < mid_layer:
            assert c == bottom_A
        else:
            assert False
    assert cnull == len(hBN_gr) // 4 * 3


def test_geom_category_xyz_none():
    hBN_gr = bilayer(1.42, Atom[5, 7], Atom[6]) * (4, 5, 1)
    mid_layer = np.average(hBN_gr.xyz[:, 2])

    A_site = AtomFracSite(graphene())
    bottom = AtomXYZ(z=(None, mid_layer))
    top = AtomXYZ(z=(mid_layer, None))

    bottom_A = A_site & bottom
    top_A = A_site & top

    cat = (bottom_A | top_A).categorize(hBN_gr)
    cnull = 0
    for i, c in enumerate(cat):
        if c == NullCategory():
            cnull += 1
        elif hBN_gr.xyz[i, 2] < mid_layer:
            assert c == bottom_A
        else:
            assert False
    assert cnull == len(hBN_gr) // 4 * 3


def test_geom_category_xyz_meta():
    """
    We check that the metaclass defined for individual direction categories works.
    """
    hBN_gr = bilayer(1.42, Atom[5, 7], Atom[6]) * (4, 5, 1)
    sc2uc = hBN_gr.asc2uc

    # Check that all classes work
    for key in ("x", "y", "z", "f_x", "f_y", "f_z", "a_x", "a_y", "a_z"):
        name = key.replace("_", "")

        # Check that the attribute is present
        assert hasattr(AtomXYZ, name)

        # Get the category class
        cls = getattr(AtomXYZ, name)

        # Assert that using the class actually does the same effect as calling the AtomXYZ
        # category with the appropiate arguments
        def get_cls(op, v):
            return AtomXYZ(**{f"{key}_{op}": v})

        assert np.all(sc2uc(cls < 0.5) == sc2uc(cls(lt=0.5)))
        assert np.all(sc2uc(cls < 0.5) == sc2uc(get_cls("lt", 0.5)))
        assert np.all(sc2uc(cls <= 0.5) == sc2uc(cls(le=0.5)))
        assert np.all(sc2uc(cls <= 0.5) == sc2uc(get_cls("le", 0.5)))
        assert np.all(sc2uc(cls > 0.5) == sc2uc(cls(gt=0.5)))
        assert np.all(sc2uc(cls > 0.5) == sc2uc(get_cls("gt", 0.5)))
        assert np.all(sc2uc(cls >= 0.5) == sc2uc(cls(ge=0.5)))
        assert np.all(sc2uc(cls >= 0.5) == sc2uc(get_cls("ge", 0.5)))

        assert np.all(
            sc2uc(cls((-1, 1))) == sc2uc(get_cls("ge", -1) & get_cls("le", 1))
        )
        assert np.all(sc2uc(cls(-1, 1)) == sc2uc(get_cls("ge", -1) & get_cls("le", 1)))
