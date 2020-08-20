import pytest

import numpy as np

from sisl import Atom, PeriodicTable, Cuboid
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
    B2 = AtomNeighbours(2, neigh_cat=B)
    N = AtomZ(7)
    N2 = AtomNeighbours(2, neigh_cat=N)
    B3 = AtomNeighbours(3, neigh_cat=B)
    N3 = AtomNeighbours(3, neigh_cat=N)

    assert B != N

    n2 = AtomNeighbours(2)
    n3 = AtomNeighbours(3)
    n3n2 = AtomNeighbours(1, 3, neigh_cat=n2) & n3
    n3n3n2 = AtomNeighbours(1, 3, neigh_cat=n3n2) & n3

    category = (B & B2) ^ (N & N2) ^ (B & B3) ^ (N & N3) ^ n2

    cat = category.categorize(hBN)


def test_geom_category_no_r():
    hBN = honeycomb(1.42, Atom[5, 7]) * (10, 11, 1)

    B = AtomZ(5)
    B2 = AtomNeighbours(2, neigh_cat=B, R=1.43)
    N = AtomZ(7)
    N2 = AtomNeighbours(min=2, max=2, neigh_cat=N, R=(0.01, 1.43))
    PT = PeriodicTable()
    B3 = AtomNeighbours(3, neigh_cat=B, R=lambda atom: (0.01, PT.radius(atom.Z)))
    N3 = AtomNeighbours(3, neigh_cat=N, R=1.43)
    Nabove3 = AtomNeighbours(min=3, neigh_cat=N, R=1.43)

    assert B != N2
    assert N2 != N3

    n2 = AtomNeighbours(2, R=1.43)

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
    B_site = AtomFracSite(graphene(), foffset=(-1/3, -1/3, 0))

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
    sc2uc = hBN_gr.sc2uc

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

        assert np.all(sc2uc(cls((-1, 1))) == sc2uc(get_cls("ge", -1) & get_cls("le", 1)))
        assert np.all(sc2uc(cls(-1, 1)) == sc2uc(get_cls("ge", -1) & get_cls("le", 1)))
