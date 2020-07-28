import pytest

from sisl import Atom, PeriodicTable
from sisl.geom import *


pytestmark = [pytest.mark.geom, pytest.mark.category, pytest.mark.geom_category]


def test_geom_category():
    hBN = honeycomb(1.42, [Atom(5, R=1.43), Atom(7, R=1.43)]) * (10, 11, 1)

    B = AtomZ(5)
    B2 = AtomNeighbours(2, neigh_cat=B)
    N = AtomZ(7)
    N2 = AtomNeighbours(2, neigh_cat=N)
    B3 = AtomNeighbours(3, neigh_cat=B)
    N3 = AtomNeighbours(3, neigh_cat=N)

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
    N2 = AtomNeighbours(2, neigh_cat=N, R=(0.01, 1.43))
    PT = PeriodicTable()
    B3 = AtomNeighbours(3, neigh_cat=B, R=lambda atom: (0.01, PT.radius(atom.Z)))
    N3 = AtomNeighbours(3, neigh_cat=N, R=1.43)

    n2 = AtomNeighbours(2, R=1.43)

    category = (B & B2) ^ (N & N2) ^ (B & B3) ^ (N & N3) ^ n2

    cat = category.categorize(hBN)
