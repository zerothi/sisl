# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl import Atom, Geometry
from sisl.geom import *
from sisl.geom._category import base as cat_base

pytestmark = [pytest.mark.geom, pytest.mark.category, pytest.mark.geom_category]


def _close_count(geom, R, neighbor=None):
    """Reference: neighbor count of each atom with `Geometry.close`"""
    n = []
    for ia in range(geom.na):
        Ria = R(geom.atoms[ia])
        idx = geom.close(ia, R=Ria)
        if np.ndim(Ria) > 0:
            idx = idx[-1]
        idx = geom.asc2uc(idx)
        if neighbor is not None:
            idx = idx[neighbor[idx]]
        n.append(len(idx))
    return np.array(n)


def _hBN():
    hBN = honeycomb(1.42, [Atom(5, R=1.43), Atom(7, R=1.6)]) * (4, 5, 1)
    return hBN.remove([0, 7, 8])


R_cases = [
    (None, lambda atom: (0.01, atom.maxR())),
    (1.43, lambda atom: (0.01, 1.43)),
    ((1.0, 1.5), lambda atom: (1.0, 1.5)),
    ((1.0, 1.5, 2.6), lambda atom: (1.0, 1.5, 2.6)),
    # a single radius counts the atom itself
    ((lambda atom: 1.5),) * 2,
    ((lambda atom: (0.01, 1.3 + atom.Z / 50)),) * 2,
]


@pytest.mark.parametrize("R, Rref", R_cases)
@pytest.mark.parametrize("nsc", [[3, 3, 1], [1, 1, 1], [1, 3, 1]])
def test_neighbors_mask_equals_close(R, Rref, nsc):
    geom = _hBN()
    geom.set_nsc(nsc)
    n = _close_count(geom, Rref)
    for nmin, nmax in [(0, 0), (2, 2), (3, 3), (1, 3), (4, 100)]:
        cat = AtomNeighbors(nmin, nmax, R=R)
        assert np.array_equal(cat.mask(geom), (nmin <= n) & (n <= nmax))


def test_neighbors_mask_neighbor_category():
    geom = _hBN()
    N = AtomZ(7)
    n = _close_count(geom, lambda atom: (0.01, 1.5), neighbor=N.mask(geom))
    cat = AtomNeighbors(2, neighbor=N, R=1.5)
    assert np.array_equal(cat.mask(geom), n == 2)
    # nested neighbor categories
    n2 = AtomNeighbors(2, R=1.5)
    n3 = AtomNeighbors(3, R=1.5)
    inner = AtomNeighbors(1, 3, neighbor=n2, R=1.5) & n3
    nn = _close_count(geom, lambda atom: (0.01, 1.5), neighbor=n2.mask(geom))
    ref = (1 <= nn) & (nn <= 3) & n3.mask(geom)
    assert np.array_equal(inner.mask(geom), ref)


def test_neighbors_mask_non_periodic_nsc():
    # close searches the supercells of nsc, also along a non-periodic direction
    geom = sisl.geom.graphene() * (3, 3, 1)
    geom.lattice.pbc = [True, False, False]
    geom.set_nsc([3, 3, 1])
    n = _close_count(geom, lambda atom: (0.01, 1.5))
    assert np.array_equal(AtomNeighbors(3, R=1.5).mask(geom), n == 3)


def test_neighbors_mask_outside_unit_cell():
    geom = _hBN()
    geom.xyz[2] += geom.cell[0] * 2 - geom.cell[1]
    geom.xyz[5] -= geom.cell[1] * 0.5
    geom.xyz[6, 2] += 10.0
    for nsc in ([3, 3, 1], [5, 5, 3], [1, 1, 1]):
        geom.set_nsc(nsc)
        n = _close_count(geom, lambda atom: (0.01, 1.5))
        for k in range(4):
            assert np.array_equal(AtomNeighbors(k, R=1.5).mask(geom), n == k)


def test_neighbors_mask_large_R():
    # R larger than the cell, the finder tiles the cell
    geom = sisl.geom.graphene()
    geom.set_nsc([7, 7, 1])
    n = _close_count(geom, lambda atom: (0.01, 4.0))
    for k in np.unique(n):
        assert np.array_equal(AtomNeighbors(int(k), R=4.0).mask(geom), n == k)


def test_neighbors_categorize_subset():
    geom = _hBN()
    cat = AtomNeighbors(2, R=1.5)
    mask = cat.mask(geom)
    sub = [3, 1, 10]
    assert np.array_equal(cat.mask(geom, sub), mask[sub])
    ret = cat.categorize(geom, sub)
    assert [not isinstance(c, NullCategory) for c in ret] == mask[sub].tolist()
    assert (cat.categorize(geom, 1) == cat) == mask[1]


def test_mask_composites_equal_categorize():
    geom = _hBN()
    B = AtomZ(5)
    n2 = AtomNeighbors(2, R=1.5)
    for cat in (B & n2, B | n2, B ^ n2, ~(B & n2), ~B | AtomOdd(), AtomEven() & ~n2):
        ref = [not isinstance(c, NullCategory) for c in cat.categorize(geom)]
        assert cat.mask(geom).tolist() == ref


def test_mask_leaves_equal_categorize():
    geom = _hBN()
    geom.atoms.replace(geom.atoms[0], Atom(5, R=1.43, tag="B1"))
    cats = [
        AtomZ([5, "N"]),
        AtomTag("B1"),
        AtomTag("N"),
        AtomEven(),
        AtomOdd(),
        AtomFracSite(sisl.geom.graphene(1.42)),
        AtomXYZ(x=(None, 3.0)),
        AtomXYZ.fy > 0.5,
        AtomIndex(1, 4, 5),
        AtomSeq("2-4"),
    ]
    for cat in cats:
        ref = [not isinstance(c, NullCategory) for c in cat.categorize(geom)]
        assert cat.mask(geom).tolist() == ref, cat


def test_mask_custom_category_fallback():
    # a user category with only categorize works in masks and composites
    class AtomFirst(AtomCategory, key="test_first"):
        def categorize(self, geometry, atoms=None):
            idx = (
                geometry._sanitize_atoms(atoms)
                if atoms is not None
                else range(geometry.na)
            )
            return [self if ia == 0 else NullCategory() for ia in idx]

    geom = _hBN()
    mask = (AtomFirst() | AtomZ(7)).mask(geom)
    assert mask[0]
    assert np.array_equal(mask[1:], geom.atoms.Z[1:] == 7)


def test_mask_vectorized_one_neighbor_search(monkeypatch):
    # performance regression: one neighbor search for a composite, no per-atom close
    geom = _hBN()
    calls = []
    find_pairs = cat_base._find_pairs

    def counted(*args, **kwargs):
        calls.append(1)
        return find_pairs(*args, **kwargs)

    def no_close(*args, **kwargs):
        raise AssertionError("Geometry.close must not be called")

    monkeypatch.setattr(cat_base, "_find_pairs", counted)
    monkeypatch.setattr(Geometry, "close", no_close)
    cat = (AtomNeighbors(2, R=1.5) & AtomZ(5)) | AtomNeighbors(
        3, neighbor=AtomNeighbors(2, R=1.5), R=1.5
    )
    geom._sanitize_atoms(cat)
    assert len(calls) == 1
