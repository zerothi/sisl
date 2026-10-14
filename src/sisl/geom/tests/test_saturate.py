# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl import Atom, Geometry, PeriodicTable
from sisl.geom import *
from sisl.geom._category import base as cat_base

pytestmark = [pytest.mark.geom]

TETRA = np.degrees(np.arccos(-1 / 3))


def _angles(xyz, center, neighbors):
    u = xyz[neighbors] - xyz[center]
    u /= np.linalg.norm(u, axis=1).reshape(-1, 1)
    cos = np.clip(u @ u.T, -1, 1)
    return np.degrees(np.arccos(cos[np.triu_indices(len(u), 1)]))


def _bonded(geom, ia, R):
    idx = geom.close(ia, R=(0.01, R))[-1]
    return geom.asc2uc(idx), geom.axyz(idx)


@pytest.mark.parametrize("ribbon", [zgnr(4), agnr(5)])
def test_saturate_ribbon_edges(ribbon):
    edge = AtomNeighbors(max=2, R=1.5)
    n_edge = edge.mask(ribbon).sum()
    sat = saturate(ribbon, [(edge, Saturate("H", bond=1.09, R=1.5))])
    assert sat.na == ribbon.na + n_edge
    assert np.all(sat.atoms.Z[ribbon.na :] == 1)
    # all carbon atoms are now 3-fold coordinated
    assert np.all(AtomNeighbors(3, R=1.5).mask(sat)[: ribbon.na])
    for ia in range(ribbon.na, sat.na):
        idx, xyz = _bonded(sat, ia, 1.5)
        assert len(idx) == 1
        assert np.linalg.norm(xyz[0] - sat.xyz[ia]) == pytest.approx(1.09)
        # the hydrogen is in the plane and bisects the C-C bonds
        ic = idx[0]
        _, cxyz = _bonded(ribbon, ic, 1.5)
        u = cxyz - ribbon.xyz[ic]
        u /= np.linalg.norm(u, axis=1).reshape(-1, 1)
        d = -u.sum(0)
        d /= np.linalg.norm(d)
        assert np.allclose(sat.xyz[ia] - sat.xyz[ic], 1.09 * d)


def test_saturate_first_rule_wins():
    ribbon = zgnr(4)
    edge = AtomNeighbors(max=2, R=1.5)
    sat = saturate(
        ribbon,
        [
            (edge & AtomOdd(), Saturate("F", bond=1.35, R=1.5)),
            (edge, Saturate("H", bond=1.09, R=1.5)),
        ],
    )
    odd_edge = np.flatnonzero(edge.mask(ribbon) & AtomOdd().mask(ribbon))
    assert sorted(sat.atoms.Z[ribbon.na :]) == sorted([9] * len(odd_edge) + [1])


def test_saturate_shares_neighbor_search(monkeypatch):
    ribbon = zgnr(4)
    calls = []
    find_pairs = cat_base._find_pairs

    def counted(*args, **kwargs):
        calls.append(1)
        return find_pairs(*args, **kwargs)

    monkeypatch.setattr(cat_base, "_find_pairs", counted)
    edge = AtomNeighbors(max=2, R=1.5)
    saturate(
        ribbon,
        [
            (edge & AtomOdd(), Saturate("F", R=1.5)),
            (edge, Saturate("H", R=1.5)),
        ],
    )
    assert len(calls) == 1


def test_saturate_defaults():
    ribbon = zgnr(4)
    # graphene atoms have orbitals with R=1.4342, coordination 3 is the most frequent
    sat = saturate(ribbon, [({"Z": "C", "neighbors": 2}, "H")])
    assert sat.na == ribbon.na + 2
    pt = PeriodicTable()
    idx, xyz = _bonded(sat, ribbon.na, 1.5)
    assert np.linalg.norm(xyz[0] - sat.xyz[ribbon.na]) == pytest.approx(
        pt.radius(6) + pt.radius(1)
    )


def test_saturate_index_selector_and_coordination():
    ribbon = zgnr(4)
    edge = np.flatnonzero(AtomNeighbors(max=2, R=1.5).mask(ribbon))
    sat = saturate(ribbon, [(edge[0], Saturate("H", bond=1.0, R=1.5))])
    assert sat.na == ribbon.na + 1
    # coordination 4 adds two atoms in tetrahedral directions
    sat = saturate(ribbon, [(edge[0], Saturate("H", bond=1.0, coordination=4, R=1.5))])
    assert sat.na == ribbon.na + 2
    idx, _ = _bonded(sat, edge[0], 1.5)
    assert len(idx) == 4
    angles = _angles(sat.xyz, edge[0], [ribbon.na, ribbon.na + 1])
    assert angles == pytest.approx(TETRA)


def _chain(angle):
    """C1-C2-C3 with C1 at the origin and C2-C3 at `angle` to C1-C2"""
    a = np.radians(180 - angle)
    xyz = [[0, 0, 0], [1.54, 0, 0], [1.54 + 1.54 * np.cos(a), 1.54 * np.sin(a), 0]]
    return Geometry(xyz, Atom(6, R=1.6), lattice=[20, 20, 20])


def test_saturate_sp3_one_bond():
    chain = _chain(TETRA)
    sat = saturate(chain, [(0, Saturate("H", bond=1.09, coordination=4))])
    assert sat.na == 6
    new = [3, 4, 5]
    assert _angles(sat.xyz, 0, [1, *new]) == pytest.approx(TETRA)
    assert np.allclose(np.linalg.norm(sat.xyz[new] - sat.xyz[0], axis=1), 1.09)
    # staggered: dihedral H-C1-C2-C3 is 60, 180 or 300 degrees
    b = sat.xyz[1] - sat.xyz[0]
    b /= np.linalg.norm(b)

    def proj(v):
        return v - v.dot(b) * b

    ref = proj(sat.xyz[2] - sat.xyz[1])
    for ih in new:
        h = proj(sat.xyz[ih] - sat.xyz[0])
        cos = h.dot(ref) / np.linalg.norm(h) / np.linalg.norm(ref)
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))) in (
            pytest.approx(60),
            pytest.approx(180),
        )


def test_saturate_sp3_two_bonds():
    chain = _chain(TETRA)
    sat = saturate(chain, [(1, Saturate("H", bond=1.09, coordination=4))])
    assert sat.na == 5
    assert _angles(sat.xyz, 1, [0, 2, 3, 4]) == pytest.approx(TETRA)


def test_saturate_sp2_one_bond():
    chain = _chain(120)
    sat = saturate(chain, [(0, Saturate("H", bond=1.09, coordination=3))])
    assert sat.na == 5
    assert _angles(sat.xyz, 0, [1, 3, 4]) == pytest.approx(120)
    # planar with the neighbor bond
    assert np.allclose(sat.xyz[:, 2], 0)


def test_saturate_full_and_unsupported():
    chain = _chain(TETRA)
    # already saturated
    sat = saturate(chain, [(1, Saturate("H", coordination=2))])
    assert sat.na == chain.na
    with pytest.raises(ValueError, match="coordination"):
        saturate(chain, [(0, Saturate("H", coordination=6))])


def test_saturate_no_orbitals_and_isolated():
    geom = Geometry([[0, 0, 0], [1.4, 0, 0]], Atom(6), lattice=[20, 20, 20])
    with pytest.raises(ValueError, match="R argument"):
        saturate(geom, [(0, "H")])
    geom = Geometry([[0, 0, 0], [5, 0, 0]], Atom(6, R=1.6), lattice=[20, 20, 20])
    with pytest.warns(sisl.SislWarning, match="no bonds"):
        sat = saturate(geom, [(0, Saturate("H", coordination=4))])
    assert sat.na == 2
