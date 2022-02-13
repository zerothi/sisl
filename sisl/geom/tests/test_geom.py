# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

from sisl import Atom
from sisl.geom import *

import math as m
import numpy as np


pytestmark = [pytest.mark.geom]


def test_basis():
    a = sc(2.52, Atom['Fe'])
    a = bcc(2.52, Atom['Fe'])
    a = bcc(2.52, Atom['Fe'], orthogonal=True)
    a = fcc(2.52, Atom['Au'])
    a = fcc(2.52, Atom['Au'], orthogonal=True)
    a = hcp(2.52, Atom['Au'])
    a = hcp(2.52, Atom['Au'], orthogonal=True)


def test_flat():
    a = graphene()
    a = graphene(atoms='C')
    a = graphene(orthogonal=True)


def test_nanotube():
    a = nanotube(1.42)
    a = nanotube(1.42, chirality=(3, 5))
    a = nanotube(1.42, chirality=(6, -3))


def test_diamond():
    a = diamond()


def test_bilayer():
    a = bilayer(1.42)
    a = bilayer(1.42, stacking='AA')
    a = bilayer(1.42, stacking='BA')
    a = bilayer(1.42, stacking='AB')
    for m in range(7):
        a = bilayer(1.42, twist=(m, m + 1))
    a = bilayer(1.42, twist=(6, 7), layer='bottom')
    a = bilayer(1.42, twist=(6, 7), layer='TOP')
    a = bilayer(1.42, bottom_atoms=(Atom['B'], Atom['N']), twist=(6, 7))
    a = bilayer(1.42, top_atoms=(Atom(5), Atom(7)), twist=(6, 7))
    a, th = bilayer(1.42, twist=(6, 7), ret_angle=True)

    with pytest.raises(ValueError):
        bilayer(1.42, twist=(6, 7), layer='undefined')

    with pytest.raises(ValueError):
        bilayer(1.42, twist=(6, 7), stacking='undefined')

    with pytest.raises(ValueError):
        bilayer(1.42, twist=('str', 7), stacking='undefined')


def test_nanoribbon():
    for w in range(0, 5):
        a = nanoribbon(w, 1.42, Atom(6), kind='armchair')
        a = nanoribbon(w, 1.42, Atom(6), kind='zigzag')
        a = nanoribbon(w, 1.42, (Atom(5), Atom(7)), kind='armchair')
        a = nanoribbon(w, 1.42, (Atom(5), Atom(7)), kind='zigzag')

    with pytest.raises(ValueError):
        nanoribbon(6, 1.42, (Atom(5), Atom(7)), kind='undefined')

    with pytest.raises(ValueError):
        nanoribbon('str', 1.42, (Atom(5), Atom(7)), kind='undefined')


def test_graphene_nanoribbon():
    a = graphene_nanoribbon(5)


def test_agnr():
    a = agnr(5)


def test_zgnr():
    a = zgnr(5)


def test_fcc100():
    g = fcc100(4.08, 79, (3, 4, 5))
    assert g.nsc[2] == 3
    g = fcc100(4.08, 'Au', (3, 4, 5), vacuum=10)
    g = fcc100(alat=4.08, atoms='Au', size=(3, 4, 5))
    g = fcc100(4.08, Atom(79, R=5), (1, 1, 5), vacuum=10)
    assert g.nsc[0] == 5
    assert g.nsc[1] == 5
    assert g.nsc[2] == 1


def test_fcc110():
    g = fcc110(4.08, 79, (3, 4, 5))
    assert g.nsc[2] == 3
    g = fcc110(4.08, 'Au', (3, 4, 5), vacuum=10)
    g = fcc110(alat=4.08, atoms='Au', size=(3, 4, 5))
    g = fcc110(4.08, Atom(79, R=5), (1, 1, 5), vacuum=10)
    assert g.nsc[0] == 3
    assert g.nsc[1] == 5
    assert g.nsc[2] == 1


def test_fcc111():
    g = fcc111(4.08, 79, (3, 4, 5))
    assert g.nsc[2] == 3
    g = fcc111(4.08, 'Au', (3, 4, 5), vacuum=10)
    g = fcc111(alat=4.08, atoms='Au', size=(3, 4, 5))
    g = fcc111(4.08, Atom(79, R=5), (1, 1, 5), vacuum=10)
    assert g.nsc[0] == 5
    assert g.nsc[1] == 5
    assert g.nsc[2] == 1
    g = fcc111(4.08, Atom(79, R=5), (1, 1, 5), vacuum=10, orthogonal=True)
    assert g.nsc[0] == 5
    assert g.nsc[1] == 3
    assert g.nsc[2] == 1
