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
        a = nanoribbon(1.42, Atom(6), w, kind='armchair')
        a = nanoribbon(1.42, Atom(6), w, kind='zigzag')
        a = nanoribbon(1.42, (Atom(5), Atom(7)), w, kind='armchair')
        a = nanoribbon(1.42, (Atom(5), Atom(7)), w, kind='zigzag')

    with pytest.raises(ValueError):
        nanoribbon(1.42, (Atom(5), Atom(7)), 6, kind='undefined')

    with pytest.raises(ValueError):
        nanoribbon(1.42, (Atom(5), Atom(7)), 'str', kind='undefined')


def test_graphene_nanoribbon():
    a = graphene_nanoribbon(5)


def test_agnr():
    a = agnr(5)


def test_zgnr():
    a = zgnr(5)
