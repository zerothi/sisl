import pytest

from sisl import Atom
from sisl.geom import *

import math as m
import numpy as np


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
    a = graphene(atom='C')
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
    a = bilayer(1.42, bottom_atom=(Atom['B'], Atom['N']), twist=(6, 7))
    a = bilayer(1.42, top_atom=(Atom(5), Atom(7)), twist=(6, 7))
    a, th = bilayer(1.42, twist=(6, 7), ret_angle=True)

def test_nanoribbon():
    for w in range(0, 7):
        a = nanoribbon(w, kind='armchair')
        a = nanoribbon(w, kind='zigzag')
    a = nanoribbon(w, bond=2.0)
    a = nanoribbon(w, sat_bond=1.3, sat_atom=Atom(3))
    a = nanoribbon(w, kind='Z', sat_bond=2.0)
    a = nanoribbon(w, kind='Z', atom=Atom(5), sat_bond=-1.0)
    a = nanoribbon(w, kind='a', atom=(Atom(5), Atom(7)))
    a = nanoribbon(w, kind='z', atom=(Atom(5), Atom(7)))
