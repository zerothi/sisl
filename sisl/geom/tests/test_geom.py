import pytest

from sisl import Atom
from sisl.geom import *

import math as m
import numpy as np


class Test:

    def test_basis(self):
        a = sc(2.52, Atom['Fe'])
        a = bcc(2.52, Atom['Fe'])
        a = bcc(2.52, Atom['Fe'], orthogonal=True)
        a = fcc(2.52, Atom['Au'])
        a = fcc(2.52, Atom['Au'], orthogonal=True)
        a = hcp(2.52, Atom['Au'])
        a = hcp(2.52, Atom['Au'], orthogonal=True)

    def test_flat(self):
        a = graphene()
        a = graphene(atom='C')
        a = graphene(orthogonal=True)

    def test_nanotube(self):
        a = nanotube(1.42)
        a = nanotube(1.42, chirality=(3, 5))
        a = nanotube(1.42, chirality=(6, -3))

    def test_diamond(self):
        a = diamond()

    def test_bilayer(self):
        a = bilayer(1.42)
        a = bilayer(1.42, stacking='AA')
        for m in range(7):
            a = bilayer(1.42, twist=(m, m + 1))
        a = bilayer(1.42, twist=(6, 7), layer='bottom')
        a = bilayer(1.42, twist=(6, 7), layer='TOP')
        a = bilayer(1.42, bottom_atom=(Atom['B'], Atom['N']), twist=(6, 7))
        a = bilayer(1.42, top_atom=(Atom(5), Atom(7)), twist=(6, 7))
        a, th = bilayer(1.42, twist=(6, 7), ret_angle=True)
