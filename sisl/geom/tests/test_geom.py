from __future__ import print_function, division

from nose.tools import *

from sisl import Atom
from sisl.geom import *

import math as m
import numpy as np


class Test(object):

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
