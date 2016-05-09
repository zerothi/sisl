from __future__ import print_function, division

from nose.tools import *

from sids import Atom
from sids.geom import *

import math as m
import numpy as np


class Test(object):

    def test_basis(self):
        a = bcc(2.52, Atom['Fe'])
        a = fcc(2.52, Atom['Au'])

    def test_flat(self):
        a = graphene()
        a = graphene(square=True)

    def test_diamond(self):
        a = diamond()
