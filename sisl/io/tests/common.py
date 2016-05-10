"""Tests suite for XVSile
"""
from __future__ import print_function, division

from nose.tools import *

from tempfile import mkstemp, mkdtemp

from sisl import Geometry, Atom, SuperCell

import os.path as osp
import math as m
import numpy as np


def setUp(self):
    # Create temporary folder
    self.d = mkdtemp()
    alat = 1.42
    sq3h = 3.**.5 * 0.5
    C = Atom(Z=6, orbs=1)
    sc = SuperCell(np.array([[1.5, sq3h, 0.],
                             [1.5, -sq3h, 0.],
                             [0., 0., 10.]], np.float64) * alat,
                   nsc=[3, 3, 1])
    self.g = Geometry(np.array([[0., 0., 0.],
                                [1., 0., 0.]], np.float64) * alat,
                      atoms=C, sc=sc)


def tearDown(self):
    # Do each removal separately
    try:
        shutil.rmtree(self.d)
    except:
        pass
