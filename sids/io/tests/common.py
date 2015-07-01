"""Tests suite for XVSile
"""
from __future__ import print_function, division

from nose.tools import *

from tempfile import mkstemp, mkdtemp

from sids.geom import Geometry, Atom

import os.path as osp
import math as m
import numpy as np

def setUp(self):
    # Create temporary folder
    self.d = mkdtemp()
    alat = 1.42
    sq3h  = 3.**.5 * 0.5
    C = Atom(Z=6,orbs=1)
    self.g = Geometry(cell=np.array([[1.5, sq3h,  0.],
                                     [1.5,-sq3h,  0.],
                                     [ 0.,   0., 10.]],np.float) * alat,
                      xyz=np.array([[ 0., 0., 0.],
                                    [ 1., 0., 0.]],np.float) * alat,
                      atoms = C, nsc = [3,3,1])

def tearDown(self):
    # Do each removal separately
    try: 
        shutil.rmtree(self.d)
    except:
        pass
