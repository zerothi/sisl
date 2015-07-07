from __future__ import print_function, division

from nose.tools import *

from sids.geom import Geometry, Atom

import math as m
import numpy as np


class TestGeometry(object):
    # Base test class for MaskedArrays.

    def setUp(self):
        alat = 1.42
        sq3h  = 3.**.5 * 0.5
        C = Atom(Z=6,R=alat * 1.01,orbs=2)
        self.g = Geometry(cell=np.array([[1.5, sq3h,  0.],
                                         [1.5,-sq3h,  0.],
                                         [ 0.,   0., 10.]],np.float) * alat,
                          xyz=np.array([[ 0., 0., 0.],
                                        [ 1., 0., 0.]],np.float) * alat,
                          atoms = C, nsc = [3,3,1])

    def tearDown(self):
        del self.g

    def test_tile1(self):
        cell = np.copy(self.g.cell)
        cell[0,:] *= 2
        t = self.g.tile(2,0)
        assert_true( np.allclose(cell,t.cell) )
        cell[1,:] *= 2
        t = t.tile(2,1)
        assert_true( np.allclose(cell,t.cell) )
        cell[2,:] *= 2
        t = t.tile(2,2)
        assert_true( np.allclose(cell,t.cell) )

    def test_tile2(self):
        cell = np.copy(self.g.cell)
        cell[:,:] *= 2
        t = self.g.tile(2,0).tile(2,1).tile(2,2)
        assert_true( np.allclose(cell,t.cell) )

    def test_repeat1(self):
        cell = np.copy(self.g.cell)
        cell[0,:] *= 2
        t = self.g.repeat(2,0)
        assert_true( np.allclose(cell,t.cell) )
        cell[1,:] *= 2
        t = t.repeat(2,1)
        assert_true( np.allclose(cell,t.cell) )
        cell[2,:] *= 2
        t = t.repeat(2,2)
        assert_true( np.allclose(cell,t.cell) )

    def test_repeat2(self):
        cell = np.copy(self.g.cell)
        cell[:,:] *= 2
        t = self.g.repeat(2,0).repeat(2,1).repeat(2,2)
        assert_true( np.allclose(cell,t.cell) )

        
    def test_a2o1(self):
        assert_true( 0 == self.g.a2o(0) )
        assert_true( self.g.atoms[0].orbs == self.g.a2o(1) )
        assert_true( self.g.no == self.g.a2o(self.g.na) )


    def test_nsc1(self):
        nsc = np.copy(self.g.nsc)
        self.g.set_supercell([5,5,0])
        assert_true( np.allclose([5,5,1],self.g.nsc) )
        assert_true( len(self.g.isc_off) == np.prod(self.g.nsc) )

    def test_nsc2(self):
        nsc = np.copy(self.g.nsc)
        self.g.set_supercell([0,1,0])
        assert_true( np.allclose([1,1,1],self.g.nsc) )
        assert_true( len(self.g.isc_off) == np.prod(self.g.nsc) )

    def test_rotation1(self):
        rot = self.g.rotate(m.pi,[0,0,1])
        rot.cell[2,2] *= -1
        assert_true( np.allclose(-rot.cell,self.g.cell) )
        assert_true( np.allclose(-rot.xyz,self.g.xyz) )

        rot = rot.rotate(m.pi,[0,0,1])
        rot.cell[2,2] *= -1
        assert_true( np.allclose(rot.cell,self.g.cell) )
        assert_true( np.allclose(rot.xyz,self.g.xyz) )

    def test_rotation2(self):
        rot = self.g.rotate(m.pi,[0,0,1],only='cell')
        rot.cell[2,2] *= -1
        assert_true( np.allclose(-rot.cell,self.g.cell) )
        assert_true( np.allclose(rot.xyz,self.g.xyz) )

        rot = rot.rotate(m.pi,[0,0,1],only='cell')
        rot.cell[2,2] *= -1
        assert_true( np.allclose(rot.cell,self.g.cell) )
        assert_true( np.allclose(rot.xyz,self.g.xyz) )

    def test_rotation3(self):
        rot = self.g.rotate(m.pi,[0,0,1],only='xyz')
        assert_true( np.allclose(rot.cell,self.g.cell) )
        assert_true( np.allclose(-rot.xyz,self.g.xyz) )

        rot = rot.rotate(m.pi,[0,0,1],only='xyz')
        assert_true( np.allclose(rot.cell,self.g.cell) )
        assert_true( np.allclose(rot.xyz,self.g.xyz) )

