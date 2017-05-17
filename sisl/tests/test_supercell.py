from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np
import scipy.linalg as sli

from sisl import SuperCell, SuperCellChild


class TestSuperCell(object):

    def setUp(self):
        alat = 1.42
        sq3h = 3.**.5 * 0.5
        self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * alat, nsc=[3, 3, 1])

    def tearDown(self):
        del self.sc

    def test_repr(self):
        print(self.sc)
        assert_false(self.sc == 'Not a SuperCell')

    def test_nsc1(self):
        sc = self.sc.copy()
        sc.set_nsc([5, 5, 0])
        assert_true(np.allclose([5, 5, 1], sc.nsc))
        assert_true(len(sc.sc_off) == np.prod(sc.nsc))

    def test_nsc2(self):
        sc = self.sc.copy()
        sc.set_nsc([0, 1, 0])
        assert_true(np.allclose([1, 1, 1], sc.nsc))
        assert_true(len(sc.sc_off) == np.prod(sc.nsc))
        sc.set_nsc(a=3)
        assert_true(np.allclose([3, 1, 1], sc.nsc))
        assert_true(len(sc.sc_off) == np.prod(sc.nsc))
        sc.set_nsc(b=3)
        assert_true(np.allclose([3, 3, 1], sc.nsc))
        assert_true(len(sc.sc_off) == np.prod(sc.nsc))
        sc.set_nsc(c=5)
        assert_true(np.allclose([3, 3, 5], sc.nsc))
        assert_true(len(sc.sc_off) == np.prod(sc.nsc))

    def test_nsc3(self):
        assert_raises(ValueError, self.sc.set_nsc, a=2)
        assert_raises(ValueError, self.sc.set_nsc, b=2)
        assert_raises(ValueError, self.sc.set_nsc, c=2)
        assert_raises(ValueError, self.sc.set_nsc, [1, 2, 3])

    def test_nsc4(self):
        assert_true(self.sc.sc_index([0, 0, 0]) == 0)

    def test_fill(self):
        sc = self.sc.swapaxes(1, 2)
        i = sc._fill([1, 1])
        assert_true(i.dtype == np.int32)
        i = sc._fill([1., 1.])
        assert_true(i.dtype == np.float64)
        for dt in [np.int32, np.int64, np.float32, np.float64, np.complex64]:
            i = sc._fill([1., 1.], dt)
            assert_true(i.dtype == dt)
            i = sc._fill(np.ones([2], dt))
            assert_true(i.dtype == dt)

    def test_add_vacuum1(self):
        sc = self.sc.copy()
        for i in range(3):
            sc.add_vacuum(10, i)
            ax = self.sc.cell[i, :]
            ax += ax / np.sum(ax ** 2) ** .5 * 10
            print(ax, sc.cell[i, :])
            assert_true(np.allclose(ax, sc.cell[i, :]))

    def test_rotation1(self):
        rot = self.sc.rotate(180, [0, 0, 1])
        rot.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.cell, self.sc.cell))

        rot = self.sc.rotate(m.pi, [0, 0, 1], radians=True)
        rot.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.cell, self.sc.cell))

        rot = rot.rotate(180, [0, 0, 1])
        rot.cell[2, 2] *= -1
        assert_true(np.allclose(rot.cell, self.sc.cell))

    def test_rotation2(self):
        rot = self.sc.rotatec(180)
        rot.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.cell, self.sc.cell))

        rot = self.sc.rotatec(m.pi, radians=True)
        rot.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.cell, self.sc.cell))

        rot = rot.rotatec(180)
        rot.cell[2, 2] *= -1
        assert_true(np.allclose(rot.cell, self.sc.cell))

    def test_rotation3(self):
        rot = self.sc.rotatea(180)
        assert_true(np.allclose(rot.cell[0, :], self.sc.cell[0, :]))
        assert_true(np.allclose(-rot.cell[2, 2], self.sc.cell[2, 2]))

        rot = self.sc.rotateb(m.pi, radians=True)
        assert_true(np.allclose(rot.cell[1, :], self.sc.cell[1, :]))
        assert_true(np.allclose(-rot.cell[2, 2], self.sc.cell[2, 2]))

    def test_swapaxes1(self):
        sab = self.sc.swapaxes(0, 1)
        assert_true(np.allclose(sab.cell[0, :], self.sc.cell[1, :]))
        assert_true(np.allclose(sab.cell[1, :], self.sc.cell[0, :]))

    def test_swapaxes2(self):
        sab = self.sc.swapaxes(0, 2)
        assert_true(np.allclose(sab.cell[0, :], self.sc.cell[2, :]))
        assert_true(np.allclose(sab.cell[2, :], self.sc.cell[0, :]))

    def test_swapaxes3(self):
        sab = self.sc.swapaxes(1, 2)
        assert_true(np.allclose(sab.cell[1, :], self.sc.cell[2, :]))
        assert_true(np.allclose(sab.cell[2, :], self.sc.cell[1, :]))

    def test_offset1(self):
        off = self.sc.offset()
        assert_true(np.allclose(off, [0, 0, 0]))
        off = self.sc.offset([1, 1, 1])
        cell = self.sc.cell[:, :]
        assert_true(np.allclose(off, cell[0, :] + cell[1, :] + cell[2, :]))

    def test_sc_index1(self):
        sc_index = self.sc.sc_index([0, 0, 0])
        assert_equal(sc_index, 0)
        assert_raises(Exception, self.sc.sc_index, [100, 100, 100])
        sc_index = self.sc.sc_index([0, 0, None])
        assert_equal(len(sc_index), self.sc.nsc[2])

    def test_cut1(self):
        cut = self.sc.cut(2, 0)
        assert_true(np.allclose(cut.cell[0, :] * 2, self.sc.cell[0, :]))
        assert_true(np.allclose(cut.cell[1, :], self.sc.cell[1, :]))

    def test_creation(self):
        # full cell
        tmp1 = SuperCell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # diagonal cell
        tmp2 = SuperCell([1, 1, 1])
        # cell parameters
        tmp3 = SuperCell([1, 1, 1, 90, 90, 90])
        tmp4 = SuperCell([1])
        assert_true(np.allclose(tmp1.cell, tmp2.cell))
        assert_true(np.allclose(tmp1.cell, tmp3.cell))
        assert_true(np.allclose(tmp1.cell, tmp4.cell))

    def test_creation2(self):
        # full cell
        class P(SuperCellChild):
            pass
        tmp1 = P()
        tmp1.set_supercell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # diagonal cell
        tmp2 = P()
        tmp2.set_supercell([1, 1, 1])
        # cell parameters
        tmp3 = P()
        tmp3.set_supercell([1, 1, 1, 90, 90, 90])
        tmp4 = P()
        tmp4.set_supercell([1])
        assert_true(np.allclose(tmp1.cell, tmp2.cell))
        assert_true(np.allclose(tmp1.cell, tmp3.cell))
        assert_true(np.allclose(tmp1.cell, tmp4.cell))
        assert_true(len(tmp1._fill([0, 0, 0])) == 3)
        assert_true(len(tmp1._fill_sc([0, 0, 0])) == 3)
        assert_true(tmp1.is_orthogonal())
        for i in range(3):
            tmp2.add_vacuum(10, i)
            assert_true(tmp1.cell[i, i] + 10 == tmp2.cell[i, i])

    def test_creation3(self):
        assert_raises(ValueError, self.sc.tocell, [3, 6])
        assert_raises(ValueError, self.sc.tocell, [3, 4, 5, 6])
        assert_raises(ValueError, self.sc.tocell, [3, 4, 5, 6, 7])
        assert_raises(ValueError, self.sc.tocell, [3, 4, 5, 6, 7, 6, 7])

    def test_rcell(self):
        # LAPACK inverse algorithm implicitly does
        # a transpose.
        rcell = sli.inv(self.sc.cell) * 2. * np.pi
        assert_true(np.allclose(rcell.T, self.sc.rcell))

    def test_translate1(self):
        sc = self.sc.translate([0, 0, 10])
        assert_true(np.allclose(sc.cell[2, :2], self.sc.cell[2, :2]))
        assert_true(np.allclose(sc.cell[2, 2], self.sc.cell[2, 2]+10))

    def test_center1(self):
        assert_true(np.allclose(self.sc.center(), np.sum(self.sc.cell, axis=0) / 2))
        for i in [0, 1, 2]:
            assert_true(np.allclose(self.sc.center(i), self.sc.cell[i, :] / 2))

    def test_pickle(self):
        import pickle as p
        s = p.dumps(self.sc)
        n = p.loads(s)
        assert_true(self.sc == n)
        assert_false(self.sc != n)
        s = SuperCell([1, 1, 1])
        assert_false(self.sc == s)

    def test_orthogonal(self):
        assert_false(self.sc.is_orthogonal())
