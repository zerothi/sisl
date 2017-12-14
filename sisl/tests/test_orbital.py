from __future__ import print_function, division

import pytest

import math as m
import numpy as np
from scipy.interpolate import interp1d

from sisl.orbital import Orbital, SphericalOrbital, AtomicOrbital


@pytest.mark.orbital
class Test_orbital(object):

    def test_init1(self):
        assert Orbital(1.) == Orbital(1.)
        assert Orbital(1., 'none') != Orbital(1.)

    def test_basic1(self):
        orb = Orbital(1.)
        repr(orb)
        orb = Orbital(1., 'none')
        repr(orb)
        assert orb == orb.copy()

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_radial1(self):
        Orbital(1.).radial(np.arange(10))

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_phi1(self):
        Orbital(1.).phi(np.arange(10))

    def test_scale1(self):
        o = Orbital(1.)
        assert o.scale(2).R == 2.

    def test_pickle1(self):
        import pickle as p
        o0 = Orbital(1.)
        o1 = Orbital(1., 'none')
        p0 = p.dumps(o0)
        p1 = p.dumps(o1)
        l0 = p.loads(p0)
        l1 = p.loads(p1)
        assert o0 == l0
        assert o1 == l1
        assert o0 != l1
        assert o1 != l0


@pytest.mark.orbital
class Test_sphericalorbital(object):

    def test_init1(self):
        n = 6
        rf = np.arange(n)
        rf = (rf, rf)
        assert SphericalOrbital(1, rf) == SphericalOrbital(1, rf)
        f = interp1d(rf[0], rf[1], fill_value=(0., 0.), bounds_error=False, kind='cubic')
        rf = [rf[0], rf[0]]
        assert SphericalOrbital(1, rf) == SphericalOrbital(1, f)
        assert SphericalOrbital(1, rf, tag='none') != SphericalOrbital(1, rf)
        SphericalOrbital(5, rf)
        for l in range(10):
            o = SphericalOrbital(l, rf)
            assert l == o.l
            assert o == o.copy()

    def test_basic1(self):
        n = 6
        rf = np.arange(n)
        rf = (rf, rf)
        orb = SphericalOrbital(1, rf)
        repr(orb)
        orb = SphericalOrbital(1, rf, tag='none')
        repr(orb)

    @pytest.mark.xfail(raises=ValueError)
    def test_set_radial1(self):
        n = 6
        rf = np.arange(n)
        rf = (rf, rf)
        o = SphericalOrbital(1, rf)
        o.set_radial(1.)

    def test_radial1(self):
        n = 6
        rf = np.arange(n)
        rf = (rf, rf)
        orb0 = SphericalOrbital(0, rf)
        orb1 = SphericalOrbital(1, rf)
        r = np.linspace(0, n, 400)
        # Note r > n - 1 should be zero, regardless of the fill-value
        r0 = orb0.radial(r)
        r1 = orb1.radial(r)
        rr = np.stack((r, np.zeros(len(r)), np.zeros(len(r))), axis=1)
        r2 = orb1.radial(rr, is_radius=False)
        assert np.allclose(r0, r1)
        assert np.allclose(r0, r2)
        r[r >= n - 1] = 0.
        assert np.allclose(r0, r)
        assert np.allclose(r1, r)

    def test_phi1(self):
        n = 6
        rf = np.arange(n)
        rf = (rf, rf)
        orb0 = SphericalOrbital(0, rf)
        orb1 = SphericalOrbital(1, rf)
        r = np.linspace(0, n, 333 * 3).reshape(-1, 3)
        p0 = orb0.phi(r)
        p1 = orb1.phi(r)
        assert not np.allclose(p0, p1)

    def test_same1(self):
        n = 6
        rf = np.arange(n)
        rf = (rf, rf)
        o0 = SphericalOrbital(0, rf)
        o1 = Orbital(5.)
        assert o0.equal(o1)
        assert not o0.equal(Orbital(3.))

    def test_toatomicorbital1(self):
        n = 6
        rf = np.arange(n)
        rf = (rf, rf)
        orb = SphericalOrbital(0, rf)
        ao = orb.toAtomicOrbital()
        assert len(ao) == 1
        assert ao[0].l == orb.l
        assert ao[0].m == 0

        # Check m and l
        for l in range(1, 5):
            orb = SphericalOrbital(l, rf)
            ao = orb.toAtomicOrbital()
            assert len(ao) == 2*l + 1
            m = -l
            for a in ao:
                assert a.l == orb.l
                assert a.m == m
                m += 1

        orb = SphericalOrbital(1, rf)
        ao = orb.toAtomicOrbital(1)
        assert ao.l == orb.l
        assert ao.m == 1
        ao = orb.toAtomicOrbital(-1)
        assert ao.l == orb.l
        assert ao.m == -1
        ao = orb.toAtomicOrbital(0)
        assert ao.l == orb.l
        assert ao.m == 0
        ao = orb.toAtomicOrbital([0, -1, 1])
        for a in ao:
            assert a.l == orb.l
        assert ao[0].m == 0
        assert ao[1].m == -1
        assert ao[2].m == 1

    @pytest.mark.xfail(raises=ValueError)
    def test_toatomicorbital2(self):
        n = 6
        rf = np.arange(n)
        rf = (rf, rf)
        orb = SphericalOrbital(1, rf)
        ao = orb.toAtomicOrbital(2)

    def test_pickle1(self):
        n = 6
        rf = np.arange(n)
        rf = (rf, rf)
        import pickle as p
        o0 = SphericalOrbital(1, rf)
        o1 = SphericalOrbital(2, rf)
        p0 = p.dumps(o0)
        p1 = p.dumps(o1)
        l0 = p.loads(p0)
        l1 = p.loads(p1)
        assert o0 == l0
        assert o1 == l1
        assert o0 != l1
        assert o1 != l0
