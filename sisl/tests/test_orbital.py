from __future__ import print_function, division

import pytest

import math as m
import numpy as np

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

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_radial1(self):
        Orbital(1.).radial(np.arange(10))

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_phi1(self):
        Orbital(1.).phi(np.arange(10))


@pytest.mark.orbital
class Test_sphericalorbital(object):

    def test_init1(self):
        n = 6
        assert SphericalOrbital(1, np.arange(n), np.arange(n)) == \
            SphericalOrbital(1, np.arange(n), np.arange(n))
        assert SphericalOrbital(1, np.arange(n), np.arange(n), tag='none') != \
            SphericalOrbital(1, np.arange(n), np.arange(n))
        SphericalOrbital(5, np.arange(n), np.arange(n))
        for l in range(10):
            o = SphericalOrbital(l, np.arange(n), np.arange(n))
            assert l == o.l

    def test_basic1(self):
        n = 6
        orb = SphericalOrbital(1, np.arange(n), np.arange(n))
        repr(orb)
        orb = SphericalOrbital(1, np.arange(n), np.arange(n), tag='none')
        repr(orb)

    def test_radial1(self):
        n = 6
        orb0 = SphericalOrbital(0, np.arange(n), np.arange(n))
        orb1 = SphericalOrbital(1, np.arange(n), np.arange(n))
        r = np.linspace(0, n, 400)
        # Note r > n - 1 should be zero, regardless of the fill-value
        r0 = orb0.radial(r)
        r1 = orb1.radial(r)
        assert np.allclose(r0, r1)
        r[r >= n - 1] = 0.
        assert np.allclose(r0, r)
        assert np.allclose(r1, r)

    def test_phi1(self):
        n = 6
        orb0 = SphericalOrbital(0, np.arange(n), np.arange(n))
        orb1 = SphericalOrbital(1, np.arange(n), np.arange(n))
        r = np.linspace(0, n, 333 * 3).reshape(-1, 3)
        p0 = orb0.phi(r)
        p1 = orb1.phi(r)
        assert not np.allclose(p0, p1)

    def test_toatomicorbital1(self):
        n = 6
        orb = SphericalOrbital(0, np.arange(n), np.arange(n))
        ao = orb.toAtomicOrbital()
        assert len(ao) == 1
        assert ao[0].l == orb.l
        assert ao[0].m == 0

        # Check m and l
        for l in range(1, 5):
            orb = SphericalOrbital(l, np.arange(n), np.arange(n))
            ao = orb.toAtomicOrbital()
            assert len(ao) == 2*l + 1
            m = -l
            for a in ao:
                assert a.l == orb.l
                assert a.m == m
                m += 1

        orb = SphericalOrbital(1, np.arange(n), np.arange(n))
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
        orb = SphericalOrbital(1, np.arange(n), np.arange(n))
        ao = orb.toAtomicOrbital(2)
