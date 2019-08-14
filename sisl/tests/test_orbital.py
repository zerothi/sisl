from __future__ import print_function, division

import pytest

import math as m
import numpy as np
from scipy import interpolate as interp

from sisl.utils.mathematics import cart2spher, spher2cart
from sisl.orbital import Orbital, SphericalOrbital, AtomicOrbital


def r_f(n):
    r = np.arange(n)
    return r, r


@pytest.mark.orbital
def test_spherical():
    rad2 = np.pi / 45
    r, theta, phi = np.ogrid[0.1:10:0.2, -np.pi:np.pi:rad2, 0:np.pi:rad2]
    xyz = spher2cart(r, theta, phi)
    s = xyz.shape[:-1]
    r1, theta1, phi1 = cart2spher(xyz)
    r1.shape = s
    theta1.shape = s
    phi1.shape = s
    assert np.allclose(r, r1)
    assert np.allclose(theta, theta1[1:2, :, 1:2])
    assert np.allclose(phi, phi1[0:1, 0:1, :])


@pytest.mark.orbital
class Test_orbital(object):

    def test_init1(self):
        assert Orbital(1.) == Orbital(1.)
        assert Orbital(1., tag='none') != Orbital(1.)
        assert Orbital(1., 1.) != Orbital(1.)
        assert Orbital(1., 1.) != Orbital(1., 1., tag='none')

    def test_basic1(self):
        orb = Orbital(1.)
        str(orb)
        orb = Orbital(1., tag='none')
        str(orb)
        orb = Orbital(1., 1., tag='none')
        str(orb)
        assert orb == orb.copy()
        assert orb != 1.

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_radial1(self):
        Orbital(1.).radial(np.arange(10))

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_psi1(self):
        Orbital(1.).psi(np.arange(10))

    def test_scale1(self):
        o = Orbital(1.)
        assert o.scale(2).R == 2.
        o = Orbital(-1)
        assert o.scale(2).R == -1.

    def test_pickle1(self):
        import pickle as p
        o0 = Orbital(1.)
        o1 = Orbital(1., tag='none')
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
        f = interp.interp1d(rf[0], rf[1], fill_value=(0., 0.), bounds_error=False, kind='cubic')
        rf = [rf[0], rf[0]]
        assert SphericalOrbital(1, rf) == SphericalOrbital(1, f)
        assert SphericalOrbital(1, rf, tag='none') != SphericalOrbital(1, rf)
        SphericalOrbital(5, rf)
        for l in range(10):
            o = SphericalOrbital(l, rf)
            assert l == o.l
            assert o == o.copy()

    def test_basic1(self):
        rf = r_f(6)
        orb = SphericalOrbital(1, rf)
        str(orb)
        orb = SphericalOrbital(1, rf, tag='none')
        str(orb)

    @pytest.mark.xfail(raises=ValueError)
    def test_set_radial1(self):
        rf = r_f(6)
        o = SphericalOrbital(1, rf)
        o.set_radial(1.)

    def test_radial1(self):
        rf = r_f(6)
        orb0 = SphericalOrbital(0, rf)
        orb1 = SphericalOrbital(1, rf)
        r = np.linspace(0, 6, 400)
        # Note r > n - 1 should be zero, regardless of the fill-value
        r0 = orb0.radial(r)
        r1 = orb1.radial(r)
        rr = np.stack((r, np.zeros(len(r)), np.zeros(len(r))), axis=1)
        r2 = orb1.radial(rr, is_radius=False)
        assert np.allclose(r0, r1)
        assert np.allclose(r0, r2)
        r[r >= rf[0].max()] = 0.
        assert np.allclose(r0, r)
        assert np.allclose(r1, r)

    def test_psi1(self):
        rf = r_f(6)
        orb0 = SphericalOrbital(0, rf)
        orb1 = SphericalOrbital(1, rf)
        r = np.linspace(0, 6, 333 * 3).reshape(-1, 3)
        p0 = orb0.psi(r)
        p1 = orb1.psi(r)
        assert not np.allclose(p0, p1)
        orb0 = SphericalOrbital(1, rf)
        assert orb0.equal(orb1, radial=True, psi=True)

        for m in range(orb0.l, orb0.l + 1):
            p0 = orb0.psi(r, -1)
            p1 = orb1.psi(r, -1)
            assert np.allclose(p0, p1)
        p0 = orb0.psi(r, -1)
        p1 = orb1.psi(r, 1)
        assert not np.allclose(p0, p1)

    def test_radial_func1(self):
        r = np.linspace(0, 4, 300)
        f = np.exp(-r)
        o = SphericalOrbital(1, (r, f))
        str(o)
        def i_univariate(r, f):
            return interp.UnivariateSpline(r, f, k=5, s=0, ext=1, check_finite=False)
        def i_interp1d(r, f):
            return interp.interp1d(r, f, kind='cubic', fill_value=(f[0], 0.), bounds_error=False)
        def i_spline(r, f):
            from functools import partial
            tck = interp.splrep(r, f, k=5, s=0)
            return partial(interp.splev, tck=tck, der=0, ext=1)

        # Interpolation radius
        R = np.linspace(0, 5, 400)

        assert np.allclose(o.f(r), f)
        f_default = o.f(R)

        o.set_radial(r, f, interp=i_univariate)
        assert np.allclose(o.f(r), f)
        f_univariate = o.f(R)
        o.set_radial(r, f, interp=i_interp1d)
        assert np.allclose(o.f(r), f)
        f_interp1d = o.f(R)

        o.set_radial(r, f, interp=i_spline)
        assert np.allclose(o.f(r), f)
        f_spline = o.f(R)

        # Checks that they are equal
        assert np.allclose(f_univariate, f_interp1d)
        assert np.allclose(f_univariate, f_spline)
        assert np.allclose(f_univariate, f_default)

    def test_same1(self):
        rf = r_f(6)
        o0 = SphericalOrbital(0, rf)
        o1 = Orbital(5.0)
        assert o0.equal(o1)
        assert not o0.equal(Orbital(3.))

    def test_toatomicorbital1(self):
        rf = r_f(6)
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
        rf = r_f(6)
        orb = SphericalOrbital(1, rf)
        ao = orb.toAtomicOrbital(2)

    def test_toatomicorbital_q0(self):
        rf = r_f(6)
        orb = SphericalOrbital(0, rf, 2.)
        ao = orb.toAtomicOrbital()
        assert len(ao) == 1
        assert ao[0].q0 == 2.

        # Check m and l
        for l in range(1, 5):
            orb = SphericalOrbital(l, rf, 2.)
            ao = orb.toAtomicOrbital()
            assert ao[0].q0 == pytest.approx(2. / (2*l+1))

    def test_pickle1(self):
        rf = r_f(6)
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

    def test_togrid1(self):
        o = SphericalOrbital(1, r_f(6))
        o.toGrid()
        o.toGrid(R=10)

    @pytest.mark.xfail(raises=ValueError)
    def test_togrid2(self):
        o = SphericalOrbital(1, r_f(6))
        o.toGrid(R=-1)


@pytest.mark.orbital
class Test_atomicorbital(object):

    def test_init1(self):
        rf = r_f(6)
        a = []
        a.append(AtomicOrbital(2, 1, 0, 1, True, rf))
        a.append(AtomicOrbital(l=1, m=0, Z=1, P=True, spherical=rf))
        f = interp.interp1d(rf[0], rf[1], fill_value=(0., 0.), bounds_error=False, kind='cubic')
        a.append(AtomicOrbital(l=1, m=0, Z=1, P=True, spherical=f))
        a.append(AtomicOrbital('pzP', f))
        a.append(AtomicOrbital('pzP', rf))
        a.append(AtomicOrbital('2pzP', rf))
        for i in range(len(a) - 1):
            for j in range(i+1, len(a)):
                assert a[i] == a[j] and a[i].equal(a[j], psi=True, radial=True)

    def test_init2(self):
        assert AtomicOrbital('pzP') == AtomicOrbital(n=2, l=1, m=0, P=True)

    def test_init3(self):
        rf = r_f(6)
        for l in range(5):
            a = AtomicOrbital(l=l, m=0, spherical=rf)
            a.name()
            a.name(True)
            str(a)
            a = AtomicOrbital(l=l, m=0, P=True, spherical=rf, tag='hello')
            a.name()
            a.name(True)
            str(a)

    def test_init4(self):
        rf = r_f(6)
        o1 = AtomicOrbital(2, 1, 0, 1, True, rf)
        o2 = AtomicOrbital('pzP', rf)
        o3 = AtomicOrbital('pzZP', rf)
        o4 = AtomicOrbital('pzZ1P', rf)
        o5 = AtomicOrbital('2pzZ1P', rf)
        assert o1 == o2
        assert o1 == o3
        assert o1 == o4
        assert o1 == o5

    @pytest.mark.xfail(raises=ValueError)
    def test_init5(self):
        AtomicOrbital(5, 5, 0)

    def test_radial1(self):
        rf = r_f(6)
        r = np.linspace(0, 6, 100)
        for l in range(5):
            so = SphericalOrbital(l, rf)
            sor = so.radial(r)
            for m in range(-l, l+1):
                o = AtomicOrbital(l=l, m=m, spherical=rf)
                assert np.allclose(sor, o.radial(r))
                o.set_radial(rf[0], rf[1])
                assert np.allclose(sor, o.radial(r))

    def test_phi1(self):
        rf = r_f(6)
        r = np.linspace(0, 6, 999).reshape(-1, 3)
        for l in range(5):
            so = SphericalOrbital(l, rf)
            for m in range(-l, l+1):
                o = AtomicOrbital(l=l, m=m, spherical=rf)
                assert np.allclose(so.psi(r, m), o.psi(r))

    def test_pickle1(self):
        import pickle as p
        rf = r_f(6)
        o0 = AtomicOrbital(2, 1, 0, 1, True, rf, tag='hello', q0=1.)
        o1 = AtomicOrbital(l=1, m=0, Z=1, P=False, spherical=rf)
        p0 = p.dumps(o0)
        p1 = p.dumps(o1)
        l0 = p.loads(p0)
        l1 = p.loads(p1)
        assert o0 == l0
        assert o1 == l1
        assert o0 != l1
        assert o1 != l0
