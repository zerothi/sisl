# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

from sisl import Atom
from sisl.geom import *

import math as m
import numpy as np


pytestmark = [pytest.mark.geom]


def test_basic():
    a = sc(2.52, Atom['Fe'])
    a = bcc(2.52, Atom['Fe'])
    a = bcc(2.52, Atom['Fe'], orthogonal=True)
    a = fcc(2.52, Atom['Au'])
    a = fcc(2.52, Atom['Au'], orthogonal=True)
    a = hcp(2.52, Atom['Au'])
    a = hcp(2.52, Atom['Au'], orthogonal=True)
    a = rocksalt(5.64, ['Na', 'Cl'])
    a = rocksalt(5.64, [Atom('Na', R=3), Atom('Cl', R=4)], orthogonal=True)


def test_flat():
    a = graphene()
    a = graphene(atoms='C')
    a = graphene(orthogonal=True)


def test_nanotube():
    a = nanotube(1.42)
    a = nanotube(1.42, chirality=(3, 5))
    a = nanotube(1.42, chirality=(6, -3))


def test_diamond():
    a = diamond()


def test_bilayer():
    a = bilayer(1.42)
    a = bilayer(1.42, stacking='AA')
    a = bilayer(1.42, stacking='BA')
    a = bilayer(1.42, stacking='AB')
    for m in range(7):
        a = bilayer(1.42, twist=(m, m + 1))
    a = bilayer(1.42, twist=(6, 7), layer='bottom')
    a = bilayer(1.42, twist=(6, 7), layer='TOP')
    a = bilayer(1.42, bottom_atoms=(Atom['B'], Atom['N']), twist=(6, 7))
    a = bilayer(1.42, top_atoms=(Atom(5), Atom(7)), twist=(6, 7))
    a, th = bilayer(1.42, twist=(6, 7), ret_angle=True)

    with pytest.raises(ValueError):
        bilayer(1.42, twist=(6, 7), layer='undefined')

    with pytest.raises(ValueError):
        bilayer(1.42, twist=(6, 7), stacking='undefined')

    with pytest.raises(ValueError):
        bilayer(1.42, twist=('str', 7), stacking='undefined')


def test_nanoribbon():
    for w in range(0, 5):
        a = nanoribbon(w, 1.42, Atom(6), kind='armchair')
        a = nanoribbon(w, 1.42, Atom(6), kind='zigzag')
        a = nanoribbon(w, 1.42, (Atom(5), Atom(7)), kind='armchair')
        a = nanoribbon(w, 1.42, (Atom(5), Atom(7)), kind='zigzag')

    with pytest.raises(ValueError):
        nanoribbon(6, 1.42, (Atom(5), Atom(7)), kind='undefined')

    with pytest.raises(ValueError):
        nanoribbon('str', 1.42, (Atom(5), Atom(7)), kind='undefined')


def test_graphene_nanoribbon():
    a = graphene_nanoribbon(5)


def test_agnr():
    a = agnr(5)


def test_zgnr():
    a = zgnr(5)


def test_fcc_slab():
    for o in [True, False]:
        fcc_slab(alat=4.08, atoms='Au', miller=(1, 0, 0), orthogonal=o)
        fcc_slab(4.08, 'Au', 100, orthogonal=o)
        fcc_slab(4.08, 'Au', 110, orthogonal=o)
        fcc_slab(4.08, 'Au', 111, orthogonal=o)
        fcc_slab(4.08, 79, '100', layers=5, vacuum=None, orthogonal=o)
        fcc_slab(4.08, 79, '110', layers=5, orthogonal=o)
        fcc_slab(4.08, 79, '111', layers=5, start=1, orthogonal=o)
        fcc_slab(4.08, 79, '111', layers=5, start='C', orthogonal=o)
        fcc_slab(4.08, 79, '111', layers=5, end=2, orthogonal=o)
        fcc_slab(4.08, 79, '111', layers=5, end='B', orthogonal=o)
    with pytest.raises(ValueError):
        fcc_slab(4.08, 'Au', 100, start=0, end=0)
    with pytest.raises(ValueError):
        fcc_slab(4.08, 'Au', 1000)
    with pytest.raises(NotImplementedError):
        fcc_slab(4.08, 'Au', 200)
    assert not np.allclose(
        fcc_slab(5.64, 'Au', 100, end=1, layers='BABAB').xyz,
        fcc_slab(5.64, 'Au', 100, end=1, layers=' BABAB ').xyz)
    assert np.allclose(
        fcc_slab(5.64, 'Au', 100, layers=' AB AB BA ', vacuum=2).xyz,
        fcc_slab(5.64, 'Au', 100, layers=(None, 2, 2, 2, None), vacuum=2, start=(0, 0, 1)).xyz)
    assert np.allclose(
        fcc_slab(5.64, 'Au', 100, layers=' AB AB BA ', vacuum=(2, 1)).xyz,
        fcc_slab(5.64, 'Au', 100, layers=(None, 2, ' ', 2, None, 2, None), vacuum=(2, 1), end=(1, 1, 0)).xyz)

    # example in documentation
    assert np.allclose(
        fcc_slab(4., 'Au', 100, layers=(' ', 3, 5, 3), start=(0, 1, 0), vacuum=(10, 1, 2)).xyz,
        fcc_slab(4., 'Au', 100, layers=' ABA BABAB ABA', vacuum=(10, 1, 2)).xyz)

    assert np.allclose(
        fcc_slab(4., 'Au', 100, layers=(' ', 3, 5, 3), start=(1, 0), vacuum=(10, 1, 2)).xyz,
        fcc_slab(4., 'Au', 100, layers=' BAB ABABA ABA', vacuum=(10, 1, 2)).xyz)


def test_bcc_slab():
    for o in [True, False]:
        bcc_slab(alat=4.08, atoms='Au', miller=(1, 0, 0), orthogonal=o)
        bcc_slab(4.08, 'Au', 100, orthogonal=o)
        bcc_slab(4.08, 'Au', 110, orthogonal=o)
        bcc_slab(4.08, 'Au', 111, orthogonal=o)
        bcc_slab(4.08, 79, '100', layers=5, vacuum=None, orthogonal=o)
        bcc_slab(4.08, 79, '110', layers=5, orthogonal=o)
        assert (bcc_slab(4.08, 79, '111', layers=5, start=1, orthogonal=o)
                .equal(
                    bcc_slab(4.08, 79, '111', layers="BCABC", orthogonal=o)
                ))
        bcc_slab(4.08, 79, '111', layers="BCABC", start='B', orthogonal=o)
        bcc_slab(4.08, 79, '111', layers="BCABC", start=1, orthogonal=o)
        bcc_slab(4.08, 79, '111', layers=5, start='C', orthogonal=o)
        bcc_slab(4.08, 79, '111', layers=5, end=2, orthogonal=o)
        bcc_slab(4.08, 79, '111', layers=5, end='B', orthogonal=o)
    with pytest.raises(ValueError):
        bcc_slab(4.08, 'Au', 100, start=0, end=0)
    with pytest.raises(ValueError):
        bcc_slab(4.08, 'Au', 1000)
    with pytest.raises(NotImplementedError):
        bcc_slab(4.08, 'Au', 200)
    assert not np.allclose(
        bcc_slab(5.64, 'Au', 100, end=1, layers='BABAB').xyz,
        bcc_slab(5.64, 'Au', 100, end=1, layers=' BABAB ').xyz)


def test_rocksalt_slab():
    rocksalt_slab(5.64, [Atom(11, R=3), Atom(17, R=4)], 100)
    assert (rocksalt_slab(5.64, ['Na', 'Cl'], 100, layers=5)
            .equal(
                rocksalt_slab(5.64, ['Na', 'Cl'], 100, layers="ABABA")
            ))
    rocksalt_slab(5.64, ['Na', 'Cl'], 110, vacuum=None)
    rocksalt_slab(5.64, ['Na', 'Cl'], 111, orthogonal=False)
    rocksalt_slab(5.64, ['Na', 'Cl'], 111, orthogonal=True)
    rocksalt_slab(5.64, 'Na', 100)
    with pytest.raises(ValueError):
        rocksalt_slab(5.64, ['Na', 'Cl'], 100, start=0, end=0)
    with pytest.raises(ValueError):
        rocksalt_slab(5.64, ['Na', 'Cl'], 1000)
    with pytest.raises(NotImplementedError):
        rocksalt_slab(5.64, ['Na', 'Cl'], 200)
    with pytest.raises(ValueError):
        rocksalt_slab(5.64, ['Na', 'Cl'], 100, start=0, layers='BABAB')
    rocksalt_slab(5.64, ['Na', 'Cl'], 100, start=1, layers='BABAB')
    with pytest.raises(ValueError):
        rocksalt_slab(5.64, ['Na', 'Cl'], 100, end=0, layers='BABAB')
    rocksalt_slab(5.64, ['Na', 'Cl'], 100, end=1, layers='BABAB')
    assert not np.allclose(
        rocksalt_slab(5.64, ['Na', 'Cl'], 100, end=1, layers='BABAB').xyz,
        rocksalt_slab(5.64, ['Na', 'Cl'], 100, end=1, layers=' BABAB ').xyz)
